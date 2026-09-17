"""The ten analyzers of spec 5, one output schema, and the certificate wiring.

Every analyzer reads only the observed data contract of spec 3 (`y`, `z`, `dose`,
`version`, `switch`, `jitter`, `C_clean`, `C_react`, `W`, and for the one forbidden arm
`belief_logged`).  Hidden truth is touched in exactly one place, `_result`, and only to
compute `corr` and `rmse` of the analyzer's own person scores against `world.theta`; no
estimator sees it.  `belief_partial` is the only reader of `belief_logged`, which
`tests/test_analyzers.py` enforces by handing every other analyzer a world that raises on
that attribute.

Occasions are 0-indexed: `t < switch[i]` is the anchor epoch (policy version 0) and
`t >= switch[i]` is the adaptive phase.  `switch` may vary across persons, so every
occasion average here is a masked mean.

`run_analyzer(name, world, bootstrap=0)` returns one schema for all ten:

    var_theta_hat  float or None   the analyzer's Var(theta) on the pi_0 scale
    ci             (lo, hi) or None
    psi_hat        float or None
    beta_clean_hat float           OLS slope of C_clean on the person scores (truth lam_C)
    scores         [N] ndarray
    corr, rmse     float           against hidden theta, metrics only
    battery        dict or None    the naive validation battery (spec 5.1)
    certificate    dict or None    {"A": {...}, "B": {...}} for the three certified arms
    extras         dict            everything else, JSON-serialisable
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats

from .certificate import RULES, certify
from .cfa import bartlett_scores, fit_longitudinal_invariance, fit_one_factor
from .shadow import fit_shadow, iv_shadow, person_mean
from .validation import validation_battery
from .world import surrogate_g1, surrogate_recency

__all__ = ["ANALYZERS", "run_analyzer", "exposure_statistics"]


# ------------------------------------------------------------------------- helpers
def _masks(world) -> tuple[np.ndarray, np.ndarray]:
    """(adaptive, anchor) occasion masks [N, T] from `switch`."""
    occ = np.arange(np.asarray(world.y).shape[1])
    adaptive = occ[None, :] >= np.asarray(world.switch, int)[:, None]
    return adaptive, ~adaptive


def _mean_over(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of `a` [N, T, ...] over the occasions selected by `mask` [N, T].

    A person with no selected occasion gets 0 rather than NaN, so nothing downstream
    blows up; that case is reported to the certificate as `has_anchor` False and the
    estimate is refused there rather than silently used.
    """
    a = np.asarray(a, float)
    m = np.asarray(mask, bool).astype(float)
    while m.ndim < a.ndim:
        m = m[..., None]
    denom = m.sum(axis=1)
    return np.where(denom > 0, (a * m).sum(axis=1) / np.where(denom > 0, denom, 1.0), 0.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(r, two-sided p) with n - 2 degrees of freedom; (0, 1) when either is constant."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        return 0.0, 1.0
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0, 1.0
    r = float(np.corrcoef(a, b)[0, 1])
    r = float(np.clip(r, -1.0 + 1e-15, 1.0 - 1e-15))
    t = r * np.sqrt((n - 2) / (1.0 - r * r))
    return r, float(2.0 * stats.t.sf(abs(t), n - 2))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, NaN when either series is constant.

    The rule statistics use `_pearson`, which reports 0 and p 1 in that case so a rule
    stays silent; a reported metric says NaN instead, because a correlation with a
    constant is undefined and the zero-person cell (Var(theta) = 0) is exactly where the
    difference would otherwise be read as a measurement.
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
        return float("nan")
    return _pearson(a, b)[0]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of `y` on `x` with an intercept."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    vx = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    if vx < 1e-12:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / vx)


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of `y` after an OLS regression on an intercept and `x`."""
    y = np.asarray(y, float).ravel()
    x = np.asarray(x, float).ravel()
    X = np.column_stack([np.ones(x.size), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def _score_error_var(fit) -> float:
    """Sampling variance of a Bartlett score: `1 / sum_j (lambda_j^2 / theta_jj)`."""
    lam = np.asarray(fit.loadings, float)
    th = np.asarray(fit.resid_var, float)
    return float(1.0 / np.sum(lam * lam / th))


def _fit_indices(fit) -> dict:
    return {"chi2": float(fit.chi2), "df": int(fit.df), "pvalue": float(fit.pvalue),
            "cfi": float(fit.cfi), "tli": float(fit.tli), "rmsea": float(fit.rmsea),
            "srmr": float(fit.srmr), "omega": float(fit.omega), "alpha": float(fit.alpha),
            "ave": float(fit.ave), "converged": bool(fit.converged),
            "std_loadings": [float(v) for v in fit.std_loadings],
            "loadings": [float(v) for v in fit.loadings],
            "resid_var": [float(v) for v in fit.resid_var]}


def _result(world, scores, var_theta_hat, ci=None, psi_hat=None, battery=None,
            certificate=None, extras=None) -> dict:
    """The common schema.  The only place hidden truth is read, and only for metrics."""
    scores = np.asarray(scores, float).ravel()
    theta = np.asarray(world.theta, float).ravel()
    return {
        "var_theta_hat": None if var_theta_hat is None else float(var_theta_hat),
        "ci": None if ci is None else (float(ci[0]), float(ci[1])),
        "psi_hat": None if psi_hat is None else float(psi_hat),
        "beta_clean_hat": _slope(scores, world.C_clean),
        "scores": scores,
        "corr": _corr(scores, theta),
        "rmse": float(np.sqrt(np.mean((scores - theta) ** 2))),
        "battery": battery,
        "certificate": certificate,
        "extras": {} if extras is None else extras,
    }


def _delta_approx_ci(estimate: float, n: int, score_error_var: float) -> tuple[float, float]:
    """The anchor arm's interval: SE of a variance under normality, inflated by the
    score error that the estimate has already subtracted off."""
    se = float(np.sqrt(2.0 / max(n - 1, 1)) * (estimate + score_error_var))
    return (float(estimate - 1.96 * se), float(estimate + 1.96 * se))


# ------------------------------------------------------- the exposure-pattern statistic
def exposure_statistics(resid: np.ndarray, adaptive: np.ndarray, x: np.ndarray,
                        windows: int = 3) -> dict:
    """Spec 11 A2: does the shadow-model residual still track the person's exposure?

    Each person's adaptive occasions are split into `windows` equal blocks by rank within
    the person; the window mean residual is correlated over persons with that person's
    exposure level `x` (`gbar_i`).  Returns the per-window correlations, the largest in
    absolute value and the smallest two-sided p.  When `x` has no variance the statistic
    is undefined by construction and comes back as corr 0, p 1, so rule R4 cannot fire.
    """
    resid = np.asarray(resid, float)
    adaptive = np.asarray(adaptive, bool)
    x = np.asarray(x, float).ravel()
    n_i = adaptive.sum(axis=1)
    rank = np.cumsum(adaptive, axis=1) - 1
    k = np.floor(rank * windows / np.maximum(n_i, 1)[:, None]).astype(int)
    k = np.clip(k, 0, windows - 1)

    corrs, ps = [], []
    for w in range(windows):
        sel = adaptive & (k == w)
        cnt = sel.sum(axis=1)
        block = np.where(cnt > 0, (resid * sel).sum(axis=1) / np.maximum(cnt, 1), np.nan)
        r, p = _pearson(block, x)
        corrs.append(float(r))
        ps.append(float(p))
    return {"exposure_corr": corrs,
            "max_exposure_corr": float(np.max(np.abs(corrs))) if corrs else 0.0,
            "exposure_p": float(np.min(ps)) if ps else 1.0}


def _implied_mean_resid_var(n: float, rho_d: float, var_d: float, var_e: float) -> float:
    """Variance of a mean of `n` model residuals: `var_e/n + var_d/n^2 sum_st rho^|s-t|`."""
    m = max(int(round(n)), 1)
    lags = np.arange(1, m)
    double_sum = float(m + 2.0 * np.sum((m - lags) * rho_d ** lags)) if m > 1 else 1.0
    n = max(float(n), 1.0)
    return float(var_e / n + var_d / (n * n) * double_sum)


# ------------------------------------------------------------------ simple analyzers
def _naive_like(world, mask, battery: bool) -> dict:
    X = _mean_over(world.y, mask)
    fit = fit_one_factor(X)
    scores = bartlett_scores(X, fit)
    extras = _fit_indices(fit)
    extras["score_error_var"] = _score_error_var(fit)
    return _result(world, scores, fit.var_theta,
                   battery=validation_battery(world, fit, scores) if battery else None,
                   extras=extras)


def analyzer_naive(world, bootstrap: int = 0) -> dict:
    """Conventional SEM on the deployed (adaptive) phase, with the full battery."""
    adaptive, _ = _masks(world)
    return _naive_like(world, adaptive, battery=True)


def analyzer_naive_pooled(world, bootstrap: int = 0) -> dict:
    """The same over all T occasions."""
    return _naive_like(world, np.ones(np.asarray(world.y).shape[:2], bool), battery=False)


def analyzer_epoch_fe(world, bootstrap: int = 0) -> dict:
    """S8 competitor: occasion fixed effects, then pooled conventional SEM."""
    y = np.asarray(world.y, float)
    y_fe = y - y.mean(axis=0, keepdims=True)
    X = y_fe.mean(axis=1)
    fit = fit_one_factor(X)
    scores = bartlett_scores(X, fit)
    extras = _fit_indices(fit)
    extras["score_error_var"] = _score_error_var(fit)
    return _result(world, scores, fit.var_theta, extras=extras)


def _partial_out(world, covariate: np.ndarray) -> dict:
    adaptive, _ = _masks(world)
    X = _mean_over(world.y, adaptive)
    fit = fit_one_factor(X)
    scores = bartlett_scores(X, fit)
    resid = _residualize(scores, covariate)
    sev = _score_error_var(fit)
    var_hat = float(np.var(resid, ddof=1) - sev)
    extras = _fit_indices(fit)
    extras.update({"score_error_var": sev,
                   "var_resid_scores": float(np.var(resid, ddof=1)),
                   "r_covariate_scores": _corr(covariate, scores)})
    return _result(world, resid, var_hat, extras=extras)


def analyzer_belief_partial(world, bootstrap: int = 0) -> dict:
    """The forbidden rule (F13): residualize on the policy's own logged belief."""
    adaptive, _ = _masks(world)
    return _partial_out(world, person_mean(np.asarray(world.belief_logged, float), adaptive))


def analyzer_dose_partial(world, bootstrap: int = 0) -> dict:
    """The synthesis's original rule K2: residualize on the mean logged dose."""
    adaptive, _ = _masks(world)
    return _partial_out(world, person_mean(np.asarray(world.dose, float), adaptive))


def analyzer_detrend(world, bootstrap: int = 0) -> dict:
    """S7 competent analyst: occasion FE, then a person-specific linear detrend of the
    adaptive phase, read off at the switch."""
    y = np.asarray(world.y, float)
    n, T, p = y.shape
    switch = np.asarray(world.switch, int)
    adaptive, _ = _masks(world)
    y_fe = y - y.mean(axis=0, keepdims=True)

    w = adaptive.astype(float)
    n_i = np.maximum(w.sum(axis=1), 1.0)
    x = (np.arange(T)[None, :] - switch[:, None]).astype(float)
    xbar = (x * w).sum(axis=1) / n_i
    ybar = (y_fe * w[:, :, None]).sum(axis=1) / n_i[:, None]
    xc = (x - xbar[:, None]) * w
    sxx = np.maximum((xc * xc).sum(axis=1), 1e-12)
    sxy = np.einsum("nt,ntp->np", xc, y_fe * w[:, :, None])
    slope = sxy / sxx[:, None]
    level = ybar - slope * xbar[:, None]          # the fitted value at t = switch_i

    fit = fit_one_factor(level)
    scores = bartlett_scores(level, fit)
    extras = _fit_indices(fit)
    extras["mean_slope"] = [float(v) for v in slope.mean(axis=0)]
    return _result(world, scores, fit.var_theta, extras=extras)


def analyzer_anchor_only(world, bootstrap: int = 0) -> dict:
    """D1 minimal: conventional SEM on the anchor epoch alone."""
    _, anchor = _masks(world)
    X = _mean_over(world.y, anchor)
    fit = fit_one_factor(X)
    scores = bartlett_scores(X, fit)
    sev = _score_error_var(fit)
    ci = _delta_approx_ci(fit.var_theta, X.shape[0], sev)
    extras = _fit_indices(fit)
    extras.update({"score_error_var": sev, "ci_method": "delta_approx",
                   "nbar_anchor": float(anchor.sum(axis=1).mean())})
    return _result(world, scores, fit.var_theta, ci=ci, extras=extras)


# --------------------------------------------------------------- the agentSEM arm
def _anchor_pipeline(y: np.ndarray, anchor: np.ndarray):
    """(a) anchor CFA on anchor person means, (b) Bartlett scores of every occasion.

    The occasion-level residual variances are the means-scale ones times the mean anchor
    length.  That rescaling leaves the Bartlett weights untouched (they are invariant to
    a common factor in Theta) and is kept because it is what the calibration claims.
    """
    X_anchor = _mean_over(y, anchor)
    afit = fit_one_factor(X_anchor)
    nbar = float(np.asarray(anchor, bool).sum(axis=1).mean())
    occ_fit = dataclasses.replace(afit, resid_var=np.asarray(afit.resid_var) * max(nbar, 1.0))
    eta = bartlett_scores(np.asarray(y, float), occ_fit)
    return afit, X_anchor, bartlett_scores(X_anchor, afit), eta


def _agentsem_diagnostics(world, anchor_scores, g, fit) -> dict:
    """Every number rule R1 to R6 looks at, computed from the observed contract."""
    y = np.asarray(world.y, float)
    n, T, _ = y.shape
    occ = np.arange(T)
    switch = np.asarray(world.switch, int)
    adaptive, anchor = _masks(world)
    dose = np.asarray(world.dose, float)
    version = np.asarray(world.version)
    resid = fit["resid"]

    n_anchor = anchor.sum(axis=1)
    aw = anchor.astype(float)
    cnt_t = aw.sum(axis=0)
    mean_t = (dose * aw).sum(axis=0) / np.maximum(cnt_t, 1.0)
    var_t = ((dose - mean_t) ** 2 * aw).sum(axis=0) / np.maximum(cnt_t - 1.0, 1.0)
    dose_sd_anchor = float(np.sqrt(np.max(np.where(cnt_t > 1, var_t, 0.0)))) if T else 0.0

    switch_corr, switch_p = _pearson(switch.astype(float), anchor_scores)

    def unit(v):
        s = float(v.std())
        return v / s if s > 1e-12 else v

    a_flat = adaptive.ravel().astype(float)
    M = np.column_stack([np.ones(a_flat.size), unit(a_flat), unit(g.ravel())])
    cond_number = float(np.linalg.cond(M))

    gbar = person_mean(g, np.ones_like(adaptive))
    var_g_total = float(np.var(g, ddof=1)) if g.size > 1 else 0.0
    between_share = (float(np.var(gbar, ddof=1)) / var_g_total) if var_g_total > 1e-15 else 0.0
    r2_gbar = _pearson(gbar, anchor_scores)[0] ** 2      # 0 when gbar is constant

    r_adapt = person_mean(resid, adaptive)
    r_anchor = person_mean(resid, anchor)
    epoch_shift = float(np.mean(r_adapt - r_anchor))

    margin = int(RULES["R4"]["changepoint_margin"])
    best = 0.0
    for tau in range(int(switch.min()) + margin, T - margin + 1):
        after = adaptive & (occ[None, :] >= tau)
        before = adaptive & (occ[None, :] < tau)
        if after.sum() < 1 or before.sum() < 1:
            continue
        best = max(best, abs(float(resid[after].mean() - resid[before].mean())))

    invariance = fit_longitudinal_invariance(_mean_over(y, anchor), _mean_over(y, adaptive))

    n_adapt = float(adaptive.sum(axis=1).mean())
    implied = _implied_mean_resid_var(n_adapt, fit["rho_d"], fit["var_d"], fit["var_e"])
    hetero_ratio = float(np.var(r_adapt, ddof=1) / implied) if implied > 0 else float("inf")

    exposure = exposure_statistics(resid, adaptive, gbar,
                                   windows=int(RULES["R4"]["exposure_windows"]))

    diag = {
        "version_complete": bool(version.shape == (n, T) and np.all(np.isin(version, (0, 1)))),
        "has_anchor": bool(n_anchor.min() > 0),
        "dose_constant_in_anchor": bool(dose_sd_anchor < 1e-9),
        "min_anchor_len": float(n_anchor.min()),
        "switch_varies": bool(switch.std() > 0),
        "switch_corr": switch_corr, "switch_p": switch_p,
        "cond_number": cond_number,
        "between_share_gbar": between_share, "r2_gbar_on_anchor": r2_gbar,
        "epoch_shift": epoch_shift, "max_changepoint_shift": float(best),
        "dcfi_metric": float(invariance["d_metric"]["dcfi"]),
        "hetero_ratio": hetero_ratio,
        "dose_sd_anchor": dose_sd_anchor,
        "n_adapt": n_adapt, "implied_mean_resid_var": implied,
    }
    diag.update(exposure)
    return diag


def _psi_flags(psi: float, ci90, ci95, smd_psi: float, identified: bool):
    """Spec 7's specificity pair: equivalence (innocent) and detection."""
    if not identified or not all(np.isfinite(v) for v in (*ci90, *ci95)):
        return None, None
    innocent = bool(ci90[0] > -smd_psi and ci90[1] < smd_psi)
    detects = bool(ci95[0] > 0.0 or ci95[1] < 0.0)
    return innocent, detects


def _agentsem(world, bootstrap: int, surrogate, label: str) -> dict:
    """D1 + D5: anchor calibration, occasion scores, shadow model, certificate."""
    y = np.asarray(world.y, float)
    n = y.shape[0]
    adaptive, anchor = _masks(world)
    g = surrogate(np.asarray(world.dose, float), np.asarray(world.switch, int))

    afit, _, anchor_scores, eta = _anchor_pipeline(y, anchor)
    fit = fit_shadow(eta, g, drift=True, between=True)

    sev = _score_error_var(afit)
    phi_A = float(afit.var_theta)
    ci_A = _delta_approx_ci(phi_A, n, sev)
    phi_B, ci_B = float(fit["phi"]), tuple(float(v) for v in fit["ci"]["phi"])
    psi, psi_ci = float(fit["psi"]), tuple(float(v) for v in fit["ci"]["psi"])
    se_psi = fit["se"]["psi"]
    psi_ci90 = ((psi - 1.645 * se_psi, psi + 1.645 * se_psi)
                if se_psi is not None and np.isfinite(se_psi) else (float("nan"),) * 2)
    ci_method = "hessian_delta"

    boot = {}
    if bootstrap > 0:
        boot = _bootstrap(world, surrogate, fit, bootstrap)
        if boot["boot_ci_phi"] is not None:
            ci_B = tuple(boot["boot_ci_phi"])
            psi_ci = tuple(boot["boot_ci_psi"])
            psi_ci90 = tuple(boot["boot_ci90_psi"])
            ci_method = "bootstrap"

    diag = _agentsem_diagnostics(world, anchor_scores, g, fit)
    diag.update({"phi_A": phi_A, "ci_A": ci_A, "phi_B": phi_B, "ci_B": ci_B,
                 "psi": psi, "psi_ci": psi_ci})
    cert = certify(diag, "D1D5")

    innocent, detects = _psi_flags(psi, psi_ci90, psi_ci,
                                   float(world.params["smd_psi"]), fit["psi_identified"])
    extras = {
        "phi_A": phi_A, "ci_A": ci_A, "phi_B_raw": phi_B,
        "ci_B_raw": tuple(float(v) for v in fit["ci"]["phi"]),
        "phi_u": float(fit["phi_u"]), "b": float(fit["b"]),
        "var_gbar": float(fit["var_gbar"]),
        "psi_identified": bool(fit["psi_identified"]),
        "between_dropped": bool(fit["between_dropped"]),
        "rho_d": float(fit["rho_d"]), "var_d": float(fit["var_d"]),
        "var_e": float(fit["var_e"]), "loglik": float(fit["loglik"]),
        "converged": bool(fit["converged"]), "anchor_score_error_var": sev,
        "psi_ci90": tuple(float(v) for v in psi_ci90),
        "innocent": innocent, "detects": detects, "ci_method": ci_method,
        "design": label,
    }
    extras.update({k: v for k, v in boot.items() if k != "boot_ci90_psi"})
    b_est = cert["B"]["estimate"]
    return _result(world, fit["scores"], b_est, ci=cert["B"]["ci"],
                   psi_hat=psi if fit["psi_identified"] else None,
                   certificate=cert, extras=extras)


def _bootstrap(world, surrogate, fit, reps: int) -> dict:
    """Person-level percentile bootstrap (spec 11 A4: warm start, no Hessian)."""
    y = np.asarray(world.y, float)
    n, T, _ = y.shape
    occ = np.arange(T)
    dose = np.asarray(world.dose, float)
    switch = np.asarray(world.switch, int)
    seed = getattr(world, "bootstrap_ss", None)
    if seed is None:
        seed = world.params.get("bootstrap_seed")
    rng = np.random.default_rng(seed)
    x0 = fit["x_transformed"]

    phis, psis, ok = [], [], 0
    for _ in range(int(reps)):
        idx = rng.integers(0, n, n)
        anchor_b = occ[None, :] < switch[idx][:, None]
        g_b = surrogate(dose[idx], switch[idx])
        try:
            *_, eta_b = _anchor_pipeline(y[idx], anchor_b)
            fit_b = fit_shadow(eta_b, g_b, drift=True, between=True, x0=x0, hessian=False)
        except (np.linalg.LinAlgError, ValueError):
            continue
        phis.append(float(fit_b["phi"]))
        psis.append(float(fit_b["psi"]))
        ok += 1
    if ok < 2:
        return {"boot_ci_phi": None, "boot_ci_psi": None, "boot_ci90_psi": None,
                "boot_reps_ok": ok}
    phis, psis = np.array(phis), np.array(psis)
    return {
        "boot_ci_phi": (float(np.percentile(phis, 2.5)), float(np.percentile(phis, 97.5))),
        "boot_ci_psi": (float(np.percentile(psis, 2.5)), float(np.percentile(psis, 97.5))),
        "boot_ci90_psi": (float(np.percentile(psis, 5.0)), float(np.percentile(psis, 95.0))),
        "boot_reps_ok": ok, "boot_sd_phi": float(phis.std(ddof=1)),
        "boot_sd_psi": float(psis.std(ddof=1)),
    }


def analyzer_agentsem(world, bootstrap: int = 0) -> dict:
    T_a = int(world.params["T"]) - int(world.params["T0"])
    return _agentsem(world, bootstrap,
                     lambda dose, switch: surrogate_g1(dose, switch, T_a), "D1D5")


def analyzer_misspecified(world, bootstrap: int = 0) -> dict:
    """The deliberately wrong surrogate: recency weighting with a two-occasion half-life."""
    out = _agentsem(world, bootstrap,
                    lambda dose, switch: surrogate_recency(dose, switch, 2.0), "D1D5")
    out["extras"]["surrogate"] = "recency_h2"
    return out


# ------------------------------------------------------------------------ the D2 arm
def analyzer_agentsem_d2(world, bootstrap: int = 0) -> dict:
    """D2: within-person IV of eta on g with the logged randomized dose as instrument."""
    y = np.asarray(world.y, float)
    n = y.shape[0]
    adaptive, anchor = _masks(world)
    switch = np.asarray(world.switch, int)
    dose = np.asarray(world.dose, float)
    T_a = int(world.params["T"]) - int(world.params["T0"])

    afit, _, anchor_scores, _ = _anchor_pipeline(y, anchor)
    phi_A = float(afit.var_theta)
    ci_A = _delta_approx_ci(phi_A, n, _score_error_var(afit))
    n_anchor = anchor.sum(axis=1)
    aw = anchor.astype(float)
    cnt_t = aw.sum(axis=0)
    mean_t = (dose * aw).sum(axis=0) / np.maximum(cnt_t, 1.0)
    var_t = ((dose - mean_t) ** 2 * aw).sum(axis=0) / np.maximum(cnt_t - 1.0, 1.0)
    dose_sd_anchor = float(np.sqrt(np.max(np.where(cnt_t > 1, var_t, 0.0))))
    switch_corr, switch_p = _pearson(switch.astype(float), anchor_scores)
    version = np.asarray(world.version)

    diag = {
        "version_complete": bool(version.shape == y.shape[:2]
                                 and np.all(np.isin(version, (0, 1)))),
        "has_anchor": bool(n_anchor.min() > 0),
        "dose_constant_in_anchor": bool(dose_sd_anchor < 1e-9),
        "min_anchor_len": float(n_anchor.min()),
        "switch_varies": bool(switch.std() > 0),
        "switch_corr": switch_corr, "switch_p": switch_p,
        "phi_A": phi_A, "ci_A": ci_A,
        "jitter_logged": world.jitter is not None,
    }

    if world.jitter is None:
        # Nothing to instrument with: the design is simply not available here.
        diag.update({"first_stage_F": 0.0, "phi_B": None, "ci_B": None,
                     "psi": None, "psi_ci": None})
        cert = certify(diag, "D2")
        X_adapt = _mean_over(y, adaptive)
        scores = bartlett_scores(X_adapt, fit_one_factor(X_adapt))
        return _result(world, scores, cert["B"]["estimate"], ci=cert["B"]["ci"],
                       certificate=cert,
                       extras={"phi_A": phi_A, "ci_A": ci_A, "phi_B_raw": None,
                               "jitter_logged": False, "first_stage_F": None,
                               "ci_method": "delta_approx", "design": "D2"})

    X_adapt = _mean_over(y, adaptive)
    sfit = fit_one_factor(X_adapt)
    nbar = float(adaptive.sum(axis=1).mean())
    occ_fit = dataclasses.replace(sfit, resid_var=np.asarray(sfit.resid_var) * max(nbar, 1.0))
    eta = bartlett_scores(y, occ_fit)

    g = surrogate_g1(dose, switch, T_a)
    z = surrogate_g1(np.asarray(world.jitter, float), switch, T_a)
    iv = iv_shadow(eta, g, z, adaptive)

    corrected = eta - iv["psi"] * g
    scores = person_mean(corrected, adaptive)
    sev = _score_error_var(sfit)
    phi_B = float(np.var(scores, ddof=1) - sev)
    ci_B = _delta_approx_ci(phi_B, n, sev)

    diag.update({"first_stage_F": float(iv["first_stage_F"]), "phi_B": phi_B, "ci_B": ci_B,
                 "psi": float(iv["psi"]), "psi_ci": tuple(float(v) for v in iv["ci"])})
    cert = certify(diag, "D2")
    se_psi = float(iv["se"])
    psi_ci90 = (iv["psi"] - 1.645 * se_psi, iv["psi"] + 1.645 * se_psi)
    innocent, detects = _psi_flags(float(iv["psi"]), psi_ci90, iv["ci"],
                                   float(world.params["smd_psi"]), True)
    extras = {"phi_A": phi_A, "ci_A": ci_A, "phi_B_raw": phi_B, "ci_B_raw": ci_B,
              "jitter_logged": True, "first_stage_F": float(iv["first_stage_F"]),
              "psi_se": se_psi, "psi_ci90": tuple(float(v) for v in psi_ci90),
              "score_error_var": sev, "innocent": innocent, "detects": detects,
              "ci_method": "delta_approx", "design": "D2"}
    return _result(world, scores, cert["B"]["estimate"], ci=cert["B"]["ci"],
                   psi_hat=float(iv["psi"]), certificate=cert, extras=extras)


# ------------------------------------------------------------------------- registry
ANALYZERS = {
    "naive": analyzer_naive,
    "naive_pooled": analyzer_naive_pooled,
    "epoch_fe": analyzer_epoch_fe,
    "belief_partial": analyzer_belief_partial,
    "dose_partial": analyzer_dose_partial,
    "detrend": analyzer_detrend,
    "anchor_only": analyzer_anchor_only,
    "agentsem": analyzer_agentsem,
    "agentsem_d2": analyzer_agentsem_d2,
    "misspecified": analyzer_misspecified,
}


def run_analyzer(name: str, world, bootstrap: int = 0) -> dict:
    """Run one analyzer by name.  `bootstrap` is honoured by the agentSEM arms only."""
    if name not in ANALYZERS:
        raise KeyError(f"unknown analyzer: {name!r}; have {sorted(ANALYZERS)}")
    return ANALYZERS[name](world, bootstrap)
