"""The policy-aware measurement model (spec 5, amended by spec 11 A1: D1 + D5, and D2's IV).

Per person i and occasion t the analyst has a Bartlett score `eta_it` from the anchored
calibration and a frozen surrogate `g_it` for the policy's exposure (`g = 0` on anchor
occasions).  The model is the correlated-random-effects (Mundlak) form:

    eta_it = theta_i + d_it + psi * g_it + e_it,   theta_i = b * gbar_i + u_i
           = u_i + d_it + psi * g_it + b * gbar_i + e_it

with `gbar_i` the person's mean surrogate over the included occasions, `u_i ~ N(0, phi_u)`
independent of `gbar_i`, `d_it` a stationary AR(1) (`rho_d`, marginal variance `var_d`) and
`e_it ~ N(0, var_e)`.  Var(theta) on the pi_0 scale is the total

    phi = phi_u + b^2 * Var(gbar)

and a person's score is the filtered `u` state plus `b * gbar_i`.

Why the b term is there: the plain random-effects likelihood (`between=False`) assumes
theta independent of g.  In this world the policy's dose is informative about theta beyond
the stream, because the belief comes from a private read, so p(theta_i | g_i) is not
N(0, phi) and the between-person channel of the likelihood moves variance out of phi and
into psi (measured on synthetic data: psi +0.06, phi -0.06; on the real base replicate phi
0.895, psi 0.576 against truth 1.0 and 0.5).  With the b term psi is identified from
within-person variation of the logged exposure path alone: the person mean of `psi * g_it`
is exactly `psi * gbar_i`, perfectly collinear with `b * gbar_i`, so the between-person
direction identifies only the sum `b + psi` and b absorbs it.

Two degenerate cases are handled rather than left to a singular Hessian.  When `gbar` has
no variance between persons (policy `none`, uniform help) b is not identified: it is
dropped, `between_dropped` is True and the fit is exactly the `between=False` fit.  When
`g` is identically zero over the included occasions (kappa 0, policy `none`) psi is not
identified either: it is fixed at 0 and reported with `psi_identified` False.

The state is `x_it = (u_i, d_it)` with transition `diag(1, rho_d)`, process noise
`diag(0, var_d (1 - rho_d^2))`, initial covariance `diag(phi_u, var_d)` and observation
`[1, 1] x + psi * g_it + b * gbar_i` with noise `var_e`.  The covariance recursion has no
data in it, so it is run once per distinct mask pattern rather than once per person; the
filtered means are the only [N, 2] object.  With the default mask there is one pattern.

`fit_shadow` maximises the exact Kalman likelihood; `iv_shadow` is the D2 arm, a
within-person instrumental-variable regression of eta on g with a logged randomized
component of the dose as the instrument.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

_LOG2PI = float(np.log(2.0 * np.pi))
_BIG = 1e12
_VAR_GBAR_MIN = 1e-12      # below this the between-person term is not identified

# Optimizer bounds on the transformed scale.  The frozen free-parameter order is
# (log phi_u, psi, b, logit rho_d, log var_d, log var_e); absent terms are simply skipped.
_BOUNDS = {"log_phi_u": (-20.0, 10.0), "psi": (-1e3, 1e3), "b": (-1e3, 1e3),
           "logit_rho_d": (-9.0, 9.0), "log_var_d": (-20.0, 10.0),
           "log_var_e": (-20.0, 10.0)}


def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _names(drift: bool, psi_free: bool, b_free: bool) -> tuple[str, ...]:
    """The free parameters, in the frozen order."""
    out = ["log_phi_u"]
    if psi_free:
        out.append("psi")
    if b_free:
        out.append("b")
    if drift:
        out += ["logit_rho_d", "log_var_d"]
    out.append("log_var_e")
    return tuple(out)


def _unpack(x: np.ndarray, names: tuple[str, ...]) -> tuple[float, float, float, float, float, float]:
    """Optimizer scale -> (phi_u, psi, b, rho_d, var_d, var_e)."""
    v = {name: x[i] for i, name in enumerate(names)}
    phi_u = float(np.exp(np.clip(v["log_phi_u"], -30.0, 30.0)))
    psi = float(v.get("psi", 0.0))
    b = float(v.get("b", 0.0))
    if "logit_rho_d" in v:
        rho_d = _expit(v["logit_rho_d"])
        var_d = float(np.exp(np.clip(v["log_var_d"], -30.0, 30.0)))
    else:
        rho_d, var_d = 0.0, 0.0
    var_e = float(np.exp(np.clip(v["log_var_e"], -30.0, 30.0)))
    return phi_u, psi, b, rho_d, var_d, var_e


def _patterns(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct rows of the mask plus each person's row index into them."""
    pats, inv = np.unique(mask, axis=0, return_inverse=True)
    return np.atleast_2d(pats), np.ravel(inv)


def _kalman(eta, g, pats, inv, phi, psi, rho_d, var_d, var_e, drift, gbar=None, b=0.0):
    """Exact filter over t, vectorized over persons.  Returns (loglik, state, ok).

    `state` is the filtered mean of the person state at each person's last included
    occasion, which for a constant state equals the smoothed mean.  With a b term that
    state is `u_i = theta_i - b * gbar_i`, so the caller adds `b * gbar_i` back.
    """
    n, T = eta.shape
    n_pat = pats.shape[0]
    if drift:
        m = 2
        F = np.array([[1.0, 0.0], [0.0, rho_d]])
        Q = np.array([[0.0, 0.0], [0.0, var_d * (1.0 - rho_d ** 2)]])
        P = np.zeros((n_pat, 2, 2))
        P[:, 0, 0] = phi
        P[:, 1, 1] = var_d
        H = np.array([1.0, 1.0])
    else:
        m = 1
        F = np.ones((1, 1))
        Q = np.zeros((1, 1))
        P = np.full((n_pat, 1, 1), phi)
        H = np.ones(1)

    shift = 0.0 if gbar is None else b * np.asarray(gbar, float)
    mean = np.zeros((n, m))
    state = np.zeros(n)
    ll = 0.0
    for t in range(T):
        if t > 0:
            mean = mean @ F.T
            P = F @ P @ F.T + Q
        sel = pats[:, t]                       # [n_pat] occasions this pattern includes
        obs = sel[inv]                         # [n]
        S = P @ H @ H + var_e                  # [n_pat] innovation variance
        if not np.all(np.isfinite(S)) or np.any(S <= 0):
            return -np.inf, state, False
        if obs.any():
            v = eta[:, t] - psi * g[:, t] - shift - mean @ H
            Si = S[inv][obs]
            vi = v[obs]
            ll -= 0.5 * float(np.sum(np.log(_LOG2PI_E * Si) + vi * vi / Si))
            K = (P @ H) / S[:, None]           # [n_pat, m] Kalman gain
            mean[obs] += K[inv][obs] * vi[:, None]
            state[obs] = mean[obs, 0]
            P_up = P - K[:, :, None] * K[:, None, :] * S[:, None, None]
            P = np.where(sel[:, None, None], P_up, P)
    return ll, state, True


_LOG2PI_E = float(np.exp(_LOG2PI))  # so log(2 pi S) is one log call on the array


def _nll(x, eta, g, gbar, pats, inv, names):
    phi_u, psi, b, rho_d, var_d, var_e = _unpack(x, names)
    ll, _, ok = _kalman(eta, g, pats, inv, phi_u, psi, rho_d, var_d, var_e,
                        "logit_rho_d" in names, gbar, b)
    if not ok or not np.isfinite(ll):
        return _BIG
    return -ll


def _hessian(f, x, h_rel=1e-3):
    """Central-difference Hessian of `f` at `x` on the transformed scale."""
    k = x.size
    h = h_rel * np.maximum(1.0, np.abs(x))
    Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            val = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej))
            Hm[i, j] = Hm[j, i] = val / (4.0 * h[i] * h[j])
    return 0.5 * (Hm + Hm.T)


def _masked_stats(eta: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(variance of person means, mean within-person variance) over included occasions."""
    w = mask.astype(float)
    n_obs = w.sum(axis=1)
    ok = n_obs > 0
    means = np.where(ok, (eta * w).sum(axis=1) / np.where(ok, n_obs, 1.0), np.nan)
    between = float(np.nanvar(means[ok], ddof=1)) if ok.sum() > 1 else 1.0
    dev = (eta - np.nan_to_num(means)[:, None]) * w
    within_n = np.maximum(n_obs - 1.0, 1.0)
    within = float(np.mean(((dev ** 2).sum(axis=1) / within_n)[n_obs > 1])) if (n_obs > 1).any() else 1.0
    return between, within


def person_mean(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of `a` [N, T] over each person's included occasions (0 where none are)."""
    a = np.asarray(a, float)
    w = np.asarray(mask, bool).astype(float)
    n_obs = w.sum(axis=1)
    return np.where(n_obs > 0, (a * w).sum(axis=1) / np.where(n_obs > 0, n_obs, 1.0), 0.0)


def fit_shadow(eta: np.ndarray, g: np.ndarray, drift: bool = True,
               mask: np.ndarray | None = None, between: bool = True,
               x0=None, hessian: bool = True) -> dict:
    """Fit `eta = u + d + psi g + b gbar + e` by exact Kalman maximum likelihood.

    `mask[N, T]` (bool) marks the occasions to include; a masked-out occasion skips the
    update and contributes nothing to the likelihood.  `drift=False` drops the AR(1)
    state and its two parameters.  `between=False` drops the correlated-random-effects
    term and gives the plain (biased in this world) random-effects fit.  `x0` is an
    optional warm start on the transformed scale, either a dict keyed by parameter name
    or an array in the frozen order.  `hessian=False` skips the finite-difference
    Hessian, leaving every standard error and interval NaN.

    Returns phi (the total Var(theta)), phi_u, psi, b, var_gbar, rho_d, var_d, var_e,
    `se` and `ci` dicts on the natural scale, `loglik`, `converged`, person `scores` [N],
    `resid` [N, T], and the flags `psi_identified` and `between_dropped`.
    """
    eta = np.asarray(eta, float)
    g = np.asarray(g, float)
    if eta.shape != g.shape or eta.ndim != 2:
        raise ValueError("eta and g must both be [N, T]")
    mask = np.ones(eta.shape, bool) if mask is None else np.asarray(mask, bool)
    if mask.shape != eta.shape:
        raise ValueError("mask must be [N, T]")
    pats, inv = _patterns(mask)

    gbar = person_mean(g, mask)
    var_gbar = float(np.var(gbar, ddof=1)) if gbar.size > 1 else 0.0
    psi_identified = bool(float(np.max(np.abs(g * mask))) > 0.0) if g.size else False
    b_free = bool(between) and var_gbar >= _VAR_GBAR_MIN
    between_dropped = bool(between) and not b_free
    names = _names(drift, psi_identified, b_free)

    between_var, within = _masked_stats(eta, mask)
    start = {"log_phi_u": float(np.log(max(between_var, 1e-3))), "psi": 0.0, "b": 0.0,
             "logit_rho_d": _logit(0.8), "log_var_d": float(np.log(0.05)),
             "log_var_e": float(np.log(max(within, 1e-3)))}
    bounds = [_BOUNDS[name] for name in names]
    if x0 is None:
        x_start = np.array([start[name] for name in names])
    elif isinstance(x0, dict):
        x_start = np.array([float(x0.get(name, start[name])) for name in names])
    else:
        x_start = np.asarray(x0, float).ravel()
        if x_start.size != len(names):
            raise ValueError(f"x0 has {x_start.size} entries, this fit has {len(names)}")
        x_start = x_start.copy()
    x_start = np.clip(x_start, [lo for lo, _ in bounds], [hi for _, hi in bounds])

    args = (eta, g, gbar, pats, inv, names)
    res = optimize.minimize(_nll, x_start, args=args, method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    polish = optimize.minimize(_nll, res.x, args=args, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    if polish.fun <= res.fun:
        res = polish
    converged = bool(np.isfinite(res.fun)) and (
        bool(res.success) or float(np.max(np.abs(res.jac))) < 1e-3)

    phi_u, psi, b, rho_d, var_d, var_e = _unpack(res.x, names)
    ll, state, _ = _kalman(eta, g, pats, inv, phi_u, psi, rho_d, var_d, var_e, drift, gbar, b)
    scores = state + b * gbar
    resid = eta - scores[:, None] - psi * g
    phi = phi_u + b * b * var_gbar

    cov = None
    if hessian:
        try:
            Hm = _hessian(lambda z: _nll(z, *args), res.x)
            cov = np.linalg.inv(Hm)
        except np.linalg.LinAlgError:
            cov = None
    sx = {}
    for i, name in enumerate(names):
        var_i = float(cov[i, i]) if cov is not None else float("nan")
        sx[name] = float(np.sqrt(var_i)) if np.isfinite(var_i) and var_i > 0 else float("nan")

    def _log_ci(value, name):
        s = sx.get(name, float("nan"))
        if not np.isfinite(s):
            return (float("nan"), float("nan"))
        return (float(value * np.exp(-1.96 * s)), float(value * np.exp(1.96 * s)))

    def _lin_ci(value, name):
        s = sx.get(name, float("nan"))
        if not np.isfinite(s):
            return (float("nan"), float("nan"))
        return (float(value - 1.96 * s), float(value + 1.96 * s))

    # phi is a function of two free parameters, so its interval is the delta method on the
    # natural scale with var(gbar) held fixed (it is a sample quantity, not a parameter).
    grad = np.array([phi_u if name == "log_phi_u" else
                     (2.0 * b * var_gbar if name == "b" else 0.0) for name in names])
    var_phi = float(grad @ cov @ grad) if cov is not None else float("nan")
    se_phi = float(np.sqrt(var_phi)) if np.isfinite(var_phi) and var_phi >= 0 else float("nan")
    ci_phi = ((float(phi - 1.96 * se_phi), float(phi + 1.96 * se_phi))
              if np.isfinite(se_phi) else (float("nan"), float("nan")))

    se = {"phi": se_phi, "phi_u": float(phi_u * sx["log_phi_u"]),
          "psi": float(sx["psi"]) if psi_identified else None,
          "b": float(sx["b"]) if b_free else None,
          "var_e": float(var_e * sx["log_var_e"])}
    ci = {"phi": ci_phi, "phi_u": _log_ci(phi_u, "log_phi_u"),
          "psi": _lin_ci(psi, "psi") if psi_identified else (float("nan"), float("nan")),
          "b": _lin_ci(b, "b") if b_free else (float("nan"), float("nan")),
          "var_e": _log_ci(var_e, "log_var_e")}
    if drift:
        se["rho_d"] = float(rho_d * (1.0 - rho_d) * sx["logit_rho_d"])
        se["var_d"] = float(var_d * sx["log_var_d"])
        xr = _logit(min(max(rho_d, 1e-12), 1 - 1e-12))
        s = sx["logit_rho_d"]
        ci["rho_d"] = ((_expit(xr - 1.96 * s), _expit(xr + 1.96 * s))
                       if np.isfinite(s) else (float("nan"), float("nan")))
        ci["var_d"] = _log_ci(var_d, "log_var_d")
    else:
        se["rho_d"] = se["var_d"] = None
        ci["rho_d"] = ci["var_d"] = None

    return {"phi": float(phi), "phi_u": float(phi_u), "psi": float(psi), "b": float(b),
            "var_gbar": float(var_gbar), "rho_d": float(rho_d), "var_d": float(var_d),
            "var_e": float(var_e), "se": se, "ci": ci, "loglik": float(ll),
            "converged": converged, "scores": scores, "resid": resid,
            "psi_identified": psi_identified, "between_dropped": between_dropped,
            "se_transformed": {k: float(v) for k, v in sx.items()},
            "x_transformed": {name: float(res.x[i]) for i, name in enumerate(names)},
            "drift": bool(drift)}


def iv_shadow(eta: np.ndarray, g: np.ndarray, z: np.ndarray, mask: np.ndarray) -> dict:
    """D2: within-person IV of eta on g, instrument z, over the masked occasions.

    Person means are swept out of all three series first, so only within-person variation
    identifies psi.  The standard error is the homoskedastic IV sandwich and the
    first-stage F is the plain (unclustered) F of g on z.
    """
    eta = np.asarray(eta, float)
    g = np.asarray(g, float)
    z = np.asarray(z, float)
    mask = np.asarray(mask, bool)
    if not (eta.shape == g.shape == z.shape == mask.shape):
        raise ValueError("eta, g, z and mask must all be [N, T]")

    w = mask.astype(float)
    n_obs_i = w.sum(axis=1)
    use = n_obs_i > 0

    def demean(a):
        means = np.where(use, (a * w).sum(axis=1) / np.where(use, n_obs_i, 1.0), 0.0)
        return (a - means[:, None]) * w

    et, gt, zt = demean(eta), demean(g), demean(z)
    n_obs = float(w.sum())
    n_person = float(use.sum())
    df = max(n_obs - n_person - 1.0, 1.0)

    szz = float(np.sum(zt * zt))
    szg = float(np.sum(zt * gt))
    sze = float(np.sum(zt * et))
    psi = sze / szg if szg != 0 else float("nan")

    u = et - psi * gt
    s2_u = float(np.sum(u * u * w)) / df
    var_psi = s2_u * szz / (szg ** 2) if szg != 0 else float("inf")
    se = float(np.sqrt(var_psi)) if np.isfinite(var_psi) and var_psi >= 0 else float("nan")

    b1 = szg / szz if szz != 0 else float("nan")
    r1 = gt - b1 * zt
    s2_1 = float(np.sum(r1 * r1 * w)) / df
    var_b1 = s2_1 / szz if szz != 0 else float("inf")
    F = float(b1 ** 2 / var_b1) if np.isfinite(var_b1) and var_b1 > 0 else float("nan")

    ci = (float(psi - 1.96 * se), float(psi + 1.96 * se)) if np.isfinite(se) else \
        (float("nan"), float("nan"))
    return {"psi": float(psi), "se": se, "ci": ci, "first_stage_F": F,
            "first_stage_beta": float(b1), "n_obs": int(n_obs), "n_person": int(n_person),
            "p": float(2.0 * stats.norm.sf(abs(psi / se))) if np.isfinite(se) and se > 0
            else float("nan")}
