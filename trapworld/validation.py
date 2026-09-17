"""The naive analyst's validation battery (spec section 5.1).

`validation_battery` reads only the observed data contract: `y`, `z`, `switch`,
`C_clean`, `C_react`, `W`.  It never touches hidden truth and never imports the
generator, so any object carrying those attributes works.

Occasions are 0-indexed: `t < switch[i]` is the anchor (policy version 0),
`t >= switch[i]` is the adaptive phase.  `switch` may vary across persons, so
every occasion average here is a masked mean.
"""

from __future__ import annotations

import numpy as np

from .cfa import CFAFit, bartlett_scores, fit_longitudinal_invariance


def _mean_over(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of `a` [N, T, ...] over the occasions selected by `mask` [N, T]."""
    a = np.asarray(a, float)
    m = mask.astype(float)
    while m.ndim < a.ndim:
        m = m[..., None]
    denom = m.sum(axis=1)
    return (a * m).sum(axis=1) / np.where(denom > 0, denom, np.nan)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _r2(y: np.ndarray, predictors: list[np.ndarray]) -> float:
    """R^2 of an OLS regression of `y` on an intercept plus `predictors`."""
    cols = [np.asarray(p, float).ravel() for p in predictors]
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(y) & np.all(np.isfinite(cols), axis=0)
    if ok.sum() < len(cols) + 2:
        return float("nan")
    X = np.column_stack([np.ones(int(ok.sum()))] + [c[ok] for c in cols])
    yk = y[ok]
    beta, *_ = np.linalg.lstsq(X, yk, rcond=None)
    resid = yk - X @ beta
    ss_tot = float(((yk - yk.mean()) ** 2).sum())
    return float(1.0 - float(resid @ resid) / ss_tot) if ss_tot > 0 else float("nan")


def validation_battery(world, fit: CFAFit, scores: np.ndarray,
                       taus=(0.2, 0.3, 0.4)) -> dict:
    """Every statistic the naive analyst reports, plus the pre-registered pass
    rules at each tau in `taus`."""
    y = np.asarray(world.y, float)
    z = np.asarray(world.z, float)
    switch = np.asarray(world.switch, int)
    scores = np.asarray(scores, float).ravel()
    occ = np.arange(y.shape[1])
    adaptive = occ[None, :] >= switch[:, None]
    anchor = ~adaptive

    nuisance = _mean_over(z.mean(axis=-1), adaptive)
    w_bar = _mean_over(np.asarray(world.W, float), adaptive)

    # test-retest: first vs second half of each person's adaptive occasions
    rank = np.cumsum(adaptive, axis=1) - 1
    half = (adaptive.sum(axis=1) // 2)[:, None]
    first = adaptive & (rank < half)
    second = adaptive & (rank >= half) & (half > 0)
    retest = _corr(bartlett_scores(_mean_over(y, first), fit),
                   bartlett_scores(_mean_over(y, second), fit))

    # two-occasion invariance: anchor person means vs adaptive person means
    x1, x2 = _mean_over(y, anchor), _mean_over(y, adaptive)
    keep = np.all(np.isfinite(x1), axis=1) & np.all(np.isfinite(x2), axis=1)
    invariance = fit_longitudinal_invariance(x1[keep], x2[keep])

    out = {
        "omega": float(fit.omega), "alpha": float(fit.alpha), "ave": float(fit.ave),
        "cfi": float(fit.cfi), "rmsea": float(fit.rmsea), "srmr": float(fit.srmr),
        "r_react": _corr(scores, world.C_react),
        "r_clean": _corr(scores, world.C_clean),
        "r_W": _corr(scores, w_bar),
        "incremental_r2": _r2(w_bar, [nuisance, scores]) - _r2(w_bar, [nuisance]),
        "test_retest": retest,
        "discriminant": abs(_corr(scores, nuisance)),
        "invariance": invariance,
    }
    for tau in taus:
        ok = (out["omega"] >= 0.70 and out["cfi"] >= 0.95 and out["rmsea"] <= 0.08
              and abs(out["r_react"]) >= tau and out["test_retest"] >= 0.70
              and out["discriminant"] <= 0.30)
        out[f"pass_{tau}"] = bool(ok)
        out[f"pass_{tau}_with_invariance"] = bool(ok and invariance["d_metric"]["dcfi"] <= 0.01)
    return out
