"""A small maximum-likelihood CFA engine with parameter tying.

The engine is general but tiny.  A model is four coefficient matrices plus two
vectors whose entries are each either a fixed number or an integer parameter
id; repeating an id ties those entries to one free parameter, which is all the
machinery the longitudinal invariance models need.  Variance parameters are
carried on the log scale so they stay positive; covariances are raw.

    Sigma = Lambda Psi Lambda' + Theta        mu = tau + Lambda alpha

Fit function (ML, with mean structure), S the ML (divide by n) covariance:

    F = log|Sigma| + tr(S Sigma^-1) + (xbar - mu)' Sigma^-1 (xbar - mu)
        - log|S| - q
    chi2 = n F            df = q(q+1)/2 + q - n_free

Public API: `fit_one_factor`, `bartlett_scores`, `fit_longitudinal_invariance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

_BIG = 1e12
_LOG2PI = float(np.log(2.0 * np.pi))
_MATS = ("lam", "psi", "th", "tau", "alpha")


# ------------------------------------------------------------------ the spec
def _spec(q: int, m: int) -> dict:
    """An empty model: everything fixed at zero, no free parameters."""
    return {
        "q": q, "m": m, "is_log": [],
        "lam_fix": np.zeros((q, m)), "lam_id": np.full((q, m), -1, int),
        "psi_fix": np.zeros((m, m)), "psi_id": np.full((m, m), -1, int),
        "th_fix": np.zeros((q, q)), "th_id": np.full((q, q), -1, int),
        "tau_fix": np.zeros(q), "tau_id": np.full(q, -1, int),
        "alpha_fix": np.zeros(m), "alpha_id": np.full(m, -1, int),
    }


def _free(spec: dict, log: bool = False) -> int:
    """Allocate a new free parameter id.  `log` marks it as a variance."""
    spec["is_log"].append(bool(log))
    return len(spec["is_log"]) - 1


def _sym(spec: dict, key: str, i: int, j: int, pid: int) -> None:
    """Put one id on both halves of a symmetric matrix."""
    spec[key + "_id"][i, j] = spec[key + "_id"][j, i] = pid


def _values(spec: dict, x: np.ndarray) -> np.ndarray:
    """Parameter vector on the optimizer's scale -> natural scale."""
    v = np.array(x, float)
    lg = np.asarray(spec["is_log"], bool)
    if lg.any():
        v[lg] = np.exp(np.clip(v[lg], -30.0, 30.0))
    return v


def _matrices(spec: dict, x: np.ndarray) -> tuple[np.ndarray, ...]:
    v = _values(spec, x)
    out = []
    for key in _MATS:
        fix, ids = spec[key + "_fix"], spec[key + "_id"]
        out.append(fix if v.size == 0 else np.where(ids >= 0, v[np.maximum(ids, 0)], fix))
    return (*out, v)


# ------------------------------------------------------------- the objective
def _objective(x, spec, S, xbar, logdet_S):
    Lam, Psi, Th, tau, al, v = _matrices(spec, x)
    Sigma = Lam @ Psi @ Lam.T + Th
    try:
        chol = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return _BIG, np.zeros_like(x)
    q = spec["q"]
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    Sinv = np.linalg.inv(Sigma)
    d = xbar - (tau + Lam @ al)
    Sd = Sinv @ d
    F = logdet + float(np.sum(Sinv * S)) + float(d @ Sd) - logdet_S - q

    G = Sinv - Sinv @ S @ Sinv - np.outer(Sd, Sd)
    grad = {
        "lam": 2.0 * (G @ Lam @ Psi) - 2.0 * np.outer(Sd, al),
        "psi": Lam.T @ G @ Lam,
        "th": G,
        "tau": -2.0 * Sd,
        "alpha": -2.0 * (Lam.T @ Sd),
    }
    g = np.zeros_like(x)
    for key in _MATS:
        ids = spec[key + "_id"]
        sel = ids >= 0
        if sel.any():
            np.add.at(g, ids[sel], grad[key][sel])
    lg = np.asarray(spec["is_log"], bool)
    if lg.any():
        g[lg] *= v[lg]  # chain rule for the log scale
    return F, g


def _start(spec: dict, S: np.ndarray, xbar: np.ndarray) -> np.ndarray:
    """First principal component scaled to each factor's marker, residuals at
    half the observed variances."""
    m, n_free = spec["m"], len(spec["is_log"])
    sd = np.sqrt(np.diag(S))
    lam0 = spec["lam_fix"].copy()
    psi0 = np.zeros((m, m))
    markers = []
    for k in range(m):
        idx = np.flatnonzero((spec["lam_id"][:, k] >= 0) | (spec["lam_fix"][:, k] != 0))
        w, V = np.linalg.eigh(S[np.ix_(idx, idx)])
        pc = V[:, -1] * np.sqrt(max(float(w[-1]), 1e-8))
        if pc.sum() < 0:
            pc = -pc
        fixed = np.flatnonzero(spec["lam_fix"][idx, k] != 0)
        if fixed.size and abs(pc[fixed[0]]) > 1e-8:
            pos = int(fixed[0])
            scale = spec["lam_fix"][idx[pos], k] / pc[pos]
        else:
            pos = int(np.argmax(np.abs(pc)))
            scale = 1.0 / max(float(np.abs(pc).max()), 1e-8)
        lam0[idx, k] = pc * scale
        mk = int(idx[pos])
        markers.append(mk)
        psi0[k, k] = 0.5 * S[mk, mk] / max(lam0[mk, k] ** 2, 1e-8)
    for k in range(m):
        for l in range(k + 1, m):
            r = S[markers[k], markers[l]] / (sd[markers[k]] * sd[markers[l]])
            psi0[k, l] = psi0[l, k] = np.clip(r, -0.9, 0.9) * np.sqrt(psi0[k, k] * psi0[l, l])
    start = {"lam": lam0, "psi": psi0, "th": np.diag(0.5 * np.diag(S)),
             "tau": np.array(xbar, float), "alpha": np.zeros(m)}

    x = np.zeros(n_free)
    cnt = np.zeros(n_free)
    for key in _MATS:
        ids = spec[key + "_id"]
        sel = ids >= 0
        if sel.any():
            np.add.at(x, ids[sel], start[key][sel])
            np.add.at(cnt, ids[sel], 1.0)
    x = np.where(cnt > 0, x / np.maximum(cnt, 1.0), 0.0)
    lg = np.asarray(spec["is_log"], bool)
    if lg.any():
        x[lg] = np.log(np.clip(x[lg], 1e-4, None))
    return x


# ------------------------------------------------------------------ the fit
def _moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    X = np.asarray(X, float)
    n = X.shape[0]
    xbar = X.mean(0)
    Xc = X - xbar
    S = (Xc.T @ Xc) / n  # ML covariance
    sign, logdet_S = np.linalg.slogdet(S)
    if sign <= 0:  # rank deficient sample; nudge so log|S| exists
        S = S + 1e-8 * np.eye(S.shape[0])
        _, logdet_S = np.linalg.slogdet(S)
    return S, xbar, n, float(logdet_S)


def _fit_spec(spec: dict, S: np.ndarray, xbar: np.ndarray, n: int, logdet_S: float) -> dict:
    q, n_free = spec["q"], len(spec["is_log"])
    lg = np.asarray(spec["is_log"], bool)
    bounds = [(-25.0, 25.0) if flag else (None, None) for flag in lg]
    opts = {"maxiter": 5000, "maxfun": 50000, "ftol": 1e-15, "gtol": 1e-10}
    args = (spec, S, xbar, logdet_S)
    res = optimize.minimize(_objective, _start(spec, S, xbar), args=args, jac=True,
                            method="L-BFGS-B", bounds=bounds, options=opts)
    polish = optimize.minimize(_objective, res.x, args=args, jac=True,
                               method="L-BFGS-B", bounds=bounds, options=opts)
    if polish.fun <= res.fun:
        res = polish
    # L-BFGS-B reports an abnormal line search when restarted from a point it
    # has already solved, so judge convergence on the gradient as well.
    converged = bool(np.isfinite(res.fun)) and (
        bool(res.success) or float(np.max(np.abs(res.jac))) < 1e-6)

    Lam, Psi, Th, tau, al, _ = _matrices(spec, res.x)
    Sigma = Lam @ Psi @ Lam.T + Th
    F = max(float(res.fun), 0.0)
    chi2 = n * F
    df = q * (q + 1) // 2 + q - n_free
    d = xbar - (tau + Lam @ al)
    Sinv = np.linalg.inv(Sigma)
    _, logdet_Sigma = np.linalg.slogdet(Sigma)
    loglik = -0.5 * n * (q * _LOG2PI + float(logdet_Sigma)
                         + float(np.sum(Sinv * S)) + float(d @ Sinv @ d))

    # independence baseline: free means and free variances only
    f0 = float(np.sum(np.log(np.diag(S)))) - logdet_S
    chi2_0, df0 = n * f0, q * (q - 1) // 2
    t, t0 = max(chi2 - df, 0.0), max(chi2_0 - df0, 0.0)
    cfi = 1.0 - t / max(t0, t, 1e-12)
    tli = float("nan")
    if df > 0 and df0 > 0 and abs(chi2_0 / df0 - 1.0) > 1e-12:
        tli = (chi2_0 / df0 - chi2 / df) / (chi2_0 / df0 - 1.0)
    rmsea = float(np.sqrt(t / (df * n))) if df > 0 else float("nan")
    sd = np.sqrt(np.diag(S))
    resid = (S - Sigma) / np.outer(sd, sd)
    tri = np.tril_indices(q)
    srmr = float(np.sqrt(np.mean(resid[tri] ** 2)))

    return {"Lambda": Lam, "Psi": Psi, "Theta": Th, "tau": tau, "alpha": al,
            "Sigma": Sigma, "chi2": chi2, "df": df,
            "pvalue": float(stats.chi2.sf(chi2, df)) if df > 0 else float("nan"),
            "cfi": float(cfi), "tli": float(tli), "rmsea": rmsea, "srmr": srmr,
            "loglik": float(loglik), "n": int(n), "n_free": n_free,
            "converged": converged, "fmin": F}


# --------------------------------------------------------------- one factor
@dataclass
class CFAFit:
    """One-factor ML CFA with marker scaling (`Lambda[0] = 1`, `alpha = 0`)."""

    loadings: np.ndarray
    intercepts: np.ndarray
    resid_var: np.ndarray
    var_theta: float
    latent_mean: float
    chi2: float
    df: int
    pvalue: float
    cfi: float
    tli: float
    rmsea: float
    srmr: float
    loglik: float
    n: int
    converged: bool
    omega: float
    alpha: float
    ave: float
    std_loadings: np.ndarray = field(repr=False)


def fit_one_factor(X: np.ndarray) -> CFAFit:
    """Fit a one-factor model with mean structure to `X` [n, q]."""
    S, xbar, n, logdet_S = _moments(X)
    q = S.shape[0]
    spec = _spec(q, 1)
    spec["lam_fix"][0, 0] = 1.0
    for j in range(1, q):
        spec["lam_id"][j, 0] = _free(spec)
    spec["psi_id"][0, 0] = _free(spec, log=True)
    for j in range(q):
        spec["th_id"][j, j] = _free(spec, log=True)
    for j in range(q):
        spec["tau_id"][j] = _free(spec)
    r = _fit_spec(spec, S, xbar, n, logdet_S)

    lam = r["Lambda"][:, 0]
    theta_jj = np.diag(r["Theta"])
    var_theta = float(r["Psi"][0, 0])
    total = lam.sum() ** 2 * var_theta
    omega = float(total / (total + theta_jj.sum()))
    cronbach = float(q / (q - 1.0) * (1.0 - np.diag(S).sum() / S.sum())) if q > 1 else float("nan")
    std = lam * np.sqrt(var_theta) / np.sqrt(np.diag(r["Sigma"]))
    return CFAFit(loadings=lam, intercepts=r["tau"], resid_var=theta_jj,
                  var_theta=var_theta, latent_mean=float(r["alpha"][0]),
                  chi2=r["chi2"], df=r["df"], pvalue=r["pvalue"], cfi=r["cfi"],
                  tli=r["tli"], rmsea=r["rmsea"], srmr=r["srmr"], loglik=r["loglik"],
                  n=r["n"], converged=r["converged"], omega=omega, alpha=cronbach,
                  ave=float(np.mean(std**2)), std_loadings=std)


def bartlett_scores(Y: np.ndarray, fit: CFAFit) -> np.ndarray:
    """`(L' Th^-1 L)^-1 L' Th^-1 (Y - tau)` along the last axis of `Y`."""
    Y = np.asarray(Y, float)
    L = np.asarray(fit.loadings, float)
    w = L / np.asarray(fit.resid_var, float)
    return (Y - np.asarray(fit.intercepts, float)) @ (w / float(L @ w))


# ------------------------------------------ two-occasion invariance sequence
def _invariance_spec(q: int, tie_loadings: bool, tie_intercepts: bool) -> dict:
    """Two factors over `[X1, X2]`: time 1 on indicators 0..q-1, time 2 on
    q..2q-1; Psi free; Theta diagonal plus the q same-indicator cross-time
    covariances; marker fixed to 1 at both times."""
    spec = _spec(2 * q, 2)
    spec["lam_fix"][0, 0] = 1.0
    spec["lam_fix"][q, 1] = 1.0
    lam_ids = [_free(spec) for _ in range(q - 1)]
    for j in range(1, q):
        spec["lam_id"][j, 0] = lam_ids[j - 1]
        spec["lam_id"][q + j, 1] = lam_ids[j - 1] if tie_loadings else _free(spec)
    spec["psi_id"][0, 0] = _free(spec, log=True)
    spec["psi_id"][1, 1] = _free(spec, log=True)
    _sym(spec, "psi", 0, 1, _free(spec))
    for j in range(2 * q):
        spec["th_id"][j, j] = _free(spec, log=True)
    for j in range(q):
        _sym(spec, "th", j, q + j, _free(spec))
    for j in range(q):
        spec["tau_id"][j] = pid = _free(spec)
        spec["tau_id"][q + j] = pid if tie_intercepts else _free(spec)
    if tie_intercepts:
        spec["alpha_id"][1] = _free(spec)
    return spec


def fit_longitudinal_invariance(X1: np.ndarray, X2: np.ndarray) -> dict:
    """Configural, metric and scalar models on `[X1, X2]` plus the two
    chi-square difference tests.  `dcfi` is less constrained minus more
    constrained."""
    X = np.column_stack([np.asarray(X1, float), np.asarray(X2, float)])
    S, xbar, n, logdet_S = _moments(X)
    q = X1.shape[1]

    levels = {}
    for name, ties in (("configural", (False, False)), ("metric", (True, False)),
                       ("scalar", (True, True))):
        r = _fit_spec(_invariance_spec(q, *ties), S, xbar, n, logdet_S)
        levels[name] = {
            "chi2": r["chi2"], "df": r["df"], "pvalue": r["pvalue"],
            "cfi": r["cfi"], "tli": r["tli"], "rmsea": r["rmsea"], "srmr": r["srmr"],
            "loglik": r["loglik"], "n": r["n"], "converged": r["converged"],
            "loadings1": r["Lambda"][:q, 0].copy(),
            "loadings2": r["Lambda"][q:, 1].copy(),
            "intercepts1": r["tau"][:q].copy(), "intercepts2": r["tau"][q:].copy(),
            "latent_mean2": float(r["alpha"][1]),
            "var_theta1": float(r["Psi"][0, 0]), "var_theta2": float(r["Psi"][1, 1]),
            "cov_theta": float(r["Psi"][0, 1]),
        }

    def delta(less: str, more: str) -> dict:
        dchi2 = max(levels[more]["chi2"] - levels[less]["chi2"], 0.0)
        ddf = levels[more]["df"] - levels[less]["df"]
        p = float(stats.chi2.sf(dchi2, ddf)) if ddf > 0 else float("nan")
        return {"dchi2": dchi2, "ddf": int(ddf), "p": p,
                "dcfi": levels[less]["cfi"] - levels[more]["cfi"]}

    return {**levels, "d_metric": delta("configural", "metric"),
            "d_scalar": delta("metric", "scalar")}
