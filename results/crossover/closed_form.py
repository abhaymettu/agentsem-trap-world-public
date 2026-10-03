"""Closed-form person-score RMSE of anchor_only, naive and naive_pooled vs psi*kappa.

Exact for the generator in trapworld/world.py (policy "ew", generator "cumulative",
proportional shadow, calendar trigger, no jitter, policy_reads_stream False). Everything
is linear-Gaussian, so each person's eta_i (the common factor of the person-mean battery
over a set M of occasions) is a fixed linear combination of theta_i, the drift path d_i and
the read-noise path r_i. The script propagates those coefficient vectors through the time
loop, so no persons are simulated and nothing is random: PARAMS in, numbers out.

    y_itj = lam_j * (theta_i + d_it + delta_it) + eps_itj          (nu = lam)
    b_t   = (1 - g) b_{t-1} + g (theta_i + d_it + r_it),  b_{-1} = 0
    dose_t = kappa * b_{t-1} for t >= T0;  delta_t = psi * sum_{s=T0..t} dose_s / T_a

Person means over M give X_i = lam * eta_i + ebar_i with Var(ebar_ij) = var_eps_j / |M|,
an exact one-factor model with marker lam_0 = 1, so the population Bartlett score is
eta_i + n_i with Var(n_i) = 1 / (|M| * sum_j lam_j^2 / var_eps_j). Intercepts are the
sample means (the mean structure is saturated), which centers the scores; theta is not
centered, so with N persons

    E[MSE] = (1 - 1/N) * (Var(eta - theta) + Var(n)) + var_theta / N.

Not modelled: sampling error in the fitted loadings and residual variances, and Jensen's
gap between E[sqrt(MSE)] and sqrt(E[MSE]). Both are O(1/N).

Run from the repo root: python results/crossover/closed_form.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from trapworld.params import PARAMS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PRODUCTS = (0.0, 0.25, 0.5, 1.0, 2.0)
T0S = (7, 14, 28)
ESTIMATORS = ("anchor_only", "naive", "naive_pooled")


def mask(T: int, T0: int, which: str) -> np.ndarray:
    """The occasions each estimator averages over."""
    occ = np.arange(T)
    return {"anchor_only": occ < T0, "naive": occ >= T0, "naive_pooled": occ >= 0}[which]


def eta_coefficients(pr: dict, T0: int, product: float, occasions: np.ndarray):
    """Coefficient vector of eta over the basis (theta, d_0..d_{T-1}, r_0..r_{T-1}) and the
    covariance of that basis."""
    T, g = pr["T"], pr["gain_ew"]
    T_a = T - T0
    K = 1 + 2 * T
    th = np.zeros(K); th[0] = 1.0
    b = np.zeros(K)                    # belief coefficients, b_{-1} = 0
    cum = np.zeros(K)
    eta = np.zeros(K)
    for t in range(T):
        if t >= T0:
            cum = cum + product * b    # psi * kappa * b_{t-1}, accumulated
        delta = cum / T_a if t >= T0 else np.zeros(K)
        state = th.copy(); state[1 + t] = 1.0   # theta + d_t
        if occasions[t]:
            eta += state + delta
        r = state.copy(); r[1 + T + t] = 1.0     # read = theta_t + r_t
        b = b + g * (r - b)
    eta /= occasions.sum()

    lags = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    Sigma = np.zeros((K, K))
    Sigma[0, 0] = pr["var_theta"]
    Sigma[1:1 + T, 1:1 + T] = pr["sd_d"] ** 2 * pr["rho_d"] ** lags  # stationary AR(1)
    Sigma[1 + T:, 1 + T:] = pr["var_read"] * np.eye(T)
    return eta, Sigma, th


def quadratic(pr: dict, T0: int, occasions: np.ndarray) -> dict:
    """eta - theta = dbar + P * w exactly, so Var(eta - theta) = c0 + 2 c1 P + c2 P^2."""
    e0, Sigma, th = eta_coefficients(pr, T0, 0.0, occasions)
    e1, _, _ = eta_coefficients(pr, T0, 1.0, occasions)
    dbar, w = e0 - th, e1 - e0
    T = pr["T"]
    part = lambda sl: float(w[sl] @ Sigma[sl, sl] @ w[sl])
    return dict(c0=float(dbar @ Sigma @ dbar), c1=float(dbar @ Sigma @ w),
                c2=float(w @ Sigma @ w), A=float(w[0]),
                c2_theta=part(slice(0, 1)), c2_drift=part(slice(1, 1 + T)),
                c2_read=part(slice(1 + T, 1 + 2 * T)))


def noise_var(pr: dict, n_occ: int) -> float:
    lam = np.asarray(pr["loadings"], float)
    return 1.0 / (n_occ * float(np.sum(lam ** 2 / np.asarray(pr["var_eps"], float))))


def crossing_closed_form(pr: dict, T0: int, which: str = "naive") -> float:
    """Root of c2 P^2 + 2 c1 P + D = 0, D = (c0 + noise) of `which` minus that of
    anchor_only. N cancels: the centering term is the same on both sides."""
    M = mask(pr["T"], T0, which)
    q = quadratic(pr, T0, M)
    qa = quadratic(pr, T0, mask(pr["T"], T0, "anchor_only"))
    D = (q["c0"] + noise_var(pr, M.sum())) - (qa["c0"] + noise_var(pr, T0))
    if D >= -1e-12:
        return 0.0
    return float((-q["c1"] + np.sqrt(q["c1"] ** 2 - q["c2"] * D)) / q["c2"])


def rmse(pr: dict, T0: int, product: float, which: str) -> float:
    M = mask(pr["T"], T0, which)
    eta, Sigma, th = eta_coefficients(pr, T0, product, M)
    dev = eta - th
    mse = float(dev @ Sigma @ dev) + noise_var(pr, M.sum())
    N = pr["N"]
    return float(np.sqrt((1 - 1 / N) * mse + pr["var_theta"] / N))


def crossing(pr: dict, T0: int, which: str = "naive") -> float | None:
    """Product at which `which` RMSE rises to the anchor_only RMSE (anchor is flat in it)."""
    target = rmse(pr, T0, 0.0, "anchor_only")
    f = lambda u: rmse(pr, T0, u, which) - target
    if f(0.0) >= 0:
        return 0.0
    return float(brentq(f, 0.0, 10.0, xtol=1e-10))


def frozen_reference() -> list[dict]:
    rows = []
    with open(ROOT / "results" / "summary.csv") as fh:
        for r in csv.DictReader(fh):
            if (r["sweep"] == "A" and float(r["gamma_C"]) == 1.0
                    and r["analyzer"] in ESTIMATORS):
                rows.append(dict(analyzer=r["analyzer"], kappa=float(r["kappa"]),
                                 psi=float(r["psi"]), rmse_mean=float(r["rmse_mean"]),
                                 rmse_mc_se=float(r["rmse_mc_se"])))
    return rows


def main() -> dict:
    pr = dict(PARAMS)
    grid, cross, quad = {}, {}, {}
    for T0 in T0S:
        grid[str(T0)] = {w: {str(u): rmse(pr, T0, u, w) for u in PRODUCTS} for w in ESTIMATORS}
        cross[str(T0)] = {w: crossing(pr, T0, w) for w in ("naive", "naive_pooled")}
        for w, c in cross[str(T0)].items():      # the quadratic root is the same number
            assert abs(c - crossing_closed_form(pr, T0, w)) < 1e-8, (T0, w)
        quad[str(T0)] = {w: quadratic(pr, T0, mask(pr["T"], T0, w)) for w in ESTIMATORS}
        quad[str(T0)].update(noise_anchor=noise_var(pr, T0), noise_naive=noise_var(pr, pr["T"] - T0),
                             noise_pooled=noise_var(pr, pr["T"]))

    # the belief error's role: crossing at T0 = 14 with var_read and gain_ew varied
    sens = []
    for vr in (0.0, 2.0, 8.0):
        for g in (0.1, 0.3, 0.6):
            q = dict(pr, var_read=vr, gain_ew=g)
            sens.append(dict(var_read=vr, gain_ew=g, cross_naive=crossing(q, 14, "naive"),
                             naive_rmse_at_1=rmse(q, 14, 1.0, "naive")))

    comp = []
    for r in frozen_reference():
        u = r["kappa"] * r["psi"]
        pred = rmse(pr, pr["T0"], u, r["analyzer"])
        comp.append(dict(r, product=u, predicted=pred, abs_err=abs(pred - r["rmse_mean"])))
    max_err = max(c["abs_err"] for c in comp)

    out = dict(method="exact linear-Gaussian propagation from PARAMS; population Bartlett "
                      "score; finite-N centering term; no loading sampling error",
               params_used={k: pr[k] for k in ("N", "T", "var_theta", "rho_d", "sd_d",
                                               "loadings", "var_eps", "var_read", "gain_ew")},
               rmse=grid, crossing=cross, variance_quadratic=quad, sensitivity_T0_14=sens,
               frozen_comparison=comp, max_abs_err=max_err)
    (HERE / "closed_form.json").write_text(json.dumps(out, indent=2))
    print(f"closed_form.json written; max abs error against frozen sweep A {max_err:.4f}")
    return out


if __name__ == "__main__":
    main()
