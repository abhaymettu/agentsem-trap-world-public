"""F1-F5: the pure-algebra falsification gate for the agentSEM trap world.

Closed forms plus a matching Monte Carlo, no generator involved.  Symbols follow
HARSH-REVIEW.md sections B1/B2:

    theta ~ N(0, s_th2)               person's true latent state
    e     ~ N(0, s_e2)                policy belief error, independent of theta
    u     = psi * kappa               adaptivity x shadow
    eta   = theta + u*(theta + e)     naive score = (1+u)*theta + u*e

Convention: ``N(mean, v)`` here means variance ``v`` (numpy's ``normal`` takes a standard
deviation, so every draw passes ``sqrt(v)``).  Under that reading F5's criterion noise
``N(0, .75)`` gives ``Var(C) = 1`` and ``corr(theta, C) = 0.5`` exactly.

Run ``python -m trapworld.closed_forms`` to write ``results/closed_forms.json``.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np

SEED = 20260916
N_DRAWS = 2_000_000
REPS = 500
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "closed_forms.json"

#: The B1.4 table as printed in HARSH-REVIEW.md, rows r = 0, .2, .4142, .6, 1.0 and
#: columns u = .25, .5, 1, 2, 8.  Kept verbatim so the check can measure its own
#: agreement with the published numbers instead of assuming it.
REVIEW_B14_TABLE = [
    [0.980, 0.949, 0.895, 0.832, 0.747],
    [1.020, 1.012, 0.984, 0.943, 0.881],
    [1.062, 1.080, 1.079, 1.062, 1.023],
    [1.098, 1.139, 1.162, 1.165, 1.146],
    [1.177, 1.265, 1.342, 1.388, 1.412],
]

#: B2 surrogate-purity table as printed in HARSH-REVIEW.md, for T = 1, 7, 28, 56.
REVIEW_B2_R2 = [0.500, 0.875, 0.966, 0.982]
REVIEW_B2_ATTENUATION = [0.707, 0.354, 0.186, 0.133]

#: B1.3 table as printed in HARSH-REVIEW.md, for u = 0, .5, 1.
REVIEW_B13_OMEGA = [0.833, 0.926, 0.962]
REVIEW_B13_CORR = [1.000, 0.949, 0.894]


# --------------------------------------------------------------------------- plumbing
def rng_for(k: int) -> np.random.Generator:
    """Independent RNG stream for check index ``k``.

    The brief writes ``SeedSequence(20260916).spawn(k)[0]``.  Taken literally that hands
    back child 0 for every k -- ``spawn`` on a freshly constructed SeedSequence always
    starts numbering at 0 -- so all five checks would share one stream.  Spawning ``k+1``
    children and taking index ``k`` is the same construction with the intended
    distinct-stream semantics.
    """
    return np.random.default_rng(np.random.SeedSequence(SEED).spawn(k + 1)[k])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def _rowwise_residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residuals of y on x (with intercept), one OLS per row."""
    yc = y - y.mean(axis=1, keepdims=True)
    xc = x - x.mean(axis=1, keepdims=True)
    slope = (xc * yc).sum(1) / (xc * xc).sum(1)
    return yc - slope[:, None] * xc


def _fl(x) -> list:
    arr = np.asarray(x, dtype=float)
    return arr.tolist()


# --------------------------------------------------------------------------- F1
def f1_validity_vs_adaptivity(
    u_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 8.0),
    lam: float = 0.5,
    s_th2: float = 1.0,
    s_e2: float = 1.0,
    n_draws: int = N_DRAWS,
) -> dict:
    """Does adaptivity inflate or attenuate the naive score's correlations?"""
    u = np.asarray(u_grid, dtype=float)

    corr_eta_theta = 1.0 / np.sqrt(1.0 + (u**2 * s_e2) / ((1.0 + u) ** 2 * s_th2))
    corr_theta_C = lam * math.sqrt(s_th2) / math.sqrt(lam**2 * s_th2 + 1.0)
    corr_eta_C = corr_eta_theta * corr_theta_C
    var_ratio = ((1.0 + u) ** 2 * s_th2 + u**2 * s_e2) / s_th2

    rng = rng_for(0)
    theta = rng.normal(0.0, math.sqrt(s_th2), n_draws)
    e = rng.normal(0.0, math.sqrt(s_e2), n_draws)
    C = lam * theta + rng.normal(0.0, 1.0, n_draws)
    var_theta = float(theta.var())

    mc_ct, mc_cC, mc_vr = [], [], []
    for uu in u:
        eta = (1.0 + uu) * theta + uu * e
        mc_ct.append(_corr(eta, theta))
        mc_cC.append(_corr(eta, C))
        mc_vr.append(float(eta.var()) / var_theta)

    mc_ct_a, mc_cC_a, mc_vr_a = map(np.asarray, (mc_ct, mc_cC, mc_vr))
    dec_theta = bool(np.all(np.diff(mc_ct_a) < 0))
    dec_C = bool(np.all(np.diff(mc_cC_a) < 0))
    inc_var = bool(np.all(np.diff(mc_vr_a) > 0))

    return {
        "name": "F1 validity vs adaptivity",
        "u_grid": _fl(u),
        "params": {"lam": lam, "s_th2": s_th2, "s_e2": s_e2, "n_draws": n_draws},
        "closed_form": {
            "corr_eta_theta": _fl(corr_eta_theta),
            "corr_eta_C": _fl(corr_eta_C),
            "var_ratio": _fl(var_ratio),
            "corr_theta_C": float(corr_theta_C),
        },
        "monte_carlo": {
            "corr_eta_theta": _fl(mc_ct_a),
            "corr_eta_C": _fl(mc_cC_a),
            "var_ratio": _fl(mc_vr_a),
            "corr_theta_C": _corr(theta, C),
        },
        "max_abs_dev_corr": float(
            max(
                np.max(np.abs(mc_ct_a - corr_eta_theta)),
                np.max(np.abs(mc_cC_a - corr_eta_C)),
            )
        ),
        "max_rel_dev_var_ratio": float(np.max(np.abs(mc_vr_a - var_ratio) / var_ratio)),
        "corr_eta_theta_strictly_decreasing": dec_theta,
        "corr_eta_C_strictly_decreasing": dec_C,
        "var_ratio_strictly_increasing": inc_var,
        "spec_predicts": "both rise",
        "review_predicts": "both fall",
        "verdict": "corr falls with u" if (dec_theta and dec_C) else "corr rises with u",
    }


# --------------------------------------------------------------------------- F2
def f2_reliability_vs_validity(
    u_grid=(0.0, 0.5, 1.0),
    p: int = 5,
    s_eps2: float = 1.0,
    s_th2: float = 1.0,
    s_e2: float = 1.0,
    n_draws: int = N_DRAWS,
) -> dict:
    """B1.3: a common policy shift raises composite reliability while validity falls."""
    u = np.asarray(u_grid, dtype=float)

    V_c = (1.0 + u) ** 2 * s_th2 + u**2 * s_e2
    omega = p * V_c / (p * V_c + s_eps2)
    corr_eta_theta = 1.0 / np.sqrt(1.0 + (u**2 * s_e2) / ((1.0 + u) ** 2 * s_th2))
    corr_comp_theta = p * (1.0 + u) * s_th2 / np.sqrt(s_th2 * (p**2 * V_c + p * s_eps2))
    # Indicators are tau-equivalent with equal error variances here, so alpha == omega.
    alpha = omega.copy()

    rng = rng_for(1)
    theta = rng.normal(0.0, math.sqrt(s_th2), n_draws)
    e = rng.normal(0.0, math.sqrt(s_e2), n_draws)

    mc_omega, mc_alpha, mc_corr, mc_comp = [], [], [], []
    for uu in u:
        eta = (1.0 + uu) * theta + uu * e
        Y = eta[:, None] + rng.normal(0.0, math.sqrt(s_eps2), (n_draws, p))
        total = Y.sum(axis=1)
        var_total = float(total.var(ddof=1))
        # omega from the oracle common part (checks the closed form directly);
        # alpha from the observed items only (what an analyst would actually compute).
        mc_omega.append(p**2 * float(eta.var(ddof=1)) / var_total)
        mc_alpha.append(p / (p - 1.0) * (1.0 - float(Y.var(axis=0, ddof=1).sum()) / var_total))
        mc_corr.append(_corr(eta, theta))
        mc_comp.append(_corr(total, theta))

    mc = {
        "omega": np.asarray(mc_omega),
        "alpha": np.asarray(mc_alpha),
        "corr_eta_theta": np.asarray(mc_corr),
        "corr_composite_theta": np.asarray(mc_comp),
    }
    closed = {
        "omega": omega,
        "alpha": alpha,
        "corr_eta_theta": corr_eta_theta,
        "corr_composite_theta": corr_comp_theta,
        "V_c": V_c,
    }
    max_dev = float(max(np.max(np.abs(mc[k] - closed[k])) for k in mc))

    omega_up = bool(np.all(np.diff(mc["omega"]) > 0))
    corr_down = bool(np.all(np.diff(mc["corr_eta_theta"]) < 0))
    reproduced = bool(
        [round(v, 3) for v in omega.tolist()] == REVIEW_B13_OMEGA
        and [round(v, 3) for v in corr_eta_theta.tolist()] == REVIEW_B13_CORR
    )

    return {
        "name": "F2 reliability vs validity",
        "u_grid": _fl(u),
        "params": {"p": p, "s_eps2": s_eps2, "s_th2": s_th2, "s_e2": s_e2, "n_draws": n_draws},
        "closed_form": {k: _fl(v) for k, v in closed.items()},
        "monte_carlo": {k: _fl(v) for k, v in mc.items()},
        "max_abs_dev": max_dev,
        "b13_table": {"omega": REVIEW_B13_OMEGA, "corr_eta_theta": REVIEW_B13_CORR},
        "b13_table_reproduced": reproduced,
        "omega_increasing": omega_up,
        "corr_decreasing": corr_down,
        "spec_predicts": "not predicted (B1.3 is the review's replacement result)",
        "review_predicts": "omega rises, corr falls, crossing",
        "verdict": "reliability up, validity down" if (omega_up and corr_down) else "no crossing",
    }


# --------------------------------------------------------------------------- F3
def _f3_ratio(u, r):
    u = np.asarray(u, dtype=float)
    return (1.0 + u + u * r) / np.sqrt(1.0 + 2.0 * u + 2.0 * u**2)


def f3_criterion_contamination(
    r_grid=(0.0, 0.2, 0.4142, 0.6, 1.0),
    u_grid=(0.25, 0.5, 1.0, 2.0, 8.0),
    n_draws: int = N_DRAWS,
    fine_u_max: float = 8.0,
    fine_u_n: int = 8001,
) -> dict:
    """B1.4: inflation of criterion validity is conditional on criterion position r.

    dose = theta + e;  O = g*dose + b*theta + N(0,1) with g + b = 1, so g = r, b = 1-r,
    which collapses to O = theta + r*e + noise.  Fixed at s_th2 = s_e2 = 1 as in B1.4.
    """
    r = np.asarray(r_grid, dtype=float)
    u = np.asarray(u_grid, dtype=float)

    ratio = np.array([_f3_ratio(u, rr) for rr in r])
    corr_theta_O = 1.0 / np.sqrt(2.0 + r**2)
    corr_eta_O = np.array(
        [(1.0 + u + u * rr) / np.sqrt(((1.0 + u) ** 2 + u**2) * (2.0 + rr**2)) for rr in r]
    )
    ceiling = (1.0 + r) / math.sqrt(2.0)

    rng = rng_for(2)
    theta = rng.normal(0.0, 1.0, n_draws)
    e = rng.normal(0.0, 1.0, n_draws)
    O = [rr * (theta + e) + (1.0 - rr) * theta + rng.normal(0.0, 1.0, n_draws) for rr in r]
    mc_corr_theta_O = np.array([_corr(theta, o) for o in O])
    mc_corr_eta_O = np.zeros_like(ratio)
    for j, uu in enumerate(u):
        eta = (1.0 + uu) * theta + uu * e
        for i in range(len(r)):
            mc_corr_eta_O[i, j] = _corr(eta, O[i])
    mc_ratio = mc_corr_eta_O / mc_corr_theta_O[:, None]

    fine_u = np.linspace(0.0, fine_u_max, fine_u_n)
    rise_u = np.linspace(0.0, 2.0, 2001)
    per_r = []
    for i, rr in enumerate(r):
        fine = _f3_ratio(fine_u, rr)
        peak_closed = None if rr >= 1.0 else float(rr / (1.0 - rr))
        per_r.append(
            {
                "r": float(rr),
                "ratio_at_u": _fl(ratio[i]),
                "inflates_for_all_u": bool(np.all(ratio[i] > 1.0)),
                "inflates_asymptotically": bool(ceiling[i] > 1.0),
                "monotone_rise_over_0_2": bool(np.all(np.diff(_f3_ratio(rise_u, rr)) > 0)),
                "peak_u_closed": peak_closed,
                "peak_u_numeric": float(fine_u[int(np.argmax(fine))]),
                "peak_ratio_numeric": float(np.max(fine)),
                "ceiling": float(ceiling[i]),
            }
        )

    review = np.asarray(REVIEW_B14_TABLE, dtype=float)
    mismatches = [
        {
            "r": float(r[i]),
            "u": float(u[j]),
            "closed_form": float(ratio[i, j]),
            "closed_form_3dp": round(float(ratio[i, j]), 3),
            "review_printed": float(review[i, j]),
            "abs_dev": float(abs(ratio[i, j] - review[i, j])),
        }
        for i in range(len(r))
        for j in range(len(u))
        if round(float(ratio[i, j]), 3) != float(review[i, j])
    ]

    thr = math.sqrt(2.0) - 1.0
    inflate_set = {row["r"] for row in per_r if row["inflates_for_all_u"]}
    # r = 0.4142 sits 1.4e-5 *below* sqrt(2)-1, so its asymptote is 0.99999; on the finite
    # u grid (max u = 8) it still inflates everywhere.  Tolerance keeps the stated
    # "True iff r >= sqrt(2)-1" reading of the grid result.
    expected_inflate = {float(x) for x in r if x >= thr - 1e-4}
    monotone_set = {row["r"] for row in per_r if row["monotone_rise_over_0_2"]}
    expected_monotone = {float(x) for x in r if x >= 2.0 / 3.0}
    conditions_hold = bool(inflate_set == expected_inflate and monotone_set == expected_monotone)

    return {
        "name": "F3 criterion contamination",
        "r_grid": _fl(r),
        "u_grid": _fl(u),
        "params": {"n_draws": n_draws, "s_th2": 1.0, "s_e2": 1.0, "noise_var_O": 1.0},
        "closed_form": {
            "ratio": _fl(ratio),
            "corr_eta_O": _fl(corr_eta_O),
            "corr_theta_O": _fl(corr_theta_O),
            "ceiling": _fl(ceiling),
        },
        "monte_carlo": {
            "ratio": _fl(mc_ratio),
            "corr_eta_O": _fl(mc_corr_eta_O),
            "corr_theta_O": _fl(mc_corr_theta_O),
        },
        "max_abs_dev": float(np.max(np.abs(mc_ratio - ratio))),
        "review_b14_table": REVIEW_B14_TABLE,
        "review_table_max_abs_dev": float(np.max(np.abs(ratio - review))),
        "review_table_rounding_mismatches": mismatches,
        "inflation_threshold_r": thr,
        "monotone_threshold_r": 2.0 / 3.0,
        "per_r": per_r,
        "spec_predicts": "monotone rise in u at any criterion position",
        "review_predicts": (
            "ratio < 1 at r=0; > 1 for all u only at r >= .4142; peak at u* = r/(1-r); "
            "monotone rise over u in [0,2] only for r >= 2/3; ceiling (1+r)/sqrt(2), max 1.414"
        ),
        "verdict": (
            "inflation conditional on r, hump-shaped in u"
            if conditions_hold
            else "review conditions not reproduced"
        ),
    }


# --------------------------------------------------------------------------- F4
def f4_surrogate_purity(
    T_grid=(1, 7, 28, 56),
    s_th2: float = 1.0,
    s_e2: float = 1.0,
    n_draws: int = N_DRAWS,
) -> dict:
    """B2: how much of theta survives partialling cumulative dose D_T."""
    T = np.asarray(T_grid, dtype=float)

    corr = 1.0 / np.sqrt(1.0 + s_e2 / (T * s_th2))
    R2 = corr**2
    theta_var_left = 1.0 - R2
    attenuation = np.sqrt(theta_var_left)

    rng = rng_for(3)
    theta = rng.normal(0.0, math.sqrt(s_th2), n_draws)
    running = np.zeros(n_draws)
    wanted = {int(t): i for i, t in enumerate(T_grid)}
    mc_corr = np.zeros(len(T))
    for t in range(1, int(max(T_grid)) + 1):
        running += rng.normal(0.0, math.sqrt(s_e2), n_draws)
        if t in wanted:
            mc_corr[wanted[t]] = _corr(t * theta + running, theta)
    mc_R2 = mc_corr**2
    mc_left = 1.0 - mc_R2
    mc_atten = np.sqrt(mc_left)

    max_dev = float(
        max(
            np.max(np.abs(mc_corr - corr)),
            np.max(np.abs(mc_R2 - R2)),
            np.max(np.abs(mc_atten - attenuation)),
        )
    )
    return {
        "name": "F4 surrogate purity",
        "T_grid": [int(t) for t in T_grid],
        "params": {"s_th2": s_th2, "s_e2": s_e2, "n_draws": n_draws},
        "closed_form": {
            "corr_D_theta": _fl(corr),
            "R2": _fl(R2),
            "theta_var_left": _fl(theta_var_left),
            "attenuation": _fl(attenuation),
        },
        "monte_carlo": {
            "corr_D_theta": _fl(mc_corr),
            "R2": _fl(mc_R2),
            "theta_var_left": _fl(mc_left),
            "attenuation": _fl(mc_atten),
        },
        "max_abs_dev": max_dev,
        "review_table": {"R2": REVIEW_B2_R2, "attenuation": REVIEW_B2_ATTENUATION},
        "review_table_max_abs_dev": {
            "R2": float(np.max(np.abs(R2 - np.asarray(REVIEW_B2_R2)))),
            "attenuation": float(np.max(np.abs(attenuation - np.asarray(REVIEW_B2_ATTENUATION)))),
        },
        "spec_predicts": "rule is safe",
        "review_predicts": "R2 = .50/.88/.97/.98; rule deletes the person",
        "verdict": "rule deletes the person" if R2[-1] > 0.95 else "rule is safe",
    }


# --------------------------------------------------------------------------- F5
def f5_benign_arm_innocence(
    kappa_grid=(0.5, 1.0, 2.0),
    N: int = 200,
    T: int = 56,
    reps: int = REPS,
    thresholds=(0.2, 0.3, 0.4),
    lam: float = 0.5,
    criterion_noise_var: float = 0.75,
    eta_noise_var: float = 0.2,
    s_th2: float = 1.0,
    s_e2: float = 1.0,
) -> dict:
    """B2 at scale: psi = 0, so nothing is contaminated.  Does the specificity rule still kill?"""
    # Population values.  kappa is a pure scale factor on D and correlations are scale
    # free, so every closed form below is kappa-invariant.
    corr_D_theta = math.sqrt(T * s_th2 / (T * s_th2 + s_e2))
    corr_eta_theta = math.sqrt(s_th2 / (s_th2 + eta_noise_var))
    var_C = lam**2 * s_th2 + criterion_noise_var
    corr_theta_C = lam * s_th2 / math.sqrt(s_th2 * var_C)
    corr_eta_C = corr_eta_theta * corr_theta_C
    cov_resid = lam * s_th2 * s_e2 / (T * s_th2 + s_e2)
    var_eta_resid = eta_noise_var + s_th2 * s_e2 / (T * s_th2 + s_e2)
    var_C_resid = criterion_noise_var + lam**2 * s_th2 * s_e2 / (T * s_th2 + s_e2)
    semipartial = cov_resid / math.sqrt(var_eta_resid * var_C)
    partial = cov_resid / math.sqrt(var_eta_resid * var_C_resid)

    rng = rng_for(4)
    cells, raw_means, semi_means, part_means = [], [], [], []
    for kappa in kappa_grid:
        theta = rng.normal(0.0, math.sqrt(s_th2), (reps, N))
        C = lam * theta + rng.normal(0.0, math.sqrt(criterion_noise_var), (reps, N))
        eta = theta + rng.normal(0.0, math.sqrt(eta_noise_var), (reps, N))
        running = np.zeros((reps, N))
        for _ in range(T):
            running += rng.normal(0.0, math.sqrt(s_e2), (reps, N))
        D = kappa * (T * theta + running)

        raw = _rowwise_corr(eta, C)
        eta_resid = _rowwise_residualize(eta, D)
        semi = _rowwise_corr(eta_resid, C)
        part = _rowwise_corr(eta_resid, _rowwise_residualize(C, D))
        raw_means.append(float(raw.mean()))
        semi_means.append(float(semi.mean()))
        part_means.append(float(part.mean()))

        for thr in thresholds:
            clears_raw = np.abs(raw) >= thr
            killed = clears_raw & (np.abs(semi) < thr)
            rate = float(killed.mean())
            n_clear = int(clears_raw.sum())
            cells.append(
                {
                    "kappa": float(kappa),
                    "threshold": float(thr),
                    "kill_rate": rate,
                    "kill_rate_se": float(math.sqrt(max(rate * (1.0 - rate), 0.0) / reps)),
                    "raw_clear_rate": float(clears_raw.mean()),
                    "kill_rate_given_raw_clears": float(killed.sum() / n_clear) if n_clear else None,
                    "kill_rate_partial_rule": float(
                        (clears_raw & (np.abs(part) < thr)).mean()
                    ),
                }
            )

    at_03 = [c["kill_rate"] for c in cells if c["threshold"] == 0.3]
    all_rates = [c["kill_rate"] for c in cells]
    if min(at_03) >= 0.95:
        verdict = "benign arm killed at near-100% (review supported)"
    elif min(at_03) > 0.5:
        verdict = "benign arm killed in the majority of reps (review supported)"
    elif max(all_rates) <= 0.05:
        verdict = "<= 5% killed (spec supported)"
    else:
        verdict = "mixed"

    return {
        "name": "F5 benign-arm innocence",
        "kappa_grid": _fl(kappa_grid),
        "thresholds": _fl(thresholds),
        "params": {
            "N": N,
            "T": T,
            "reps": reps,
            "psi": 0.0,
            "lam": lam,
            "criterion_noise_var": criterion_noise_var,
            "eta_noise_var": eta_noise_var,
            "s_th2": s_th2,
            "s_e2": s_e2,
        },
        "closed_form": {
            "corr_D_theta": corr_D_theta,
            "corr_eta_theta": corr_eta_theta,
            "corr_theta_C": corr_theta_C,
            "corr_eta_C": corr_eta_C,
            "semipartial_eta_C_given_D": semipartial,
            "partial_eta_C_given_D": partial,
            "note": "kappa cancels out of every correlation; it only rescales D",
        },
        "monte_carlo": {
            "cells": cells,
            "mean_raw_corr": float(np.mean(raw_means)),
            "mean_semipartial_corr": float(np.mean(semi_means)),
            "mean_partial_corr": float(np.mean(part_means)),
            "mean_raw_corr_by_kappa": raw_means,
            "mean_semipartial_corr_by_kappa": semi_means,
            "mean_partial_corr_by_kappa": part_means,
        },
        "spec_predicts": "<= 5% killed at every kappa",
        "review_predicts": "near 100% killed at T=56",
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- driver
def run_all(n_draws: int = N_DRAWS, reps: int = REPS) -> dict:
    checks = {
        "F1": f1_validity_vs_adaptivity(n_draws=n_draws),
        "F2": f2_reliability_vs_validity(n_draws=n_draws),
        "F3": f3_criterion_contamination(n_draws=n_draws),
        "F4": f4_surrogate_purity(n_draws=n_draws),
        "F5": f5_benign_arm_innocence(reps=reps),
    }
    return {
        **checks,
        "verdicts": {k: v["verdict"] for k, v in checks.items()},
        "seed": SEED,
        "n_draws": n_draws,
        "reps": reps,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def main(out_path=DEFAULT_OUT, n_draws: int = N_DRAWS, reps: int = REPS) -> dict:
    payload = run_all(n_draws=n_draws, reps=reps)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    for key in ("F1", "F2", "F3", "F4", "F5"):
        print(f"{key}: {payload[key]['verdict']}")
    print(f"wrote {out}")
    return payload


if __name__ == "__main__":
    main()
