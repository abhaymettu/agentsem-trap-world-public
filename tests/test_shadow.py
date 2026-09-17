"""Tests for the policy-aware state-space shadow model (task 4: D1 + D5, and D2's IV).

The data here are generated from the shadow model itself, not from `trapworld.world`:
the point of this file is that the estimator recovers what it claims to recover and
reports honest uncertainty.  The trap world is what `tests/test_analyzers.py` is for.

The correlated-random-effects (Mundlak) term of spec 11 A1 is what makes psi recoverable
here.  `test_between_false_reproduces_the_between_person_bias` keeps the old model in the
suite so the mechanism stays pinned: with `between=False` the same data give psi +0.06 and
phi -0.06, because the likelihood treats theta as independent of `g` while the trap makes
the person's exposure informative about theta.

One of the task brief's pre-registered predictions still does not hold and is marked xfail
with the observed numbers rather than loosened: the IV estimator is unbiased but far
noisier than one draw of +-.05 allows (SD 0.101 at N = 2000 over 40 seeds), and its
homoskedastic standard error understates that SD by about a factor of 1.5 because the
structural error is autocorrelated and the instrument is a random walk.
"""

from __future__ import annotations

import numpy as np
import pytest

from trapworld.shadow import _kalman, _patterns, fit_shadow, iv_shadow

T = 56
SWITCH = 14
T_A = T - SWITCH


def _drift(rng, n, rho_d, var_d):
    """Stationary AR(1) with marginal variance `var_d`."""
    d = np.zeros((n, T))
    if var_d <= 0:
        return d
    z = rng.standard_normal((n, T))
    d[:, 0] = np.sqrt(var_d) * z[:, 0]
    inn = np.sqrt(var_d * (1.0 - rho_d ** 2))
    for t in range(1, T):
        d[:, t] = rho_d * d[:, t - 1] + inn * z[:, t]
    return d


def make_data(n=2000, seed=0, phi=1.0, psi=0.5, rho_d=0.9, var_d=0.09, var_e=0.3,
              g_noise=0.5, exogenous=False):
    """`eta = theta + d + psi * g + e` with `g` a person-specific ramp, as in the world.

    `g` is deliberately correlated with theta (that is the trap: the policy doses the
    people it believes are high).  `exogenous=True` breaks that correlation, which is the
    control condition for the psi bias.
    """
    rng = np.random.default_rng(seed)
    theta = np.sqrt(phi) * rng.standard_normal(n)
    d = _drift(rng, n, rho_d, var_d)
    occ = np.arange(T)
    ramp = np.where(occ >= SWITCH, (occ - SWITCH + 1) / T_A, 0.0)
    drive = (0.0 if exogenous else 1.0) * theta + g_noise * rng.standard_normal(n)
    g = drive[:, None] * ramp[None, :]
    e = np.sqrt(var_e) * rng.standard_normal((n, T))
    return theta[:, None] + d + psi * g + e, g, theta


def make_iv_data(n=2000, seed=0, phi=1.0, psi=0.5, rho_d=0.9, var_d=0.09, var_e=0.3,
                 kappa=1.0, var_jitter=0.25):
    """The D2 structure: dose = kappa * theta + jitter, `g` and `z` both cumulative."""
    rng = np.random.default_rng(seed)
    theta = np.sqrt(phi) * rng.standard_normal(n)
    d = _drift(rng, n, rho_d, var_d)
    occ = np.arange(T)
    adaptive = np.tile(occ >= SWITCH, (n, 1))
    jitter = np.sqrt(var_jitter) * rng.standard_normal((n, T))
    dose = np.where(adaptive, kappa * theta[:, None] + jitter, 0.0)
    g = np.where(adaptive, np.cumsum(dose, axis=1) / T_A, 0.0)
    z = np.where(adaptive, np.cumsum(np.where(adaptive, jitter, 0.0), axis=1) / T_A, 0.0)
    e = np.sqrt(var_e) * rng.standard_normal((n, T))
    eta = theta[:, None] + d + psi * g + e
    return eta, g, z, adaptive, theta


def same(a, b) -> bool:
    """Equality that treats NaN as equal, for comparing two fit results."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


@pytest.fixture(scope="module")
def base_fit():
    eta, g, theta = make_data()
    return fit_shadow(eta, g, drift=True), theta


@pytest.fixture(scope="module")
def iv_replicates():
    """40 independent draws of the D2 structure at N = 2000."""
    psi, se, F = [], [], []
    for seed in range(40):
        eta, g, z, mask, _ = make_iv_data(seed=seed)
        iv = iv_shadow(eta, g, z, mask)
        psi.append(iv["psi"]); se.append(iv["se"]); F.append(iv["first_stage_F"])
    return np.array(psi), np.array(se), np.array(F)


# --------------------------------------------------------------- the filter itself
def test_kalman_loglik_equals_direct_multivariate_normal():
    """The recursion must be the exact likelihood, masked or not."""
    rng = np.random.default_rng(0)
    n, q = 5, 7
    phi, psi, rho_d, var_d, var_e = 1.3, 0.4, 0.8, 0.07, 0.25
    eta = rng.standard_normal((n, q))
    g = rng.standard_normal((n, q))
    idx = np.arange(q)
    Sigma = phi + var_d * rho_d ** np.abs(idx[:, None] - idx[None, :]) + var_e * np.eye(q)
    r = eta - psi * g

    def direct(keep):
        S = Sigma[np.ix_(keep, keep)]
        Si = np.linalg.inv(S)
        _, logdet = np.linalg.slogdet(S)
        return sum(-0.5 * (len(keep) * np.log(2 * np.pi) + logdet + r[i, keep] @ Si @ r[i, keep])
                   for i in range(n))

    for keep in (np.arange(q), np.arange(3, q)):
        mask = np.zeros((n, q), bool)
        mask[:, keep] = True
        pats, inv = _patterns(mask)
        ll, scores, ok = _kalman(eta, g, pats, inv, phi, psi, rho_d, var_d, var_e, True)
        assert ok
        assert ll == pytest.approx(direct(keep), rel=1e-10)

    # scores are the smoother E[theta | eta] for a constant state
    mask = np.ones((n, q), bool)
    pats, inv = _patterns(mask)
    _, scores, _ = _kalman(eta, g, pats, inv, phi, psi, rho_d, var_d, var_e, True)
    Si = np.linalg.inv(Sigma)
    smooth = np.array([phi * np.ones(q) @ Si @ r[i] for i in range(n)])
    assert np.allclose(scores, smooth, atol=1e-12)


def test_kalman_b_term_is_an_observation_offset():
    """`b * gbar` shifts the observation and nothing else in the recursion."""
    rng = np.random.default_rng(1)
    n, q = 6, 8
    eta = rng.standard_normal((n, q))
    g = rng.standard_normal((n, q))
    gbar = g.mean(axis=1)
    mask = np.ones((n, q), bool)
    pats, inv = _patterns(mask)
    args = (pats, inv, 1.0, 0.4, 0.8, 0.07, 0.25, True)
    ll_b, state_b, _ = _kalman(eta, g, *args, gbar=gbar, b=0.7)
    ll_0, state_0, _ = _kalman(eta - 0.7 * gbar[:, None], g, *args)
    assert ll_b == pytest.approx(ll_0, rel=1e-12)
    assert np.allclose(state_b, state_0, atol=1e-12)


# 1. recovery (the brief's tolerances: phi .06, psi .04, var_e .05 at N = 2000)
def test_recovers_phi_and_var_e(base_fit):
    fit, _ = base_fit
    assert fit["converged"]
    assert abs(fit["phi"] - 1.0) < 0.06, fit["phi"]
    assert abs(fit["var_e"] - 0.3) < 0.05, fit["var_e"]
    assert 0.8 < fit["rho_d"] < 0.97, fit["rho_d"]
    assert abs(fit["var_d"] - 0.09) < 0.04, fit["var_d"]


def test_recovers_psi(base_fit):
    """The Mundlak term is what makes this pass; see the between=False test below."""
    fit, _ = base_fit
    assert fit["psi_identified"] and not fit["between_dropped"]
    assert abs(fit["psi"] - 0.5) < 0.04, fit["psi"]


def test_phi_is_the_total_variance(base_fit):
    fit, _ = base_fit
    assert fit["phi"] == pytest.approx(fit["phi_u"] + fit["b"] ** 2 * fit["var_gbar"], rel=1e-12)
    assert fit["phi_u"] < fit["phi"]          # the b term carries real between-person variance
    assert fit["se"]["phi"] > 0 and np.isfinite(fit["se"]["phi"])
    lo, hi = fit["ci"]["phi"]
    assert lo < fit["phi"] < hi


def test_between_false_reproduces_the_between_person_bias():
    """The old random-effects model, kept so the mechanism stays pinned in the suite."""
    eta, g, _ = make_data(seed=0)
    old = fit_shadow(eta, g, drift=True, between=False)
    assert old["psi"] - 0.5 > 0.03, old["psi"]           # observed +0.057
    assert 1.0 - old["phi"] > 0.03, old["phi"]           # observed -0.056, the mirror image
    new = fit_shadow(eta, g, drift=True, between=True)
    assert abs(new["psi"] - 0.5) < abs(old["psi"] - 0.5)
    assert abs(new["phi"] - 1.0) < abs(old["phi"] - 1.0)


def test_exogenous_g_is_the_control_for_that_bias():
    """Same estimator, `g` no longer correlated with theta: the old model is fine too."""
    eta, g, _ = make_data(seed=0, exogenous=True, g_noise=1.0)
    clean = fit_shadow(eta, g, drift=True, between=False)
    assert abs(clean["psi"] - 0.5) < 0.04, clean["psi"]
    assert abs(clean["phi"] - 1.0) < 0.06, clean["phi"]


def test_between_term_is_null_on_exogenous_g():
    """(i) b is indistinguishable from 0 and psi does not move when it is added."""
    eta, g, _ = make_data(seed=0, exogenous=True, g_noise=1.0)
    with_b = fit_shadow(eta, g, drift=True, between=True)
    without = fit_shadow(eta, g, drift=True, between=False)
    assert abs(with_b["b"]) < 2.0 * with_b["se"]["b"], (with_b["b"], with_b["se"]["b"])
    assert abs(with_b["psi"] - without["psi"]) < 0.01, (with_b["psi"], without["psi"])


def test_no_between_variation_drops_the_b_term():
    """(ii) every person on the same exposure path: b is not identified, so it goes."""
    eta, _, _ = make_data(n=300, seed=3)
    occ = np.arange(T)
    ramp = np.where(occ >= SWITCH, (occ - SWITCH + 1) / T_A, 0.0)
    g = np.tile(ramp, (300, 1))
    dropped = fit_shadow(eta, g, drift=True, between=True)
    plain = fit_shadow(eta, g, drift=True, between=False)
    assert dropped["between_dropped"] and not plain["between_dropped"]
    assert dropped["var_gbar"] < 1e-12 and dropped["b"] == 0.0
    assert dropped["se"]["b"] is None and plain["se"]["b"] is None
    keys = set(dropped) - {"between_dropped"}
    assert all(same(dropped[k], plain[k]) for k in keys), [
        k for k in keys if not same(dropped[k], plain[k])]


def test_zero_g_leaves_psi_unidentified():
    """(iii) no exposure anywhere: psi is fixed at 0 rather than fitted to noise."""
    eta, _, _ = make_data(n=300, seed=3)
    fit = fit_shadow(eta, np.zeros_like(eta), drift=True, between=True)
    assert fit["psi_identified"] is False
    assert fit["psi"] == 0.0 and fit["se"]["psi"] is None
    assert all(np.isnan(v) for v in fit["ci"]["psi"])
    assert fit["between_dropped"] and fit["b"] == 0.0
    assert np.isfinite(fit["phi"]) and fit["phi"] > 0
    assert np.isfinite(fit["se"]["phi"])
    assert "psi" not in fit["x_transformed"] and "b" not in fit["x_transformed"]


def test_warm_start_reaches_the_same_optimum():
    """(iv) the bootstrap's warm start must not move the answer."""
    eta, g, _ = make_data(n=500, seed=11)
    base = fit_shadow(eta, g, drift=True)
    warm = fit_shadow(eta, g, drift=True, x0=base["x_transformed"])
    assert abs(warm["loglik"] - base["loglik"]) < 1e-6
    assert warm["phi"] == pytest.approx(base["phi"], rel=1e-5)
    assert warm["psi"] == pytest.approx(base["psi"], rel=1e-5)
    array_start = fit_shadow(eta, g, drift=True,
                             x0=[base["x_transformed"][k] for k in base["x_transformed"]])
    assert abs(array_start["loglik"] - base["loglik"]) < 1e-6
    with pytest.raises(ValueError):
        fit_shadow(eta, g, drift=True, x0=[0.0, 0.0])


def test_hessian_false_keeps_the_point_estimates_and_drops_the_errors():
    """(v) what the bootstrap fits use."""
    eta, g, _ = make_data(n=500, seed=11)
    full = fit_shadow(eta, g, drift=True)
    fast = fit_shadow(eta, g, drift=True, hessian=False)
    for key in ("phi", "phi_u", "psi", "b", "rho_d", "var_d", "var_e", "loglik"):
        assert fast[key] == full[key], key
    assert np.array_equal(fast["scores"], full["scores"])
    assert all(v is None or np.isnan(v) for v in fast["se"].values())
    assert all(v is None or all(np.isnan(x) for x in v) for v in fast["ci"].values())


# 2. the reported standard error is honest
def test_reported_se_matches_sampling_sd():
    phis, ses = [], []
    for seed in range(20):
        eta, g, _ = make_data(n=500, seed=100 + seed)
        fit = fit_shadow(eta, g, drift=True)
        phis.append(fit["phi"])
        ses.append(fit["se"]["phi"])
    emp_sd = float(np.std(phis, ddof=1))
    mean_se = float(np.mean(ses))
    ratio = emp_sd / mean_se
    assert abs(float(np.mean(phis)) - 1.0) < 0.05, float(np.mean(phis))
    assert 1.0 / 1.6 <= ratio <= 1.6, (emp_sd, mean_se, ratio)


# 3. person scores track theta
def test_scores_track_theta(base_fit):
    fit, theta = base_fit
    assert np.corrcoef(fit["scores"], theta)[0, 1] > 0.9
    assert fit["scores"].shape == theta.shape
    assert fit["resid"].shape == (theta.size, T)
    assert abs(float(fit["resid"].mean())) < 0.05


# 4. the no-drift model
def test_no_drift_model_recovers_phi_and_psi():
    eta, g, _ = make_data(n=2000, seed=7, var_d=0.0)
    fit = fit_shadow(eta, g, drift=False)
    assert fit["converged"]
    assert abs(fit["phi"] - 1.0) < 0.06, fit["phi"]
    assert abs(fit["psi"] - 0.5) < 0.04, fit["psi"]
    assert abs(fit["var_e"] - 0.3) < 0.05, fit["var_e"]
    assert fit["rho_d"] == 0.0 and fit["var_d"] == 0.0


# 5. the D2 instrumental-variable estimator
@pytest.mark.xfail(strict=True, reason=(
    "observed psi 0.5801 on one draw at N=2000 (brief predicts within .05). The estimator is "
    "unbiased (mean 0.4845 over 40 seeds) but has SD 0.1006, so one draw inside +-.05 happens "
    "38% of the time. first_stage_F holds: mean 1015, min 382."))
def test_iv_shadow_recovers_psi_on_one_draw():
    eta, g, z, mask, _ = make_iv_data()
    iv = iv_shadow(eta, g, z, mask)
    assert iv["first_stage_F"] > 10.0, iv["first_stage_F"]
    assert abs(iv["psi"] - 0.5) < 0.05, iv["psi"]


def test_iv_shadow_is_unbiased_and_strong(iv_replicates):
    psi, se, F = iv_replicates
    assert abs(psi.mean() - 0.5) < 0.05, psi.mean()
    assert F.min() > 10.0, F.min()
    assert np.all(se > 0)


def test_iv_homoskedastic_se_understates_the_sampling_sd(iv_replicates):
    """Reported for the coverage audit: the plain IV sandwich is anti-conservative here."""
    psi, se, _ = iv_replicates
    ratio = float(psi.std(ddof=1) / se.mean())
    assert ratio > 1.2, ratio


# 6. mask semantics
def test_mask_all_true_equals_default():
    eta, g, _ = make_data(n=300, seed=3)
    a = fit_shadow(eta, g, drift=True)
    b = fit_shadow(eta, g, drift=True, mask=np.ones(eta.shape, bool))
    assert a["phi"] == pytest.approx(b["phi"], rel=1e-12)
    assert a["loglik"] == pytest.approx(b["loglik"], rel=1e-12)


def test_masked_occasions_are_ignored():
    """Corrupting occasions that the mask excludes must not move the fit."""
    eta, g, _ = make_data(n=300, seed=4)
    mask = np.ones(eta.shape, bool)
    mask[:, :SWITCH] = False
    clean = fit_shadow(eta, g, drift=True, mask=mask)
    eta2 = eta.copy()
    eta2[:, :SWITCH] += 25.0
    dirty = fit_shadow(eta2, g, drift=True, mask=mask)
    assert clean["phi"] == pytest.approx(dirty["phi"], rel=1e-8)
    assert clean["psi"] == pytest.approx(dirty["psi"], rel=1e-8)
    assert clean["loglik"] == pytest.approx(dirty["loglik"], rel=1e-8)


def test_gbar_follows_the_mask():
    """`gbar` is the mean over included occasions, so the mask changes it."""
    eta, g, _ = make_data(n=300, seed=4)
    mask = np.ones(eta.shape, bool)
    mask[:, :SWITCH] = False
    full = fit_shadow(eta, g, drift=True)
    adaptive_only = fit_shadow(eta, g, drift=True, mask=mask)
    assert adaptive_only["var_gbar"] > full["var_gbar"]
    ratio = (T / (T - SWITCH)) ** 2
    assert adaptive_only["var_gbar"] == pytest.approx(full["var_gbar"] * ratio, rel=1e-10)


def test_person_specific_masks_are_supported():
    """Ragged anchors (the performance trigger) must not break the shared recursion."""
    eta, g, _ = make_data(n=400, seed=5)
    switch = np.repeat([8, 11, 14, 17], 100)
    mask = np.arange(T)[None, :] >= switch[:, None]
    fit = fit_shadow(eta, g, drift=True, mask=mask)
    assert np.isfinite(fit["loglik"]) and fit["scores"].shape == (400,)
