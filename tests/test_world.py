"""Tests for trapworld.params and trapworld.world (Task 2): the trap-world generator.

Occasions are 0-indexed throughout: ``version[i, t] = 0`` for ``t < switch[i]`` and 1
otherwise, so ``switch[i]`` is the index of the first adaptive occasion and the adaptive
phase has ``T_a = T - T0`` occasions under the calendar trigger.

Every test uses the frozen table's N = 200 except ``test_g1_slope_matches_belief_cov``
(N = 20000, run once) and the permutation null (N = 5000, so that a null correlation of
+-0.05 is a real threshold rather than a coin flip at N = 200).
"""

from __future__ import annotations

import numpy as np
import pytest

from trapworld import world as tw
from trapworld.params import PARAMS, canonical_hash, resolve

BASE: dict = {}  # the base cell is the frozen table itself
BIG = {"N": 20000, "T": 56, "sd_d": 0.0, "policy": "ew", "psi": 0.5, "kappa": 1.0}
SEED = 20260916

OBSERVED = ("y", "z", "dose", "version", "switch", "C_clean", "C_react", "W", "belief_logged")
HIDDEN = ("theta", "theta_t", "delta", "psi_i", "belief")


def adaptive_mean(arr: np.ndarray, version: np.ndarray) -> np.ndarray:
    """Per-person mean of ``arr[N, T]`` over that person's adaptive occasions."""
    mask = version == 1
    return (arr * mask).sum(axis=1) / mask.sum(axis=1)


@pytest.fixture(scope="module")
def base_world():
    return tw.simulate(BASE, SEED)


@pytest.fixture(scope="module")
def big_world():
    """One large, drift-free replicate; the closed-form checks share it."""
    return tw.simulate(BIG, SEED)


# --------------------------------------------------------------------------- 1. contract
def test_shapes_dtypes_version_and_delta_gate(base_world):
    w = base_world
    pr = w.params
    N, T, p, k = pr["N"], pr["T"], pr["p"], pr["n_nuisance"]

    assert w.y.shape == (N, T, p)
    assert w.z.shape == (N, T, k)
    assert w.dose.shape == (N, T)
    assert w.version.shape == (N, T)
    assert w.switch.shape == (N,)
    assert w.C_clean.shape == (N,)
    assert w.C_react.shape == (N,)
    assert w.W.shape == (N, T)
    assert w.belief_logged.shape == (N, T)
    assert w.theta.shape == (N,)
    assert w.theta_t.shape == (N, T)
    assert w.delta.shape == (N, T)
    assert w.psi_i.shape == (N,)
    assert w.belief.shape == (N, T)

    for name in ("y", "z", "dose", "C_clean", "C_react", "W", "theta", "theta_t", "delta",
                 "psi_i", "belief", "belief_logged"):
        assert getattr(w, name).dtype == np.float64, name
    assert w.version.dtype.kind == "i"
    assert w.switch.dtype.kind == "i"
    assert isinstance(w.implied_r, float)
    assert isinstance(w.dW, float)
    assert isinstance(w.params["dW_se"], float)
    assert w.jitter is None  # var_jitter = 0 in the frozen table, so there is no D2 instrument

    for name in OBSERVED + HIDDEN:
        assert np.all(np.isfinite(getattr(w, name))), name

    occ = np.arange(T)[None, :]
    assert np.array_equal(w.version, (occ >= w.switch[:, None]).astype(w.version.dtype))
    assert set(np.unique(w.version).tolist()) <= {0, 1}
    assert np.all(w.switch == pr["T0"])  # calendar trigger

    assert np.all(w.delta[w.version == 0] == 0.0)
    assert np.all(w.dose[w.version == 0] == 0.0)
    # and the gate opens exactly at the switch: every adaptive occasion is dosed and shadowed,
    # which pins the adaptive phase to T_a = T - T0 occasions with no off-by-one.
    assert np.count_nonzero(w.dose) == N * (T - pr["T0"])
    assert np.count_nonzero(w.delta) == N * (T - pr["T0"])
    assert w.belief_logged is w.belief


# --------------------------------------------------------------------------- 2. determinism
def test_determinism_and_seed_sensitivity():
    a = tw.simulate(BASE, 7)
    b = tw.simulate(BASE, 7)
    for name in OBSERVED + HIDDEN:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert a.dW == b.dW
    assert a.implied_r == b.implied_r
    assert a.params["dW_se"] == b.params["dW_se"]

    c = tw.simulate(BASE, 8)
    assert not np.array_equal(a.theta, c.theta)
    assert not np.array_equal(a.y, c.y)
    assert not np.array_equal(a.dose, c.dose)

    # S15: the spec's seed chain hands simulate a SeedSequence. Re-running the same replicate
    # object, in any order or any process, must give the same world, so simulate must not
    # consume the caller's spawn counter.
    ss = np.random.SeedSequence(PARAMS["master_seed"]).spawn(3)[1].spawn(5)[2]
    d, e = tw.simulate(BASE, ss), tw.simulate(BASE, ss)
    np.testing.assert_array_equal(d.y, e.y)
    np.testing.assert_array_equal(d.W, e.W)
    np.testing.assert_array_equal(tw.simulate(BASE, np.random.SeedSequence(7)).y, a.y)
    assert not np.array_equal(d.y, a.y)


# --------------------------------------------------------------------------- 3. streams
def test_streams_are_independent():
    a = tw.simulate({"var_jitter": 0.0}, SEED)
    b = tw.simulate({"var_jitter": 0.25}, SEED)
    for name in ("theta", "theta_t", "C_clean", "z"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
    assert not np.array_equal(a.dose, b.dose)  # the jitter arm does move the logged dose
    assert b.jitter is not None and b.jitter.shape == (b.params["N"], b.params["T"])

    c = tw.simulate({"psi": 0.0}, SEED)
    d = tw.simulate({"psi": 0.5}, SEED)
    for name in ("theta", "theta_t", "dose", "belief"):
        np.testing.assert_array_equal(getattr(c, name), getattr(d, name), err_msg=name)
    assert not np.array_equal(c.y, d.y)  # the shadow only shows up in the indicators


# --------------------------------------------------------------------------- 4. innocence 1
def test_innocence_kappa_zero():
    w = tw.simulate({"kappa": 0.0}, SEED)
    assert np.all(w.dose == 0.0)
    assert np.all(w.delta == 0.0)
    assert w.dW == 0.0
    assert w.implied_r == 0.0

    # policy = "none" forces kappa to 0, whatever the cell says
    n = tw.simulate({"policy": "none", "kappa": 2.0}, SEED)
    assert np.all(n.dose == 0.0)
    assert np.all(n.delta == 0.0)
    assert n.dW == 0.0
    assert n.implied_r == 0.0


# --------------------------------------------------------------------------- 5. innocence 2
def test_innocence_psi_zero_dose_still_targets():
    w = tw.simulate({"psi": 0.0}, SEED)
    assert np.all(w.delta == 0.0)
    dose_mean = adaptive_mean(w.dose, w.version)
    assert np.corrcoef(dose_mean, w.theta)[0, 1] > 0.5


# --------------------------------------------------------------------------- 6. good agent
def test_good_agent_dW_positive_and_beats_uniform(base_world):
    w = base_world
    assert w.dW > 0
    assert w.dW > 3 * w.params["dW_se"]
    u = tw.simulate({"policy": "uniform"}, SEED)
    assert u.dW > 0  # uniform help still helps, it just does not target
    assert u.dW < w.dW
    assert np.all(u.dose[u.version == 1] == 1.0)
    # a constant dose carries no person information, so it contaminates no criterion
    assert np.all(u.belief == 0.0)
    assert u.implied_r == 0.0
    assert np.var(adaptive_mean(u.delta, u.version)) == 0.0  # R7: uniform help adds no person variance


# --------------------------------------------------------------------------- 7. closed form
def test_g1_slope_matches_belief_cov(big_world):
    w = big_world
    last = w.delta[:, -1]
    theta = w.theta

    assert np.corrcoef(last, theta)[0, 1] > 0.8
    assert np.var(last) > 0.1

    slope = np.polyfit(theta, last, 1)[0]
    belief_mean = adaptive_mean(w.belief, w.version)
    target = (BIG["psi"] * BIG["kappa"]
              * np.cov(belief_mean, theta, ddof=1)[0, 1] / np.var(theta, ddof=1))
    assert abs(slope - target) <= 0.10 * abs(target)


def test_belief_recursions_match_spec_closed_forms(big_world):
    """Spec 2.2: ew steady-state error variance K/(2-K)*var_read; kalman P_T for a static trait."""
    K, var_read = big_world.params["gain_ew"], big_world.params["var_read"]
    ew_err = np.var(big_world.belief[:, -1] - big_world.theta)
    assert abs(ew_err - K / (2 - K) * var_read) <= 0.05 * (K / (2 - K) * var_read)
    # spec 2.6 justifies var_read = 2.0 by corr(bhat, theta) of about .86 at steady state
    assert abs(np.corrcoef(big_world.belief[:, -1], big_world.theta)[0, 1] - 0.86) < 0.02

    k = tw.simulate({**BIG, "N": 5000, "policy": "kalman"}, SEED)
    T, var_theta = k.params["T"], k.params["var_theta"]
    P_T = var_theta / (1.0 + T * var_theta / var_read)
    assert abs(np.var(k.belief[:, -1] - k.theta) - P_T) <= 0.05 * P_T


# --------------------------------------------------------------------------- 8. F8 null
def test_permutation_null_kills_the_shadow():
    cell = {"N": 5000}
    a = tw.simulate(cell, SEED)
    b = tw.simulate({**cell, "permute_dose": True}, SEED)

    corr_real = np.corrcoef(adaptive_mean(a.delta, a.version), a.theta)[0, 1]
    corr_null = np.corrcoef(adaptive_mean(b.delta, b.version), b.theta)[0, 1]
    assert corr_real > 0.8
    assert abs(corr_null) < 0.05

    np.testing.assert_array_equal(a.dose, b.dose)  # the logged dose is untouched
    assert not np.array_equal(a.delta, b.delta)


# --------------------------------------------------------------------------- 9. surrogates
def test_surrogate_g1_equals_generator_D(base_world):
    w = base_world
    T_a = w.params["T"] - w.params["T0"]
    g1 = tw.surrogate_g1(w.dose, w.switch, T_a)
    np.testing.assert_array_equal(w.delta, w.params["psi"] * g1)

    r = tw.simulate({"generator": "recency"}, SEED)
    rec = tw.surrogate_recency(r.dose, r.switch, r.params["halflife"])
    np.testing.assert_allclose(r.delta, r.params["psi"] * rec, rtol=0, atol=1e-12)


# --------------------------------------------------------------------------- 10. G4
def test_g4_unlogged_step():
    w = tw.simulate({"generator": "unlogged"}, SEED)
    t_cb = w.params["t_cb"]
    assert np.all(w.delta[:, :t_cb] == 0.0)
    expect = w.params["c0"] + w.params["c1"] * w.theta
    for t in (t_cb, w.params["T"] - 1):
        np.testing.assert_allclose(w.delta[:, t], expect)
    assert np.all(w.psi_i == 0.0)  # psi acts as 0 under G4

    dose_mean = adaptive_mean(w.dose, w.version)
    assert dose_mean.std() > 0
    assert np.corrcoef(dose_mean, w.theta)[0, 1] > 0.5


def test_nonproportional_shadow_pattern():
    """nu = (1, 1, .2, .2, 0) instead of nu = lambda, and nothing else moves."""
    a = tw.simulate({"shadow_pattern": "proportional"}, SEED)
    b = tw.simulate({"shadow_pattern": "nonproportional"}, SEED)
    for name in ("theta_t", "dose", "belief", "delta"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)

    lam = np.asarray(a.params["loadings"])
    nu = np.asarray(tw.NU_NONPROPORTIONAL)
    for j in range(a.params["p"]):
        np.testing.assert_allclose(b.y[:, -1, j] - a.y[:, -1, j],
                                   (nu[j] - lam[j]) * a.delta[:, -1], atol=1e-12)
    # channel 5 carries no shadow at all under the nonproportional pattern
    np.testing.assert_allclose(b.y[:, :, 4], a.y[:, :, 4] - lam[4] * a.delta, atol=1e-12)


# --------------------------------------------------------------------------- 11. trigger
def test_performance_trigger():
    w = tw.simulate({"trigger": "performance"}, SEED)
    sw = w.switch
    assert sw.min() >= 8
    assert sw.max() <= w.params["T0"]
    assert sw.std() > 0

    early = float((sw < w.params["T0"]).mean())
    assert 0.10 <= early <= 0.90
    assert np.corrcoef(sw, w.theta)[0, 1] < 0  # high theta trips the threshold sooner

    occ = np.arange(w.params["T"])[None, :]
    assert np.array_equal(w.version, (occ >= sw[:, None]).astype(w.version.dtype))
    assert np.all(w.dose[w.version == 0] == 0.0)


# --------------------------------------------------------------------------- 12. implied r
def test_implied_r(base_world):
    assert 0.5 < base_world.implied_r < 0.8
    assert tw.simulate({"gamma_C": 0.0}, SEED).implied_r == 0.0


# --------------------------------------------------------------------------- 13. params api
def test_resolve_and_canonical_hash():
    with pytest.raises(KeyError):
        resolve({"not_a_real_parameter": 1})

    r = resolve({"kappa": 2.0})
    assert r["kappa"] == 2.0
    assert PARAMS["kappa"] == 1.0  # PARAMS itself is never mutated
    assert r["loadings"] == PARAMS["loadings"]
    assert isinstance(r["loadings"], tuple)
    assert isinstance(resolve({"var_eps": [1.0] * 5})["var_eps"], tuple)

    assert canonical_hash({"b": 1, "a": 2}) == canonical_hash({"a": 2, "b": 1})
    assert canonical_hash({"a": 2, "b": 1}) != canonical_hash({"a": 2, "b": 2})
    assert len(canonical_hash(PARAMS)) == 64


# --------------------------------------------------------------------------- 14. F9 world
def test_zero_person_world_runs():
    w = tw.simulate({"var_theta": 0.0, "policy": "kalman"}, SEED)
    assert np.all(w.theta == 0.0)
    for name in OBSERVED + HIDDEN:
        assert np.all(np.isfinite(getattr(w, name))), name
    assert w.implied_r == 0.0
    assert np.isfinite(w.dW)
    assert np.var(w.delta[:, -1]) > 0  # the shadow is all that is left
