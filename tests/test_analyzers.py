"""The ten analyzers against the real trap world (task 4, tests (1) to (17)).

Everything here runs on `simulate` at N = 1000, seed 1, so the checks are tight enough to
be worth making.  Worlds and analyzer results are memoized across tests because an
agentSEM fit costs about 1.5 s at this N.

One pre-registered prediction does not hold and is marked xfail with the observed number
rather than loosened: under G3 heterogeneity the certificate refuses claim B through R4's
exposure test, not through R6, because R6's ratio never rises above 1 in this world (the
person effect has been estimated out of the residual it is computed from).  See
`.superpowers/sdd/trap-world/task-04-report.md`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from trapworld.analyzers import ANALYZERS, exposure_statistics, run_analyzer
from trapworld.world import simulate

N = 1000
SEED = 1

CELLS = {
    "base": {},
    "psi0": {"psi": 0.0},
    "recency": {"generator": "recency"},
    "hetero": {"generator": "hetero", "rho_psitheta": 0.6},
    "unlogged": {"generator": "unlogged"},
    "nonproportional": {"shadow_pattern": "nonproportional"},
    "performance": {"trigger": "performance"},
    "jitter": {"var_jitter": 0.25},
}

_WORLDS: dict[str, object] = {}
_RESULTS: dict[tuple[str, str], dict] = {}

SCHEMA = {"var_theta_hat", "ci", "psi_hat", "beta_clean_hat", "scores", "corr", "rmse",
          "battery", "certificate", "extras"}


def world(key: str):
    if key not in _WORLDS:
        _WORLDS[key] = simulate({**CELLS[key], "N": N}, SEED)
    return _WORLDS[key]


def result(key: str, name: str) -> dict:
    if (key, name) not in _RESULTS:
        _RESULTS[(key, name)] = run_analyzer(name, world(key))
    return _RESULTS[(key, name)]


class NoHiddenTruth:
    """A world whose hidden fields raise, except `theta` (metrics may read that one)."""

    BLOCKED = ("belief_logged", "belief", "theta_t", "delta", "psi_i", "implied_r", "dW")

    def __init__(self, base):
        object.__setattr__(self, "_base", base)

    def __getattr__(self, name):
        if name in NoHiddenTruth.BLOCKED:
            raise AssertionError(f"analyzer read hidden field {name!r}")
        return getattr(object.__getattribute__(self, "_base"), name)


# 1. the common schema -----------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ANALYZERS))
def test_every_analyzer_returns_the_common_keys(name):
    r = result("base", name)
    assert set(r) == SCHEMA, set(r) ^ SCHEMA
    assert isinstance(r["scores"], np.ndarray) and r["scores"].shape == (N,)
    assert r["var_theta_hat"] is None or isinstance(r["var_theta_hat"], float)
    assert r["ci"] is None or (len(r["ci"]) == 2 and all(isinstance(v, float) for v in r["ci"]))
    assert r["psi_hat"] is None or isinstance(r["psi_hat"], float)
    assert isinstance(r["beta_clean_hat"], float)
    assert isinstance(r["corr"], float) and isinstance(r["rmse"], float)
    assert isinstance(r["extras"], dict)
    assert (r["battery"] is not None) == (name == "naive")
    assert (r["certificate"] is not None) == (
        name in ("agentsem", "agentsem_d2", "misspecified"))
    json.dumps({"extras": r["extras"], "certificate": r["certificate"]})


def test_run_analyzer_rejects_an_unknown_name():
    with pytest.raises(KeyError):
        run_analyzer("not_an_analyzer", world("base"))


# 2, 3, 8, 9, 10: the competitors -------------------------------------------------
def test_naive_inflates_var_theta():
    assert result("base", "naive")["var_theta_hat"] > 1.3, result("base", "naive")["var_theta_hat"]


def test_anchor_only_recovers_var_theta():
    r = result("base", "anchor_only")
    assert abs(r["var_theta_hat"] - 1.0) < 0.2, r["var_theta_hat"]
    assert r["extras"]["ci_method"] == "delta_approx"
    assert r["ci"][0] < 1.0 < r["ci"][1]


def test_belief_partial_over_corrects():
    assert result("base", "belief_partial")["var_theta_hat"] < 0.5


def test_dose_partial_deletes_the_person():
    assert result("base", "dose_partial")["corr"] < 0.6


def test_epoch_fe_does_not_repair():
    assert result("base", "epoch_fe")["var_theta_hat"] > 1.3


def test_epoch_fe_equals_naive_pooled():
    """Occasion FE shifts every person's mean by the same constant, so a pooled CFA on
    person means cannot see it.  That is F12's point, exactly."""
    fe, pooled = result("base", "epoch_fe"), result("base", "naive_pooled")
    assert fe["var_theta_hat"] == pytest.approx(pooled["var_theta_hat"], rel=1e-6)
    assert fe["corr"] == pytest.approx(pooled["corr"], rel=1e-9)


# 11. the competent analyst --------------------------------------------------------
def test_detrend_repairs_g1_but_not_g2():
    g1 = result("base", "detrend")["var_theta_hat"]
    g2 = result("recency", "detrend")["var_theta_hat"]
    assert abs(g1 - 1.0) < 0.25, g1
    assert abs(g2 - 1.0) - abs(g1 - 1.0) > 0.1, (g1, g2)


# 4, 7: agentSEM where its assumptions hold ---------------------------------------
def test_agentsem_identifies_and_recovers():
    r = result("base", "agentsem")
    cert = r["certificate"]
    assert cert["B"]["status"] == "identified", (cert["B"]["status"], cert["B"]["triggered"])
    assert cert["A"]["status"] == "identified"
    assert abs(r["var_theta_hat"] - 1.0) < 0.15, r["var_theta_hat"]
    assert abs(r["psi_hat"] - 0.5) < 0.15, r["psi_hat"]
    assert r["corr"] > 0.9, r["corr"]
    assert r["extras"]["converged"] and not r["extras"]["between_dropped"]
    assert r["extras"]["psi_identified"]


def test_agentsem_beats_naive_and_the_anchor_on_person_scores():
    agent, naive, anchor = (result("base", k) for k in ("agentsem", "naive", "anchor_only"))
    assert agent["rmse"] < naive["rmse"] and agent["rmse"] < anchor["rmse"]


def test_agentsem_is_innocent_at_psi_zero():
    r = result("psi0", "agentsem")
    assert r["extras"]["innocent"] is True, r["extras"]["psi_ci90"]
    assert r["extras"]["detects"] is False
    assert r["certificate"]["B"]["status"] == "identified"
    assert abs(r["psi_hat"]) < 0.05, r["psi_hat"]


def test_psi_shifts_by_exactly_the_generated_amount():
    """With a proportional shadow the surrogate is exactly the generated exposure, so the
    psi = 0 and psi = .5 cells differ in psi alone."""
    base, zero = result("base", "agentsem"), result("psi0", "agentsem")
    assert base["psi_hat"] - zero["psi_hat"] == pytest.approx(0.5, abs=0.01)
    assert base["extras"]["phi_B_raw"] == pytest.approx(zero["extras"]["phi_B_raw"], rel=1e-4)


# 5, 6, 14, 15, 16, 17: the certificate refusing ------------------------------------
def test_unlogged_driver_makes_claim_b_unsupported():
    cert = result("unlogged", "agentsem")["certificate"]
    assert cert["B"]["status"] == "unsupported"
    assert "R4" in cert["B"]["triggered"]
    assert cert["B"]["estimate"] is None
    assert cert["B"]["numbers"]["phi_B_raw"] is not None


def test_nonproportional_loading_pattern_trips_r5():
    cert = result("nonproportional", "agentsem")["certificate"]
    assert cert["B"]["status"] == "unsupported"
    assert "R5" in cert["B"]["triggered"], cert["B"]["triggered"]
    assert cert["B"]["numbers"]["dcfi_metric"] > 0.01


@pytest.mark.xfail(strict=True, reason=(
    "observed: B unsupported through R4 with hetero_ratio 0.510, not bounded through R6 "
    "(N=1000 seed 1, generator hetero, rho_psitheta .6). R6's ratio is below 1 in every "
    "cell measured (0.26 to 0.96) because the person effect has been estimated out of the "
    "residual the numerator is computed from, so the rule as defined cannot fire; the "
    "refusal here comes from R4's exposure test at p 0.0098, which is the same borderline "
    "value the base cell shows (p 0.0105). See task-04-report.md."))
def test_hetero_generator_bounds_claim_b_through_r6():
    cert = result("hetero", "agentsem")["certificate"]
    assert cert["B"]["status"] == "bounded"
    assert "R6" in cert["B"]["triggered"]
    assert cert["B"]["estimate"] == pytest.approx(result("hetero", "agentsem")["extras"]["phi_A"])


def test_hetero_generator_is_refused_somehow():
    """What actually happens, kept next to the xfail so the record is not only a failure."""
    r = result("hetero", "agentsem")
    cert = r["certificate"]
    assert cert["B"]["status"] == "unsupported" and cert["B"]["triggered"] == ["R4"]
    assert cert["B"]["numbers"]["hetero_ratio"] < 1.5
    assert cert["B"]["numbers"]["exposure_p"] < 0.01


def test_performance_trigger_flags_the_anchor():
    cert = result("performance", "agentsem")["certificate"]
    assert "R2" in cert["A"]["triggered"] and "R2" in cert["B"]["triggered"]
    assert cert["A"]["status"] == "bounded" and cert["B"]["status"] == "bounded"
    assert cert["B"]["estimate"] == cert["A"]["estimate"]        # the anchor fallback
    assert abs(cert["B"]["numbers"]["switch_corr"]) > 0.1
    assert cert["B"]["numbers"]["switch_p"] < 0.01


def test_recency_world_trips_the_exposure_rule():
    """(16) the G1 surrogate against a G2 world."""
    cert = result("recency", "agentsem")["certificate"]
    numbers = cert["B"]["numbers"]
    assert cert["B"]["status"] == "unsupported" and "R4" in cert["B"]["triggered"]
    assert numbers["exposure_p"] < 0.01, numbers["exposure_p"]
    assert numbers["max_exposure_corr"] > 0.2, numbers["max_exposure_corr"]
    assert abs(numbers["epoch_shift"]) < numbers["shift_threshold"]   # the pooled test is blind


def test_misspecified_surrogate_is_caught_by_the_exposure_rule():
    """(17) the same world, the wrong surrogate."""
    r = result("base", "misspecified")
    cert = r["certificate"]
    assert r["extras"]["surrogate"] == "recency_h2"
    assert cert["B"]["status"] == "unsupported" and "R4" in cert["B"]["triggered"]
    assert cert["B"]["numbers"]["exposure_p"] < 0.01
    assert cert["B"]["estimate"] is None
    assert isinstance(cert["B"]["numbers"]["phi_B_raw"], float)
    assert isinstance(r["extras"]["phi_B_raw"], float)


# 12. the D2 arm --------------------------------------------------------------------
def test_d2_recovers_psi_when_the_jitter_is_logged():
    r = result("jitter", "agentsem_d2")
    assert abs(r["psi_hat"] - 0.5) < 0.15, r["psi_hat"]
    assert r["certificate"]["B"]["status"] == "identified", r["certificate"]["B"]["triggered"]
    assert r["extras"]["first_stage_F"] > 10.0


def test_d2_without_jitter_is_unsupported():
    r = result("base", "agentsem_d2")
    cert = r["certificate"]
    assert cert["B"]["status"] == "unsupported" and "D2" in cert["B"]["triggered"]
    assert r["var_theta_hat"] is None and r["ci"] is None
    assert cert["A"]["status"] == "identified"      # the anchor still certifies claim A


# 13. the information guard ----------------------------------------------------------
@pytest.mark.parametrize("name", sorted(set(ANALYZERS) - {"belief_partial"}))
def test_no_analyzer_but_belief_partial_reads_hidden_fields(name):
    guarded = NoHiddenTruth(world("base"))
    r = run_analyzer(name, guarded)
    assert set(r) == SCHEMA


def test_belief_partial_is_the_one_that_needs_the_belief():
    with pytest.raises(AssertionError):
        run_analyzer("belief_partial", NoHiddenTruth(world("base")))


# the exposure statistic itself -------------------------------------------------------
def test_exposure_statistic_is_silent_without_exposure_variance():
    rng = np.random.default_rng(0)
    resid = rng.standard_normal((50, 30))
    adaptive = np.ones((50, 30), bool)
    out = exposure_statistics(resid, adaptive, np.ones(50), windows=3)
    assert out["exposure_corr"] == [0.0, 0.0, 0.0]
    assert out["exposure_p"] == 1.0 and out["max_exposure_corr"] == 0.0


def test_exposure_statistic_finds_a_planted_pattern():
    rng = np.random.default_rng(0)
    n, T = 400, 30
    x = rng.standard_normal(n)
    adaptive = np.ones((n, T), bool)
    resid = rng.standard_normal((n, T)) * 0.3
    resid[:, :10] += 0.5 * x[:, None]              # window 1 only
    out = exposure_statistics(resid, adaptive, x, windows=3)
    assert out["exposure_corr"][0] > 0.8
    assert abs(out["exposure_corr"][1]) < 0.15 and abs(out["exposure_corr"][2]) < 0.15
    assert out["exposure_p"] < 1e-10


def test_exposure_windows_split_by_rank_within_the_person():
    """Ragged adaptive phases still give three equal windows per person."""
    n, T = 6, 20
    switch = np.array([2, 5, 8, 11, 14, 2])
    adaptive = np.arange(T)[None, :] >= switch[:, None]
    resid = np.zeros((n, T))
    x = np.arange(n, dtype=float)
    out = exposure_statistics(resid, adaptive, x, windows=3)
    assert out["exposure_corr"] == [0.0, 0.0, 0.0]   # constant residual, no pattern


# the bootstrap ------------------------------------------------------------------------
def test_bootstrap_replaces_the_interval_and_is_deterministic():
    small = simulate({"N": 200}, 7)
    a = run_analyzer("agentsem", small, bootstrap=8)
    b = run_analyzer("agentsem", small, bootstrap=8)
    assert a["extras"]["ci_method"] == "bootstrap"
    assert a["extras"]["boot_ci_phi"] == b["extras"]["boot_ci_phi"]
    assert a["extras"]["boot_ci_psi"] == b["extras"]["boot_ci_psi"]
    assert a["extras"]["boot_reps_ok"] == 8
    assert a["ci"] == tuple(a["extras"]["boot_ci_phi"])
    assert a["extras"]["ci_B_raw"] != a["extras"]["boot_ci_phi"]
    plain = run_analyzer("agentsem", small)
    assert plain["extras"]["ci_method"] == "hessian_delta"
    assert plain["extras"]["phi_B_raw"] == a["extras"]["phi_B_raw"]
