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


class GuardedParams(dict):
    """`world.params` with only the design constants the analyst is allowed to know.

    The frozen table carries the answer (`psi`, `kappa`, `var_theta`, `generator`,
    `kappa_eff`) next to genuine design facts, so a guard that only blocks attributes
    leaves `world.params["psi"]` wide open.  Everything outside `ALLOWED` raises.
    """

    #: T and T0 are the logged occasion count and the deployment's own switch time;
    #: `smd_psi` is the pre-registered equivalence margin; `bootstrap_seed` is an RNG
    #: handle for F17.  Nothing else in the table is observable.
    ALLOWED = frozenset({"T", "T0", "smd_psi", "bootstrap_seed"})

    def __getitem__(self, key):
        self._check(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._check(key)
        return super().get(key, default)

    @staticmethod
    def _check(key):
        if key not in GuardedParams.ALLOWED:
            raise AssertionError(f"analyzer read hidden parameter params[{key!r}]")


class NoHiddenTruth:
    """A world whose hidden fields raise, whose params are guarded, and which counts
    every read of `theta`.

    `theta` is not blocked because `analyzers._result` reads it once to report `corr`
    and `rmse`.  It is counted instead, so an estimator that used it would show up as a
    second read even though the first one is legitimate.
    """

    BLOCKED = ("belief_logged", "belief", "theta_t", "delta", "psi_i", "implied_r", "dW")

    def __init__(self, base, allow=()):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_allow", frozenset(allow))
        object.__setattr__(self, "_params", GuardedParams(base.params))
        object.__setattr__(self, "reads", {})

    def __getattr__(self, name):
        object.__getattribute__(self, "reads")[name] = \
            object.__getattribute__(self, "reads").get(name, 0) + 1
        if name in NoHiddenTruth.BLOCKED and name not in object.__getattribute__(self, "_allow"):
            raise AssertionError(f"analyzer read hidden field {name!r}")
        if name == "params":
            return object.__getattribute__(self, "_params")
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
    # spec 4's primary tolerance is 0.10 absolute, so that is what the gate asserts.
    # Observed at N = 1000 seed 1: 0.93554, |bias| 0.06446, margin 0.0355.
    assert abs(r["var_theta_hat"] - 1.0) <= 0.10, r["var_theta_hat"]
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
    """No hidden attribute, no hidden parameter, and exactly one read of `theta`.

    The single read is `analyzers._result`, which reports `corr` and `rmse`.  Any
    estimator that consulted the truth would read it a second time.
    """
    guarded = NoHiddenTruth(world("base"))
    r = run_analyzer(name, guarded)
    assert set(r) == SCHEMA
    assert guarded.reads.get("theta") == 1, guarded.reads
    assert not (set(guarded.reads) & set(NoHiddenTruth.BLOCKED)), guarded.reads


def test_the_guard_would_catch_a_hidden_parameter_read():
    """The guard has to be able to fail, or it is decoration."""
    guarded = NoHiddenTruth(world("base"))
    with pytest.raises(AssertionError):
        guarded.params["psi"]
    with pytest.raises(AssertionError):
        guarded.params["kappa"]
    with pytest.raises(AssertionError):
        guarded.params["var_theta"]
    with pytest.raises(AssertionError):
        guarded.params.get("generator")
    assert guarded.params["T0"] == world("base").params["T0"]


def test_the_bootstrap_path_reads_nothing_hidden_either():
    guarded = NoHiddenTruth(world("base"))
    run_analyzer("agentsem", guarded, bootstrap=3)
    assert guarded.reads.get("theta") == 1, guarded.reads
    assert not (set(guarded.reads) & set(NoHiddenTruth.BLOCKED)), guarded.reads


def test_belief_partial_is_the_one_that_needs_the_belief():
    with pytest.raises(AssertionError):
        run_analyzer("belief_partial", NoHiddenTruth(world("base")))
    # and with the belief allowed through it is the only truth it needs
    guarded = NoHiddenTruth(world("base"), allow=("belief_logged",))
    r = run_analyzer("belief_partial", guarded)
    assert set(r) == SCHEMA
    assert guarded.reads.get("theta") == 1, guarded.reads
    assert guarded.reads.get("belief") is None


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
    """Ragged adaptive phases still give three windows per person, by the person's own rank.

    The pattern is planted in each person's own first third, computed here by an
    independent loop.  A split on absolute occasion instead would smear it: measured,
    that wrong split gives window correlations [0.999, 0.919, -0.008] on this data,
    so the window-1 assertion below is what makes the test able to fail.
    """
    rng = np.random.default_rng(0)
    n, T, windows = 300, 30, 3
    switch = np.tile([2, 5, 8, 11, 14], n // 5)
    adaptive = np.arange(T)[None, :] >= switch[:, None]
    x = rng.standard_normal(n)
    resid = 0.05 * rng.standard_normal((n, T))
    for i in range(n):
        occ = np.flatnonzero(adaptive[i])
        block = np.floor(np.arange(occ.size) * windows / occ.size).astype(int)
        resid[i, occ[block == 0]] += 0.8 * x[i]

    out = exposure_statistics(resid, adaptive, x, windows=windows)
    assert out["exposure_corr"][0] > 0.9, out["exposure_corr"]
    assert abs(out["exposure_corr"][1]) < 0.3, out["exposure_corr"]
    assert abs(out["exposure_corr"][2]) < 0.3, out["exposure_corr"]
    assert out["exposure_p"] < 1e-10


def test_exposure_statistic_is_constant_safe_on_a_ragged_mask():
    """A constant residual has no pattern to find, ragged phases or not."""
    n, T = 6, 20
    switch = np.array([2, 5, 8, 11, 14, 2])
    adaptive = np.arange(T)[None, :] >= switch[:, None]
    out = exposure_statistics(np.zeros((n, T)), adaptive, np.arange(n, dtype=float), windows=3)
    assert out["exposure_corr"] == [0.0, 0.0, 0.0]
    assert out["exposure_p"] == 1.0


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


# ---------------------------------------- defects found by the task 6 adversarial pass
def test_d2_without_jitter_reports_no_person_metrics():
    """Task 6 defect D3, fixed: an arm whose design is unavailable has no person scores,
    so it reports NaN rather than borrowing the naive analyzer's numbers."""
    d2, naive = result("base", "agentsem_d2"), result("base", "naive")
    assert d2["var_theta_hat"] is None and d2["ci"] is None
    assert d2["certificate"]["B"]["status"] == "unsupported"
    assert np.isnan(d2["corr"]) and np.isnan(d2["rmse"]) and np.isnan(d2["beta_clean_hat"])
    assert np.all(np.isnan(d2["scores"]))
    assert np.isfinite(naive["corr"])


def test_zero_exposure_keeps_the_r3_number_finite_and_reportable():
    """Task 6 defect D4, fixed: with policy `none` the surrogate is identically zero and is
    not in the model, so it is not in R3's design either.  The condition number stays
    finite and reported, R3 stays silent, and claim B is certified from the plain
    measurement model with psi fixed at 0."""
    from trapworld import metrics

    w = simulate({"N": 200, "policy": "none"}, 5)
    r = run_analyzer("agentsem", w)
    cert = r["certificate"]
    assert np.isfinite(cert["B"]["numbers"]["cond_number"])
    assert cert["B"]["numbers"]["cond_number"] < 30
    assert "R3" not in cert["B"]["triggered"]
    assert not r["extras"]["psi_identified"] and r["extras"]["between_dropped"]

    block = metrics.replicate_metrics(w, {"agentsem": r})["analyzers"]["agentsem"]
    assert "R3" not in block["triggered_B"]
    assert block["cond_number"] is not None
    assert block["false_refusal_strict"] in (True, False)


def test_d2_interval_carries_the_instrument_error():
    """Task 6 defect D2, fixed: the D2 arm's interval for Var(theta) includes the IV
    error in psi through the delta method, so it is wider than the variance-only band."""
    w = simulate({"N": 200, "var_jitter": 0.25}, 11)
    d2 = run_analyzer("agentsem_d2", w)
    assert d2["extras"]["ci_method"] == "delta_iv"
    lo, hi = d2["extras"]["ci_B_raw"]
    variance_only = np.sqrt(2.0 / 199) * (d2["extras"]["phi_B_raw"]
                                          + d2["extras"]["score_error_var"]) * 2 * 1.96
    assert hi - lo > variance_only
    assert np.isfinite(d2["extras"]["psi_se"]) and d2["extras"]["psi_se"] > 0


# -------------------------------------- spec 11 A1 on the real world (task 6b item 1)
def _eta_and_g(cell, n, seed=SEED):
    """The anchored occasion scores and the frozen surrogate for one real replicate."""
    from trapworld.analyzers import _anchor_pipeline, _masks
    from trapworld.world import surrogate_g1

    w = simulate({**cell, "N": n}, seed)
    _, anchor = _masks(w)
    g = surrogate_g1(np.asarray(w.dose, float), np.asarray(w.switch, int),
                     w.params["T"] - w.params["T0"])
    *_, eta = _anchor_pipeline(np.asarray(w.y, float), anchor)
    return eta, g


def test_mundlak_total_is_exactly_phi_u_plus_b2_var_gbar():
    """phi = phi_u + b^2 Var(gbar) on the real base world, to the bit, and within
    tolerance of the truth (task-04-report: 0.936 at N = 1000 seed 1)."""
    extras = result("base", "agentsem")["extras"]
    total = extras["phi_u"] + extras["b"] ** 2 * extras["var_gbar"]
    assert extras["phi_B_raw"] == total          # exact, not approx
    assert abs(extras["phi_B_raw"] - 1.0) <= 0.10, extras["phi_B_raw"]
    assert extras["var_gbar"] > 0 and not extras["between_dropped"]


def test_between_false_still_shows_the_psi_bias_on_the_real_world():
    """The mechanism pin of spec 11 A1, measured on `simulate` rather than on the
    shadow model's own synthetic data: dropping the Mundlak term puts the between-person
    signal back into psi.  Observed at N = 400 seed 1: psi +0.597 against +0.524."""
    from trapworld.shadow import fit_shadow

    eta, g = _eta_and_g({}, 400)
    with_b = fit_shadow(eta, g, drift=True, between=True, hessian=False)
    without = fit_shadow(eta, g, drift=True, between=False, hessian=False)
    assert without["psi"] - 0.5 > 0.06, without["psi"]
    assert 1.0 - without["phi"] > 0.06, without["phi"]
    assert abs(with_b["psi"] - 0.5) < abs(without["psi"] - 0.5)
    assert abs(with_b["phi"] - 1.0) < abs(without["phi"] - 1.0)


def test_uniform_help_drops_the_between_term_bit_for_bit():
    """The drop rule on the real uniform-help arm: a dose that is the same for everyone
    leaves Var(gbar) at exactly 0, so the fit must equal the `between=False` fit."""
    from trapworld.shadow import fit_shadow

    eta, g = _eta_and_g({"policy": "uniform"}, 200)
    dropped = fit_shadow(eta, g, drift=True, between=True, hessian=False)
    plain = fit_shadow(eta, g, drift=True, between=False, hessian=False)
    assert dropped["var_gbar"] == 0.0 and dropped["between_dropped"]
    assert dropped["b"] == 0.0 and dropped["se"]["b"] is None
    for key in ("phi", "phi_u", "psi", "rho_d", "var_d", "var_e", "loglik"):
        assert dropped[key] == plain[key], key
    assert np.array_equal(dropped["scores"], plain["scores"])
    assert np.array_equal(dropped["resid"], plain["resid"])


def test_uniform_help_leaves_the_exposure_rule_silent():
    """R4's exposure half is undefined without between-person exposure, so it cannot
    fire on the uniform arm (spec 11 A2)."""
    w = simulate({"policy": "uniform", "N": 400}, SEED)
    numbers = run_analyzer("agentsem", w)["certificate"]["B"]["numbers"]
    assert numbers["exposure_p"] == 1.0
    assert numbers["max_exposure_corr"] == 0.0
    assert numbers["exposure_corr"] == [0.0, 0.0, 0.0]
