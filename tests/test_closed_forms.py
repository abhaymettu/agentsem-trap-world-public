"""Falsification gate F1-F5: tests for trapworld.closed_forms.

Monte Carlo draw counts are cut to 200_000 here (the module default, used by
``python -m trapworld.closed_forms``, is 2_000_000) so the suite stays well under a
minute.  Every check is seeded, so these assertions are deterministic.
"""

from __future__ import annotations

import datetime as _dt
import json
import math

import numpy as np
import pytest

from trapworld import closed_forms as cf

TEST_DRAWS = 200_000
ABS_TOL = 0.005  # brief: closed form vs Monte Carlo within 0.005 at every F1-F4 point


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def f1():
    return cf.f1_validity_vs_adaptivity(n_draws=TEST_DRAWS)


@pytest.fixture(scope="module")
def f2():
    return cf.f2_reliability_vs_validity(n_draws=TEST_DRAWS)


@pytest.fixture(scope="module")
def f3():
    return cf.f3_criterion_contamination(n_draws=TEST_DRAWS)


@pytest.fixture(scope="module")
def f4():
    return cf.f4_surrogate_purity(n_draws=TEST_DRAWS)


@pytest.fixture(scope="module")
def f5():
    return cf.f5_benign_arm_innocence()


def _maxdev(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# --------------------------------------------------------------------------- F1
def test_f1_correlations_closed_vs_mc(f1):
    for key in ("corr_eta_theta", "corr_eta_C"):
        dev = _maxdev(f1["monte_carlo"][key], f1["closed_form"][key])
        assert dev < ABS_TOL, f"F1 {key}: max |MC - closed| = {dev}"


def test_f1_var_ratio_closed_vs_mc(f1):
    # Var(eta)/Var(theta) reaches 145 at u=8, where Monte Carlo error is ~0.3% relative
    # (~0.4 absolute) at n=200k, far above a flat 0.005 band.  Relative band instead;
    # this is the one documented departure from the brief's blanket 0.005.
    c = np.asarray(f1["closed_form"]["var_ratio"], float)
    m = np.asarray(f1["monte_carlo"]["var_ratio"], float)
    rel = float(np.max(np.abs(m - c) / c))
    assert rel < 0.02, f"F1 var_ratio: max relative dev = {rel}"


def test_f1_corr_strictly_decreasing_in_u(f1):
    for key in ("corr_eta_theta", "corr_eta_C"):
        v = np.asarray(f1["monte_carlo"][key], float)
        assert np.all(np.diff(v) < 0), f"F1 {key} not strictly decreasing: {v}"


def test_f1_var_ratio_strictly_increasing_in_u(f1):
    v = np.asarray(f1["monte_carlo"]["var_ratio"], float)
    assert np.all(np.diff(v) > 0), f"F1 var_ratio not strictly increasing: {v}"


def test_f1_verdict_follows_from_mc(f1):
    falls = f1["corr_eta_theta_strictly_decreasing"] and f1["corr_eta_C_strictly_decreasing"]
    assert f1["verdict"] == ("corr falls with u" if falls else "corr rises with u")
    assert f1["spec_predicts"] == "both rise"
    assert f1["review_predicts"] == "both fall"


# --------------------------------------------------------------------------- F2
def test_f2_closed_vs_mc(f2):
    for key in ("omega", "alpha", "corr_eta_theta", "corr_composite_theta"):
        dev = _maxdev(f2["monte_carlo"][key], f2["closed_form"][key])
        assert dev < ABS_TOL, f"F2 {key}: max |MC - closed| = {dev}"


def test_f2_reproduces_b13_table(f2):
    omega = [round(v, 3) for v in f2["closed_form"]["omega"]]
    corr = [round(v, 3) for v in f2["closed_form"]["corr_eta_theta"]]
    assert omega == [0.833, 0.926, 0.962]
    assert corr == [1.000, 0.949, 0.894]
    assert f2["b13_table_reproduced"] is True


def test_f2_reliability_up_validity_down(f2):
    om = np.asarray(f2["monte_carlo"]["omega"], float)
    cr = np.asarray(f2["monte_carlo"]["corr_eta_theta"], float)
    assert np.all(np.diff(om) > 0)
    assert np.all(np.diff(cr) < 0)
    assert f2["verdict"] == "reliability up, validity down"


# --------------------------------------------------------------------------- F3
def test_f3_ratio_closed_vs_mc(f3):
    dev = _maxdev(f3["monte_carlo"]["ratio"], f3["closed_form"]["ratio"])
    assert dev < ABS_TOL, f"F3 ratio: max |MC - closed| = {dev}"


def test_f3_component_correlations_closed_vs_mc(f3):
    for key in ("corr_eta_O", "corr_theta_O"):
        dev = _maxdev(f3["monte_carlo"][key], f3["closed_form"][key])
        assert dev < ABS_TOL, f"F3 {key}: max |MC - closed| = {dev}"


def test_f3_matches_review_b14_table(f3):
    # The review printed its B1.4 table to 3 dp from a 4M-draw sim; several cells sit one
    # unit off the exact closed form (worst: r=1.0, u=2 -> 5/sqrt(13) = 1.3868 vs 1.388).
    # Agreement is asserted at 0.002 and every rounding mismatch is recorded in the result.
    dev = _maxdev(f3["closed_form"]["ratio"], cf.REVIEW_B14_TABLE)
    assert dev < 0.002, f"F3 vs review table: max dev = {dev}"
    assert dev == pytest.approx(f3["review_table_max_abs_dev"], abs=1e-12)


def test_f3_inflates_for_all_u_exactly_at_r_ge_sqrt2_minus_1(f3):
    got = {row["r"] for row in f3["per_r"] if row["inflates_for_all_u"]}
    assert got == {0.4142, 0.6, 1.0}


def test_f3_monotone_rise_over_0_2_only_at_r_1(f3):
    got = {row["r"] for row in f3["per_r"] if row["monotone_rise_over_0_2"]}
    assert got == {1.0}  # r=0.6 peaks at u*=1.5 < 2, so it turns over inside [0, 2]


def test_f3_peak_matches_r_over_1_minus_r(f3):
    for row in f3["per_r"]:
        if row["r"] in (0.2, 0.4142, 0.6):
            assert abs(row["peak_u_numeric"] - row["r"] / (1 - row["r"])) < 0.05


def test_f3_ceiling(f3):
    ceilings = {row["r"]: row["ceiling"] for row in f3["per_r"]}
    assert ceilings[0.0] == pytest.approx(1 / math.sqrt(2), abs=1e-9)
    assert round(ceilings[1.0], 3) == 1.414


# --------------------------------------------------------------------------- F4
def test_f4_closed_vs_mc(f4):
    for key in ("corr_D_theta", "R2", "attenuation"):
        dev = _maxdev(f4["monte_carlo"][key], f4["closed_form"][key])
        assert dev < ABS_TOL, f"F4 {key}: max |MC - closed| = {dev}"


def test_f4_r2_matches_review(f4):
    assert [round(v, 3) for v in f4["closed_form"]["R2"]] == [0.500, 0.875, 0.966, 0.982]


def test_f4_attenuation_matches_review(f4):
    # Review prints .707/.354/.186/.133; the exact value at T=56 is sqrt(1/57) = 0.13245,
    # which rounds to .132, so this is asserted at 0.001 rather than as a 3-dp string match.
    dev = _maxdev(f4["closed_form"]["attenuation"], [0.707, 0.354, 0.186, 0.133])
    assert dev < 0.001, f"F4 attenuation vs review: max dev = {dev}"


def test_f4_verdict(f4):
    assert f4["closed_form"]["R2"][-1] > 0.95
    assert f4["verdict"] == "rule deletes the person"


# --------------------------------------------------------------------------- F5
def test_f5_kill_rate_at_threshold_point_three(f5):
    cells = [c for c in f5["monte_carlo"]["cells"] if c["threshold"] == 0.3]
    assert {c["kappa"] for c in cells} == {0.5, 1.0, 2.0}
    for c in cells:
        assert c["kill_rate"] > 0.5, f"F5 kappa={c['kappa']} kill_rate={c['kill_rate']}"


def test_f5_reports_monte_carlo_se(f5):
    for c in f5["monte_carlo"]["cells"]:
        assert c["kill_rate_se"] >= 0.0
        assert 0.0 <= c["kill_rate"] <= 1.0


def test_f5_semipartial_matches_closed_form(f5):
    dev = abs(f5["monte_carlo"]["mean_semipartial_corr"] - f5["closed_form"]["semipartial_eta_C_given_D"])
    assert dev < 0.02, f"F5 semipartial: |MC - closed| = {dev}"


# --------------------------------------------------------------------------- main()
def test_main_writes_roundtrippable_json(tmp_path):
    out = tmp_path / "closed_forms.json"
    cf.main(out_path=out, n_draws=TEST_DRAWS, reps=100)
    payload = json.loads(out.read_text())
    assert {"F1", "F2", "F3", "F4", "F5"} <= set(payload)
    assert set(payload["verdicts"]) == {"F1", "F2", "F3", "F4", "F5"}
    assert payload["seed"] == cf.SEED
    assert payload["n_draws"] == TEST_DRAWS
    _dt.datetime.fromisoformat(payload["generated_at"])


def test_seed_streams_are_distinct():
    firsts = [cf.rng_for(k).random() for k in range(5)]
    assert len(set(firsts)) == 5
