"""Tests for the frozen certificate (spec 5.2, thresholds in `trapworld/rules.json`).

Every diagnostic dict here is hand built, so each test isolates one rule.  The analyzer
end of this (real numbers off a real world) is `tests/test_analyzers.py`.
"""

from __future__ import annotations

import pytest

from trapworld.certificate import RULES, certify

CI_A = (0.9, 1.1)
CI_B = (0.85, 1.15)


def clean(**over) -> dict:
    """A diagnostic dict on which no rule fires."""
    d = {
        "version_complete": True,
        "has_anchor": True, "dose_constant_in_anchor": True, "min_anchor_len": 14,
        "switch_varies": False, "switch_corr": 0.0, "switch_p": 1.0,
        "cond_number": 12.0, "between_share_gbar": 0.8, "r2_gbar_on_anchor": 0.9,
        "epoch_shift": 0.01, "max_changepoint_shift": 0.05,
        "exposure_corr": [0.02, -0.01, 0.03], "max_exposure_corr": 0.03, "exposure_p": 0.6,
        "dcfi_metric": 0.001, "hetero_ratio": 1.1,
        "phi_A": 1.02, "ci_A": CI_A, "phi_B": 0.98, "ci_B": CI_B,
        "psi": 0.5, "psi_ci": (0.45, 0.55),
    }
    d.update(over)
    return d


def clean_d2(**over) -> dict:
    d = clean(jitter_logged=True, first_stage_F=40.0, psi=0.5)
    d.update(over)
    return d


# ------------------------------------------------------------------ the frozen file
def test_rules_json_carries_the_frozen_thresholds():
    assert RULES["R2"]["min_anchor_len"] == 7
    assert RULES["R2"]["switch_corr_max"] == 0.10
    assert RULES["R2"]["switch_p_max"] == 0.01
    assert RULES["R3"]["cond_max"] == 30
    assert RULES["R4"]["shift_sd_max"] == 0.15
    assert RULES["R4"]["changepoint_margin"] == 3
    assert RULES["R4"]["exposure_windows"] == 3
    assert RULES["R4"]["exposure_p_max"] == 0.01
    assert RULES["R5"]["dcfi_max"] == 0.01
    assert RULES["R6"]["hetero_ratio_max"] == 1.5
    assert RULES["D2"]["first_stage_F_min"] == 10
    assert RULES["precedence"] == ["unknown", "unsupported", "bounded", "identified"]


def test_output_shape():
    out = certify(clean(), "D1D5")
    assert set(out) == {"A", "B"}
    for claim in out.values():
        assert set(claim) == {"status", "estimate", "ci", "triggered", "numbers"}
        assert isinstance(claim["triggered"], list)
        assert isinstance(claim["numbers"], dict)


def test_unknown_design_raises():
    with pytest.raises(ValueError):
        certify(clean(), "D3")


# ------------------------------------------------------------------------- clean run
def test_clean_diagnostics_identify_both_claims():
    out = certify(clean(), "D1D5")
    assert out["A"]["status"] == "identified"
    assert out["B"]["status"] == "identified"
    assert out["A"]["estimate"] == 1.02 and out["A"]["ci"] == CI_A
    assert out["B"]["estimate"] == 0.98 and out["B"]["ci"] == CI_B
    assert out["A"]["triggered"] == [] and out["B"]["triggered"] == []
    assert out["B"]["numbers"]["psi"] == 0.5


# ------------------------------------------------------------------------------- R1
def test_r1_missing_version_makes_both_unknown():
    out = certify(clean(version_complete=False), "D1D5")
    assert out["A"]["status"] == "unknown" and out["B"]["status"] == "unknown"
    assert "R1" in out["A"]["triggered"] and "R1" in out["B"]["triggered"]
    assert out["A"]["estimate"] is None and out["B"]["estimate"] is None
    assert out["A"]["ci"] is None and out["B"]["ci"] is None
    assert out["B"]["numbers"]["phi_B_raw"] == 0.98


def test_r1_beats_every_other_status():
    out = certify(clean(version_complete=False, hetero_ratio=9.0, dcfi_metric=0.5), "D1D5")
    assert out["B"]["status"] == "unknown"
    assert {"R1", "R5", "R6"} <= set(out["B"]["triggered"])


# ------------------------------------------------------------------------------- R2
@pytest.mark.parametrize("over", [
    {"has_anchor": False},
    {"dose_constant_in_anchor": False},
    {"min_anchor_len": 6},
])
def test_r2_anchor_failures_make_both_unsupported(over):
    out = certify(clean(**over), "D1D5")
    assert out["A"]["status"] == "unsupported" and out["B"]["status"] == "unsupported"
    assert "R2" in out["A"]["triggered"] and "R2" in out["B"]["triggered"]
    assert out["A"]["estimate"] is None and out["B"]["estimate"] is None
    assert out["B"]["numbers"]["phi_B_raw"] == 0.98


def test_r2_anchor_length_boundary_is_inclusive():
    out = certify(clean(min_anchor_len=7), "D1D5")
    assert out["A"]["status"] == "identified"


def test_r2_switch_exogeneity_bounds_both_claims():
    out = certify(clean(switch_varies=True, switch_corr=-0.22, switch_p=0.002), "D1D5")
    assert out["A"]["status"] == "bounded" and out["B"]["status"] == "bounded"
    assert "R2" in out["A"]["triggered"] and "R2" in out["B"]["triggered"]
    assert out["A"]["estimate"] == 1.02 and out["A"]["ci"] == CI_A
    assert out["B"]["estimate"] == 1.02 and out["B"]["ci"] == CI_A   # the anchor fallback
    assert out["B"]["numbers"]["phi_B_raw"] == 0.98


@pytest.mark.parametrize("over", [
    {"switch_varies": False, "switch_corr": 0.9, "switch_p": 0.0},   # constant switch
    {"switch_varies": True, "switch_corr": 0.05, "switch_p": 0.0},   # correlation too small
    {"switch_varies": True, "switch_corr": 0.9, "switch_p": 0.05},   # not significant
])
def test_r2_switch_rule_needs_all_three_conditions(over):
    assert certify(clean(**over), "D1D5")["A"]["status"] == "identified"


# ------------------------------------------------------------------------------- R3
def test_r3_conditioning_bounds_b_only():
    out = certify(clean(cond_number=31.0), "D1D5")
    assert out["A"]["status"] == "identified"
    assert out["B"]["status"] == "bounded" and "R3" in out["B"]["triggered"]
    assert out["B"]["estimate"] == 1.02 and out["B"]["ci"] == CI_A


def test_r3_numbers_are_always_echoed():
    out = certify(clean(), "D1D5")
    assert out["B"]["numbers"]["cond_number"] == 12.0
    assert out["B"]["numbers"]["between_share_gbar"] == 0.8
    assert out["B"]["numbers"]["r2_gbar_on_anchor"] == 0.9


# ------------------------------------------------------------------------------- R4
@pytest.mark.parametrize("over", [
    {"epoch_shift": 0.2}, {"epoch_shift": -0.2}, {"max_changepoint_shift": 0.2},
])
def test_r4_unexplained_shift_makes_b_unsupported(over):
    out = certify(clean(**over), "D1D5")
    assert out["A"]["status"] == "identified"
    assert out["B"]["status"] == "unsupported" and "R4" in out["B"]["triggered"]
    assert out["B"]["estimate"] is None and out["B"]["ci"] is None
    assert out["B"]["numbers"]["phi_B_raw"] == 0.98


def test_r4_threshold_scales_with_the_anchor_sd():
    """0.15 SD of the anchor score, not 0.15 raw."""
    assert certify(clean(epoch_shift=0.2, phi_A=4.0), "D1D5")["B"]["status"] == "identified"
    assert certify(clean(epoch_shift=0.2, phi_A=0.25), "D1D5")["B"]["status"] == "unsupported"


# ------------------------------------------------------- R4 exposure pattern (A2)
def test_r4_exposure_pattern_makes_b_unsupported():
    """A residual that tracks the person's exposure level kills claim B."""
    out = certify(clean(exposure_corr=[0.31, 0.02, -0.28], max_exposure_corr=0.31,
                        exposure_p=0.001), "D1D5")
    assert out["A"]["status"] == "identified"
    assert out["B"]["status"] == "unsupported" and "R4" in out["B"]["triggered"]
    assert out["B"]["estimate"] is None and out["B"]["ci"] is None
    assert out["B"]["numbers"]["phi_B_raw"] == 0.98
    assert out["B"]["numbers"]["max_exposure_corr"] == 0.31


def test_r4_exposure_boundary_is_inclusive():
    assert certify(clean(exposure_p=0.01), "D1D5")["B"]["status"] == "identified"
    assert certify(clean(exposure_p=0.0099), "D1D5")["B"]["status"] == "unsupported"


def test_r4_exposure_is_silent_when_exposure_has_no_variance():
    """The analyzer reports corr 0 and p 1 when var(gbar) is 0, so the rule cannot fire."""
    out = certify(clean(exposure_corr=[0.0, 0.0, 0.0], max_exposure_corr=0.0,
                        exposure_p=1.0), "D1D5")
    assert out["B"]["status"] == "identified" and out["B"]["triggered"] == []


def test_r4_exposure_numbers_are_echoed():
    numbers = certify(clean(), "D1D5")["B"]["numbers"]
    assert numbers["exposure_corr"] == [0.02, -0.01, 0.03]
    assert numbers["max_exposure_corr"] == 0.03
    assert numbers["exposure_p"] == 0.6
    assert numbers["exposure_windows"] == 3 and numbers["exposure_p_max"] == 0.01


def test_r4_either_half_fires_on_its_own():
    """The shift test and the exposure test are independent halves of one rule."""
    shift_only = certify(clean(epoch_shift=0.4), "D1D5")["B"]
    exposure_only = certify(clean(exposure_p=1e-6), "D1D5")["B"]
    assert shift_only["status"] == exposure_only["status"] == "unsupported"
    assert shift_only["triggered"] == exposure_only["triggered"] == ["R4"]


# ------------------------------------------------------------------------------- R5
def test_r5_loading_pattern_makes_b_unsupported():
    out = certify(clean(dcfi_metric=0.02), "D1D5")
    assert out["B"]["status"] == "unsupported" and "R5" in out["B"]["triggered"]
    assert out["A"]["status"] == "identified"


def test_r5_boundary_is_inclusive():
    assert certify(clean(dcfi_metric=0.01), "D1D5")["B"]["status"] == "identified"


# ------------------------------------------------------------------------------- R6
def test_r6_heterogeneity_bounds_b():
    out = certify(clean(hetero_ratio=1.6), "D1D5")
    assert out["B"]["status"] == "bounded" and "R6" in out["B"]["triggered"]
    assert out["B"]["estimate"] == 1.02 and out["B"]["ci"] == CI_A


def test_r6_boundary_is_inclusive():
    assert certify(clean(hetero_ratio=1.5), "D1D5")["B"]["status"] == "identified"


# ------------------------------------------------------------------------ precedence
def test_unsupported_beats_bounded():
    out = certify(clean(hetero_ratio=9.0, dcfi_metric=0.5), "D1D5")
    assert out["B"]["status"] == "unsupported"
    assert {"R5", "R6"} <= set(out["B"]["triggered"])


def test_bounded_beats_identified():
    out = certify(clean(cond_number=99.0, hetero_ratio=9.0), "D1D5")
    assert out["B"]["status"] == "bounded"


# -------------------------------------------------------------------------- D2 gate
def test_d2_clean_identifies_b_and_a():
    out = certify(clean_d2(), "D2")
    assert out["B"]["status"] == "identified" and out["B"]["estimate"] == 0.98
    assert out["A"]["status"] == "identified" and out["A"]["estimate"] == 1.02


def test_d2_without_jitter_is_unsupported():
    out = certify(clean_d2(jitter_logged=False), "D2")
    assert out["B"]["status"] == "unsupported" and "D2" in out["B"]["triggered"]
    assert out["B"]["estimate"] is None
    assert out["B"]["numbers"]["phi_B_raw"] == 0.98


def test_d2_weak_instrument_bounds_b():
    out = certify(clean_d2(first_stage_F=5.0), "D2")
    assert out["B"]["status"] == "bounded" and "D2" in out["B"]["triggered"]
    assert out["B"]["estimate"] == 1.02 and out["B"]["ci"] == CI_A


def test_d2_first_stage_boundary():
    assert certify(clean_d2(first_stage_F=10.0), "D2")["B"]["status"] == "identified"
    assert certify(clean_d2(first_stage_F=9.99), "D2")["B"]["status"] == "bounded"


def test_d2_without_an_anchor_leaves_a_unsupported():
    out = certify(clean_d2(has_anchor=False), "D2")
    assert out["A"]["status"] == "unsupported" and out["A"]["estimate"] is None
    assert out["B"]["status"] == "identified"      # B rides the instrument, not the anchor


def test_d2_applies_r2_to_a_when_an_anchor_exists():
    out = certify(clean_d2(switch_varies=True, switch_corr=0.3, switch_p=0.001), "D2")
    assert out["A"]["status"] == "bounded" and out["A"]["estimate"] == 1.02
    assert out["B"]["status"] == "identified"


def test_d2_shadow_model_rules_apply_only_when_their_numbers_are_supplied():
    """The D2 arm fits no shadow model, so R3 to R6 have nothing to look at there."""
    d = clean_d2()
    for key in ("cond_number", "epoch_shift", "max_changepoint_shift", "exposure_corr",
                "max_exposure_corr", "exposure_p", "dcfi_metric", "hetero_ratio"):
        d.pop(key)
    assert certify(d, "D2")["B"]["status"] == "identified"
    assert "exposure_p" not in certify(d, "D2")["B"]["numbers"]
    assert certify(clean_d2(cond_number=99.0), "D2")["B"]["status"] == "bounded"


def test_d2_applies_the_exposure_rule_when_its_number_is_supplied():
    assert certify(clean_d2(exposure_p=0.001), "D2")["B"]["status"] == "unsupported"


# ---------------------------------------------------------------- estimate plumbing
def test_missing_phi_b_leaves_the_estimate_none():
    d = clean()
    d["phi_B"] = None
    d["ci_B"] = None
    out = certify(d, "D1D5")
    assert out["B"]["status"] == "identified" and out["B"]["estimate"] is None


def test_numbers_record_what_each_rule_looked_at():
    out = certify(clean(), "D1D5")
    assert {"version_complete", "has_anchor", "min_anchor_len", "switch_corr",
            "switch_p", "phi_A"} <= set(out["A"]["numbers"])
    assert {"cond_number", "epoch_shift", "max_changepoint_shift", "dcfi_metric",
            "hetero_ratio", "phi_B_raw", "ci_B_raw", "psi", "shift_threshold",
            "exposure_corr", "max_exposure_corr", "exposure_p"} <= set(
        out["B"]["numbers"])
