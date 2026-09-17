"""The frozen certificate: spec 5.2, with every threshold read from `rules.json`.

Two claims are certified separately from one dict of diagnostics.

* Claim A: "Var(theta) on the pi_0 scale" (the anchor CFA's estimate).
* Claim B: "adaptive-phase person scores are on the pi_0 scale after correction".

Statuses are ordered `unknown > unsupported > bounded > identified` (the `precedence`
list in `rules.json`); every rule that fires downgrades, and the worst status wins.  A
rule that fires is recorded in `triggered` by its id, and every quantity any rule looked
at is echoed in `numbers`, including the ones that did not fire, so a replicate's record
shows why the certificate said what it said.

R4 carries two tests since spec 11 A2: the pooled unexplained shift (an epoch difference
and a change-point scan) and the exposure-pattern test, which is the one that can see a
mis-specified surrogate shape in a world where every shadow is signed by the person.

Design "D1D5" runs R1 to R6.  Design "D2" identifies claim B from the logged randomized
dose component rather than from the anchor epoch and the shadow model, so its B is gated
by R1 and the D2 rule; claim A still goes through R2 (an anchor that is not there leaves
A `unsupported`).  R3 to R6 are still applied under D2 if a caller supplies their
numbers, but the D2 analyzer fits no shadow model and does not.

`numbers["phi_B_raw"]` is kept whatever the status, including `unsupported` and
`unknown`, because the false-refusal metric (spec 7) needs to know what the correction
would have said had it not refused.
"""

from __future__ import annotations

import json
import pathlib

RULES = json.loads((pathlib.Path(__file__).with_name("rules.json")).read_text())

_RANK = {name: i for i, name in enumerate(RULES["precedence"])}   # 0 is the strongest
DESIGNS = ("D1D5", "D2")


def _worse(current: str, candidate: str) -> str:
    return current if _RANK[current] <= _RANK[candidate] else candidate


class _Claim:
    """One claim's running status, triggered list and numbers."""

    def __init__(self) -> None:
        self.status = "identified"
        self.triggered: list[str] = []
        self.numbers: dict = {}

    def fire(self, rule: str, status: str) -> None:
        self.status = _worse(self.status, status)
        if rule not in self.triggered:
            self.triggered.append(rule)

    def out(self, estimate, ci) -> dict:
        return {"status": self.status, "estimate": estimate, "ci": ci,
                "triggered": list(self.triggered), "numbers": dict(self.numbers)}


def _look(diag: dict, key, default, *claims):
    """Read one diagnostic and record it on every claim whose rules consulted it."""
    value = diag.get(key, default)
    for claim in claims:
        claim.numbers[key] = value
    return value


def certify(diag: dict, design: str) -> dict:
    """Apply the frozen rules to `diag` and return `{"A": {...}, "B": {...}}`."""
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
    A, B = _Claim(), _Claim()

    # R1 provenance -------------------------------------------------------------
    if not bool(_look(diag, "version_complete", True, A, B)):
        A.fire("R1", "unknown")
        B.fire("R1", "unknown")

    # R2 anchor and switch exogeneity -------------------------------------------
    r2 = RULES["R2"]
    anchor_claims = (A, B) if design == "D1D5" else (A,)
    has_anchor = bool(_look(diag, "has_anchor", True, *anchor_claims))
    dose_const = bool(_look(diag, "dose_constant_in_anchor", True, *anchor_claims))
    min_len = float(_look(diag, "min_anchor_len", r2["min_anchor_len"], *anchor_claims))
    varies = bool(_look(diag, "switch_varies", False, *anchor_claims))
    s_corr = float(_look(diag, "switch_corr", 0.0, *anchor_claims))
    s_p = float(_look(diag, "switch_p", 1.0, *anchor_claims))
    if (not has_anchor) or (not dose_const) or min_len < r2["min_anchor_len"]:
        for claim in anchor_claims:
            claim.fire("R2", "unsupported")
    elif varies and abs(s_corr) > r2["switch_corr_max"] and s_p < r2["switch_p_max"]:
        for claim in anchor_claims:
            claim.fire("R2", "bounded")

    phi_A = _look(diag, "phi_A", None, A, B)
    ci_A = _look(diag, "ci_A", None, A, B)
    phi_B = diag.get("phi_B")
    ci_B = diag.get("ci_B")
    B.numbers["phi_B_raw"] = phi_B          # kept whatever the status (false refusal)
    B.numbers["ci_B_raw"] = ci_B
    B.numbers["psi"] = diag.get("psi")
    B.numbers["psi_ci"] = diag.get("psi_ci")

    optional = design == "D2"               # under D2 these rules apply only if supplied

    # R3 conditioning ------------------------------------------------------------
    if not optional or "cond_number" in diag:
        cond = float(_look(diag, "cond_number", 0.0, B))
        if cond > RULES["R3"]["cond_max"]:
            B.fire("R3", "bounded")
    # informational, always echoed (spec 5.2 R3)
    _look(diag, "between_share_gbar", None, B)
    _look(diag, "r2_gbar_on_anchor", None, B)

    # R4 unexplained shift -------------------------------------------------------
    sd_anchor = float(phi_A) ** 0.5 if isinstance(phi_A, (int, float)) and phi_A and phi_A > 0 else 1.0
    threshold = RULES["R4"]["shift_sd_max"] * sd_anchor
    B.numbers["shift_threshold"] = threshold
    B.numbers["changepoint_margin"] = RULES["R4"]["changepoint_margin"]
    if not optional or "epoch_shift" in diag or "max_changepoint_shift" in diag:
        shift = float(_look(diag, "epoch_shift", 0.0, B))
        cp = float(_look(diag, "max_changepoint_shift", 0.0, B))
        if abs(shift) > threshold or cp > threshold:
            B.fire("R4", "unsupported")

    # R4 exposure pattern (spec 11 A2) --------------------------------------------
    # Every shadow in this world is signed by the person, so a pooled mean shift cannot
    # see a mis-specified surrogate shape.  D5's own assumption E[resid_it | exposure_i]
    # = 0 is what gets tested: per-person window-mean residuals against the person's
    # exposure level.  The statistic is undefined when exposure has no variance, and the
    # analyzer then supplies corr 0 and p 1, so the rule stays silent by construction.
    B.numbers["exposure_windows"] = RULES["R4"]["exposure_windows"]
    B.numbers["exposure_p_max"] = RULES["R4"]["exposure_p_max"]
    if not optional or "exposure_p" in diag:
        _look(diag, "exposure_corr", None, B)
        _look(diag, "max_exposure_corr", None, B)
        exposure_p = float(_look(diag, "exposure_p", 1.0, B))
        if exposure_p < RULES["R4"]["exposure_p_max"]:
            B.fire("R4", "unsupported")

    # R5 loading pattern ---------------------------------------------------------
    if not optional or "dcfi_metric" in diag:
        dcfi = float(_look(diag, "dcfi_metric", 0.0, B))
        if dcfi > RULES["R5"]["dcfi_max"]:
            B.fire("R5", "unsupported")

    # R6 heterogeneity -----------------------------------------------------------
    if not optional or "hetero_ratio" in diag:
        ratio = float(_look(diag, "hetero_ratio", 0.0, B))
        if ratio > RULES["R6"]["hetero_ratio_max"]:
            B.fire("R6", "bounded")

    # D2 gate --------------------------------------------------------------------
    if design == "D2":
        if not bool(_look(diag, "jitter_logged", False, B)):
            B.fire("D2", "unsupported")
        first_F = float(_look(diag, "first_stage_F", 0.0, B))
        if first_F < RULES["D2"]["first_stage_F_min"]:
            B.fire("D2", "bounded")

    # estimates ------------------------------------------------------------------
    a_est = (phi_A, ci_A) if A.status in ("identified", "bounded") else (None, None)
    if B.status == "identified":
        b_est = (phi_B, ci_B)
    elif B.status == "bounded":
        b_est = (phi_A, ci_A)               # the anchor fallback
    else:
        b_est = (None, None)
    return {"A": A.out(*a_est), "B": B.out(*b_est)}
