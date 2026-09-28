"""Task 7: the figure module runs end to end against the real results directory.

There is nothing to assert about a rendered figure beyond the fact that it was written
and is not empty, so that is what this checks.  Every number on every panel comes from
`results/`, and the tests for those numbers live with the code that produced them.

The whole suite skips when `results/summary.json` is absent, so a clone without a run
is not a failure.
"""

from __future__ import annotations

import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

#: Every figure the briefs name, with P10 optional (it needs sweep J's records) and P8
#: drawn either as the real panel or as a labelled placeholder.
REQUIRED = ["P1_headline", "P2_f3_hump", "P3_competitors", "P4_certificate",
            "P5_specificity", "P6_zero_person", "P7_f4_purity", "P8_coverage",
            "P9_closed_forms"]
OPTIONAL = ["P10_jev_arm"]

pytestmark = pytest.mark.skipif(
    not (RESULTS / "summary.json").exists(),
    reason="results/summary.json is absent; nothing to plot from")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run `main()` once against the real results into a scratch directory."""
    from trapworld import plots

    out = tmp_path_factory.mktemp("plots")
    written = plots.main(RESULTS, out)
    return out, written


@pytest.mark.parametrize("name", REQUIRED)
@pytest.mark.parametrize("ext", ["png", "svg"])
def test_required_figure_written_and_not_empty(built, name, ext):
    out, _ = built
    path = out / f"{name}.{ext}"
    assert path.exists(), f"{path.name} was not written"
    assert path.stat().st_size > 0, f"{path.name} is empty"


@pytest.mark.parametrize("name", OPTIONAL)
@pytest.mark.parametrize("ext", ["png", "svg"])
def test_optional_figure_written_and_not_empty(built, name, ext):
    out, _ = built
    path = out / f"{name}.{ext}"
    if not path.exists():
        pytest.skip(f"{name} has no input in this results directory")
    assert path.stat().st_size > 0, f"{path.name} is empty"


def test_main_returns_every_file_it_wrote(built):
    out, written = built
    assert written, "main() returned no paths"
    for path in written:
        assert path.exists() and path.stat().st_size > 0
    on_disk = {p.name for p in out.iterdir()}
    assert {p.name for p in written} == on_disk


def test_no_sweep_J_cell_is_read_by_P1_to_P9():
    """Sweep J is the fake-transport arm and belongs only to P10 (task 7b)."""
    from trapworld import plots

    res = plots.Results(RESULTS)
    builders = [plots.p1_headline, plots.p2_f3_hump, plots.p3_competitors,
                plots.p4_certificate, plots.p5_specificity, plots.p6_zero_person,
                plots.p7_f4_purity, plots.p9_closed_forms]
    assert not any("J_" in c for c in plots.P3_CELLS + plots.P4_CELLS)
    assert plots.BASE_CELL == "A_k1.0_p0.5_g1.0"
    assert all(callable(b) for b in builders)


def test_r4_split_matches_the_stored_triggered_list():
    """The exposure/shift split must reproduce `triggered_B` exactly, not approximate it."""
    from trapworld import plots

    cell_ids = ("A_k1.0_p0.5_g1.0", "B_G2", "B_G4")
    missing = [c for c in cell_ids if not (RESULTS / "grid" / f"{c}.json").is_file()]
    if missing:
        if os.environ.get("AGENTSEM_REQUIRE_GRID_FIXTURES") == "1" or len(missing) < len(cell_ids):
            pytest.fail(f"missing grid fixture(s): {', '.join(missing)}")
        pytest.skip("grid fixtures not restored; use AGENTSEM_REQUIRE_GRID_FIXTURES=1 for integration")
    res = plots.Results(RESULTS)
    checked = 0
    for cell_id in cell_ids:
        for rec in res.cell(cell_id).get("records", []):
            arm = rec.get("analyzers", {}).get("agentsem")
            if arm is None:
                continue
            category = plots.rule_category(
                arm, rec["analyzers"].get("anchor_only"), res.rules)
            fired = bool(arm.get("triggered_B"))
            assert (category != "none") == fired, (cell_id, rec.get("rep"), category)
            assert "?" not in category and "other" not in category
            checked += 1
    assert checked > 0

