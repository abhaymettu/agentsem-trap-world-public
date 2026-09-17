"""Task 5 tests: the frozen grid, the metrics layer and the runner.

Written before the modules they cover.  The smoke run is shared by every test that
needs one (module-scoped fixture) because it is the one slow thing here: eight cells
at two replicates each.
"""

from __future__ import annotations

import csv
import importlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from trapworld import certificate, grid, metrics, run_grid
from trapworld.analyzers import ANALYZERS, run_analyzer
from trapworld.params import PARAMS
from trapworld.world import simulate

SWEEP_COUNTS = {"A": 36, "B": 7, "C": 2, "D": 4, "E": 1, "F": 15, "G": 6, "J": 2}


# ------------------------------------------------------------------ 1. the cell list
def test_cell_count_and_unique_ids():
    assert len(grid.CELLS) == 73
    ids = [c["cell_id"] for c in grid.CELLS]
    assert len(set(ids)) == len(ids)
    counts = {}
    for cell in grid.CELLS:
        counts[cell["sweep"]] = counts.get(cell["sweep"], 0) + 1
    assert counts == SWEEP_COUNTS
    assert sum(SWEEP_COUNTS.values()) == 73


def test_headline_sweeps_and_audit_ids():
    assert grid.HEADLINE_SWEEPS == ("A", "B", "C", "D", "E", "F", "G")
    assert "J" not in grid.HEADLINE_SWEEPS
    for cell_id in grid.AUDIT:
        assert grid.cell_by_id(cell_id)["cell_id"] == cell_id
    assert grid.AUDIT == ["A_k1.0_p0.5_g1.0", "B_G2", "B_G3_r03"]


def test_sweep_j_shape():
    fake, live = grid.cell_by_id("J_fake"), grid.cell_by_id("J_live")
    assert fake["jev"] == "fake" and fake["reps"] == 50
    assert live["jev"] == "live" and live["reps"] == 5
    for cell in (fake, live):
        assert cell["policy"] == "external"
        assert cell["sweep"] == "J"


def test_reps_follow_the_spec():
    heavy = [c["cell_id"] for c in grid.CELLS if c["sweep"] == "A" and c["reps"] == 500]
    assert sorted(heavy) == sorted(
        f"A_k1.0_p{p}_g{g}" for p in ("0.0", "0.5") for g in ("0.0", "1.0"))
    for cell in grid.CELLS:
        if cell["sweep"] in ("A", "B", "C", "D", "E", "F", "G"):
            assert cell["reps"] in (200, 500)


def test_kappa_zero_cells_have_policy_none():
    for cell in grid.CELLS:
        if cell["sweep"] == "A" and cell.get("kappa") == 0.0:
            assert cell["policy"] == "none"


# ------------------------------------------------------------- 2. hash and key safety
def test_grid_hash_stable_across_imports():
    first = grid.GRID_HASH
    reloaded = importlib.reload(grid)
    assert reloaded.GRID_HASH == first
    out = subprocess.run(
        [sys.executable, "-c", "from trapworld.grid import GRID_HASH; print(GRID_HASH)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == first


def test_every_override_is_a_params_key():
    for cell in grid.CELLS:
        for key in grid.cell_overrides(cell):
            assert key in PARAMS, f"{cell['cell_id']} overrides unknown key {key!r}"
        assert set(cell) - set(grid.CELL_META) - set(PARAMS) == set()


def test_seed_tree_matches_the_spawn_construction():
    root = np.random.SeedSequence(PARAMS["master_seed"])
    cell_seeds = root.spawn(len(grid.CELLS))
    for cell_index in (0, 7, len(grid.CELLS) - 1):
        kids = cell_seeds[cell_index].spawn(grid.REPS_MAX)
        for rep in (0, 1, grid.REPS_MAX - 1):
            mine = run_grid.replicate_seed(cell_index, rep)
            assert list(mine.generate_state(4)) == list(kids[rep].generate_state(4))


# ------------------------------------------------------------------ 3. determinism
def test_work_unit_is_deterministic_and_order_independent():
    first = run_grid.run_replicate(0, 0)
    again = run_grid.run_replicate(0, 0)
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)

    forward = [run_grid.run_replicate(0, rep) for rep in (0, 1)]
    backward = {rep: run_grid.run_replicate(0, rep) for rep in (1, 0)}
    for rep, record in zip((0, 1), forward):
        assert json.dumps(record, sort_keys=True) == json.dumps(backward[rep], sort_keys=True)


# ------------------------------------------------------------------ 4. the metrics
@pytest.fixture(scope="module")
def base_records():
    cell = grid.cell_by_id("A_k1.0_p0.5_g1.0")
    out = []
    for rep in range(3):
        world = simulate(grid.cell_overrides(cell), run_grid.replicate_seed(0, rep))
        results = {name: run_analyzer(name, world) for name in ANALYZERS}
        out.append(metrics.replicate_metrics(world, results, cell_id=cell["cell_id"], rep=rep))
    return out


def test_replicate_metrics_has_the_documented_keys(base_records):
    rec = base_records[0]
    for key in ("cell_id", "rep", "implied_r", "dW", "dW_se", "var_theta", "psi_true",
                "r_theta_react", "r_theta_clean", "r_theta_W", "u_eff", "shadow_share",
                "policy", "kappa_eff", "analyzers"):
        assert key in rec, key
    assert set(rec["analyzers"]) == set(ANALYZERS)

    for name in ANALYZERS:
        block = rec["analyzers"][name]
        for key in ("var_theta_hat", "bias", "abs_bias_ok", "ci", "covers", "psi_hat",
                    "psi_true", "beta_clean_hat", "corr", "rmse"):
            assert key in block, (name, key)

    naive = rec["analyzers"]["naive"]
    for key in ("omega", "alpha", "ave", "cfi", "rmsea", "srmr", "r_react", "r_clean",
                "r_W", "incremental_r2", "test_retest", "discriminant", "inv_dcfi_metric",
                "inv_p_metric", "inv_dcfi_scalar", "inv_p_scalar", "ratio_react"):
        assert key in naive, key
    for tau in ("0.2", "0.3", "0.4"):
        for stem in (f"pass_{tau}", f"pass_{tau}_with_invariance",
                     f"false_conf_{tau}", f"false_conf_{tau}_with_invariance"):
            assert stem in naive, stem

    for name in metrics.CERT_ANALYZERS:
        block = rec["analyzers"][name]
        for key in ("status_A", "status_B", "triggered", "false_confidence",
                    "false_refusal", "innocent", "detects", "phi_B_raw", "psi",
                    "max_exposure_corr", "exposure_p", "hetero_ratio", "cond_number",
                    "epoch_shift", "max_changepoint_shift", "dcfi_metric",
                    "b", "phi_u", "psi_identified", "between_dropped", "ci_method"):
            assert key in block, (name, key)


def test_records_are_strict_json(base_records):
    for rec in base_records:
        json.dumps(rec, allow_nan=False)


def test_aggregate_rates_and_mc_se(base_records):
    agg = metrics.aggregate(base_records)
    assert agg["n_reps"] == 3
    assert agg["cell_id"] == "A_k1.0_p0.5_g1.0"
    assert isinstance(agg["dW_mean"], float) and agg["dW_mc_se"] >= 0.0
    assert isinstance(agg["implied_r_mean"], float)

    def walk(node):
        if isinstance(node, dict):
            if "rate" in node:
                assert 0.0 <= node["rate"] <= 1.0
                assert node["mc_se"] >= 0.0
                assert node["n"] >= 0
            if "mc_se" in node and node["mc_se"] is not None:
                assert node["mc_se"] >= 0.0
            for value in node.values():
                walk(value)

    walk(agg)
    cert = agg["analyzers"]["agentsem"]
    assert set(cert["status_B"]["rates"]) <= {"identified", "bounded", "unsupported", "unknown"}
    assert abs(sum(cert["status_B"]["rates"].values()) - 1.0) < 1e-9
    assert "agentsem_minus_epoch_fe_bias" in agg["paired"]
    json.dumps(agg, allow_nan=False)


def test_jsonable_rejects_arrays():
    with pytest.raises(TypeError):
        metrics.jsonable({"bad": np.zeros(3)})
    assert metrics.jsonable({"a": np.float64(1.5), "b": np.bool_(True),
                             "c": (1, 2)}) == {"a": 1.5, "b": True, "c": [1, 2]}
    assert metrics.jsonable({"nan": float("nan"), "inf": float("inf")}) == \
        {"nan": None, "inf": None}


# ------------------------------------------------------------------ 5 to 8. the runner
@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    assert run_grid.main(["--smoke", "--out", str(out)]) == 0
    return out


def test_smoke_writes_every_output(smoke):
    assert (smoke / "hashes.json").exists()
    assert (smoke / "summary.csv").exists()
    assert (smoke / "summary.json").exists()
    assert (smoke / "run_manifest.json").exists()

    selected = grid.first_of_each_sweep()
    assert len(selected) == 8
    for cell_id in selected:
        assert (smoke / "grid" / f"{cell_id}.json").exists(), cell_id

    hashes = json.loads((smoke / "hashes.json").read_text())
    assert hashes["hashes_match"] is True
    assert set(run_grid.HASH_KEYS) <= set(hashes)
    assert hashes["grid_hash"] == grid.GRID_HASH

    manifest = json.loads((smoke / "run_manifest.json").read_text())
    assert manifest["workers"] == 2
    assert manifest["reps_per_cell"]["B_G2"] == 2
    assert manifest["J_live"] == "not_run"

    with open(smoke / "summary.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {r["analyzer"] for r in rows} == set(ANALYZERS)
    assert {r["cell_id"] for r in rows} == set(selected)


def test_smoke_runs_the_fake_jev_arm(smoke):
    payload = json.loads((smoke / "grid" / "J_fake.json").read_text())
    assert len(payload["records"]) == 2
    for rec in payload["records"]:
        assert rec["jev_status"] == "ok"
        assert rec["jev"]["n_calls"] > 0
    log = smoke / "jev" / "J_fake_rep0.jsonl"
    assert log.exists()
    lines = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert lines
    assert {line["person"] for line in lines} == {0, 1}
    assert {line["arm"] for line in lines} == {"jev_fake"}

    live = json.loads((smoke / "grid" / "J_live.json").read_text())
    assert live["status"] == "not_run"
    assert "JEV-LIVE-RUN.md" in live["reason"]


def test_check_hashes_passes_then_fails(smoke, monkeypatch, tmp_path):
    assert run_grid.check_hashes(smoke) == 0

    source = pathlib.Path(certificate.__file__).with_name("rules.json")
    copy = tmp_path / "rules.json"
    copy.write_text(source.read_text())
    edited = json.loads(copy.read_text())
    edited["R6"]["hetero_ratio_max"] = 99.0
    copy.write_text(json.dumps(edited))

    monkeypatch.setattr(certificate, "RULES", json.loads(copy.read_text()))
    assert run_grid.check_hashes(smoke) == 1


def test_sweep_j_rows_are_labelled_and_nothing_pools_cells(smoke):
    """Sweep J is reported on its own; the summary never averages across cells (S13)."""
    summary = json.loads((smoke / "summary.json").read_text())
    assert "J_fake" in {row["cell_id"] for row in summary["rows"]}
    assert {row["sweep"] for row in summary["rows"] if row["cell_id"] == "J_fake"} == {"J"}
    assert summary["headline_sweeps"] == list(grid.HEADLINE_SWEEPS)
    assert "J" not in summary["headline_sweeps"]
    assert set(summary) == {"generated_utc", "headline_sweeps", "rows"}
