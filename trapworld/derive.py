"""Post-run derived metrics and a summary rebuild (spec 11 A7; exploratory by construction).

Run after the grid: ``python -m trapworld.derive [--out results]``.

The pre-registered bias compares an estimate with the population ``Var(theta)`` (1.0).  At
N = 200 the realized variance of the 200 drawn persons sits about 0.10 from 1.0 on average,
the same size as the pre-registered tolerance, so the per-replicate flags ``abs_bias_ok``,
``false_confidence`` and ``false_conf_{tau}`` fire on sampling noise even for an estimator
with no bias (measured on the first finished cell: mean bias 0.007, flag rate 0.39).  The
per-replicate records already carry ``var_theta_sample``, the realized variance of that
replicate's persons, so this module adds the same flags against it, suffixed ``_sample``,
re-aggregates every cell and rebuilds ``summary.csv`` and ``summary.json`` from the per-cell
files.  Nothing stored by the run is altered: every derived field is a function of two
numbers already in the record, the original fields stay, and the run's own manifest and
hashes are untouched.  Because the definition was written after results were seen, the
``_sample`` fields are exploratory and are labelled so wherever they are reported.

The rebuild is also what a later ``--cells`` run (the live Jev cell) needs: it makes the
summary a function of whatever ``results/grid/*.json`` files exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from trapworld import grid, metrics, run_grid

__all__ = ["derive_record", "main", "DERIVED_FIELDS"]

DERIVED_FIELDS = ("bias_sample", "abs_bias_sample_ok", "false_confidence_sample",
                  "false_refusal_sample", "false_refusal_strict_sample",
                  "false_conf_{tau}_sample", "false_conf_{tau}_with_invariance_sample")


def derive_record(record: dict, tolerance: float) -> dict:
    """Add the ``_sample`` fields to every analyzer block of one replicate record."""
    truth = record.get("var_theta_sample")
    for block in (record.get("analyzers") or {}).values():
        if truth is None or block.get("var_theta_hat") is None:
            block["bias_sample"] = None
            block["abs_bias_sample_ok"] = None
            ok = None
        else:
            bias = float(block["var_theta_hat"]) - float(truth)
            block["bias_sample"] = bias
            ok = bool(abs(bias) <= tolerance)
            block["abs_bias_sample_ok"] = ok
        if "status_B" in block:
            status = block["status_B"]
            raw = block.get("phi_B_raw")
            close = raw is not None and truth is not None and abs(float(raw) - float(truth)) <= tolerance
            block["false_confidence_sample"] = (None if ok is None
                                                else bool(status == "identified" and not ok))
            block["false_refusal_sample"] = bool(status in ("bounded", "unsupported") and close)
            block["false_refusal_strict_sample"] = bool(status == "unsupported" and close)
        for tau in metrics.TAUS:
            for suffix in ("", "_with_invariance"):
                key = f"pass_{tau}{suffix}"
                if key in block:
                    block[f"false_conf_{tau}{suffix}_sample"] = (
                        None if ok is None else bool(block[key] and not ok))
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trapworld.derive")
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    grid_dir = out_dir / "grid"
    if not grid_dir.exists():
        print(f"no grid directory at {grid_dir}", file=sys.stderr)
        return 1

    rows, cells_done = [], []
    for path in sorted(grid_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if "records" not in payload:                  # the J_live not_run stub
            continue
        cell = payload["cell"]
        tolerance = float(payload["params"]["tolerance"])
        records = [derive_record(r, tolerance) for r in payload["records"]]
        payload["records"] = records
        payload["aggregate"] = metrics.aggregate(records)
        payload["derived"] = {"fields": list(DERIVED_FIELDS),
                              "derived_utc": datetime.now(timezone.utc).isoformat(),
                              "note": "exploratory: defined after results were seen (spec 11 A7)"}
        run_grid._write_json(path, payload)
        rows += run_grid._summary_rows(cell, payload["aggregate"])
        cells_done.append(cell["cell_id"])
        print(f"derived {cell['cell_id']} ({len(records)} records)", file=sys.stderr)

    run_grid.write_summary_csv(out_dir / "summary.csv", rows)
    run_grid._write_json(out_dir / "summary.json",
                         {"generated_utc": datetime.now(timezone.utc).isoformat(),
                          "headline_sweeps": list(grid.HEADLINE_SWEEPS), "rows": rows,
                          "derived_fields": list(DERIVED_FIELDS),
                          "derived_note": "fields ending in _sample are exploratory (spec 11 A7)"})
    run_grid._write_json(out_dir / "derived_manifest.json",
                         {"derived_utc": datetime.now(timezone.utc).isoformat(),
                          "cells": cells_done, "fields": list(DERIVED_FIELDS)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
