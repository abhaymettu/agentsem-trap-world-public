"""The grid runner: seeds, work units, hashes, results files (spec sections 6 and 9).

One command runs the frozen grid end to end.  What it guarantees:

* **Seeds.** A replicate's nine RNG streams come from child ``rep`` of child
  ``cell_index`` of ``SeedSequence(PARAMS["master_seed"])``, with the cell's position in
  ``grid.CELLS`` as the index and 500 replicate seeds reserved per cell whatever that
  cell's ``reps``.  Nothing about a replicate depends on how many replicates ran, in what
  order they finished, or how many workers there were.
* **Hashes.** Every frozen object is hashed into ``results/hashes.json`` before the first
  replicate and re-checked after the last one, so a results directory says on its face
  whether the code it came from moved underneath it.
* **Failure is recorded, not hidden.** An analyzer that raises leaves an ``error`` string
  in its own block; a Jev call that fails leaves the replicate with no analyzer fields at
  all and is counted in ``n_failed``.  Nothing is ever filled in from another arm.

Sweep J (the external policy arm, spec 11 A3) is run like any other cell but is kept out
of every cross-cell aggregate, and ``J_live`` does not run at all unless a live or
recorded transport was asked for on the command line.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

from trapworld import analyzers as analyzers_module
from trapworld import certificate, grid, metrics
from trapworld import jev as jev_module
from trapworld import params as params_module
from trapworld import world as world_module
from trapworld.analyzers import ANALYZERS, run_analyzer
from trapworld.params import PARAMS

__all__ = ["main", "run_replicate", "replicate_seed", "compute_hashes", "check_hashes",
           "HASH_KEYS"]

#: The frozen objects, in the order the hash table prints them (spec 9, 11 A5).
HASH_KEYS = ("params_hash", "grid_hash", "surrogate_hash", "rules_hash",
             "certificate_src_sha256", "analyzers_src_sha256", "jev_policy_hash",
             "runner_src_sha256")

#: Replicates timed before deciding whether a cell is too slow, and the limit (spec 6, relaxed
#: by amendment A6: the 3 s rule fired on scheduling contention from unrelated processes,
#: not on analyzer cost, which is 1.3 s per replicate in one process).
PROBE_REPS = 10
SLOW_SECONDS = 8.0
HALVE_ABOVE_REPS = 100

SMOKE_REPS = 2
SMOKE_WORKERS = 2

J_LIVE_REASON = ("requires a separately orchestrated live session (JEV-LIVE-RUN.md)")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------- hashes
def compute_hashes() -> dict:
    """Recompute every frozen hash from the code as it is right now.

    The modules are read through their module objects rather than through ``from ...
    import``, so a test that swaps one frozen object in (``certificate.RULES``) changes
    the hash it is supposed to change.
    """
    canonical_hash = params_module.canonical_hash
    return {
        "params_hash": canonical_hash(params_module.PARAMS),
        "grid_hash": canonical_hash(grid.CELLS),
        "surrogate_hash": canonical_hash(inspect.getsource(world_module.surrogate_g1)),
        "rules_hash": canonical_hash(certificate.RULES),
        "certificate_src_sha256": _sha256_bytes(Path(certificate.__file__)),
        "analyzers_src_sha256": _sha256_bytes(Path(analyzers_module.__file__)),
        "jev_policy_hash": canonical_hash(jev_module.JEV_POLICY),
        # The runner decides which replicates run and how they are scored; a change to it
        # after a run is as visible as a change to the rules.
        "runner_src_sha256": _sha256_bytes(Path(__file__)),
    }


def write_hashes(out_dir: Path) -> dict:
    payload = compute_hashes()
    payload["written_utc"] = _now()
    _write_json(Path(out_dir) / "hashes.json", payload)
    return payload


def recheck_hashes(out_dir: Path, before: dict) -> bool:
    after = compute_hashes()
    ok = all(after[key] == before.get(key) for key in HASH_KEYS)
    path = Path(out_dir) / "hashes.json"
    payload = json.loads(path.read_text())
    payload["hashes_match"] = ok
    payload["checked_utc"] = _now()
    if not ok:
        payload["recheck"] = after
    _write_json(path, payload)
    return ok


def check_hashes(out_dir) -> int:
    """Recompute every hash, compare with the stored file, print the table.

    Returns a process exit code: 0 when every frozen object still hashes to what the run
    recorded, 1 otherwise (including when there is no file to compare with).
    """
    path = Path(out_dir) / "hashes.json"
    if not path.exists():
        print(f"no hashes file at {path}", file=sys.stderr)
        return 1
    stored = json.loads(path.read_text())
    now = compute_hashes()
    width = max(len(key) for key in HASH_KEYS)
    ok = True
    print(f"{'object':<{width}}  {'stored':<16}  {'now':<16}  match")
    for key in HASH_KEYS:
        was, is_now = stored.get(key), now[key]
        same = was == is_now
        ok = ok and same
        print(f"{key:<{width}}  {str(was)[:16]:<16}  {is_now[:16]:<16}  "
              f"{'yes' if same else 'NO'}")
    print("all hashes match" if ok else "HASH MISMATCH: results and code disagree")
    return 0 if ok else 1


# ------------------------------------------------------------------------- seeds
def replicate_seed(cell_index: int, rep: int) -> np.random.SeedSequence:
    """The SeedSequence of one replicate.

    This is child ``rep`` of child ``cell_index`` of ``SeedSequence(master_seed)``, built
    from the spawn key directly.  ``SeedSequence.spawn`` mutates its parent's child
    counter, so a worker that spawned a cell's children twice would hand out the next 500
    children the second time; the spawn key is the same object without that hazard, and
    ``tests/test_grid.py`` checks the two constructions against each other.
    """
    return np.random.SeedSequence(entropy=PARAMS["master_seed"],
                                  spawn_key=(int(cell_index), int(rep)))


# ------------------------------------------------------------------- the work unit
def run_replicate(cell_index: int, rep: int, bootstrap: int = 0, jev_kind: str = "fake",
                  log_dir=None, recorded_dir=None, analyzer_names=None) -> dict:
    """Simulate one replicate, run the analyzers, return its metrics record."""
    cell = grid.CELLS[cell_index]
    overrides = grid.cell_overrides(cell)
    seed = replicate_seed(cell_index, rep)
    names = tuple(analyzer_names) if analyzer_names else tuple(ANALYZERS)

    jev_record = None
    if cell["sweep"] == "J":
        # J_fake is the fake arm by definition; J_live takes the requested transport.
        kind = "fake" if cell.get("jev") == "fake" else jev_kind
        world, jev_record = jev_module.run_jev_replicate(
            overrides, seed, kind, log_dir, cell["cell_id"], rep, recorded_dir)
        if world is None or jev_record.get("status") == "failed":
            # A failed call makes the replicate missing. It is never substituted.
            return metrics.jsonable({
                "cell_id": cell["cell_id"], "rep": rep,
                "policy": overrides.get("policy"), "jev_status": "failed",
                "jev_failure": jev_record.get("failure"),
                "jev": {"n_calls": jev_record.get("n_calls"),
                        "n_failed": jev_record.get("n_failed"),
                        "mean_latency_ms": jev_record.get("mean_latency_ms"),
                        "returned_models": list(jev_record.get("returned_models") or []),
                        "failure_classes": dict(jev_record.get("failure_classes") or {})}})
    else:
        world = world_module.simulate(overrides, seed)

    results: dict[str, dict] = {}
    for name in names:
        try:
            results[name] = run_analyzer(name, world,
                                         bootstrap if name == "agentsem" else 0)
        except Exception as exc:                          # recorded, never fatal
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}

    return metrics.replicate_metrics(world, results, cell_id=cell["cell_id"], rep=rep,
                                     jev=jev_record)


def _work(task):
    """Pool entry point: ``(cell_index, rep, options) -> (rep, record, seconds)``."""
    cell_index, rep, options = task
    started = time.perf_counter()
    try:
        record = run_replicate(cell_index, rep, **options)
    except Exception as exc:                              # one replicate, not the run
        record = {"cell_id": grid.CELLS[cell_index]["cell_id"], "rep": rep,
                  "error": f"{type(exc).__name__}: {exc}"}
    return rep, record, time.perf_counter() - started


# ------------------------------------------------------------------------- the run
def _map(pool, tasks):
    if pool is None:
        return [_work(task) for task in tasks]
    return list(pool.imap_unordered(_work, tasks))


def run_cell(pool, cell_index: int, reps: int, options: dict) -> tuple[list, dict]:
    """Run one cell's replicates, honouring the 3 s halving rule of spec section 6."""
    done = _map(pool, [(cell_index, rep, options) for rep in range(min(PROBE_REPS, reps))])
    halved = False
    if reps > PROBE_REPS:
        median = float(np.median([seconds for _, _, seconds in done]))
        if median > SLOW_SECONDS and reps > HALVE_ABOVE_REPS:
            reps = reps // 2
            halved = True
        done += _map(pool, [(cell_index, rep, options)
                            for rep in range(PROBE_REPS, reps)])

    done.sort(key=lambda item: item[0])          # output must not depend on finish order
    records = [record for _, record, _ in done]
    info = {"reps": len(records), "reps_halved": halved,
            "seconds": sum(seconds for _, _, seconds in done),
            "seconds_per_replicate": (sum(s for _, _, s in done) / len(done)
                                      if done else 0.0),
            "n_errors": sum(1 for record in records if "error" in record)}
    return records, info


# ------------------------------------------------------------------------- writing
def _write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics.jsonable(payload), indent=2,
                                     sort_keys=True, allow_nan=False) + "\n")


def _flatten(prefix: str, node: dict, out: dict) -> dict:
    """Flatten a nested summary to ``field_sub`` columns; lists are not columns."""
    for key, value in node.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten(name, value, out)
        elif not isinstance(value, list):
            out[name] = value
    return out


def _summary_rows(cell: dict, aggregate: dict) -> list[dict]:
    head = {
        "cell_id": cell["cell_id"], "sweep": cell["sweep"],
        "reps_run": aggregate.get("n_reps"), "n_scored": aggregate.get("n_scored"),
        "n_failed": aggregate.get("n_failed"),
        "policy": cell.get("policy", PARAMS["policy"]),
        "generator": cell.get("generator", PARAMS["generator"]),
        "kappa": cell.get("kappa", PARAMS["kappa"]),
        "psi": cell.get("psi", PARAMS["psi"]),
        "gamma_C": cell.get("gamma_C", PARAMS["gamma_C"]),
        "dW_mean": aggregate.get("dW_mean"), "dW_mc_se": aggregate.get("dW_mc_se"),
        "implied_r_mean": aggregate.get("implied_r_mean"),
    }
    rows = []
    for name, fields in aggregate.get("analyzers", {}).items():
        row = dict(head, analyzer=name)
        rows.append(_flatten("", fields, row))
    return rows


IDENTITY_COLUMNS = ("cell_id", "sweep", "analyzer", "policy", "generator", "kappa",
                    "psi", "gamma_C", "reps_run", "n_scored", "n_failed",
                    "dW_mean", "dW_mc_se", "implied_r_mean")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row} - set(IDENTITY_COLUMNS))
    header = list(IDENTITY_COLUMNS) + keys
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def coverage_audit(aggregates: dict, records: dict) -> dict:
    """F17: the Hessian interval against the bootstrap one, per audit cell."""
    out = {}
    for cell_id, aggregate in aggregates.items():
        arm = aggregate.get("analyzers", {}).get("agentsem", {})
        out[cell_id] = {
            "n_reps": aggregate.get("n_reps"),
            "hessian_coverage": arm.get("covers_hessian"),
            "bootstrap_coverage": arm.get("covers_bootstrap"),
            "certificate_coverage": arm.get("covers"),
            "ci_hessian_width": arm.get("ci_hessian_width"),
            "ci_bootstrap_width": arm.get("ci_bootstrap_width"),
            "bias": arm.get("bias"),
            "boot_reps_ok": arm.get("boot_reps_ok"),
            "status_B": arm.get("status_B"),
            "records": records[cell_id],
        }
    return out


# ------------------------------------------------------------------------- the CLI
def _parse(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m trapworld.run_grid",
        description="Run the frozen trap-world grid (spec sections 6 and 9).")
    parser.add_argument("--reps", type=int, default=None,
                        help="override every cell's replicate count")
    parser.add_argument("--cells", default=None, help="comma separated cell ids")
    parser.add_argument("--workers", type=int, default=None,
                        help="worker processes (default: one fewer than the cores)")
    parser.add_argument("--bootstrap", action="store_true",
                        help="bootstrap coverage audit on the audit cells only")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--smoke", action="store_true",
                        help="2 replicates, 2 workers, the first cell of each sweep")
    parser.add_argument("--jev", choices=("fake", "live", "recorded"), default="fake",
                        help="transport for the J_live cell (default: fake, so it is skipped)")
    parser.add_argument("--recorded-dir", default=None,
                        help="directory of recorded Jev logs for --jev recorded")
    parser.add_argument("--check-hashes", action="store_true",
                        help="recompare the frozen hashes with --out and exit")
    return parser.parse_args(argv)


def _selected_ids(args) -> list[str]:
    if args.cells:
        return [part.strip() for part in args.cells.split(",") if part.strip()]
    if args.bootstrap:
        return list(grid.AUDIT)
    if args.smoke:
        return grid.first_of_each_sweep()
    return [cell["cell_id"] for cell in grid.CELLS]


def _reps_for(cell: dict, args) -> int:
    if args.reps is not None:
        return int(args.reps)
    if args.smoke:
        return SMOKE_REPS
    if args.bootstrap:
        return grid.AUDIT_REPS
    return int(cell["reps"])


def main(argv=None) -> int:
    args = _parse(argv)
    out_dir = Path(args.out)
    if args.check_hashes:
        return check_hashes(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.jev == "live":
        # A live session spends a real credential. It does not start against code that has
        # moved since the grid it is meant to join.
        if not (out_dir / "hashes.json").exists():
            print(f"--jev live needs {out_dir / 'hashes.json'} to exist first",
                  file=sys.stderr)
            return 2
        if check_hashes(out_dir) != 0:
            print("--jev live refused: the frozen hashes do not match", file=sys.stderr)
            return 2

    selected = _selected_ids(args)
    cells = {cell_id: grid.cell_by_id(cell_id) for cell_id in selected}
    workers = args.workers or (SMOKE_WORKERS if args.smoke
                               else max(1, (os.cpu_count() or 2) - 1))
    bootstrap = grid.AUDIT_BOOTSTRAP if args.bootstrap else 0
    analyzer_names = ("agentsem",) if args.bootstrap else None

    # J_live runs only in a separately orchestrated live session. Any run that touches
    # sweep J at all leaves the stub behind, so a results directory always says what
    # happened to the live arm rather than staying silent about it.
    touches_j = any(cell["sweep"] == "J" for cell in cells.values())
    live_requested = args.jev in ("live", "recorded")
    j_live_state = "not_selected"
    if "J_live" in cells and live_requested:
        j_live_state = f"run_{args.jev}"
    elif touches_j:
        selected = [cell_id for cell_id in selected if cell_id != "J_live"]
        j_live_state = "not_run"
        _write_json(out_dir / "grid" / "J_live.json",
                    {"status": "not_run", "reason": J_LIVE_REASON,
                     "cell": grid.cell_by_id("J_live")})

    before = write_hashes(out_dir)
    started = _now()
    clock = time.perf_counter()
    options = dict(bootstrap=bootstrap, jev_kind=args.jev,
                   log_dir=str(out_dir / "jev"), recorded_dir=args.recorded_dir,
                   analyzer_names=analyzer_names)

    aggregates, all_records, info_by_cell, rows = {}, {}, {}, []
    context = mp.get_context("spawn")
    pool = context.Pool(workers) if workers > 1 else None
    try:
        for position, cell_id in enumerate(selected, start=1):
            cell = cells[cell_id]
            cell_index = grid.CELL_INDEX[cell_id]
            records, info = run_cell(pool, cell_index, _reps_for(cell, args), options)
            aggregate = metrics.aggregate(records)
            aggregates[cell_id] = aggregate
            all_records[cell_id] = records
            info_by_cell[cell_id] = info
            if not args.bootstrap:
                _write_json(out_dir / "grid" / f"{cell_id}.json",
                            {"cell": cell, "params": params_module.resolve(
                                grid.cell_overrides(cell)),
                             "records": records, "aggregate": aggregate})
                rows += _summary_rows(cell, aggregate)
            print(f"[{position}/{len(selected)}] {cell_id} reps={info['reps']}"
                  f"{' (halved)' if info['reps_halved'] else ''} "
                  f"{info['seconds_per_replicate']:.2f}s/rep "
                  f"errors={info['n_errors']}", file=sys.stderr, flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.perf_counter() - clock
    if args.bootstrap:
        _write_json(out_dir / "coverage_audit.json",
                    {"bootstrap": bootstrap, "cells": coverage_audit(aggregates,
                                                                     all_records)})
    else:
        write_summary_csv(out_dir / "summary.csv", rows)
        # One row per cell x analyzer, each labelled with its sweep.  Nothing pools cells:
        # a mean bias over G1 and G4 worlds would be a number without a meaning (S13).
        _write_json(out_dir / "summary.json",
                    {"generated_utc": _now(), "headline_sweeps": list(grid.HEADLINE_SWEEPS),
                     "rows": rows})

    _write_json(out_dir / "run_manifest.json", {
        "mode": "bootstrap" if args.bootstrap else "grid",
        "start_utc": started, "end_utc": _now(), "elapsed_seconds": elapsed,
        "workers": workers, "jev": args.jev, "bootstrap": bootstrap,
        "slow_seconds": SLOW_SECONDS, "probe_reps": PROBE_REPS,
        "J_live": j_live_state,
        "master_seed": PARAMS["master_seed"], "grid_hash": grid.GRID_HASH,
        "cells": selected,
        "reps_per_cell": {cell_id: info_by_cell[cell_id]["reps"] for cell_id in selected},
        "reps_requested": {cell_id: _reps_for(cells[cell_id], args)
                           for cell_id in selected},
        "seconds_per_cell": {cell_id: info_by_cell[cell_id]["seconds"]
                             for cell_id in selected},
        "seconds_per_replicate": {cell_id: info_by_cell[cell_id]["seconds_per_replicate"]
                                  for cell_id in selected},
        "reps_halved": {cell_id: info_by_cell[cell_id]["reps_halved"]
                        for cell_id in selected},
        "errors_per_cell": {cell_id: info_by_cell[cell_id]["n_errors"]
                            for cell_id in selected},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
    })

    matched = recheck_hashes(out_dir, before)
    if not matched:
        print("HASH MISMATCH after the run: the code moved underneath the results",
              file=sys.stderr)
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
