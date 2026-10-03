"""Crossover sweep (2026-10-02): where the adaptive-phase estimators stop beating the anchor.

The frozen 2026-09-17 grid (results/summary.csv, sweep A, mean over the psi x gamma_C cells)
records the person-score RMSE against theta on the raw scale.  anchor_only sits at 0.291 at
every kappa.  The estimators that also use the adaptive phase start below it and end above
it: misspecified 0.184 / 0.213 / 0.277 / 0.448 and naive 0.208 / 0.276 / 0.388 / 0.640 at
kappa 0 / 0.5 / 1 / 2, epoch_fe 0.188 / 0.235 / 0.315 / 0.500, while agentsem stays at
0.184 / 0.195 / 0.195 / 0.195.  Cells with equal psi * kappa agree to the third decimal
(naive 0.5945 at kappa 2 psi 0.5 against 0.5961 at kappa 1 psi 1), so the sweep is laid
out on the product psi * kappa and on the anchor length T0, the two quantities the
closed form `eta = theta (1 + psi kappa) + psi kappa e` and the anchor score's error
variance (proportional to 1 / T0) say control the sign.

Nothing in `trapworld/` is edited.  The generator, the analyzers and PARAMS are the hashed
frozen objects; this script only registers new cells, draws their replicates from its own
seed block, and scores paired RMSE differences against anchor_only from the records.

Predictions, written with the registry before any replicate exists (paired RMSE difference
`d_X = rmse_X - rmse_anchor_only`, per replicate, same world; a crossing is the product at
which the ladder's mean `d_X` changes sign, linearly interpolated between neighbours):

  P1  product invariance: at a matched product the mean `d_X` of two ladders with different
      kappa agree within 0.02 for naive, epoch_fe, misspecified and agentsem (T0 = 14).
  P2  crossings at T0 = 14: naive in [0.30, 0.40]; epoch_fe in [0.45, 0.55]; misspecified in
      [0.60, 0.80]; detrend has no crossing (worse than the anchor at every product);
      agentsem has no crossing (better than the anchor at every product up to 1.5).
  P3  anchor length: every crossing moves to a higher product at T0 = 7 and to a lower
      product at T0 = 28 than at T0 = 14, and anchor_only's own RMSE falls with T0.

Seed block: `SeedSequence(master_seed, spawn_key=(3000 + i, rep))`, disjoint from the frozen
grid's cell indices 0..72.  Usage:

  uv run python scripts/crossover_sweep.py register        # write the registry + hash
  uv run python scripts/crossover_sweep.py smoke           # one replicate of three cells
  uv run python scripts/crossover_sweep.py run --workers 10   # cells/*.json + hashes_after
  uv run python scripts/crossover_sweep.py score              # crossings.json + crossings.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trapworld import analyzers as analyzers_module  # noqa: E402
from trapworld import metrics  # noqa: E402
from trapworld import world as world_module  # noqa: E402
from trapworld.params import PARAMS, canonical_hash  # noqa: E402
from trapworld.run_grid import _sha256_bytes, _write_json, check_hashes  # noqa: E402

OUT = ROOT / "results" / "crossover"
SEED_BLOCK = 3000
REPS = 200
ANALYZERS = ("anchor_only", "naive", "naive_pooled", "epoch_fe", "detrend",
             "misspecified", "agentsem")
COMPARED = ANALYZERS[1:]

# Frozen cells on the same ladders, scored from results/grid/*.json alongside the new ones.
FROZEN = {
    ("P", 1.0, 14): ["F_p0.3_k1.0", "A_k1.0_p0.5_g1.0", "A_k1.0_p1.0_g1.0"],
    ("P", 2.0, 14): ["F_p0.2_k2.0", "F_p0.3_k2.0", "A_k2.0_p0.5_g1.0", "A_k2.0_p1.0_g1.0"],
    ("P", 0.5, 14): ["F_p0.3_k0.5", "A_k0.5_p0.5_g1.0", "A_k0.5_p1.0_g1.0"],
}


def _f(value: float) -> str:
    return f"{value:g}"


def _build() -> list[dict]:
    cells = []

    def add(block, kappa, psi, T0=14):
        cid = f"X_{block}_k{_f(kappa)}_p{_f(psi)}" + ("" if T0 == 14 else f"_a{T0}")
        over = dict(kappa=kappa, psi=psi, gamma_C=1.0)
        if T0 != 14:
            over["T0"] = T0                   # a parameter override, so not a META key
        cells.append(dict(cell_id=cid, sweep="X", block=block, product=round(kappa * psi, 6),
                          reps=REPS, **over))

    # Block P: the product ladder at T0 = 14.  Frozen cells already cover psi 0.3 / 0.5 / 1.0
    # at kappa 1, psi 0.2 / 0.3 / 0.5 / 1.0 at kappa 2 and psi 0.3 / 0.5 / 1.0 at kappa 0.5.
    for psi in (0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9):
        add("P", 1.0, psi)
    for psi in (0.15, 0.175, 0.225, 0.25, 0.275, 0.325, 0.35, 0.375, 0.4, 0.45, 0.6, 0.75):
        add("P", 2.0, psi)
    for psi in (0.6, 0.7, 0.8, 0.9):
        add("P", 0.5, psi)
    # Block T: the anchor-length lever at kappa = 1.
    for psi in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        add("T", 1.0, psi, T0=7)
    for psi in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        add("T", 1.0, psi, T0=28)
    ids = [c["cell_id"] for c in cells]
    assert len(ids) == len(set(ids)), "duplicate cell id"
    return cells


CELLS = _build()
CELLS_HASH = canonical_hash(CELLS)
META = ("cell_id", "sweep", "block", "product", "reps")


def overrides(cell: dict) -> dict:
    return {k: v for k, v in cell.items() if k not in META}


def replicate_seed(i: int, rep: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(entropy=PARAMS["master_seed"],
                                  spawn_key=(SEED_BLOCK + int(i), int(rep)))


def run_replicate(i: int, rep: int) -> dict:
    cell = CELLS[i]
    world = world_module.simulate(overrides(cell), replicate_seed(i, rep))
    results = {}
    for name in ANALYZERS:
        try:
            results[name] = analyzers_module.run_analyzer(name, world, 0)
        except Exception as exc:                          # recorded, never fatal
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return metrics.replicate_metrics(world, results, cell_id=cell["cell_id"], rep=rep)


def _work(task):
    i, rep = task
    started = time.perf_counter()
    try:
        record = run_replicate(i, rep)
    except Exception as exc:                              # one replicate, not the run
        record = {"cell_id": CELLS[i]["cell_id"], "rep": rep,
                  "error": f"{type(exc).__name__}: {exc}"}
    return i, rep, record, time.perf_counter() - started


def hashes() -> dict:
    return {
        "cells_hash": CELLS_HASH,
        "params_hash": canonical_hash(PARAMS),
        "script_src_sha256": _sha256_bytes(Path(__file__)),
        "world_src_sha256": _sha256_bytes(Path(world_module.__file__)),
        "analyzers_src_sha256": _sha256_bytes(Path(analyzers_module.__file__)),
        "frozen_hashes_mismatches": check_hashes(ROOT / "results"),
        "seed_block": SEED_BLOCK, "reps": REPS, "analyzers": list(ANALYZERS),
    }


def register() -> None:
    _write_json(OUT / "registry.json", {"cells": CELLS, **hashes(),
                                        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(f"{len(CELLS)} cells, cells_hash {CELLS_HASH}")


def smoke() -> None:
    for cid in ("X_P_k1_p0.6", "X_T_k1_p0.5_a7", "X_T_k1_p0.5_a28"):
        i = next(j for j, c in enumerate(CELLS) if c["cell_id"] == cid)
        _, _, rec, secs = _work((i, 0))
        if "error" in rec:
            print(cid, "ERROR", rec["error"])
            continue
        print(cid, f"{secs:.2f}s", {n: (round(b["rmse"], 4) if b.get("rmse") is not None
                                        else b.get("error", "none"))
                                    for n, b in rec["analyzers"].items()})


def run(workers: int) -> None:
    reg = OUT / "registry.json"
    if not reg.exists():
        raise SystemExit("register first")
    if json.loads(reg.read_text())["cells_hash"] != CELLS_HASH:
        raise SystemExit("registry hash differs from the cells in this file; re-register")
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Pool(workers) as pool:
        for i, cell in enumerate(CELLS):
            items = sorted(pool.imap_unordered(_work, [(i, r) for r in range(REPS)], chunksize=4),
                           key=lambda t: t[1])
            records = [r for _, _, r, _ in items]
            info = {"reps": len(records), "seconds": sum(t[3] for t in items),
                    "n_errors": sum(1 for r in records if "error" in r)}
            _write_json(OUT / "cells" / f"{cell['cell_id']}.json",
                        {"cell": cell, "info": info, "aggregate": metrics.aggregate(records),
                         "records": records})
            print(f"{time.strftime('%H:%M:%S')} {cell['cell_id']} done errors {info['n_errors']} "
                  f"elapsed {time.perf_counter() - started:.0f}s ({i + 1}/{len(CELLS)} cells)",
                  flush=True)
    _write_json(OUT / "hashes_after.json", hashes())


# ------------------------------------------------------------------------- scoring
def _payload(cell_id: str) -> dict:
    """A sweep cell file, or the frozen grid's record of the same cell id."""
    for path in (OUT / "cells" / f"{cell_id}.json", ROOT / "results" / "grid" / f"{cell_id}.json"):
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(cell_id)


def _verdict(d: dict) -> str:
    """H: the adaptive-phase estimator holds (beats the anchor by more than 2 mc_se);
    F: it flips (worse than the anchor by more than 2 mc_se); T: tie within 2 mc_se."""
    if d["mean"] is None:
        return "none"
    if abs(d["mean"]) < 2 * d["mc_se"]:
        return "T"
    return "F" if d["mean"] > 0 else "H"


def _crossing(points: list[tuple[float, float, float]]):
    """First sign change of the mean along the ladder, linearly interpolated.

    `points` are (product, mean, mc_se) sorted by product.  Returns None when the sign
    never changes; the band is the product range over which |mean| < 2 mc_se.
    """
    cross = None
    for (x0, m0, _), (x1, m1, _) in zip(points, points[1:]):
        if m0 == 0 or m0 * m1 < 0:
            cross = x0 + (x1 - x0) * (-m0) / (m1 - m0)
            break
    band = [x for x, m, se in points if se is not None and abs(m) < 2 * se]
    return cross, ([min(band), max(band)] if band else None)


def score() -> None:
    ladders: dict[tuple, list[str]] = {}
    for key, ids in FROZEN.items():
        ladders.setdefault(key, []).extend(ids)
    for c in CELLS:
        ladders.setdefault((c["block"], c["kappa"], c.get("T0", 14)), []).append(c["cell_id"])
    out = {"ladders": {}, "crossings": {}}
    lines = []
    for (block, kappa, T0), ids in sorted(ladders.items()):
        key = f"{block}_k{_f(kappa)}_T0{T0}"
        table = []
        for cid in ids:
            try:
                payload = _payload(cid)
            except FileNotFoundError:
                continue
            row = {"cell_id": cid,
                   "product": round(payload["cell"]["kappa"] * payload["cell"]["psi"], 6),
                   "anchor_rmse": payload["aggregate"]["analyzers"]["anchor_only"]["rmse"]["mean"]}
            for name in COMPARED:
                row[name] = metrics._paired_difference(payload["records"], name, "anchor_only", "rmse")
            table.append(row)
        table.sort(key=lambda r: r["product"])
        out["ladders"][key] = table
        out["crossings"][key] = {}
        for name in COMPARED:
            pts = [(r["product"], r[name]["mean"], r[name]["mc_se"]) for r in table
                   if r[name]["mean"] is not None]
            cross, band = _crossing(pts)
            out["crossings"][key][name] = {"crossing": cross, "band": band}
        lines.append(f"\n## ladder {key}  (paired rmse_X - rmse_anchor_only, mean (mc_se) verdict H/F/T)")
        lines.append("| cell | product | anchor rmse | " + " | ".join(COMPARED) + " |")
        lines.append("|---|---|---|" + "---|" * len(COMPARED))
        for r in table:
            cells_txt = " | ".join(
                f"{r[n]['mean']:+.4f} ({r[n]['mc_se']:.4f}) {_verdict(r[n])}"
                if r[n]["mean"] is not None else "none" for n in COMPARED)
            lines.append(f"| {r['cell_id']} | {r['product']:g} | {r['anchor_rmse']:.4f} | {cells_txt} |")
        lines.append("crossings: " + "; ".join(
            f"{n} {out['crossings'][key][n]['crossing']:.3f}" if out['crossings'][key][n]['crossing'] is not None
            else f"{n} none" for n in COMPARED))
    _write_json(OUT / "crossings.json", out)
    text = "\n".join(lines)
    (OUT / "crossings.md").write_text(text + "\n")
    print(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["register", "smoke", "run", "score"])
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    run(args.workers) if args.mode == "run" else {"register": register, "smoke": smoke,
                                                  "score": score}[args.mode]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
