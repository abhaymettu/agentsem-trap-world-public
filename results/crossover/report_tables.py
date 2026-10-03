"""Print the tables for CROSSOVER-SWEEP-RESULT.md from the recorded files only.

Reads results/summary.csv (the frozen grid) and results/crossover/{registry,crossings,closed_form,
hashes_after}.json.  Writes nothing.  The per-ladder tables are in
crossings.md (written by `scripts/crossover_sweep.py score`) and are not reprinted here.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from statistics import mean
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "crossover"
COMPARED = ["naive", "naive_pooled", "epoch_fe", "detrend", "misspecified", "agentsem"]
K1 = (("T", 7), ("P", 14), ("T", 28))            # the kappa-1 ladders, by anchor length


def load(name):
    return json.loads((OUT / name).read_text())


def fmt(c):
    return "none" if c is None else f"{c:.3f}"


def main():
    reg, cross, cf, after = (load(n) for n in
                             ("registry.json", "crossings.json", "closed_form.json", "hashes_after.json"))

    # ------------------------------------------------------------------ frozen sweep A baseline
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "summary.csv"))
            if r["sweep"] == "A" and r["gamma_C"] == "1.0" and r.get("rmse_mean")]
    agg = defaultdict(list)
    cellv = {}
    for r in rows:
        agg[(r["analyzer"], float(r["kappa"]))].append(float(r["rmse_mean"]))
        cellv[(r["analyzer"], float(r["kappa"]), float(r["psi"]))] = float(r["rmse_mean"])
    kappas = (0.0, 0.5, 1.0, 2.0)
    means = {key: mean(v) for key, v in agg.items()}
    print("## Frozen sweep A, gamma_C 1.0, mean rmse over psi 0 / 0.5 / 1 at each kappa (summary.csv)\n")
    print("| estimator | " + " | ".join(f"kappa {k:g}" for k in kappas) + " | minus anchor_only: " +
          " | ".join(f"k{k:g}" for k in kappas) + " |")
    print("|---|" + "---|" * (2 * len(kappas)))
    for a in ["anchor_only"] + [c for c in COMPARED if c != "epoch_fe"]:
        label = "naive_pooled = epoch_fe" if a == "naive_pooled" else a
        print(f"| {label} | " + " | ".join(f"{means[(a, k)]:.3f}" for k in kappas) + " | " +
              " | ".join(f"{means[(a, k)] - means[('anchor_only', k)]:+.3f}" for k in kappas) + " |")
    print("\nmatched-product frozen cells (naive rmse): "
          f"product 1: k2 p0.5 {cellv[('naive', 2.0, 0.5)]:.4f} vs k1 p1 {cellv[('naive', 1.0, 1.0)]:.4f}; "
          f"product 0.5: k1 p0.5 {cellv[('naive', 1.0, 0.5)]:.4f} vs k0.5 p1 {cellv[('naive', 0.5, 1.0)]:.4f}")
    anchors = [v for (a, _, _), v in cellv.items() if a == "anchor_only"]
    print(f"anchor_only range over the 12 cells: {min(anchors):.4f} to {max(anchors):.4f}\n")

    # ------------------------------------------------------------------ hashes
    t = lambda v: v[:12] if isinstance(v, str) and len(v) > 16 else v   # noqa: E731
    fields = ("cells_hash", "params_hash", "world_src_sha256", "analyzers_src_sha256",
              "script_src_sha256", "frozen_hashes_mismatches", "seed_block", "reps")
    print("## Hashes (registry.json vs hashes_after.json): fields that differ\n")
    print({k: (t(reg[k]), t(after[k])) for k in fields if reg[k] != after[k]} or "none", "\n")

    # ------------------------------------------------------------------ crossings, with the closed form
    print("## Crossings by ladder against the closed form (crossings.json, closed_form.json)\n")
    print("| ladder | estimator | measured | tie band | closed form | diff |")
    print("|---|---|---|---|---|---|")
    for key, cr in cross["crossings"].items():
        T0 = re.search(r"T0(\d+)", key).group(1)
        for n in ("naive", "naive_pooled"):
            pred = cf["crossing"][T0].get(n)
            meas, band = cr[n]["crossing"], cr[n]["band"]
            diff = f"{meas - pred:+.3f}" if pred is not None and meas is not None else "-"
            band_txt = f"{band[0]:g} to {band[1]:g}" if band else "none"
            print(f"| {key} | {n} | {fmt(meas)} | {band_txt} | {fmt(pred)} | {diff} |")
    print()

    # ------------------------------------------------------------------ P1 product invariance
    print("## P1 product invariance at T0 = 14 (max pairwise |delta mean d_X| at a matched product)\n")
    by_prod = defaultdict(list)
    for key, table in cross["ladders"].items():
        if key.endswith("_T014"):
            for r in table:
                by_prod[round(r["product"], 6)].append(r)
    names = ("naive", "epoch_fe", "misspecified", "agentsem")
    print("| product | cells | " + " | ".join(names) + " | worst |")
    print("|---|---|" + "---|" * (len(names) + 1))
    worst_all = 0.0
    for p in sorted(by_prod):
        rs = by_prod[p]
        if len(rs) < 2:
            continue
        spreads = {n: max(r[n]["mean"] for r in rs) - min(r[n]["mean"] for r in rs) for n in names}
        w = max(spreads.values())
        worst_all = max(worst_all, w)
        print(f"| {p:g} | {', '.join(r['cell_id'] for r in rs)} | " +
              " | ".join(f"{spreads[n]:.4f}" for n in names) + f" | {w:.4f} |")
    print(f"\nworst matched-product spread: {worst_all:.4f} (P1 bar 0.02: {'HOLDS' if worst_all <= 0.02 else 'MISSES'})\n")

    # ------------------------------------------------------------------ P2 / P3
    print("## P2 crossings at T0 = 14 against the preregistered bands\n")
    bands = {"naive": (0.30, 0.40), "epoch_fe": (0.45, 0.55), "misspecified": (0.60, 0.80)}
    print("| ladder | products spanned | estimator | band | measured | hit |")
    print("|---|---|---|---|---|---|")
    for key, cr in cross["crossings"].items():
        if not key.endswith("_T014"):
            continue
        prods = [r["product"] for r in cross["ladders"][key]]
        span = f"{min(prods):g} to {max(prods):g}"
        for n, (lo, hi) in bands.items():
            c = cr[n]["crossing"]
            res = ("not spanned" if min(prods) > lo or max(prods) < hi
                   else "yes" if c is not None and lo <= c <= hi else "NO")
            print(f"| {key} | {span} | {n} | [{lo}, {hi}] | {fmt(c)} | {res} |")
        for n in ("detrend", "agentsem"):
            c = cr[n]["crossing"]
            print(f"| {key} | {span} | {n} | none | {fmt(c)} | {'yes' if c is None else 'NO'} |")
    print()
    print("## P3 anchor length (kappa = 1 ladders)\n")
    k1 = {T0: cross["crossings"][f"{b}_k1_T0{T0}"] for b, T0 in K1}
    print("| estimator | T0=7 | T0=14 | T0=28 | higher at 7 | lower at 28 |")
    print("|---|---|---|---|---|---|")
    for n in [c for c in COMPARED if c != "epoch_fe"]:
        c7, c14, c28 = (k1[T0][n]["crossing"] for T0 in (7, 14, 28))
        hi7 = "n/a" if c7 is None or c14 is None else "yes" if c7 > c14 else "NO"
        lo28 = "n/a" if c28 is None or c14 is None else "yes" if c28 < c14 else "NO"
        label = "naive_pooled = epoch_fe" if n == "naive_pooled" else n
        print(f"| {label} | {fmt(c7)} | {fmt(c14)} | {fmt(c28)} | {hi7} | {lo28} |")
    print("\nanchor_only rmse by T0, mean over the ladder's cells (closed form): " + ", ".join(
        f"T0={T0}: {mean(r['anchor_rmse'] for r in cross['ladders'][f'{b}_k1_T0{T0}']):.4f} "
        f"({cf['rmse'][str(T0)]['anchor_only']['0.0']:.4f})" for b, T0 in K1))


if __name__ == "__main__":
    main()
