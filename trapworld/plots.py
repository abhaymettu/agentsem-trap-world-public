"""Figures P1 to P10, regenerated from ``results/`` only (task 7).

Nothing here recomputes a statistic.  Every number on every panel is read from
``results/summary.json``, ``results/closed_forms.json``, ``results/grid/<cell_id>.json``
or ``results/coverage_audit.json``; the only arithmetic is the split of rule R4 into its
two tests, which is a re-read of the stored diagnostics against the frozen thresholds in
``rules.json`` and is validated against the stored ``triggered_B`` list.

Colour is Okabe-Ito and never carries meaning alone: every series also differs in marker
shape and line style and carries a direct text label, and bar identity is the axis label.
The six-slot order below was chosen by running the data-viz skill's validator
(``validate_palette.js``) on candidate orderings against a white surface; this order is
the one that passes the lightness band, the chroma floor, the adjacent-pair CVD floor
(worst 11.0 deutan) and the normal-vision floor (worst 18.4), and whose first four slots
also pass all-pairs.  Slots 4 and 5 sit below 3:1 contrast on white, so the skill's
relief rule applies and they are used only where a visible direct label is present.

No panel carries two y-scales: where the brief names a left and a right axis, the two
scales are drawn as separate rows of small multiples instead (the data-viz skill's
no-dual-axis rule), which keeps every quantity and changes only the layout.

Sweep J is never read by P1 to P9.  P10 is the fake-transport plumbing panel and says so
in its own title.

Run as ``python -m trapworld.plots``.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ------------------------------------------------------------------ palette and chrome

#: Okabe-Ito, in the validated slot order (see the module docstring).
OI = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "purple": "#CC79A7",
}
SLOTS = [OI["blue"], OI["vermillion"], OI["green"], OI["sky"], OI["orange"], OI["purple"]]

INK = "#000000"          # Okabe-Ito's own black, used for text only
SECOND_INK = "#4a4a4a"
MUTED = "#767676"
GRIDLINE = "#dcdcdc"
BAND = "#e8e8e8"

MARKERS = ["o", "s", "^", "D", "v", "P"]
DASHES = ["-", "--", "-.", (0, (1, 1)), (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1))]

EXPLORATORY = "exploratory (defined post hoc)"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#b4b4b4",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SECOND_INK,
    "ytick.color": SECOND_INK,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
})

TOLERANCE = 0.10          # the spec's |bias| <= .10 tolerance on Var(theta)

# ------------------------------------------------------------------------- the results

BASE_CELL = "A_k1.0_p0.5_g1.0"
P3_CELLS = ["A_k1.0_p0.5_g1.0", "B_G2", "B_G3_r03", "B_G3_r06", "B_G4", "B_G4b", "B_G4c"]
P3_ANALYZERS = ["naive", "epoch_fe", "detrend", "dose_partial", "belief_partial",
                "anchor_only", "misspecified", "agentsem"]
P4_CELLS = ["A_k1.0_p0.5_g1.0", "B_G2", "B_G3_r0", "B_G3_r03", "B_G3_r06",
            "B_G4", "B_G4b", "B_G4c"]
#: The pre-registered rules, in the order their segments stack.  Both R4 tests share the
#: rule id "R4" in ``triggered_B`` and are told apart by the stored diagnostics.
RULE_ORDER = ["none", "R4 exposure only", "R4 shift", "R5", "R6", "R2", "other"]
#: Slot order validated with the data-viz skill's validator on a white surface: worst
#: adjacent CVD deltaE 18.0, worst normal-vision 18.7, both gates PASS.
RULE_COLOUR = {
    "none": OI["green"],
    "R4 exposure only": OI["blue"],
    "R4 shift": OI["vermillion"],
    "R5": OI["sky"],
    "R6": OI["orange"],
    "R2": OI["purple"],
    "other": MUTED,
}
RULE_HATCH = {"none": "", "R4 exposure only": "//", "R4 shift": "\\\\",
              "R5": "xx", "R6": "..", "R2": "oo", "other": "++"}


class Results:
    """Lazy reader for one results directory.  Grid cells are read once each."""

    def __init__(self, root: pathlib.Path) -> None:
        self.root = pathlib.Path(root)
        self.summary = self._json("summary.json")
        self.closed = self._json("closed_forms.json")
        self.manifest = self._json("run_manifest.json") or {}
        self.rules = json.loads((pathlib.Path(__file__).with_name("rules.json")).read_text())
        rows = (self.summary or {}).get("rows", [])
        self.rows = {(r["cell_id"], r["analyzer"]): r for r in rows}
        self._cells: dict[str, dict] = {}

    def _json(self, name: str):
        path = self.root / name
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def row(self, cell_id: str, analyzer: str) -> dict:
        return self.rows.get((cell_id, analyzer), {})

    def val(self, cell_id: str, analyzer: str, field: str, default=None):
        value = self.row(cell_id, analyzer).get(field, default)
        return default if value is None else value

    def cell(self, cell_id: str) -> dict:
        if cell_id not in self._cells:
            path = self.root / "grid" / f"{cell_id}.json"
            self._cells[cell_id] = json.loads(path.read_text()) if path.exists() else {}
        return self._cells[cell_id]

    def reps(self, cell_id: str, analyzer: str = "naive") -> int:
        return int(self.val(cell_id, analyzer, "reps_run", 0) or 0)

    def coverage_audit(self):
        return self._json("coverage_audit.json")


# --------------------------------------------------------------------------- utilities

def _style(ax, *, xlabel="", ylabel="", title="", grid_axis="y"):
    ax.set_title(title, loc="left", color=INK, pad=6)
    ax.set_xlabel(xlabel, color=SECOND_INK)
    ax.set_ylabel(ylabel, color=SECOND_INK)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def _footnote(fig, text: str) -> None:
    fig.text(0.01, 0.006, text, ha="left", va="bottom", fontsize=7, color=MUTED,
             wrap=True)


def _label(ax, x, y, text, colour, *, dx=4, dy=0, ha="left", va="center", size=8,
           weight="normal"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                fontsize=size, color=colour, weight=weight, zorder=6)


def _save(fig, out_dir: pathlib.Path, name: str) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * float(value):.0f}%"


def _num(value, digits=3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _series_line(ax, x, y, se, colour, marker, dash, label, *, band=True, fill=True):
    y = list(y)
    ax.plot(x, y, color=colour, marker=marker, linestyle=dash, label=label,
            markerfacecolor=colour if fill else "white",
            markeredgecolor=colour, markeredgewidth=1.4, zorder=4)
    if band and se is not None and any(s is not None for s in se):
        lo = [a - (b or 0.0) for a, b in zip(y, se)]
        hi = [a + (b or 0.0) for a, b in zip(y, se)]
        ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0, zorder=2)


# ------------------------------------------------------------- the R4 split (P4 and P6)

def rule_category(agentsem: dict, anchor_only: dict, rules: dict) -> str:
    """The triggered rule set of one replicate, with R4's two tests told apart.

    ``triggered_B`` records both R4 tests under the one id, so the two are recovered from
    the diagnostics the certificate stored: the exposure test fires when
    ``exposure_p < R4.exposure_p_max``, the pooled shift test when the epoch difference or
    the change-point scan exceeds ``R4.shift_sd_max`` anchor SDs.  The anchor SD is the
    anchor CFA's Var(theta), which is what the ``anchor_only`` arm reports for the same
    replicate (both fit the same one-factor model to the same anchor person means).
    """
    triggered = set(agentsem.get("triggered_B") or [])
    if not triggered:
        return "none"
    phi_a = (anchor_only or {}).get("var_theta_hat")
    sd = math.sqrt(phi_a) if isinstance(phi_a, (int, float)) and phi_a and phi_a > 0 else 1.0
    threshold = float(rules["R4"]["shift_sd_max"]) * sd
    shift = abs(agentsem.get("epoch_shift") or 0.0) > threshold
    shift = shift or (agentsem.get("max_changepoint_shift") or 0.0) > threshold
    exposure_p = agentsem.get("exposure_p")
    exposure = (1.0 if exposure_p is None else float(exposure_p)) < float(
        rules["R4"]["exposure_p_max"])
    parts = []
    if "R2" in triggered:
        parts.append("R2")
    if "R4" in triggered:
        parts.append("R4 shift" if shift else ("R4 exposure only" if exposure else "other"))
    if "R5" in triggered:
        parts.append("R5")
    if "R6" in triggered:
        parts.append("R6")
    if triggered - {"R2", "R4", "R5", "R6"}:
        parts.append("other")
    if len(parts) == 1:
        return parts[0]
    return " + ".join(parts) if parts else "other"


def rule_mix(res: Results, cell_id: str) -> tuple[dict, int]:
    """Rates of each triggered-rule category for the agentSEM arm of one cell."""
    records = res.cell(cell_id).get("records", [])
    counts: dict[str, int] = {}
    n = 0
    for rec in records:
        arm = rec.get("analyzers", {}).get("agentsem")
        if arm is None:
            continue
        n += 1
        key = rule_category(arm, rec.get("analyzers", {}).get("anchor_only"), res.rules)
        counts[key] = counts.get(key, 0) + 1
    return ({k: v / n for k, v in counts.items()} if n else {}), n


def _mix_order(mix: dict) -> list[str]:
    known = [k for k in RULE_ORDER if k in mix]
    rest = sorted(k for k in mix if k not in RULE_ORDER)
    return known + rest


def _mix_colour(key: str) -> str:
    if key in RULE_COLOUR:
        return RULE_COLOUR[key]
    first = key.split(" + ")[0]
    return RULE_COLOUR.get(first, MUTED)


def _segment_text(key: str, share: float) -> str:
    """Name plus share when the segment is wide enough to hold both, else the share."""
    if share >= 0.30:
        return f"{key}\n{share * 100:.0f}%"
    if share >= 0.11:
        return f"{share * 100:.0f}%"
    return ""


def _mix_legend(ax, keys, **kw):
    handles = [Patch(facecolor=_mix_colour(k), edgecolor="white",
                     hatch=_mix_hatch(k), label=k) for k in keys]
    ax.legend(handles=handles, **kw)


def _mix_hatch(key: str) -> str:
    if key in RULE_HATCH:
        return RULE_HATCH[key]
    return "xx"


# ------------------------------------------------------------------------------ P1 ----

PSI_VALUES = [0.0, 0.5, 1.0]
PSI_STYLE = {
    0.0: (OI["green"], "^", "--", "benign (ψ = 0)"),
    0.5: (OI["blue"], "o", "-", "ψ = 0.5"),
    1.0: (OI["vermillion"], "s", "-", "ψ = 1.0"),
}
KAPPAS = [0.0, 0.5, 1.0, 2.0]
GAMMAS = [0.0, 1.0, 2.0]


def _a_cell(kappa, psi, gamma) -> str:
    return f"A_k{float(kappa)}_p{float(psi)}_g{float(gamma)}"


def p1_headline(res: Results, out_dir):
    rows = [
        ("naive reliability ω and person correlation\n(unitless, 0 to 1)", "quality"),
        ("naive pass rate at τ = 0.30\n(share of replicates, pre-registered)", "pass"),
        ("naive false-confidence rate at τ = 0.30\n(share of replicates)", "fconf"),
        ("naive Var(θ) bias\n(Var(θ) units; truth = 1.00)", "bias"),
    ]
    fig, axes = plt.subplots(len(rows), len(GAMMAS), figsize=(13.2, 12.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 0.7, 1.0, 1.2]})
    fig.suptitle("P1  Convergent validity against the policy-downstream criterion "
                 "(γ_C shown per column)",
                 x=0.012, y=0.998, ha="left", va="top", fontsize=13, weight="bold",
                 color=INK)
    fig.text(0.012, 0.975,
             "Every audit number the naive analyst can see stays at its ceiling while the "
             "quantity it certifies goes wrong.  Series differ in colour, marker and line "
             "style and are labelled directly.  dW under each κ tick is agent quality in "
             "W units against the κ = 0 counterfactual.",
             ha="left", fontsize=8.5, color=SECOND_INK)

    reps = set()
    for col, gamma in enumerate(GAMMAS):
        dw_by_kappa = []
        for kappa in KAPPAS:
            values = [res.val(_a_cell(kappa, psi, gamma), "naive", "dW_mean")
                      for psi in PSI_VALUES]
            values = [v for v in values if v is not None]
            dw_by_kappa.append(sum(values) / len(values) if values else None)

        for row, (ylabel, kind) in enumerate(rows):
            ax = axes[row][col]
            for psi in PSI_VALUES:
                colour, marker, dash, name = PSI_STYLE[psi]
                cells = [_a_cell(k, psi, gamma) for k in KAPPAS]
                for c in cells:
                    reps.add(res.reps(c))
                if kind == "quality":
                    omega = [res.val(c, "naive", "omega_mean") for c in cells]
                    omega_se = [res.val(c, "naive", "omega_mc_se") for c in cells]
                    corr = [res.val(c, "naive", "corr_mean") for c in cells]
                    corr_se = [res.val(c, "naive", "corr_mc_se") for c in cells]
                    _series_line(ax, KAPPAS, omega, omega_se, colour, marker, dash, name)
                    _series_line(ax, KAPPAS, corr, corr_se, colour, marker, dash, None,
                                 fill=False)
                    if psi == 1.0:
                        _label(ax, KAPPAS[1], 1.0015, "reliability ω (filled marker)",
                               SECOND_INK, dx=0, ha="center", size=7.6)
                        _label(ax, KAPPAS[1], 0.9735,
                               "person correlation (open marker)", SECOND_INK, dx=0,
                               ha="center", size=7.6)
                    ax.set_ylim(0.9555, 1.0055)
                elif kind == "pass":
                    y = [res.val(c, "naive", "pass_0.3_rate") for c in cells]
                    se = [res.val(c, "naive", "pass_0.3_mc_se") for c in cells]
                    _series_line(ax, KAPPAS, y, se, colour, marker, dash, name)
                    ax.set_ylim(0.0, 1.06)
                elif kind == "fconf":
                    y = [res.val(c, "naive", "false_conf_0.3_rate") for c in cells]
                    se = [res.val(c, "naive", "false_conf_0.3_mc_se") for c in cells]
                    ys = [res.val(c, "naive", "false_conf_0.3_sample_rate") for c in cells]
                    _series_line(ax, KAPPAS, y, se, colour, marker, dash, name)
                    _series_line(ax, KAPPAS, ys, None, colour, marker, (0, (1, 1)), None,
                                 band=False, fill=False)
                    ax.set_ylim(-0.04, 1.10)
                else:
                    y = [res.val(c, "naive", "bias_mean") for c in cells]
                    se = [res.val(c, "naive", "bias_mc_se") for c in cells]
                    _series_line(ax, KAPPAS, y, se, colour, marker, dash, name)
                    ax.set_ylim(-0.25, 3.6)

            # matplotlib hides any label starting with "_", so the twin's legend
            # entry is built as an explicit handle instead.
            sample_handle = Line2D([], [], color=MUTED, linestyle=(0, (1, 1)),
                                   marker="o", markerfacecolor="white",
                                   markeredgecolor=MUTED,
                                   label=f"`_sample` twin, {EXPLORATORY}")
            if kind == "bias":
                ax.axhspan(-TOLERANCE, TOLERANCE, color=BAND, zorder=1)
                _label(ax, 1.55, TOLERANCE, "tolerance ± 0.10", MUTED, dx=0,
                       dy=8, ha="center", size=7.2)
                ax.axhline(0.0, color=MUTED, linewidth=0.8, zorder=1)

            _style(ax, ylabel=ylabel if col == 0 else "",
                   title=f"γ_C = {gamma:g}" if row == 0 else "")
            ax.set_xticks(KAPPAS)
            if row == len(rows) - 1:
                ax.set_xticklabels(
                    [f"{k:g}\ndW {_num(dw, 2)}" for k, dw in zip(KAPPAS, dw_by_kappa)])
                if col == 1:
                    ax.set_xlabel("policy strength κ (unitless), with dW beneath it",
                                  color=SECOND_INK)
            if row == 0 and col == 0:
                ax.legend(loc="lower right", ncol=1)
            if kind == "fconf" and col == 2:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles=handles + [sample_handle],
                          labels=labels + [sample_handle.get_label()],
                          loc="center", bbox_to_anchor=(0.54, 0.63), ncol=1,
                          fontsize=7.4)

    r = sorted(x for x in reps if x)
    _footnote(fig, "Source: results/summary.json, naive analyzer, sweep A only "
                   f"(sweep J is never shown).  R = {min(r)} to {max(r)} replicates per "
                   "cell.  Solid = ψ 0.5 and 1.0, dashed = benign ψ = 0.  "
                   "Dotted lines in row 3 are the _sample twin of the pre-registered "
                   f"flag: {EXPLORATORY}.  Mean-bias rows use the pre-registered "
                   "population-truth `bias`.  Bands are ± 1 Monte Carlo SE.")
    fig.subplots_adjust(top=0.935, hspace=0.30)
    return _save(fig, out_dir, "P1_headline")


# ------------------------------------------------------------------------------ P2 ----

def p2_f3_hump(res: Results, out_dir):
    f3 = res.closed["F3"]
    u = f3["u_grid"]
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    _style(ax, xlabel="adaptivity u  (shadow variance share, unitless)",
           ylabel="ratio  corr(η, O) / corr(θ, O)   (unitless)",
           title="P2  Criterion contamination: the ratio is hump-shaped in u and its "
                 "sign depends on r", grid_axis="both")
    ax.set_xscale("log")
    ax.axhline(1.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    _label(ax, u[-1], 1.0, "ratio = 1 (no inflation)", MUTED, dx=-4, dy=-12, ha="right",
           size=7.5)

    for i, (r, cf, mc) in enumerate(zip(f3["r_grid"], f3["closed_form"]["ratio"],
                                        f3["monte_carlo"]["ratio"])):
        colour, marker, dash = SLOTS[i], MARKERS[i], DASHES[i]
        ax.plot(u, cf, color=colour, linestyle=dash, zorder=4)
        ax.plot(u, mc, color=colour, linestyle="none", marker=marker,
                markerfacecolor="white", markeredgecolor=colour, markeredgewidth=1.5,
                zorder=5)
        _label(ax, u[-1], cf[-1], f"r = {r:g}", colour, dx=6, size=8.5, weight="bold")

    marks = [(kappa, _a_cell(kappa, 0.5, 1.0)) for kappa in (0.5, 1.0, 2.0)]
    xs, ys, reps = [], [], []
    stagger = [(14, -46), (14, -100), (14, 34)]
    for i, (kappa, cell_id) in enumerate(marks):
        records = res.cell(cell_id).get("records", [])
        u_eff = [rec["u_eff"] for rec in records if rec.get("u_eff") is not None]
        if not u_eff:
            continue
        x = sum(u_eff) / len(u_eff)
        y = res.val(cell_id, "naive", "ratio_react_mean")
        r_hat = res.val(cell_id, "naive", "implied_r_mean")
        xs.append(x)
        ys.append(y)
        reps.append(res.reps(cell_id))
        ax.errorbar(x, y, yerr=res.val(cell_id, "naive", "ratio_react_mc_se", 0.0),
                    fmt="*", markersize=15, color=INK, markerfacecolor=INK, capsize=3,
                    zorder=7)
        dx, dy = stagger[i]
        ax.annotate(f"{cell_id}\nκ {kappa:g}, u_eff {x:.3f}, implied r {_num(r_hat, 2)},"
                    f"\nratio_react {_num(y, 3)}",
                    (x, y), textcoords="offset points", xytext=(dx, dy), ha="left",
                    va="center", fontsize=7.2, color=INK, zorder=8,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})

    handles = [
        Line2D([], [], color=MUTED, linestyle="-", label="closed form (F3)"),
        Line2D([], [], color=MUTED, linestyle="none", marker="o", markerfacecolor="white",
               markeredgecolor=MUTED, label="Monte Carlo check (F3)"),
        Line2D([], [], color=INK, linestyle="none", marker="*", markersize=13,
               label="observed naive ratio_react in the grid"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_xlim(0.085, u[-1] * 1.9)
    ax.set_xticks([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ax.set_xticklabels(["0.125", "0.25", "0.5", "1", "2", "4", "8"])
    ax.minorticks_off()
    _footnote(fig, "Source: results/closed_forms.json F3 (n_draws = "
                   f"{f3['params']['n_draws']:,}) and results/summary.json naive arm of "
                   f"cells {', '.join(c for _, c in marks)} "
                   f"(R = {', '.join(str(x) for x in reps)} replicates); u_eff is the "
                   "per-replicate mean from results/grid/<cell_id>.json.  "
                   f"F3 verdict as stored: \"{f3['verdict']}\".")
    return _save(fig, out_dir, "P2_f3_hump")


# ------------------------------------------------------------------------------ P3 ----

def p3_competitors(res: Results, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 9.0), sharex=True)
    fig.suptitle("P3  Var(θ) bias by analyzer, one panel per world "
                 "(mean ± 1 Monte Carlo SE)",
                 x=0.012, ha="left", fontsize=13, weight="bold", color=INK)
    fig.text(0.012, 0.952,
             "Analyzer identity is the axis label, never the colour.  Filled = the "
             "certified Var(θ) estimate; hatched = bias_B_raw, the shadow model's own "
             "estimate before the certificate acted on it (all replicates).",
             ha="left", fontsize=8.5, color=SECOND_INK)

    reps = {}
    y = list(range(len(P3_ANALYZERS)))[::-1]
    for i, cell_id in enumerate(P3_CELLS):
        ax = axes[i // 4][i % 4]
        reps[cell_id] = res.reps(cell_id)
        ax.axvspan(-TOLERANCE, TOLERANCE, color=BAND, zorder=1)
        ax.axvline(0.0, color=MUTED, linewidth=0.9, zorder=2)
        for pos, analyzer in zip(y, P3_ANALYZERS):
            n_cert = res.val(cell_id, analyzer, "bias_n", 0) or 0
            n_reps = res.reps(cell_id, analyzer) or res.reps(cell_id)
            raw = res.val(cell_id, analyzer, "bias_B_raw_mean")
            has_raw = analyzer in ("agentsem", "misspecified") and raw is not None
            offset = 0.19 if has_raw else 0.0
            if n_cert and n_cert >= 0.5 * max(n_reps, 1):
                ax.barh(pos + offset, res.val(cell_id, analyzer, "bias_mean", 0.0),
                        height=0.36 if has_raw else 0.62, color=OI["blue"],
                        edgecolor="white", linewidth=1.5, zorder=4)
                ax.errorbar(res.val(cell_id, analyzer, "bias_mean", 0.0), pos + offset,
                            xerr=res.val(cell_id, analyzer, "bias_mc_se", 0.0),
                            fmt="none", ecolor=INK, capsize=2.5, elinewidth=1.1, zorder=6)
            else:
                refusal = res.val(cell_id, analyzer, "status_B_rates_unsupported")
                if refusal is None and n_reps:
                    refusal = 1.0 - n_cert / n_reps
                ax.plot([0.0], [pos + offset], marker="o", markersize=9,
                        markerfacecolor="white", markeredgecolor=OI["blue"],
                        markeredgewidth=1.8, zorder=6)
                ax.annotate(f"refused {_pct(refusal)}", (-1.30, pos + offset),
                            textcoords="offset points", xytext=(0, 0), ha="left",
                            va="center", fontsize=7.2, color=OI["blue"], zorder=6)
            if has_raw:
                ax.barh(pos - 0.19, raw, height=0.36, color="white",
                        edgecolor=OI["vermillion"], linewidth=1.4, hatch="///", zorder=4)
                ax.errorbar(raw, pos - 0.19,
                            xerr=res.val(cell_id, analyzer, "bias_B_raw_mc_se", 0.0),
                            fmt="none", ecolor=OI["vermillion"], capsize=2.5,
                            elinewidth=1.1, zorder=6)
        ax.set_yticks(y)
        ax.set_yticklabels(P3_ANALYZERS if i % 4 == 0 else [])
        ax.set_xlim(-1.35, 1.45)
        label = "base" if cell_id == BASE_CELL else "sweep B"
        _style(ax, title=f"{cell_id}\n{label}, R = {reps[cell_id]}",
               xlabel="Var(θ) bias (Var(θ) units)" if i // 4 == 1 else "",
               grid_axis="x")
        ax.title.set_fontsize(9)

    legend_ax = axes[1][3]
    legend_ax.axis("off")
    handles = [
        Patch(facecolor=OI["blue"], edgecolor="white",
              label="certified Var(θ) estimate (`bias`)"),
        Patch(facecolor="white", edgecolor=OI["vermillion"], hatch="///",
              label="bias_B_raw: shadow model's own estimate,\nall replicates "
                    "(agentsem and misspecified)"),
        Line2D([], [], color=OI["blue"], marker="o", linestyle="none",
               markerfacecolor="white", markeredgewidth=1.8,
               label="refused (unsupported): no certified estimate\nin most replicates"),
        Patch(facecolor=BAND, edgecolor="none", label="tolerance ± 0.10"),
    ]
    legend_ax.legend(handles=handles, loc="upper left", labelspacing=1.1, fontsize=8)
    legend_ax.text(0.0, 0.14,
                   "No `between=False` agentSEM arm exists in these results,\n"
                   "so none is drawn (task 7b: do not compute it here).",
                   fontsize=7.5, color=MUTED, transform=legend_ax.transAxes)

    _footnote(fig, "Source: results/summary.json, one row per cell × analyzer.  "
                   "R per cell: "
                   + ", ".join(f"{c} {n}" for c, n in reps.items())
                   + ".  Mean bias is the pre-registered population-truth `bias`; "
                     "`bias_B_raw` and its _sample twin are derived fields "
                     f"({EXPLORATORY}) and only the population-truth one is drawn.")
    fig.subplots_adjust(top=0.885, wspace=0.10, hspace=0.42)
    return _save(fig, out_dir, "P3_competitors")


# ------------------------------------------------------------------------------ P4 ----

def p4_certificate(res: Results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.4),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    fig.suptitle("P4  What the certificate did, and what it cost",
                 x=0.012, ha="left", fontsize=13, weight="bold", color=INK)

    ax = axes[0]
    mixes, ns = {}, {}
    for cell_id in P4_CELLS:
        mixes[cell_id], ns[cell_id] = rule_mix(res, cell_id)
    keys = []
    for mix in mixes.values():
        for k in _mix_order(mix):
            if k not in keys:
                keys.append(k)
    keys = [k for k in RULE_ORDER if k in keys] + [k for k in keys if k not in RULE_ORDER]

    positions = list(range(len(P4_CELLS)))[::-1]
    for pos, cell_id in zip(positions, P4_CELLS):
        left = 0.0
        for key in keys:
            share = mixes[cell_id].get(key, 0.0)
            if share <= 0:
                continue
            ax.barh(pos, share, left=left, height=0.62, color=_mix_colour(key),
                    edgecolor="white", linewidth=1.6, hatch=_mix_hatch(key), zorder=4)
            text = _segment_text(key, share)
            if text:
                ax.text(left + share / 2, pos, text, ha="center", va="center",
                        fontsize=7.4, color=INK, zorder=6)
            left += share
        unsupported = res.val(cell_id, "agentsem", "status_B_rates_unsupported", 0.0)
        _label(ax, 1.0, pos, f"B unsupported {_pct(unsupported)}", SECOND_INK,
               dx=8, size=7.4)
    _mix_legend(ax, keys, loc="lower center", bbox_to_anchor=(0.42, -0.155), ncol=3,
                fontsize=7.8)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{c}\n(R = {ns[c]})" for c in P4_CELLS])
    ax.set_xlim(0.0, 1.30)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _style(ax, xlabel="share of replicates",
           title="Triggered rule set per replicate (R4 split into its two tests)",
           grid_axis="x")
    absent = [k for k in ("R2", "R5", "R6") if k not in keys]
    ax.text(0.0, -0.245,
            "R4's two tests share one rule id in `triggered_B` and are told apart by the "
            "stored numbers: exposure_p < "
            f"{res.rules['R4']['exposure_p_max']} versus the pooled epoch and "
            f"change-point shift > {res.rules['R4']['shift_sd_max']} anchor SD.\n"
            "\"R4 shift\" is any replicate whose pooled test fired; \"R4 exposure only\" "
            "is the exposure test firing on its own."
            + (f"  Never fired in these cells: {', '.join(absent)}." if absent else ""),
            transform=ax.transAxes, fontsize=7.4, color=MUTED, va="top")

    ax = axes[1]
    #: Cells whose claim B is never certified have no false-confidence rate at all
    #: (the denominator is zero), so they sit in their own "undefined" strip rather
    #: than at zero, which would read as a measured value.
    undefined_x = -0.075
    points, undefined = [], []
    for i, cell_id in enumerate(P4_CELLS):
        marker = MARKERS[i % len(MARKERS)]
        for key, colour, fill, tag in (
                ("", OI["blue"], True, "pre-registered"),
                ("_sample", OI["vermillion"], False, "_sample")):
            fc = res.val(cell_id, "agentsem", f"false_confidence{key}_rate")
            fr = res.val(cell_id, "agentsem", f"false_refusal{key}_rate")
            if fr is None:
                continue
            fr_se = res.val(cell_id, "agentsem", f"false_refusal{key}_mc_se", 0.0)
            if fc is None:
                undefined.append((undefined_x, fr, fr_se, colour, marker, fill, cell_id,
                                  tag))
            else:
                points.append((fc, fr, res.val(cell_id, "agentsem",
                                               f"false_confidence{key}_mc_se", 0.0),
                               fr_se, colour, marker, fill, cell_id, tag))

    for x, y, yerr, colour, marker, fill, cell_id, tag in undefined:
        ax.errorbar(x, y, yerr=yerr, fmt=marker, markersize=8, color=colour,
                    markerfacecolor=colour if fill else "white", markeredgewidth=1.6,
                    capsize=3, elinewidth=1.1, zorder=5)
    for x, y, xerr, yerr, colour, marker, fill, cell_id, tag in points:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=marker, markersize=8, color=colour,
                    markerfacecolor=colour if fill else "white", markeredgewidth=1.6,
                    capsize=3, elinewidth=1.1, zorder=5)

    #: the certified cells sit almost on top of each other, so their labels are fanned
    #: into a column to the right with leader lines rather than nudged.
    cluster = sorted([(e[0], e[1], e[-2]) for e in points if e[-1] == "pre-registered"],
                     key=lambda t: t[1])
    if cluster:
        x_max = max(e[0] for e in points)
        label_x = x_max * 1.10
        span = 0.055 * max(len(cluster) - 1, 1)
        base = max(e[1] for e in cluster) + 0.03
        for j, (x, y, cell_id) in enumerate(cluster):
            ax.annotate(cell_id, (x, y), xytext=(label_x, base + span - j * 0.055),
                        textcoords="data", ha="left", va="center", fontsize=7.4,
                        color=INK, zorder=7,
                        arrowprops={"arrowstyle": "-", "color": MUTED,
                                    "linewidth": 0.7, "shrinkA": 1, "shrinkB": 4})
    for j, entry in enumerate(undefined):
        x, y, _, _, _, _, cell_id, tag = entry
        if tag != "pre-registered":
            continue
        ax.annotate(cell_id, (x, y), textcoords="offset points", xytext=(9, 0),
                    ha="left", va="center", fontsize=7.4, color=INK, zorder=7)
    #: join each cell's pre-registered point to its _sample twin
    for cell_id in P4_CELLS:
        pair = [(e[0], e[1], e[-1]) for e in points if e[-2] == cell_id]
        pair += [(e[0], e[1], e[-1]) for e in undefined if e[-2] == cell_id]
        if len(pair) == 2:
            ax.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], color=MUTED,
                    linewidth=0.8, linestyle=(0, (1, 1)), zorder=3)

    ax.axvline(-0.028, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)
    all_fc = [e[0] for e in points]
    all_fr = [e[1] for e in points] + [e[1] for e in undefined]
    ax.set_xlim(undefined_x - 0.045, max(all_fc + [0.4]) * 1.55)
    ax.set_ylim(-0.06, max(all_fr + [0.4]) * 1.14)
    ticks = [t for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
             if t <= max(all_fc + [0.4]) * 1.1]
    ax.set_xticks([undefined_x] + ticks)
    ax.set_xticklabels(["undefined\n(never\ncertified)"] + [f"{t:g}" for t in ticks])
    handles = [
        Line2D([], [], color=OI["blue"], marker="o", linestyle="none",
               label="pre-registered flags"),
        Line2D([], [], color=OI["vermillion"], marker="o", linestyle="none",
               markerfacecolor="white", markeredgewidth=1.6,
               label=f"_sample twin, {EXPLORATORY}"),
    ]
    ax.legend(handles=handles, loc="upper left")
    _style(ax, xlabel="false-confidence rate (share of certified replicates)",
           ylabel="false-refusal rate (share of replicates)",
           title="Cost of the certificate: false confidence against false refusal",
           grid_axis="both")

    _footnote(fig, "Source: results/grid/<cell_id>.json per-replicate `triggered_B`, "
                   "`exposure_p`, `epoch_shift`, `max_changepoint_shift` and the "
                   "`anchor_only` Var(θ) (left); results/summary.json agentsem arm "
                   "(right).  Thresholds from trapworld/rules.json.  R per cell: "
                   + ", ".join(f"{c} {ns[c]}" for c in P4_CELLS)
                   + f".  Fields ending in _sample are {EXPLORATORY}.")
    fig.subplots_adjust(top=0.90, bottom=0.30, wspace=0.30)
    return _save(fig, out_dir, "P4_certificate")


# ------------------------------------------------------------------------------ P5 ----

F_PSI = [0.0, 0.05, 0.1, 0.2, 0.3]
F_KAPPA = [0.5, 1.0, 2.0]


def _f_cell(psi, kappa) -> str:
    return f"F_p{float(psi)}_k{float(kappa)}"


def p5_specificity(res: Results, out_dir):
    fig, ax = plt.subplots(figsize=(10.2, 6.8))
    _style(ax, xlabel="true shadow loading ψ (unitless)",
           ylabel="share of replicates", grid_axis="both",
           title="P5  Specificity in sweep F: detection rises with ψ, and the benign "
                 "arm's innocence verdict depends on κ")
    ax.axhline(0.90, color=MUTED, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
    _label(ax, 0.028, 0.90, "0.90: the innocence bar", MUTED, dx=0, dy=8, ha="center",
           size=7.5)

    lines, reps = [], []
    for i, kappa in enumerate(F_KAPPA):
        colour, marker, dash = SLOTS[i], MARKERS[i], DASHES[i]
        cells = [_f_cell(psi, kappa) for psi in F_PSI]
        reps += [res.reps(c, "agentsem") for c in cells]
        detects = [res.val(c, "agentsem", "detects_rate") for c in cells]
        se = [res.val(c, "agentsem", "detects_mc_se") for c in cells]
        _series_line(ax, F_PSI, detects, se, colour, marker, dash,
                     f"detection rate, κ = {kappa:g}")
        anchor, dx, dy, ha = {0: (3, 8, -26, "left"), 1: (2, 10, -20, "left"),
                              2: (1, -10, -24, "right")}[i]
        _label(ax, F_PSI[anchor], detects[anchor], f"detection, κ = {kappa:g}", colour,
               dx=dx, dy=dy, ha=ha, weight="bold", size=8.5)
        innocent = res.val(_f_cell(0.0, kappa), "agentsem", "innocent_rate")
        innocent_se = res.val(_f_cell(0.0, kappa), "agentsem", "innocent_mc_se", 0.0)
        ax.errorbar(0.0, innocent, yerr=innocent_se, fmt=marker, markersize=10,
                    color=colour, markerfacecolor="white", markeredgewidth=2.0,
                    capsize=3, zorder=6)
        lines.append(f"κ = {kappa:g}:  innocent rate at ψ = 0 is "
                     f"{_pct(innocent)} ± {100 * innocent_se:.1f} pts")

    ax.text(0.176, 0.70, "innocent rate at ψ = 0 (open markers at ψ = 0)\n"
            + "\n".join(lines), fontsize=8.4, color=INK, va="top",
            bbox={"facecolor": "white", "edgecolor": GRIDLINE, "boxstyle": "round,pad=0.5"})
    handles = [
        Line2D([], [], color=MUTED, marker="o", label="detection rate (line + filled)"),
        Line2D([], [], color=MUTED, marker="o", linestyle="none", markerfacecolor="white",
               markeredgewidth=2.0, label="innocent rate at ψ = 0 (open)"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_xticks(F_PSI)
    ax.set_xlim(-0.012, 0.345)
    ax.set_ylim(-0.05, 1.08)
    span = (f"R = {min(reps)}" if min(reps) == max(reps)
            else f"R = {min(reps)} to {max(reps)}")
    _footnote(fig, "Source: results/summary.json, agentsem arm of the 15 sweep-F cells, "
                   f"{span} replicates each.  Bands are ± 1 "
                   "Monte Carlo SE.  Both rates are pre-registered (spec 7); no _sample "
                   "twin is defined for them.")
    return _save(fig, out_dir, "P5_specificity")


# ------------------------------------------------------------------------------ P6 ----

def p6_zero_person(res: Results, out_dir):
    cells = [BASE_CELL, "G_zero_person"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    fig.suptitle("P6  A world with no person in it: G_zero_person against the base cell",
                 x=0.012, ha="left", fontsize=13, weight="bold", color=INK)

    ax = axes[0]
    quantities = [("reliability ω", "omega_mean", "omega_mc_se"),
                  ("reactive correlation\nr_react", "r_react_mean", "r_react_mc_se"),
                  ("pass rate\nat τ = 0.30", "pass_0.3_rate", "pass_0.3_mc_se")]
    width = 0.34
    for j, cell_id in enumerate(cells):
        colour, marker = SLOTS[j], MARKERS[j]
        xs = [i + (j - 0.5) * width for i in range(len(quantities))]
        for x, (_, mean_key, se_key) in zip(xs, quantities):
            value = res.val(cell_id, "naive", mean_key, 0.0)
            ax.bar(x, value, width=width, color=colour, edgecolor="white", linewidth=1.6,
                   zorder=4)
            ax.errorbar(x, value, yerr=res.val(cell_id, "naive", se_key, 0.0), fmt="none",
                        ecolor=INK, capsize=3, elinewidth=1.1, zorder=6)
            ax.text(x, value + 0.028, f"{value:.3f}", ha="center", fontsize=7.8,
                    color=INK, zorder=6)
    ax.legend(handles=[Patch(facecolor=SLOTS[j], edgecolor="white",
                             label=f"{c} (R = {res.reps(c)})")
                       for j, c in enumerate(cells)], loc="upper right")
    ax.set_xticks(range(len(quantities)))
    ax.set_xticklabels([q[0] for q in quantities], fontsize=8)
    ax.set_ylim(0.0, 1.18)
    _style(ax, ylabel="value (unitless; rates are shares of replicates)",
           title="What the naive analyst sees")

    ax = axes[1]
    positions = [1, 0]
    drawn = []
    for pos, cell_id in zip(positions, cells):
        mix, n = rule_mix(res, cell_id)
        left = 0.0
        for key in _mix_order(mix):
            share = mix[key]
            if key not in drawn:
                drawn.append(key)
            ax.barh(pos, share, left=left, height=0.5, color=_mix_colour(key),
                    edgecolor="white", linewidth=1.6, hatch=_mix_hatch(key), zorder=4)
            text = _segment_text(key, share)
            if text:
                ax.text(left + share / 2, pos, text, ha="center", va="center",
                        fontsize=7.2, color=INK, zorder=6)
            left += share
        status = res.row(cell_id, "agentsem")
        parts = []
        for name in ("identified", "bounded", "unsupported"):
            rate = status.get(f"status_B_rates_{name}")
            if rate:
                parts.append(f"{name} {_pct(rate)}")
        _label(ax, 1.0, pos, "claim B: " + (", ".join(parts) or "n/a"), SECOND_INK, dx=8,
               size=7.6)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{c}\n(R = {res.reps(c)})" for c in cells])
    ax.set_xlim(0.0, 1.46)
    ax.set_ylim(-0.75, 1.7)
    _mix_legend(ax, _mix_order({k: 1 for k in drawn}), loc="lower center",
                bbox_to_anchor=(0.40, -0.34), ncol=3, fontsize=7.4)
    _style(ax, xlabel="share of replicates",
           title="agentSEM claim-B status, by triggered rule set", grid_axis="x")

    _footnote(fig, "Source: results/summary.json naive and agentsem arms, and "
                   "results/grid/<cell_id>.json per-replicate `triggered_B` for the rule "
                   "split.  G_zero_person sets Var(θ) = 0, so `bias` is not a "
                   "ratio-scale quantity there and is not drawn; the panel shows the "
                   "audit numbers and the certificate's verdict only.")
    fig.subplots_adjust(top=0.90, bottom=0.26, wspace=0.28)
    return _save(fig, out_dir, "P6_zero_person")


# ------------------------------------------------------------------------------ P7 ----

def p7_f4_purity(res: Results, out_dir):
    f4 = res.closed["F4"]
    T = f4["T_grid"]
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    _style(ax, xlabel="occasions T averaged into the dose surrogate (count)",
           ylabel="share of variance (unitless, 0 to 1)", grid_axis="both",
           title="P7  Surrogate purity: R²(dose ~ θ) rises to ~0.98 by T = 56, "
                 "so partialling the dose deletes the person")
    _series_line(ax, T, f4["closed_form"]["R2"], None, OI["blue"], MARKERS[0], "-",
                 "closed-form R²(dose ~ θ)", band=False)
    ax.plot(T, f4["monte_carlo"]["R2"], linestyle="none", marker=MARKERS[1],
            markerfacecolor="white", markeredgecolor=OI["vermillion"],
            markeredgewidth=1.6, markersize=9, label="Monte Carlo R²", zorder=5)
    for t, value in zip(T, f4["closed_form"]["R2"]):
        _label(ax, t, value, f"{value:.3f}", OI["blue"], dy=9, ha="center", size=7.8)

    corr = res.val(BASE_CELL, "dose_partial", "corr_mean")
    corr_se = res.val(BASE_CELL, "dose_partial", "corr_mc_se", 0.0)
    ax.errorbar(T[-1], corr, yerr=corr_se, fmt=MARKERS[2], markersize=12, color=OI["green"],
                markerfacecolor=OI["green"], capsize=3, zorder=6,
                label="dose_partial person correlation, κ = 1")
    _label(ax, T[-1], corr, f"dose_partial person corr = {_num(corr, 3)}\n"
                            f"({BASE_CELL}, κ = 1, T = {T[-1]})\n"
                            "a correlation, not an R²", OI["green"], dx=-10, dy=-4,
           ha="right", size=7.8)
    ax.legend(loc="center right")
    ax.set_xticks(T)
    ax.set_ylim(0.0, 1.06)
    ax.set_xlim(-2, T[-1] + 12)
    _footnote(fig, "Source: results/closed_forms.json F4 (n_draws = "
                   f"{f4['params']['n_draws']:,}) and results/summary.json dose_partial "
                   f"arm of {BASE_CELL} "
                   f"(R = {res.reps(BASE_CELL, 'dose_partial')} replicates).  "
                   f"F4 verdict as stored: \"{f4['verdict']}\".")
    return _save(fig, out_dir, "P7_f4_purity")


# ------------------------------------------------------------------------------ P8 ----

def _coverage_pending(out_dir):
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.axis("off")
    ax.text(0.5, 0.62, "P8  Coverage audit: PENDING", ha="center", va="center",
            fontsize=20, weight="bold", color=OI["vermillion"])
    ax.text(0.5, 0.40,
            "results/coverage_audit.json was not present when these figures were built.\n"
            "The bootstrap coverage run writes it for the three audit cells\n"
            "(A_k1.0_p0.5_g1.0, B_G2, B_G3_r03).\n\n"
            "Re-running `python -m trapworld.plots` after that file lands draws the real\n"
            "panel with no change to this module.",
            ha="center", va="center", fontsize=10.5, color=INK)
    _footnote(fig, "Source: results/coverage_audit.json (absent).  This panel is a "
                   "placeholder and carries no result.")
    return _save(fig, out_dir, "P8_coverage")


def p8_coverage(res: Results, out_dir):
    audit = res.coverage_audit()
    if not audit or not audit.get("cells"):
        return _coverage_pending(out_dir), False

    cells = audit["cells"]
    names = list(cells.keys())
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    _style(ax, ylabel="coverage of the nominal 95% interval (share of replicates)",
           grid_axis="y",
           title="P8  Interval coverage per audit cell: Hessian delta against the person "
                 "bootstrap")
    ax.axhline(0.95, color=MUTED, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
    _label(ax, len(names) - 0.5, 0.95, "0.95 nominal", MUTED, dx=6, size=8)

    methods = [("hessian_coverage", "Hessian delta", OI["blue"], MARKERS[0], ""),
               ("bootstrap_coverage", "person bootstrap", OI["vermillion"], MARKERS[1],
                "///")]
    width = 0.34
    reported = []
    for j, (key, label, colour, marker, hatch) in enumerate(methods):
        for i, name in enumerate(names):
            entry = cells[name].get(key) or {}
            rate = entry.get("rate")
            if rate is None:
                ax.text(i + (j - 0.5) * width, 0.02, "not run", rotation=90, ha="center",
                        va="bottom", fontsize=8, color=MUTED)
                continue
            se = entry.get("mc_se") or 0.0
            ax.bar(i + (j - 0.5) * width, rate, width=width, color=colour,
                   edgecolor="white", linewidth=1.6, hatch=hatch, zorder=4)
            ax.errorbar(i + (j - 0.5) * width, rate, yerr=se, fmt="none", ecolor=INK,
                        capsize=3, elinewidth=1.1, zorder=6)
            ax.text(i + (j - 0.5) * width, rate + 0.02,
                    f"{rate:.3f}\n(n = {entry.get('n')})", ha="center", fontsize=7.6,
                    color=INK, zorder=6)
            reported.append(f"{name} {label} {rate:.3f}")
    ax.legend(handles=[Patch(facecolor=colour, edgecolor="white", hatch=hatch,
                             label=label)
                       for _, label, colour, _, hatch in methods],
              loc="upper right")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n(R = {cells[n].get('n_reps')})" for n in names])
    ax.set_ylim(0.0, 1.14)
    _footnote(fig, "Source: results/coverage_audit.json, B = "
                   f"{audit.get('bootstrap')} bootstrap resamples, agentsem arm.  "
                   "Bars are the stored coverage rates; error bars are ± 1 Monte "
                   "Carlo SE as stored.")
    return _save(fig, out_dir, "P8_coverage"), True


# ------------------------------------------------------------------------------ P9 ----

def p9_closed_forms(res: Results, out_dir):
    f1, f2 = res.closed["F1"], res.closed["F2"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6))
    fig.suptitle("P9  The two closed forms behind the trap "
                 "(closed form as lines, Monte Carlo as open markers)",
                 x=0.012, ha="left", fontsize=13, weight="bold", color=INK)

    ax = axes[0]
    _style(ax, xlabel="adaptivity u (unitless)", ylabel="corr(η, θ) (unitless)",
           title="F1  validity falls as u rises", grid_axis="both")
    _series_line(ax, f1["u_grid"], f1["closed_form"]["corr_eta_theta"], None, OI["blue"],
                 MARKERS[0], "-", "closed form", band=False)
    ax.plot(f1["u_grid"], f1["monte_carlo"]["corr_eta_theta"], linestyle="none",
            marker=MARKERS[1], markerfacecolor="white", markeredgecolor=OI["vermillion"],
            markeredgewidth=1.6, markersize=9, label="Monte Carlo", zorder=5)
    for x, y in zip(f1["u_grid"], f1["closed_form"]["corr_eta_theta"]):
        _label(ax, x, y, f"{y:.3f}", OI["blue"], dx=8, dy=7, ha="left", size=7.4)
    ax.set_ylim(0.70, 1.055)
    ax.set_xlim(-0.35, 9.6)
    ax.legend(loc="lower left")

    ax = axes[1]
    _style(ax, xlabel="adaptivity u (unitless)",
           ylabel="Var(η) / Var(θ) (ratio, log scale)",
           title="F1  variance ratio rises without bound", grid_axis="both")
    ax.set_yscale("log")
    _series_line(ax, f1["u_grid"], f1["closed_form"]["var_ratio"], None, OI["green"],
                 MARKERS[2], "-.", "closed form", band=False)
    ax.plot(f1["u_grid"], f1["monte_carlo"]["var_ratio"], linestyle="none",
            marker=MARKERS[3], markerfacecolor="white", markeredgecolor=OI["purple"],
            markeredgewidth=1.6, markersize=9, label="Monte Carlo", zorder=5)
    for x, y in zip(f1["u_grid"], f1["closed_form"]["var_ratio"]):
        _label(ax, x, y, f"{y:g}", OI["green"], dx=-6, dy=7, ha="right", size=7.4)
    ax.set_xlim(-0.55, 9.2)
    ax.legend(loc="upper left")

    ax = axes[2]
    _style(ax, xlabel="adaptivity u (unitless)", ylabel="value (unitless, 0 to 1)",
           title="F2  reliability up, validity down", grid_axis="both")
    _series_line(ax, f2["u_grid"], f2["closed_form"]["omega"], None, OI["blue"],
                 MARKERS[0], "-", "coefficient ω (reliability)", band=False)
    _series_line(ax, f2["u_grid"], f2["closed_form"]["corr_eta_theta"], None,
                 OI["vermillion"], MARKERS[1], "--",
                 "corr(η, θ) (validity)", band=False)
    _series_line(ax, f2["u_grid"], f2["closed_form"]["corr_composite_theta"], None,
                 OI["green"], MARKERS[2], "-.",
                 "corr(composite, θ)", band=False)
    ax.plot(f2["u_grid"], f2["monte_carlo"]["omega"], linestyle="none", marker=MARKERS[0],
            markerfacecolor="white", markeredgecolor=OI["blue"], markeredgewidth=1.6,
            markersize=10, zorder=5)
    ax.plot(f2["u_grid"], f2["monte_carlo"]["corr_composite_theta"], linestyle="none",
            marker=MARKERS[2], markerfacecolor="white", markeredgecolor=OI["green"],
            markeredgewidth=1.6, markersize=10, zorder=5)
    ax.plot(f2["u_grid"], f2["monte_carlo"]["corr_eta_theta"], linestyle="none",
            marker=MARKERS[1], markerfacecolor="white",
            markeredgecolor=OI["vermillion"], markeredgewidth=1.6, markersize=10,
            zorder=5)
    for x, y in zip(f2["u_grid"], f2["closed_form"]["omega"]):
        _label(ax, x, y, f"{y:.3f}", OI["blue"], dx=8, dy=-6, ha="left", size=7.4)
    for x, y in zip(f2["u_grid"], f2["closed_form"]["corr_eta_theta"]):
        _label(ax, x, y, f"{y:.3f}", OI["vermillion"], dy=8, ha="center", size=7.4)
    ax.set_ylim(0.80, 1.035)
    ax.set_xlim(-0.06, 1.22)
    ax.legend(loc="lower right")

    _footnote(fig, "Source: results/closed_forms.json F1 and F2 "
                   f"(n_draws = {f1['params']['n_draws']:,}); no replicates of the grid "
                   "are involved.  Stored verdicts: F1 \""
                   f"{f1['verdict']}\", F2 \"{f2['verdict']}\".  Largest closed-form "
                   f"versus Monte Carlo deviation: F1 {f1['max_abs_dev_corr']:.2e}, "
                   f"F2 {f2['max_abs_dev']:.2e}.")
    fig.subplots_adjust(top=0.86)
    return _save(fig, out_dir, "P9_closed_forms")


# ----------------------------------------------------------------------------- P10 ----

def p10_jev_arm(res: Results, out_dir):
    cell = res.cell("J_fake")
    if not cell.get("records"):
        return None
    cells = [BASE_CELL, "J_fake"]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.0),
                             gridspec_kw={"width_ratios": [1.0, 1.2]})
    fig.suptitle("P10  Fake-transport plumbing check, not a Jev result",
                 x=0.012, ha="left", fontsize=13, weight="bold", color=INK)
    fig.text(0.012, 0.915,
             "Sweep J uses an external policy transport and is never pooled with sweeps A "
             "to G.  Nothing on this panel is evidence about Jev.",
             ha="left", fontsize=8.5, color=SECOND_INK)

    ax = axes[0]
    positions = list(range(len(cells)))[::-1]
    for pos, cell_id in zip(positions, cells):
        value = res.val(cell_id, "naive", "bias_mean", 0.0)
        ax.barh(pos, value, height=0.5, color=SLOTS[positions.index(pos)],
                edgecolor="white", linewidth=1.6, zorder=4)
        ax.errorbar(value, pos, xerr=res.val(cell_id, "naive", "bias_mc_se", 0.0),
                    fmt="none", ecolor=INK, capsize=3, elinewidth=1.1, zorder=6)
        _label(ax, value, pos, f"{value:.3f}", INK, dx=7, size=8.5)
    ax.axvspan(-TOLERANCE, TOLERANCE, color=BAND, zorder=1)
    ax.axvline(0.0, color=MUTED, linewidth=0.9, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{c}\n(R = {res.reps(c)})" for c in cells])
    ax.set_xlim(-0.2, 0.85)
    _style(ax, xlabel="naive Var(θ) bias (Var(θ) units; truth = 1.00)",
           title="Naive bias, J_fake beside the base cell", grid_axis="x")

    ax = axes[1]
    for pos, cell_id in zip(positions, cells):
        mix, n = rule_mix(res, cell_id)
        left = 0.0
        for key in _mix_order(mix):
            share = mix[key]
            ax.barh(pos, share, left=left, height=0.5, color=_mix_colour(key),
                    edgecolor="white", linewidth=1.6, hatch=_mix_hatch(key), zorder=4)
            text = _segment_text(key, share)
            if text:
                ax.text(left + share / 2, pos, text, ha="center", va="center",
                        fontsize=7.2, color=INK, zorder=6)
            left += share
        status = res.row(cell_id, "agentsem")
        parts = [f"{name} {_pct(status.get(f'status_B_rates_{name}'))}"
                 for name in ("identified", "bounded", "unsupported")
                 if status.get(f"status_B_rates_{name}")]
        _label(ax, 1.0, pos, "claim B: " + (", ".join(parts) or "n/a"), SECOND_INK, dx=8,
               size=7.6)
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{c}\n(R = {res.reps(c)})" for c in cells])
    ax.set_xlim(0.0, 1.50)
    ax.set_ylim(-0.6, 1.6)
    _style(ax, xlabel="share of replicates",
           title="agentSEM claim-B status, by triggered rule set", grid_axis="x")

    live = res.cell("J_live")
    if live.get("status") == "not_run":
        fig.text(0.5, 0.035,
                 f"J_live: not_run. {live.get('reason', 'no reason recorded')}",
                 ha="center", fontsize=10, color=OI["vermillion"], weight="bold")

    _footnote(fig, "Source: results/summary.json naive and agentsem arms, and "
                   "results/grid/J_fake.json per-replicate `triggered_B`.  "
                   f"R: {BASE_CELL} {res.reps(BASE_CELL)}, J_fake {res.reps('J_fake')}.")
    fig.subplots_adjust(top=0.86, bottom=0.16)
    return _save(fig, out_dir, "P10_jev_arm")


# ----------------------------------------------------------------------------- main ---

def main(results_dir="results", out_dir="plots") -> list[pathlib.Path]:
    """Build every figure from ``results_dir`` into ``out_dir``; return the files."""
    res = Results(pathlib.Path(results_dir))
    if res.summary is None:
        raise FileNotFoundError(f"{results_dir}/summary.json not found")
    out = pathlib.Path(out_dir)
    written: list[pathlib.Path] = []

    written += p1_headline(res, out)
    written += p2_f3_hump(res, out)
    written += p3_competitors(res, out)
    written += p4_certificate(res, out)
    written += p5_specificity(res, out)
    written += p6_zero_person(res, out)
    written += p7_f4_purity(res, out)
    written += p9_closed_forms(res, out)
    p10 = p10_jev_arm(res, out)
    if p10:
        written += p10
    # P8 last: its input is written by the separate bootstrap run.
    p8_files, p8_real = p8_coverage(res, out)
    written += p8_files
    print(f"P8 coverage: {'real figure drawn' if p8_real else 'PENDING placeholder'}",
          file=sys.stderr)
    for path in written:
        print(path, file=sys.stderr)
    return written


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["results"]))
