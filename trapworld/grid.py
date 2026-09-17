"""The frozen cell list: spec section 6 plus amendment 11 A3 (sweep J).

Nothing here is computed from a result.  ``CELLS`` is built programmatically so the
sweeps read like the spec table, and it is hashed as one object into
``results/hashes.json`` before the grid runs, so the order of the list and the value of
every override are part of the frozen record.

A cell carries three bookkeeping keys that are not parameters (``cell_id``, ``sweep``,
``reps``) and sweep J carries a fourth (``jev``).  ``resolve`` raises on any key it does
not know, so ``cell_overrides`` strips those four before a cell reaches the generator.

Cell order matters twice over: it fixes ``GRID_HASH``, and the runner's seed tree uses a
cell's position in ``CELLS`` as its spawn index.  Appending a cell is therefore safe and
inserting one is not.

Floats in cell ids are written in Python's shortest round-trip form (``1.0`` stays
``"1.0"``, ``0.05`` stays ``"0.05"``), so an id can be read back as the number it names.
"""

from __future__ import annotations

from trapworld.params import PARAMS, canonical_hash

__all__ = ["CELLS", "GRID_HASH", "AUDIT", "HEADLINE_SWEEPS", "BASE", "CELL_META",
           "REPS_MAX", "cell_by_id", "cell_overrides", "first_of_each_sweep"]

#: Base cell B of spec section 6.  Every other key sits at its ``PARAMS`` default.
BASE = dict(kappa=1.0, psi=0.5, gamma_C=1.0)

#: Sweeps that may be mixed in a cross-cell aggregate.  Sweep J is a plumbing arm with a
#: different policy source and is reported on its own (spec 11 A3).
HEADLINE_SWEEPS = ("A", "B", "C", "D", "E", "F", "G")

#: Keys of a cell that are not parameters.
CELL_META = ("cell_id", "sweep", "reps", "jev")

REPS_BASE = 200
REPS_HEAVY = 500

#: Replicate seeds spawned per cell, whatever that cell's ``reps``.  Fixed so that
#: halving a cell's replicates (the runner's 3 s rule) cannot move the seed of any
#: replicate that still runs.
REPS_MAX = 500

#: Bootstrap coverage audit (spec section 6, F17): these three cells at 100 replicates
#: with B = 99 resamples.
AUDIT = ["A_k1.0_p0.5_g1.0", "B_G2", "B_G3_r03"]
AUDIT_REPS = 100
AUDIT_BOOTSTRAP = 99


def _f(value: float) -> str:
    """Shortest round-trip decimal form of a float, for cell ids."""
    return repr(float(value))


def _cell(cell_id: str, sweep: str, reps: int = REPS_BASE, **overrides) -> dict:
    return {"cell_id": cell_id, "sweep": sweep, "reps": reps, **overrides}


def _build() -> list[dict]:
    cells: list[dict] = []

    # A: the headline factorial.  kappa 0 is run as the "none" policy, which doses nobody.
    for kappa in (0.0, 0.5, 1.0, 2.0):
        for psi in (0.0, 0.5, 1.0):
            for gamma_C in (0.0, 1.0, 2.0):
                over = dict(kappa=kappa, psi=psi, gamma_C=gamma_C)
                if kappa == 0.0:
                    over["policy"] = "none"
                heavy = kappa == 1.0 and psi in (0.0, 0.5) and gamma_C in (0.0, 1.0)
                cells.append(_cell(f"A_k{_f(kappa)}_p{_f(psi)}_g{_f(gamma_C)}", "A",
                                   REPS_HEAVY if heavy else REPS_BASE, **over))

    # B: the shadow generator is not the one the surrogate assumes.
    cells += [
        _cell("B_G2", "B", generator="recency", **BASE),
        _cell("B_G3_r0", "B", generator="hetero", rho_psitheta=0.0, **BASE),
        _cell("B_G3_r03", "B", generator="hetero", rho_psitheta=0.3, **BASE),
        _cell("B_G3_r06", "B", generator="hetero", rho_psitheta=0.6, **BASE),
        _cell("B_G4", "B", generator="unlogged", **BASE),
        _cell("B_G4b", "B", generator="unlogged_plus", **BASE),
        # c0 = 0 leaves the unlogged step proportional to theta alone: the undetectable limit.
        _cell("B_G4c", "B", generator="unlogged", c0=0.0, **BASE),
    ]

    # C: the shadow does not load like the trait (R5).
    cells += [
        _cell("C_nonprop", "C", shadow_pattern="nonproportional", **BASE),
        _cell("C_nonprop_psi0", "C", shadow_pattern="nonproportional",
              **{**BASE, "psi": 0.0}),
    ]

    # D: other policies form other beliefs.
    cells += [
        _cell("D_kalman_k1", "D", policy="kalman", **BASE),
        _cell("D_kalman_k05", "D", policy="kalman", **{**BASE, "kappa": 0.5}),
        _cell("D_kalman_k2", "D", policy="kalman", **{**BASE, "kappa": 2.0}),
        _cell("D_heuristic", "D", policy="heuristic", **BASE),
    ]

    # E: the switch time is chosen by the stream (R2).
    cells += [_cell("E_perf", "E", trigger="performance", **BASE)]

    # F: the specificity ROC.
    for psi in (0.0, 0.05, 0.1, 0.2, 0.3):
        for kappa in (0.5, 1.0, 2.0):
            cells.append(_cell(f"F_p{_f(psi)}_k{_f(kappa)}", "F",
                               psi=psi, kappa=kappa, gamma_C=1.0))

    # G: the special worlds.
    cells += [
        _cell("G_uniform", "G", policy="uniform", **BASE),
        _cell("G_zero_person", "G", var_theta=0.0, **BASE),
        _cell("G_nodrift", "G", sd_d=0.0, **BASE),
        _cell("G_d2", "G", var_jitter=0.25, **BASE),
        _cell("G_feedback_p025", "G", policy_reads_stream=True,
              **{**BASE, "psi": 0.25}),
        _cell("G_feedback_p05", "G", policy_reads_stream=True, **BASE),
    ]

    # J: the external policy arm (spec 11 A3).  Never averaged with the rest.
    cells += [
        _cell("J_fake", "J", 50, policy="external", jev="fake", **BASE),
        _cell("J_live", "J", 5, policy="external", jev="live", **BASE),
    ]
    return cells


CELLS: list[dict] = _build()
CELL_INDEX = {cell["cell_id"]: i for i, cell in enumerate(CELLS)}
GRID_HASH = canonical_hash(CELLS)


def cell_by_id(cell_id: str) -> dict:
    """The cell with this id, or KeyError naming it."""
    if cell_id not in CELL_INDEX:
        raise KeyError(f"unknown cell_id: {cell_id!r}")
    return CELLS[CELL_INDEX[cell_id]]


def cell_overrides(cell: dict) -> dict:
    """The parameter overrides of a cell: everything but the bookkeeping keys."""
    return {k: v for k, v in cell.items() if k not in CELL_META}


def first_of_each_sweep() -> list[str]:
    """The first cell id of every sweep, in sweep order.  The smoke run's selection."""
    seen: dict[str, str] = {}
    for cell in CELLS:
        seen.setdefault(cell["sweep"], cell["cell_id"])
    return list(seen.values())


# A typo in an override would otherwise sit in the frozen hash unnoticed until the
# generator refused it mid-run, so the grid checks itself at import.
for _cell_dict in CELLS:
    for _key in cell_overrides(_cell_dict):
        if _key not in PARAMS:
            raise KeyError(f"{_cell_dict['cell_id']} overrides unknown parameter {_key!r}")
del _cell_dict, _key
