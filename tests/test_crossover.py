"""The crossover sweep's registry and scorer (scripts/crossover_sweep.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from trapworld.params import resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import crossover_sweep as xs  # noqa: E402


def test_registry_is_frozen():
    """40 unique cells, every override a real parameter, hash as registered."""
    ids = [c["cell_id"] for c in xs.CELLS]
    assert len(ids) == 40 and len(set(ids)) == 40
    for cell in xs.CELLS:
        resolve(xs.overrides(cell))                    # KeyError on a typo
        assert cell["product"] == pytest.approx(cell["kappa"] * cell["psi"])
    assert xs.CELLS_HASH == "6709bb1f08c70dbbac93dd54e22b21d2d16369a177d96c75dedee2a9ab912810"
    assert xs.SEED_BLOCK + len(xs.CELLS) < 4000 and xs.SEED_BLOCK > 72   # off the frozen grid


def test_seed_block_is_disjoint_from_the_frozen_grid():
    from trapworld.run_grid import replicate_seed
    assert xs.replicate_seed(0, 0).spawn_key != replicate_seed(0, 0).spawn_key
    assert xs.replicate_seed(3, 7).spawn_key == (3003, 7)


def test_crossing_interpolates_the_first_sign_change():
    pts = [(0.3, -0.03, 0.001), (0.5, 0.01, 0.001), (1.0, 0.1, 0.001)]
    cross, band = xs._crossing(pts)
    assert cross == pytest.approx(0.3 + 0.2 * 0.03 / 0.04)
    assert band is None
    assert xs._crossing([(0.3, -0.03, 0.001), (0.5, -0.001, 0.001)]) == (None, [0.5, 0.5])
    assert xs._verdict({"mean": -0.05, "mc_se": 0.001}) == "H"
    assert xs._verdict({"mean": 0.05, "mc_se": 0.001}) == "F"
    assert xs._verdict({"mean": 0.001, "mc_se": 0.001}) == "T"
