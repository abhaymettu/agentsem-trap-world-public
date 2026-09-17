"""Frozen parameter table for the trap-world simulation (spec 2.6).

Every value here comes from the spec's frozen table and is used verbatim; nothing in the
package may invent a parameter. ``PARAMS`` is hashed into ``results/hashes.json`` before
the grid runs, so its contents and key order must not drift.

Occasion indices are 0-based in code: ``t_cb = 36`` and ``T0 = 14`` are indices, and the
adaptive phase runs over occasions ``T0 .. T-1`` (``T_a = T - T0 = 42`` occasions).
"""

from __future__ import annotations

import hashlib
import json

PARAMS = dict(
    N=200, T=56, T0=14, var_theta=1.0, rho_d=0.9, sd_d=0.3,
    p=5, loadings=(1.0, 0.9, 0.8, 0.7, 0.6), intercepts=(0.0,) * 5, var_eps=(1.0,) * 5,
    shadow_pattern="proportional",            # or "nonproportional" -> nu = NU_NONPROPORTIONAL
    n_nuisance=3, nuisance_loading=0.7, nuisance_var_eps=1.0,
    policy="ew",                              # "ew" | "kalman" | "heuristic" | "uniform" | "none"
    var_read=2.0, gain_ew=0.3, kappa=1.0, policy_reads_stream=False, var_jitter=0.0,
    trigger="calendar",                       # or "performance"
    generator="cumulative",                   # "cumulative"|"recency"|"hetero"|"unlogged"|"unlogged_plus"|"none"
    psi=0.5, halflife=7.0, tau_psi=0.5, rho_psitheta=0.0, t_cb=36, c0=0.5, c1=0.3,
    lam_C=0.5, var_C=0.75, gamma_C=1.0,
    beta_W=-0.5, gamma_W1=0.2, gamma_W2=0.3, var_W=0.25,
    tolerance=0.10, smd_psi=0.10, master_seed=20260916,
    permute_dose=False,                       # F8 permutation null (tests only, never a grid cell)
)

# Shadow loading pattern when shadow_pattern = "nonproportional" (spec 2.4 and 2.6).
NU_NONPROPORTIONAL = (1.0, 1.0, 0.2, 0.2, 0.0)

# Performance trigger (spec 2.2): first occasion at or after TRIGGER_MIN_T (1-indexed) at
# which the TRIGGER_WINDOW-occasion running mean of the stream mean exceeds TRIGGER_THRESHOLD.
TRIGGER_MIN_T = 8
TRIGGER_WINDOW = 7
TRIGGER_THRESHOLD = 0.3

# policy_reads_stream = True variant (spec 2.2): the stream-mean read carries variance 0.31
# of its own, so the added noise is var_read - 0.31 and total read noise stays at var_read.
READ_STREAM_VAR_DROP = 0.31

# Kalman P_0 floor for the zero-person world (var_theta = 0), so K_0 is 0/0-free.
KALMAN_P0_FLOOR = 1e-6

# The nine spawned RNG streams, in the fixed order required by spec 3 and plan.md.
STREAM_ORDER = ("theta", "drift", "eps", "nuisance", "read", "jitter", "criterion",
                "outcome", "bootstrap")


def resolve(cell: dict) -> dict:
    """Return a copy of PARAMS updated by ``cell``.

    An unknown key raises KeyError (a typo in a grid cell must not silently do nothing).
    Keys whose frozen value is a tuple stay tuples, so a cell may pass a list.
    """
    out = dict(PARAMS)
    for key, value in cell.items():
        if key not in PARAMS:
            raise KeyError(f"unknown parameter: {key!r}")
        if isinstance(PARAMS[key], tuple) and isinstance(value, (list, tuple)):
            value = tuple(value)
        out[key] = value
    return out


def canonical_hash(obj) -> str:
    """sha256 over canonical JSON, the one hashing rule for every frozen object."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
