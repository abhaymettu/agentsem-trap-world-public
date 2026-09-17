# agentSEM trap-world Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. One Opus
> implementer per task; tasks with disjoint files may run in parallel. Subagents do not commit.

**Goal:** the smallest reproducible simulation that kills or confirms the helpful-agent trap, with the
harsh review as a binding falsification gate.

**Architecture:** one Python package `trapworld/` with a pure generator (`world.py`), a small ML-CFA
engine (`cfa.py`), analyzers that read only the observed data contract (`analyzers.py`, `shadow.py`,
`certificate.py`), a frozen grid runner (`grid.py`, `run_grid.py`), metrics (`metrics.py`), and plots
(`plots.py`). Results are JSON/CSV; plots regenerate from results only.

**Tech Stack:** Python 3.13 (uv), numpy, scipy, matplotlib, pytest. Optional R/lavaan cross-check test
(skipped when Rscript is absent).

**Spec:** `specs/001-trap-world/spec.md` (authority). Parameter values live ONLY there and in
`trapworld/params.py`.

## Global Constraints

- numpy/scipy/matplotlib only; no new dependencies.
- Every random draw comes from a `numpy.random.Generator` built from a `SeedSequence` child; nine named
  streams per replicate in the fixed order theta, drift, eps, nuisance, read, jitter, criterion,
  outcome, bootstrap.
- Analyzers read only the observed contract (spec section 3); `belief_partial` is the only reader of
  `belief_logged`, enforced by a test.
- Occasions are 0-indexed in code: `version[i, t] = 0` for `t < switch[i]`, else 1; `T_a = T - T0`.
- Frozen objects (params, grid, surrogate set, rules.json, certificate.py source) are hashed with
  sha256 over canonical JSON (`sort_keys=True, separators=(',', ':')`) before the grid runs.
- No em dashes anywhere in prose or docstrings.
- Tests first; every module ships with its test file; the suite must run in under 5 minutes.

---

## File map

| file | responsibility | task |
|---|---|---|
| `trapworld/closed_forms.py`, `tests/test_closed_forms.py` | F1 to F5 algebra + MC, `results/closed_forms.json` | 1 |
| `trapworld/params.py` | frozen PARAMS dict, `resolve(cell)`, `canonical_hash(obj)` | 2 |
| `trapworld/world.py`, `tests/test_world.py` | `World` dataclass, `simulate(cell, seed)`, `surrogate_g1`, policies, shadow generators, dW counterfactual | 2 |
| `trapworld/cfa.py`, `tests/test_cfa.py` | general ML-CFA with parameter tying; `fit_one_factor`, `bartlett_scores`, `fit_longitudinal_invariance`, fit indices, omega/alpha/AVE; lavaan cross-check | 3 |
| `trapworld/validation.py` | naive validation battery + pass rules | 3 |
| `trapworld/shadow.py` | state-space shadow model (Kalman likelihood, L-BFGS, Hessian SE, person scores), D2 IV | 4 |
| `trapworld/rules.json`, `trapworld/certificate.py` | frozen certificate thresholds + logic | 4 |
| `trapworld/analyzers.py`, `tests/test_analyzers.py`, `tests/test_certificate.py` | the ten analyzers with one output schema; bootstrap | 4 |
| `trapworld/grid.py`, `trapworld/metrics.py`, `trapworld/run_grid.py`, `tests/test_grid.py` | frozen cell list, per-replicate metrics, multiprocessing runner, results writer, hashes, manifest | 5 |
| `trapworld/plots.py` | eight figures from results | 7 |
| `README.md`, `HANDOFF.md`, `REVIEW-RESPONSE.md` | docs | 8 |

## Interfaces (binding across tasks)

```python
# params.py
PARAMS: dict            # spec 2.6 defaults; keys listed in the Task 2 brief
def resolve(cell: dict) -> dict     # PARAMS | cell; unknown key -> KeyError
def canonical_hash(obj) -> str      # sha256 hex of canonical JSON

# world.py
@dataclass
class World:
    y: np.ndarray; z: np.ndarray; dose: np.ndarray; version: np.ndarray; switch: np.ndarray
    jitter: np.ndarray | None; C_clean: np.ndarray; C_react: np.ndarray; W: np.ndarray
    belief_logged: np.ndarray
    theta: np.ndarray; theta_t: np.ndarray; delta: np.ndarray; psi_i: np.ndarray; belief: np.ndarray
    params: dict; implied_r: float; dW: float
def simulate(cell: dict, seed: int | np.random.SeedSequence) -> World
def surrogate_g1(dose: np.ndarray, switch: np.ndarray, T_a: int) -> np.ndarray   # [N,T]
def surrogate_recency(dose, switch, halflife: float) -> np.ndarray               # [N,T]

# cfa.py
@dataclass
class CFAFit: loadings, intercepts, resid_var, var_theta, latent_mean, chi2, df, pvalue, cfi, tli,
              rmsea, srmr, loglik, n, converged, omega, alpha, ave, std_loadings
def fit_one_factor(X: np.ndarray) -> CFAFit                     # X [n,p]
def bartlett_scores(Y: np.ndarray, fit: CFAFit) -> np.ndarray   # Y [...,p] -> [...]
def fit_longitudinal_invariance(X1: np.ndarray, X2: np.ndarray) -> dict
    # keys: configural, metric, scalar (each: chi2, df, cfi, rmsea), d_metric (dchi2, ddf, p, dcfi),
    #       d_scalar (dchi2, ddf, p, dcfi)

# validation.py
def validation_battery(world, fit: CFAFit, scores: np.ndarray, taus=(.2,.3,.4)) -> dict
    # uses world.y, world.z, world.switch, world.C_clean, world.C_react, world.W only
    # keys: omega, alpha, ave, cfi, rmsea, srmr, r_react, r_clean, r_W, incremental_r2, test_retest,
    #       discriminant, invariance (dict from fit_longitudinal_invariance),
    #       pass_{tau} (bool), pass_{tau}_with_invariance (bool)

# analyzers.py
ANALYZERS = {"naive", "naive_pooled", "epoch_fe", "belief_partial", "dose_partial", "detrend",
             "anchor_only", "agentsem", "agentsem_d2", "misspecified"}
def run_analyzer(name: str, world: World, bootstrap: int = 0) -> dict
    # common keys: var_theta_hat, ci (lo, hi) or None, psi_hat or None, beta_clean_hat, scores [N],
    #   corr, rmse (hidden truth used ONLY here for corr/rmse), battery (naive only, else None),
    #   certificate (agentsem/agentsem_d2 only: {"A": {...}, "B": {...}}), extras (dict)

# certificate.py
RULES = json.load(open("trapworld/rules.json"))
def certify(diagnostics: dict, design: str) -> dict   # {"A": {status, estimate, ci, triggered, numbers}, "B": {...}}

# grid.py
CELLS: list[dict]      # each: {"cell_id": str, "sweep": str, **overrides}
# metrics.py
def replicate_metrics(world: World, results: dict[str, dict]) -> dict
def aggregate(records: list[dict]) -> dict
# run_grid.py
python -m trapworld.run_grid [--reps R] [--cells id,id] [--workers K] [--bootstrap]
```

## Tasks

1. Closed forms F1 to F5 (brief: `.superpowers/sdd/trap-world/task-01-brief.md`).
2. Params + generator (`task-02-brief.md`). Parallel with 3.
3. CFA engine + validation battery (`task-03-brief.md`). Parallel with 2.
4. Shadow model + certificate + analyzers (`task-04-brief.md`). After 2 and 3.
5. Grid, metrics, runner, hashes (`task-05-brief.md`). After 4.
6. Adversarial test pass (tests agent) over 1 to 5; smoke run with R = 3.
7. Full grid run (orchestrator, background) then plots (`task-07-brief.md`).
8. Methods audit (methods agent) of identification, certificate, results; then README, HANDOFF,
   REVIEW-RESPONSE (orchestrator).
