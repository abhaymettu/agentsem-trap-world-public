"""Per-replicate metrics and per-cell aggregates (spec section 7).

This is the one layer that is allowed to read hidden truth, and it reads it only to
score what the analyzers produced: bias against ``Var(theta)``, coverage, the
correlations the plots need, the contamination share the trap implies.  No estimator
sees anything computed here.

Two shapes come out:

* ``replicate_metrics(world, results)`` scores one replicate.  It flattens the naive
  battery to scalars and the certificate to statuses, flags and numbers, because the raw
  analyzer output carries numpy arrays (scores, the invariance fits) that must never
  reach a results file.
* ``aggregate(records)`` reduces a cell's records field by field: mean, SD, MC SE and
  quantiles for a number; rate and MC SE for a flag; a distribution for a status.  Every
  summary carries ``n``, the count of non-null values, because a flag that is undefined
  in this cell (psi where the policy doses nobody) is missing, not false.

Everything a record holds is strict JSON.  ``jsonable`` converts numpy scalars, turns a
non-finite float into None (a correlation with a constant series is not a number and
must not be read as one) and raises on anything else, so a results file can never hold
an object that only pickled by accident.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["replicate_metrics", "aggregate", "jsonable", "CERT_ANALYZERS", "TAUS"]

#: The arms that return a certificate.
CERT_ANALYZERS = ("agentsem", "misspecified", "agentsem_d2")

#: Pass thresholds of the naive battery (spec 5.1).
TAUS = ("0.2", "0.3", "0.4")

#: Battery statistics that are already scalars.
BATTERY_SCALARS = ("omega", "alpha", "ave", "cfi", "rmsea", "srmr", "r_react", "r_clean",
                   "r_W", "incremental_r2", "test_retest", "discriminant")

#: Numbers every certificate echoes, whatever it decided (spec 5.2, 11 A2).
CERT_NUMBERS = ("phi_B_raw", "psi", "max_exposure_corr", "exposure_p", "hetero_ratio",
                "cond_number", "epoch_shift", "max_changepoint_shift", "dcfi_metric",
                "between_share_gbar", "r2_gbar_on_anchor", "first_stage_F")

#: Shadow-model quantities from the analyzer's extras (spec 11 A1).
CERT_EXTRAS = ("b", "phi_u", "var_gbar", "psi_identified", "between_dropped", "ci_method")

#: Statuses in the order they are reported.
STATUSES = ("identified", "bounded", "unsupported", "unknown")

#: Paired per-replicate bias differences (F12 and the two competent competitors).
PAIRED = (("agentsem", "epoch_fe"), ("agentsem", "detrend"), ("agentsem", "anchor_only"))


# ------------------------------------------------------------------ small helpers
def _num(value):
    """A float, or None when the value is missing or not finite."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    out = float(value)
    return out if math.isfinite(out) else None


def _corr(a, b):
    """Pearson r, or None when either series is constant or too short."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3 or a[ok].std() < 1e-12 or b[ok].std() < 1e-12:
        return None
    return _num(np.corrcoef(a[ok], b[ok])[0, 1])


def _person_mean(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of ``a`` [N, T] over the occasions ``mask`` selects; 0 where none are."""
    a = np.asarray(a, float)
    m = np.asarray(mask, bool).astype(float)
    denom = m.sum(axis=1)
    return np.where(denom > 0, (a * m).sum(axis=1) / np.where(denom > 0, denom, 1.0), 0.0)


def jsonable(obj, path: str = ""):
    """Return ``obj`` as strict JSON data, or raise TypeError naming the offender.

    numpy scalars become Python scalars, tuples become lists, non-finite floats become
    None.  An ndarray, a dataclass or anything else raises: the results files are the
    record of the run and nothing may enter them by accident.
    """
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string key at {path or '<root>'}: {key!r}")
            out[key] = jsonable(value, f"{path}.{key}" if path else key)
        return out
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    raise TypeError(f"non-serialisable value at {path or '<root>'}: {type(obj).__name__}")


# ------------------------------------------------------------------ per replicate
def _psi_true(world) -> float:
    """The shadow strength the world actually used."""
    generator = world.params["generator"]
    if generator == "hetero":
        return float(np.mean(world.psi_i))
    if generator in ("unlogged", "none"):
        return 0.0
    return float(world.params["psi"])


def _battery_block(battery: dict, abs_bias_ok, r_theta_react) -> dict:
    """The naive battery flattened to scalars, plus the false-confidence flags."""
    out = {key: _num(battery[key]) for key in BATTERY_SCALARS}
    invariance = battery["invariance"]
    out["inv_dcfi_metric"] = _num(invariance["d_metric"]["dcfi"])
    out["inv_p_metric"] = _num(invariance["d_metric"]["p"])
    out["inv_dcfi_scalar"] = _num(invariance["d_scalar"]["dcfi"])
    out["inv_p_scalar"] = _num(invariance["d_scalar"]["p"])
    for tau in TAUS:
        for suffix in ("", "_with_invariance"):
            passed = bool(battery[f"pass_{tau}{suffix}"])
            out[f"pass_{tau}{suffix}"] = passed
            out[f"false_conf_{tau}{suffix}"] = (None if abs_bias_ok is None
                                                else bool(passed and not abs_bias_ok))
    # How much of the reactivity the analyst reads is the trait's own doing.
    out["ratio_react"] = (None if not r_theta_react or out["r_react"] is None
                          else _num(out["r_react"] / r_theta_react))
    return out


def _certificate_block(result: dict, truth: float, tolerance: float,
                       abs_bias_ok) -> dict:
    """Statuses, the two error flags, the specificity pair and every rule number."""
    cert = result["certificate"]
    extras = result.get("extras") or {}
    claim_a, claim_b = cert["A"], cert["B"]
    numbers = claim_b["numbers"]
    status_b = claim_b["status"]

    phi_b_raw = _num(numbers.get("phi_B_raw"))
    out = {
        "status_A": claim_a["status"],
        "status_B": status_b,
        "triggered_A": list(claim_a["triggered"]),
        "triggered_B": list(claim_b["triggered"]),
        "triggered": sorted(set(claim_a["triggered"]) | set(claim_b["triggered"])),
        "false_confidence": (None if abs_bias_ok is None
                             else bool(status_b == "identified" and not abs_bias_ok)),
        # A refusal is wrong only if the correction it refused to report was in fact good.
        # The pre-registered form (spec 7) counts `bounded`, which still returns the anchor
        # estimate; the strict form counts only `unsupported`, where nothing is returned.
        "false_refusal": bool(status_b in ("bounded", "unsupported")
                              and phi_b_raw is not None
                              and abs(phi_b_raw - truth) <= tolerance),
        "false_refusal_strict": bool(status_b == "unsupported"
                                     and phi_b_raw is not None
                                     and abs(phi_b_raw - truth) <= tolerance),
        "innocent": extras.get("innocent"),
        "detects": extras.get("detects"),
    }
    out.update({key: _num(numbers.get(key)) for key in CERT_NUMBERS})
    for key in CERT_EXTRAS:
        value = extras.get(key)
        out[key] = value if isinstance(value, str) or value is None else _num(value)

    # F17: the Hessian interval is always kept, the bootstrap one only when it was run.
    raw_ci = extras.get("ci_B_raw")
    out["ci_hessian"] = None if raw_ci is None else [_num(raw_ci[0]), _num(raw_ci[1])]
    out["covers_hessian"] = (None if raw_ci is None
                             else bool(raw_ci[0] <= truth <= raw_ci[1]))
    out["ci_hessian_width"] = (None if raw_ci is None else _num(raw_ci[1] - raw_ci[0]))
    boot_ci = extras.get("boot_ci_phi")
    out["ci_bootstrap"] = None if boot_ci is None else [_num(boot_ci[0]), _num(boot_ci[1])]
    out["covers_bootstrap"] = (None if boot_ci is None
                               else bool(boot_ci[0] <= truth <= boot_ci[1]))
    out["ci_bootstrap_width"] = (None if boot_ci is None
                                 else _num(boot_ci[1] - boot_ci[0]))
    boot_psi = extras.get("boot_ci_psi")
    out["ci_bootstrap_psi"] = (None if boot_psi is None
                               else [_num(boot_psi[0]), _num(boot_psi[1])])
    out["boot_reps_ok"] = _num(extras.get("boot_reps_ok"))
    return out


def _analyzer_block(name: str, result: dict, truth: float, tolerance: float,
                    psi_true: float, r_theta_react) -> dict:
    """One analyzer's score on one replicate."""
    if "error" in result:
        return {"error": str(result["error"]), "var_theta_hat": None, "bias": None,
                "abs_bias_ok": None, "ci": None, "covers": None, "psi_hat": None,
                "psi_true": psi_true, "beta_clean_hat": None, "corr": None, "rmse": None}
    estimate = _num(result["var_theta_hat"])
    bias = None if estimate is None else estimate - truth
    abs_bias_ok = None if bias is None else bool(abs(bias) <= tolerance)
    ci = result["ci"]
    block = {
        "var_theta_hat": estimate,
        "bias": _num(bias),
        "abs_bias_ok": abs_bias_ok,
        "ci": None if ci is None else [_num(ci[0]), _num(ci[1])],
        "ci_lo": None if ci is None else _num(ci[0]),
        "ci_hi": None if ci is None else _num(ci[1]),
        "ci_width": None if ci is None else _num(ci[1] - ci[0]),
        "covers": None if ci is None else bool(ci[0] <= truth <= ci[1]),
        "psi_hat": _num(result["psi_hat"]),
        "psi_true": psi_true,
        "beta_clean_hat": _num(result["beta_clean_hat"]),
        "corr": _num(result["corr"]),
        "rmse": _num(result["rmse"]),
    }
    if result.get("battery") is not None:
        block.update(_battery_block(result["battery"], abs_bias_ok, r_theta_react))
    if result.get("certificate") is not None:
        block.update(_certificate_block(result, truth, tolerance, abs_bias_ok))
        # Coverage is only a claim where the certificate made one.
        block["covers_when_identified"] = (block["covers"]
                                           if block["status_B"] == "identified" else None)
    return block


def replicate_metrics(world, results: dict[str, dict], cell_id=None, rep=None,
                      jev: dict | None = None) -> dict:
    """Score one replicate.  ``results`` is ``{analyzer name: run_analyzer output}``."""
    params = world.params
    truth = float(params["var_theta"])
    tolerance = float(params["tolerance"])
    psi_true = _psi_true(world)

    theta = np.asarray(world.theta, float).ravel()
    occasions = np.arange(np.asarray(world.y).shape[1])
    adaptive = occasions[None, :] >= np.asarray(world.switch, int)[:, None]
    delta_bar = _person_mean(np.asarray(world.delta, float), adaptive)
    w_bar = _person_mean(np.asarray(world.W, float), adaptive)

    var_theta_sample = float(np.var(theta, ddof=1)) if theta.size > 1 else 0.0
    var_delta_bar = float(np.var(delta_bar, ddof=1)) if delta_bar.size > 1 else 0.0
    total = var_delta_bar + var_theta_sample
    r_theta_react = _corr(theta, world.C_react)

    record = {
        "cell_id": cell_id,
        "rep": rep,
        "policy": params["policy"],
        "generator": params["generator"],
        "kappa": _num(params["kappa"]),
        "kappa_eff": _num(params.get("kappa_eff")),
        "implied_r": _num(world.implied_r),
        "dW": _num(world.dW),
        "dW_se": _num(params.get("dW_se")),
        "var_theta": truth,
        "var_theta_sample": _num(var_theta_sample),
        "psi_true": psi_true,
        "r_theta_react": r_theta_react,
        "r_theta_clean": _corr(theta, world.C_clean),
        "r_theta_W": _corr(theta, w_bar),
        # How much of the person ordering the shadow alone would reproduce.
        "u_eff": (0.0 if var_theta_sample <= 0.0
                  else _num(np.cov(delta_bar, theta, ddof=1)[0, 1] / var_theta_sample)),
        "shadow_share": None if total <= 0.0 else _num(var_delta_bar / total),
        "analyzers": {
            name: _analyzer_block(name, result, truth, tolerance, psi_true, r_theta_react)
            for name, result in results.items()
        },
    }
    if jev is not None:
        record["jev_status"] = jev.get("status")
        record["jev"] = {"n_calls": _num(jev.get("n_calls")),
                         "n_failed": _num(jev.get("n_failed")),
                         "mean_latency_ms": _num(jev.get("mean_latency_ms")),
                         "returned_models": list(jev.get("returned_models") or []),
                         "failure_classes": dict(jev.get("failure_classes") or {})}
    return jsonable(record)


# ------------------------------------------------------------------ per cell
def _numeric_summary(values: list) -> dict:
    arr = np.asarray(values, float)
    n = arr.size
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {"mean": float(arr.mean()), "sd": sd, "mc_se": sd / math.sqrt(n),
            "q05": float(np.percentile(arr, 5)), "q50": float(np.percentile(arr, 50)),
            "q95": float(np.percentile(arr, 95)), "n": int(n)}


def _rate_summary(values: list) -> dict:
    n = len(values)
    rate = float(np.mean([1.0 if v else 0.0 for v in values]))
    return {"rate": rate, "mc_se": math.sqrt(max(rate * (1.0 - rate), 0.0) / n),
            "n": int(n)}


def _category_summary(values: list) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    n = len(values)
    order = [s for s in STATUSES if s in counts] + sorted(set(counts) - set(STATUSES))
    return {"counts": {k: counts[k] for k in order},
            "rates": {k: counts[k] / n for k in order}, "n": int(n)}


def _summarize(values: list):
    """Summarize one field across replicates, dispatching on what the values are."""
    kept = [v for v in values if v is not None]
    if not kept:
        return {"n": 0}
    if all(isinstance(v, bool) for v in kept):
        return _rate_summary(kept)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in kept):
        return _numeric_summary(kept)
    if all(isinstance(v, str) for v in kept):
        return _category_summary(kept)
    return None                      # lists and mixed fields are not summarized


def _rule_rates(lists: list) -> dict:
    """Fire rate per certificate rule id over the replicates that produced a list."""
    present = [v for v in lists if isinstance(v, list)]
    if not present:
        return {"n": 0, "rates": {}}
    rules = sorted({rule for row in present for rule in row})
    n = len(present)
    return {"n": n,
            "rates": {rule: sum(rule in row for row in present) / n for rule in rules}}


def _summarize_fields(rows: list[dict]) -> dict:
    """Summarize every field of a list of same-shaped flat dicts."""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    out = {}
    for key in sorted(keys):
        values = [row.get(key) for row in rows]
        if key in ("triggered", "triggered_A", "triggered_B"):
            out[key] = _rule_rates(values)
            continue
        summary = _summarize(values)
        if summary is not None:
            out[key] = summary
    return out


def _paired_difference(records: list[dict], left: str, right: str, field: str) -> dict:
    """Mean and MC SE of a within-replicate difference, over replicates with both."""
    diffs = []
    for record in records:
        blocks = record.get("analyzers") or {}
        a = (blocks.get(left) or {}).get(field)
        b = (blocks.get(right) or {}).get(field)
        if a is not None and b is not None:
            diffs.append(float(a) - float(b))
    if not diffs:
        return {"mean": None, "mc_se": None, "n": 0}
    arr = np.asarray(diffs, float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "mc_se": sd / math.sqrt(arr.size),
            "sd": sd, "n": int(arr.size)}


def _jev_summary(records: list[dict]) -> dict:
    rows = [r["jev"] for r in records if isinstance(r.get("jev"), dict)]
    if not rows:
        return {}
    models = sorted({m for row in rows for m in (row.get("returned_models") or [])})
    classes: dict[str, int] = {}
    for row in rows:
        for name, count in (row.get("failure_classes") or {}).items():
            classes[name] = classes.get(name, 0) + int(count)
    return {
        "n_replicates": len(rows),
        "n_failed": sum(1 for r in records if r.get("jev_status") == "failed"),
        "n_calls": _summarize([row.get("n_calls") for row in rows]),
        "calls_failed": _summarize([row.get("n_failed") for row in rows]),
        "mean_latency_ms": _summarize([row.get("mean_latency_ms") for row in rows]),
        "returned_models": models,
        "failure_classes": dict(sorted(classes.items())),
        "status": _summarize([r.get("jev_status") for r in records]),
    }


def aggregate(records: list[dict]) -> dict:
    """Reduce one cell's replicate records to the per-cell summary."""
    if not records:
        return {"cell_id": None, "n_reps": 0, "n_failed": 0, "cell": {}, "analyzers": {},
                "paired": {}}

    scored = [r for r in records if r.get("analyzers")]
    cell_rows = [{k: v for k, v in r.items() if k not in ("analyzers", "jev", "cell_id")}
                 for r in records]
    cell_fields = _summarize_fields(cell_rows)

    names: list[str] = []
    for record in scored:
        for name in record["analyzers"]:
            if name not in names:
                names.append(name)
    analyzers = {name: _summarize_fields([r["analyzers"][name] for r in scored
                                          if name in r["analyzers"]])
                 for name in sorted(names)}

    dW = cell_fields.get("dW", {})
    implied = cell_fields.get("implied_r", {})
    out = {
        "cell_id": records[0].get("cell_id"),
        "n_reps": len(records),
        "n_failed": sum(1 for r in records if r.get("jev_status") == "failed"),
        "n_scored": len(scored),
        "dW_mean": dW.get("mean"),
        "dW_mc_se": dW.get("mc_se"),
        "implied_r_mean": implied.get("mean"),
        "cell": cell_fields,
        "analyzers": analyzers,
        "paired": {},
    }
    for left, right in PAIRED:
        out["paired"][f"{left}_minus_{right}_bias"] = _paired_difference(
            scored, left, right, "bias")
        out["paired"][f"{left}_minus_{right}_abs_bias"] = _paired_difference(
            [{"analyzers": {k: {"bias": None if v.get("bias") is None
                                else abs(v["bias"])}
                            for k, v in r["analyzers"].items()}} for r in scored],
            left, right, "bias")
    jev = _jev_summary(records)
    if jev:
        out["jev"] = jev
    return jsonable(out)
