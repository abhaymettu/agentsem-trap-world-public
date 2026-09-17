# agentSEM trap-world Constitution

## Core Principles

### I. Falsification over demonstration
HARSH-REVIEW.md is a binding gate. Closed-form checks run before any generator. A claim the arithmetic
kills is reported as killed and removed from the demonstration; nothing is tuned to resurrect it.

### II. Pre-registration
Parameter table, grid cell list, surrogate set, certificate rules, estimand and tolerance are frozen and
hashed before the main grid runs. Hashes are written into the results file. Anything added after seeing
results is labelled exploratory.

### III. Named identification, never bare adjustment
The correction instantiates a named design (D1 anchor epoch; D2 randomized action component) and states
the extra information it consumes. Covariate adjustment on cumulative dose is a competitor, not agentSEM.
When exogeneity or positivity fails the certificate returns `unsupported`, never a corrected point estimate.

### IV. Reproducibility
Independent RNG streams per component via SeedSequence spawning; deterministic tests; results as JSON/CSV;
plots regenerate from results only. No network, no paid APIs, numpy/scipy/matplotlib only.

### V. Minimal code
Smallest simulation that settles the question. Stdlib and installed deps before new ones; no abstractions
beyond what two call sites need; delete before adding.

## Governance
The spec under specs/001-trap-world/ is the authority; the plan argues from it. Deviations are recorded in
REVIEW-RESPONSE.md (harsh-review items) and HANDOFF.md (open items).

**Version**: 1.0.0 | **Ratified**: 2026-09-16
