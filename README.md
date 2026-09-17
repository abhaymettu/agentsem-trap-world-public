# agentSEM trap-world

The smallest reproducible simulation that settles the "helpful-agent trap" claimed in the agentSEM
synthesis of 2026-09-16, built with the independent HARSH-REVIEW.md as a binding falsification gate.

The claim under test, in one sentence: a genuinely good adaptive agent improves outcomes and leaves a
policy shadow in the logs; standard construct validation passes on the shadow; conventional SEM reports
inflated person signal; a policy-aware correction recovers the truth.

**Headline result: see `MORNING-HANDOFF.md`** (killed claims, supported replacement, open gaps, next runs). The
numbers below are filled from `results/` after the grid run; nothing in this file is hand-typed from
memory.

## What was killed by arithmetic before any simulation ran

`python -m trapworld.closed_forms` writes `results/closed_forms.json`. Verdicts as written there: F1 "corr
falls with u"; F2 "reliability up, validity down"; F3 "inflation conditional on r, hump-shaped in u"; F4
"rule deletes the person"; F5 "benign arm killed at near-100% (review supported)".

- K1 "validation correlation is inflated by (1 + psi*kappa)": the factor cancels out of any correlation.
  Under the synthesis's own linearization, correlation-based validity falls monotonically in adaptivity.
- K2 "partial cumulative dose to separate the person from the policy": at T = 56 cumulative dose is a
  0.99 proxy for the trait; partialling it deletes the person and kills the benign arm.
- K3 "the mechanism is Theorem 1": the trap world logs its actions, so Theorem 1's premise fails. This
  is a logged-action world, and the finding is the practitioner-relevant one: ordinary validation does not
  notice a fully logged driver, and the fix is cheap when the deployment has an anchor.

## The world (linear-Gaussian, 200 persons, 56 daily occasions)

A person has a stable trait `theta_i` (task-deferral propensity, unit variance on the reference-policy
scale) plus slow AR(1) drift. Five stream channels (snooze, reschedule, late start, abandon, idle) load on
it. A reminder agent forms a daily belief `bhat` about the person from a private noisy read, and after a
two-week nonadaptive epoch (policy version 0, same dose for everyone) it sets a logged reminder dose
`a = kappa * bhat`. The dose does two things:

1. It helps. The true outcome (task completion) rises with targeted dose through an interaction with
   need: `W = -0.5 theta + 0.2 a + 0.3 a theta + noise`. `dW` (adaptive minus nonadaptive, same person
   draws) is reported per cell; the agent is good when it is positive.
2. It shadows. Reminders create logged deferral-like events: the stream shifts by `delta = psi * g(A)`
   where `g` is a function of the logged dose path (cumulative exposure by default; recency-weighted,
   person-specific, or an unlogged context-builder change in the misspecification cells).

The analyst validates a stream-derived construct against three criteria computed in every replicate:
a clean criterion (`0.5 theta + noise`), a policy-downstream criterion (`0.5 theta + gamma_C * mean dose
+ noise`, the reactivity case), and the true outcome `W`.

**The sign condition (review S10, steering item 3).** Convergent validity inflates only against the
policy-downstream criterion, and only when the share `r` of the theta-criterion association running
through the policy satisfies `r >= sqrt(2) - 1`; it peaks at `u* = r/(1-r)` and is bounded by
`(1+r)/sqrt(2)`. Against the clean criterion and against `W` (where help has the compensating sign)
validity attenuates. What rises regardless is reliability: omega, alpha, AVE and one-factor fit all
improve with adaptivity because the shadow is a common shift across channels. That is the
review-supported replacement claim (reliability up, validity down), and it is what the headline figure
shows.

Full equations, parameter table with a justification per value, grid, certificate rules and
falsification map: `specs/001-trap-world/spec.md`.

## The correction is a named design, not covariate adjustment

`agentsem` instantiates **D1 + D5**: the logged version-0 epoch is an exogenous anchor (same dose for
everyone, so the shadow is zero there by design) that calibrates the measurement model and pins the
reference scale; the adaptive-phase shadow is modelled as a common proportional shift `psi * g1` with `g1`
the normalized cumulative logged dose since the switch (the frozen surrogate set, hashed before the run).
A state-space model `eta_it = theta_i + d_it + psi * g1_it + b * gbar_i + e_it` fitted by maximum
likelihood gives `Var(theta)`, `psi`, and person scores. The `b * gbar_i` term (spec section 11, A1) is
there because the agent reads the person privately: its dose carries information about the person beyond
the stream, so the person effect and the person's mean exposure `gbar_i` are correlated, and a plain
random-effects likelihood pushes part of `Var(theta)` into `psi` (base replicate: `psi` 0.58,
`Var(theta)` 0.89). A sequential-likelihood argument ("actions depend only on the past, so conditioning
on them is free") does not hold here for that reason, and it is not what the correction rests on. With
the term, `psi` is identified from within-person variation of the logged exposure path and
`Var(theta) = phi_u + b^2 Var(gbar)`; what that still assumes is that the drift does not feed back into
the dose (the no-drift cell isolates that channel, the base cell measures it).

Two things about that estimator that the grid makes plain. Almost all of the recovered variance is the
Mundlak term (`b^2 Var(gbar)` is 95% of the number at the base cell; `phi_u` is 0.05): the estimator
recovers `Var(E[theta | logged exposure])` plus a small residual, which is F4's "dose is a .99 proxy for
the trait" turned into an estimator rather than a mistake. And on `Var(theta)` alone the anchor epoch does
the work: `anchor_only` is within tolerance (+.05, the drift term) in every cell, and the shadow model
improves on it by .01 in absolute bias. What the shadow model adds is the person scores (rmse .19 against
.29 for the anchor scores at the base cell) and the refusals: the certificate is what says no under the
recency shadow and the unlogged driver.

`agentsem_d2` instantiates **D2**: when the deployment logs a randomized component of the dose, the
cumulative logged jitter instruments the surrogate within person; no anchor needed.

**Information the correction consumes beyond the logs:** (1) the logged policy version per occasion and
the deployment's statement that version 0 applied the same dose to everyone; (2) the logged dose path;
(3) the parametric restriction on the shadow. Nothing about the policy's belief. Conditioning on the
policy's belief is run as a labelled positive control (`belief_partial`) and over-corrects; partialling
cumulative dose is run as the synthesis's original rule (`dose_partial`) and deletes the person.

### The certificate (frozen in `trapworld/rules.json`)

Two claims are certified separately: A, `Var(theta)` on the reference scale (anchor-based); B, adaptive-phase
person scores on the reference scale after correction.

| rule | fires when | effect |
|---|---|---|
| R1 | policy version missing | both `unknown` |
| R2 | no version-0 epoch, dose not constant in it, or shorter than 7 occasions; switch time correlated with anchor score | `unsupported`; `bounded` |
| R3 | surrogate design ill-conditioned (condition number > 30) | B `bounded`; conditioning numbers always emitted |
| R4 | residual epoch shift or change-point not explained by logged actions (> 0.15 SD), or a residual that still tracks the person's exposure level (window-mean residual vs `gbar_i`, p < .01; spec 11 A2) | B `unsupported` |
| R5 | metric invariance fails across epochs (dCFI > .01) | B `unsupported` |
| R6 | person-level residual heterogeneity ratio > 1.5 | B `bounded` (anchor fallback) |
| D2 | jitter not logged; first-stage F < 10 | `unsupported`; `bounded` |

`unsupported` returns no point estimate. False confidence (identified but wrong) and false refusal
(refused but would have been right) are reported as a pair for every cell.

## Competitors and falsification arms

naive (conventional SEM on adaptive-phase logs, full validation battery including longitudinal
invariance), naive_pooled, epoch_fe (occasion fixed effects), detrend (competent-analyst preprocessing),
belief_partial (the forbidden rule), dose_partial (the original rule), anchor_only (D1 minimal),
agentsem, agentsem_d2, misspecified (agentsem with the wrong surrogate). Generators G1 cumulative, G2
recency, G3 heterogeneous psi correlated with theta, G4 unlogged change (with the detectable, the
combined, and the undetectable-limit variants). Policies: greedy exponentially weighted, learning Kalman,
single-channel heuristic, uniform help, none. Special worlds: zero-person, no-drift, closed-loop feedback
(the policy reads its own shadow), performance-triggered switch.

## The optional Jev policy arm

`trapworld/jev.py` adds one more adaptive policy: a TypeSafe Jev Score question ("how much reminder
support does this person need this week", four levels written as situations) asked once a week per person
from a bounded seven-day state of the five stream signals and the past dose. The answer's
probability-weighted level maps to a dose on the same scale as the deterministic policy. That dose is the
only thing that reaches the world: the trait, drift, shadow, criteria and outcome are generated by the same
code as every other arm and never see a Jev output; a failed call makes the replicate missing, never
substituted; the deterministic `ew` policy remains the central result and Jev cells are excluded from
every headline aggregate. The repo ships a fake transport (`fake-jev-stub`, a plumbing check that is not a
Jev result), a recorded-replay transport, the logging schema, and the live path in `JEV-LIVE-RUN.md`,
which runs in a separate session with an ephemeral credential and cannot change any frozen object.

## How to run

```bash
uv sync --group dev                       # python 3.13, numpy, scipy, matplotlib, pytest
.venv/bin/pytest -q                       # unit and falsification tests
.venv/bin/python -m trapworld.closed_forms      # F1..F5 -> results/closed_forms.json
.venv/bin/python -m trapworld.run_grid --workers 11   # 73 cells -> results/grid, summary, hashes, manifest
.venv/bin/python -m trapworld.run_grid --bootstrap    # coverage audit -> results/coverage_audit.json
.venv/bin/python -m trapworld.run_grid --check-hashes # re-verify every frozen object against results/hashes.json
.venv/bin/python -m trapworld.plots             # plots/P1..P9
```

The Jev cells run with the fake transport by default; `J_live` runs only with `--jev live` (see
`JEV-LIVE-RUN.md`) and replays offline with `--jev recorded --recorded-dir results/jev`.

Every draw comes from `SeedSequence(20260916)` spawned per cell, replicate and component, so cells and
replicates can run in any order or in parallel with identical output (tested).

## Results

Grid of 2026-09-17 (72 cells, 15,450 replicates, `results/summary.csv`; MC SE about .035 on a rate near
0.5 and .005 to .013 on a bias mean). Full verdicts and named limits: `MORNING-HANDOFF.md`; the item-by-item
review response: `REVIEW-RESPONSE.md`; the methods audit: `AUDIT.md`.

- **Killed by arithmetic** (`results/closed_forms.json`): K1 (correlations fall in adaptivity; only the
  variance rises) and K2 (cumulative dose is a .99 proxy for the trait at T = 56; partialling it kills the
  benign arm .99 of the time).
- **Reliability up, validity down** (R1/R3): naive omega .993 to .998 and naive Var(theta) bias +.03 to
  +3.25 as u = psi kappa runs 0 to 2, person correlation .981 to .977, battery pass rate .93 to .98
  everywhere with a person, including the longitudinal invariance test when the shadow is proportional to
  the loadings (dCFI .0001). Invariance testing catches a nonproportional shadow (pass rate .94 to .15).
- **Criterion contamination** (R2): present in direction, 1 to 2% in size against the policy-downstream
  criterion; attenuation (.98) against the clean one.
- **The correction** (R4): base-cell bias +.013 (MC SE .005), certified identified in .96, psi .48 for a
  truth of .5; refuses claim B in 200 of 200 replicates under the recency shadow (G2) and the unlogged
  driver (G4, G4b); bias +.001 to +.020 under heterogeneous responsiveness (G3), where epoch fixed effects
  sit at +.45 (paired margin -.45, MC SE .005). Its Hessian interval covers only .55 to .63; the
  full-pipeline person bootstrap covers .94 at the base cell and .93 under G3
  (`results/coverage_audit.json`), so that is the interval to quote.
- **Competitors** (R5): epoch fixed effects equal pooled naive (+.44 at the base cell); detrending repairs a
  linear ramp (+.11) but not a saturating shadow (+.58); belief conditioning deletes the person (-.95); dose
  partialling leaves a person correlation of .18.
- **Specificity** (R6): innocence at psi 0 is .06 to .09 / .83 to .88 / 1.00 for kappa .5 / 1 / 2; detection reaches 1.00
  by psi .2 at kappa 1.
- **Uniform help** (R7) inflates nothing (+.036 naive); the **zero-person world** (R8) fools omega, fit and
  convergent validity but fails test-retest, and the certificate refuses it in .88.
- **Jev**: no live grid ran; the fake-transport cell is a plumbing check (naive +.48, agentsem +.055).

## Limitations and open items

See `MORNING-HANDOFF.md` section 3, `AUDIT.md` and `REVIEW-RESPONSE.md` for the item-by-item status against the review.

## Layout

```
trapworld/   closed_forms, params, world, cfa, validation, shadow, certificate (+rules.json),
             analyzers, jev, grid, metrics, run_grid, plots
tests/       one test file per module plus adversarial tests
results/     machine-readable outputs (JSON, CSV) and hashes
plots/       figures regenerated from results
specs/001-trap-world/   spec.md (authority), plan.md
```
