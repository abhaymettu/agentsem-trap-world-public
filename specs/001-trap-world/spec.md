# Feature Specification: agentSEM trap-world simulation

**Feature Branch**: `001-trap-world` | **Created**: 2026-09-16 | **Status**: frozen before grid run
**Sources**: BUILD-BRIEF.md, STEERING-20260916-1946.md, HARSH-REVIEW.md, and the three Obsidian specs
(synthesis 2026-09-16, round 1, round 2). This document is the single authority for the build; the plan
argues from it.

## 1. What this build settles

The synthesis claims a "helpful-agent trap": a genuinely good adaptive policy improves outcomes and
leaves a policy shadow in the logs; standard construct validation passes on the shadow; conventional
SEM recovers inflated person signal; a policy-aware correction recovers truth.

HARSH-REVIEW.md shows by algebra that two load-bearing sentences of the synthesis are wrong, and the
steering note makes the review a binding gate. The build therefore has two jobs:

1. **Kill or confirm the original claims by arithmetic first** (F1 to F5, no generator).
2. **Test the review-supported replacement claims in a full simulation** with named identification,
   a misspecification grid on the correction itself, and a certificate that refuses when it should.

### 1.1 Original claims under test (expected to be killed; see F1 to F5)

- K1 "Validation correlation is inflated by (1 + psi*kappa) as adaptivity rises." Prediction (review):
  corr(eta, theta) and corr(eta, clean criterion) fall monotonically in u = psi*kappa. If F1 confirms,
  K1 is removed from the demonstration and recorded in HANDOFF.md as killed.
- K2 "Partial cumulative dose (what the policy did) to separate person from policy." Prediction (review):
  cumulative dose is a 0.99 proxy for theta at T = 56; partialling deletes the person; the benign arm
  fails specificity at near 100%. If F4/F5 confirm, K2 is removed as *the correction* and kept only as a
  labelled competitor (`dose_partial`) that demonstrates the failure.
- K3 "The mechanism is Theorem 1's aliasing." Ruled by inspection (B3): the trap world logs its actions,
  so Theorem 1's premise fails; this is a logged-action world. Stated in README, not simulated.

### 1.2 Replacement claims (pre-registered, tested in the grid)

- R1 **Reliability up, validity down** (B1.3). A common policy-induced shift across stream channels
  raises composite reliability (omega, alpha, AVE, one-factor fit) monotonically in u while
  corr(person score, theta) falls. Guaranteed by the algebra; the grid must reproduce it at the effective
  u of each cell.
- R2 **Criterion contamination with a stated condition** (B1.4). Convergent validity against a criterion
  that is downstream of the policy's dose inflates only when the share r of the theta-criterion
  association running through the policy satisfies r >= sqrt(2) - 1, peaks at u* = r/(1-r), ceiling
  (1+r)/sqrt(2). Against a clean criterion it attenuates. The grid varies gamma_C and reports implied r.
- R3 **Scale inflation of the person signal.** Conventional SEM on adaptive-phase logs estimates
  Var(theta) on the reference scale with relative bias approximately (1+u_eff)^2 - 1 + u_eff^2*s_e2
  (proportional shadow). Person-level rank survives (Corollary 1.1); scale does not.
- R4 **Named identification recovers truth where its assumptions hold and refuses where they do not.**
  The agentSEM correction instantiates D1 (an exogenous nonadaptive anchor epoch, logged as policy
  version 0) plus D5 (a frozen parametric surrogate for the shadow), and separately D2 (a logged
  randomized dose component) in one design arm. Under the matched generator (G1) it recovers Var(theta)
  within tolerance with nominal coverage; under G2/G3 the certificate downgrades to `bounded`; under G4
  (unlogged driver) it returns `unsupported` for adaptive-phase person scores.
- R5 **Competitors.** Epoch fixed effects alone do not repair the person-specific shadow (kill test
  S8/F12). Competent preprocessing (person-specific linear detrend) repairs G1's ramp but not G2's step
  (S7/F11). Belief conditioning over-corrects (F13). Dose partialling deletes the person (F4/F5).
  Longitudinal invariance testing catches the shadow only when its loading pattern differs from the
  trait's (S6/F10; narrowed claim).
- R6 **Specificity is an equivalence decision** (S4): psi = 0 arms are declared innocent at >= 90% and
  detection power rises in psi (ROC over psi). **False confidence and false refusal are reported as a
  pair** (S5).
- R7 **The trap is personalized help, not help** (S9): a uniform-help arm (dose constant across persons)
  produces no person-variance inflation.
- R8 **Zero-person world** (F9): with Var(theta) = 0 and psi > 0, conventional SEM reports a reliable,
  convergent "construct" that is 100% policy shadow, and the certificate refuses or bounds it.

Each claim has a falsification test in section 8 with the outcome that would kill it.

## 2. The world

Linear-Gaussian, discrete occasions. Persons i = 1..N, occasions t = 1..T. All symbols below are the
names used in code (`trapworld/params.py`).

### 2.1 Person

- Trait: `theta_i ~ N(0, var_theta)`, the construct on the reference-policy (pi_0) scale.
- Drift: `d_it = rho_d * d_i,t-1 + sqrt(1 - rho_d^2) * sd_d * z_it`, stationary AR(1), `d_i0 ~ N(0, sd_d^2)`.
  State: `theta_it = theta_i + d_it`.
- Nuisance construct: `xi_i ~ N(0, 1)`, independent of theta; 3 indicators `z_ikt = 0.7 * xi_i + N(0, 1)`.
  Never touched by the policy. Used for discriminant validity and the passive baseline.

### 2.2 Policy (the agent)

Policy version `v_it`: 0 (nonadaptive) for `t <= switch_i`, 1 (adaptive) after. Calendar trigger:
`switch_i = T0 = 14` for everyone. Performance trigger (S1 cell): `switch_i` = first `t >= 8` at which
the 7-occasion running mean of the stream mean `m_it` exceeds 0.3, else 14. Version and switch time are
logged.

Belief `bhat_it` is formed from a private noisy read of the person, `r_it = theta_it + N(0, var_read)`,
which does not contain the stream indicators (this matches the review's linearization, where the
belief error e is independent of the shadow). One grid variant `policy_reads_stream = True` replaces
the read with the stream mean `mean_j(y_ijt - mu_j)/mean(lambda)` plus `N(0, var_read - 0.31)`, so the
policy reads its own shadow (closed-loop feedback). Prediction, stated in ramp form because under G1
the quantity fed back is the running average of the dose, not its terminal value: with
`x = (t - t*)/T_a`, `delta(x) = theta * (exp(u x) - 1)` instead of `theta * u * x`, so the terminal
shadow SD is amplified by `(exp(u) - 1)/u` (1.30 at u = 0.5, 1.72 at u = 1), not by `1/(1 - u)`.
The generator's own check at u = 0.5 gave 1.56 in variance against 1.68 predicted with drift and
read noise ignored.

Policy types:
- `ew` (greedy, default): exponentially weighted belief, `bhat_it = bhat_i,t-1 + K * (r_it - bhat_i,t-1)`,
  `K = gain_ew`, `bhat_i0 = 0`. Steady-state belief error variance `K/(2-K) * var_read`.
- `kalman` (learning): Kalman gain for a static trait, `P_0 = var_theta`, `K_t = P_{t-1}/(P_{t-1} + var_read)`,
  `P_t = (1 - K_t) P_{t-1}`; belief converges to `theta_i`.
- `heuristic` (S11): no filter; `bhat_it = (y_i1t - mu_1)/lambda_1` (yesterday's marker channel; reads the
  stream, so it carries the shadow and the analyst's own indicator noise).

Dose (logged): `a_it = 0` if `v_it = 0`; else `a_it = kappa * bhat_i,t-1 + xi_jit_it` where `xi_jit_it ~ N(0, var_jitter)`
is the logged randomized component (D2; `var_jitter = 0` by default). Uniform-help arm: `a_it = 1` for
all persons after the switch. Actions at t depend only on information through t-1.

### 2.3 Shadow generators (what the policy's actions do to the measurement)

`T_a = T - T0 = 42` normalizes exposure. With `t* = switch_i + 1`:
- G1 `cumulative` (default): `D_it = (1/T_a) * sum_{s = t*..t} a_is`, `delta_it = psi * D_it`. A linear
  ramp from 0 to about `psi * mean dose` over the adaptive phase.
- G2 `recency`: `R_it = sum_{s = t*..t} (1 - rho_r) rho_r^{t-s} a_is`, `rho_r = 2^(-1/halflife)`,
  `delta_it = psi * R_it`. Saturates to about `psi * mean dose` within two half-lives.
- G3 `hetero`: as G1 with `psi_i = psi * (1 + tau_psi * (rho_psitheta * theta_i/sd_theta + sqrt(1 - rho_psitheta^2) * w_i))`,
  `w_i ~ N(0,1)`; `Cov(psi_i, theta_i) = psi * tau_psi * rho_psitheta * sd_theta`.
- G4 `unlogged`: `delta_it = (c0 + c1 * theta_i) * 1[t >= t_cb]` with `t_cb = 36`, no dose-driven term
  (psi acts as 0). The change is not logged anywhere. G4b `unlogged_plus`: G1 term plus the G4 term.
- `none`: `delta = 0` (used when psi = 0).

`delta_it = 0` for `t <= switch_i` by construction in every generator.

### 2.4 Stream indicators (what the analyst sees)

`y_ijt = mu_j + lambda_j * theta_it + nu_j * delta_it + eps_ijt`, `eps_ijt ~ N(0, var_eps_j)`, j = 1..p.
Loading pattern of the shadow: `proportional` (`nu = lambda`, invisible to a one-factor CFA) or
`nonproportional` (`nu = (1.0, 1.0, 0.2, 0.2, 0.0)`, the policy hits two channels).

### 2.5 Criteria and outcome

- Clean criterion: `C_clean_i = lam_C * theta_i + N(0, var_C)`.
- Reactive (policy-downstream) criterion: `C_react_i = lam_C * theta_i + gamma_C * abar_i + N(0, var_C)`,
  `abar_i` = mean logged dose over the adaptive phase. Implied contamination share
  `r = gamma_C * kappa_eff / (lam_C + gamma_C * kappa_eff)`, `kappa_eff = kappa * Cov(bhat, theta)/Var(theta)`,
  computed from the generator's hidden truth and written per cell.
- True outcome (what the agent is for): `W_it = beta_W * theta_it + gamma_W1 * a_it + gamma_W2 * a_it * theta_it + N(0, var_W)`.
  Agent quality metric: `dW = mean(W adaptive phase) - mean(W adaptive phase under kappa = 0 with the same
  person and noise draws)`. "Genuinely good" means `dW > 0` with its Monte Carlo SE, reported per cell.
  The interaction term is why targeting beats uniform dose: `gamma_W2 * kappa * Cov(bhat, theta)`.

Sign condition (S10), stated here and in README: R2's inflation requires the policy's dose to enter the
score and the criterion with the same sign as theta enters each. `C_react` satisfies it by construction
(reactivity: more reminders, more self-reported deferral awareness). `C_clean` does not (attenuation
predicted). `W` has a compensating sign (`beta_W < 0`, help raises W): attenuation predicted. All three
are computed in every replicate; the pre-registered headline convergent statistic is against `C_react`,
and the abstract states this condition.

### 2.6 Frozen parameter table (S16)

| name | value | justification |
|---|---|---|
| N | 200 | synthesis phase 7 |
| T | 56 | synthesis phase 7 |
| T0 | 14 | two weeks of v0; person-mean reliability of the marker over 14 occasions = 14/(14+1) = .93 |
| var_theta | 1.0 | unit pi_0 scale |
| rho_d, sd_d | 0.9, 0.3 | slow AR drift, marginal SD 30% of the trait SD (synthesis: "trait + slow AR drift") |
| p | 5 | five stream channels (round 1, 7.1) |
| lambda | (1.0, 0.9, 0.8, 0.7, 0.6) | marker = 1 fixes the scale; realistic spread |
| mu | (0, 0, 0, 0, 0) | intercepts |
| var_eps | 1.0 each | noisy daily logs: single-occasion marker reliability .5 |
| nu (proportional) | = lambda | the CFA-invisible shadow |
| nu (nonproportional) | (1.0, 1.0, 0.2, 0.2, 0.0) | two channels hit hard, others weakly |
| nuisance | 3 indicators, loading 0.7, var_eps 1 | discriminant target |
| var_read | 2.0 | corr(bhat, theta) about .86 at steady state under `ew`: good but imperfect agent |
| gain_ew | 0.3 | effective window about 6 occasions |
| kappa grid | {0, 0.5, 1, 2} | adaptivity axis; 0 is the nonadaptive arm |
| psi grid | {0, 0.5, 1.0} headline; {0, .05, .1, .2, .3} ROC | u = psi*kappa spans 0 to 2 |
| gamma_C grid | {0, 1, 2} | criterion contamination; implied r reported |
| lam_C, var_C | 0.5, 0.75 | corr(theta, C_clean) = .5 |
| beta_W, gamma_W1, gamma_W2, var_W | -0.5, 0.2, 0.3, 0.25 | trait lowers the outcome; help adds; targeting benefit |
| halflife (G2) | 7 | one-week memory |
| tau_psi, rho_psitheta (G3) | 0.5, {0, .3, .6} | B3.3 axis |
| t_cb, c0, c1 (G4) | 36, 0.5, 0.3 | unlogged change three weeks after the logged switch |
| var_jitter (D2 arm) | 0.25 | logged randomized dose component |
| tolerance (primary) | 0.10 | absolute bias in Var(theta) (10% of truth) |
| smd_psi | 0.10 | smallest meaningful shadow for the equivalence decision |
| reps R | 200 (500 for the four headline cells) | MC SE on a rate at .5: .035 (.022) |
| bootstrap | B = 99 person-level, 3 cells x 100 reps | coverage audit (F17) |
| master seed | 20260916 | SeedSequence root |

The table is written as `trapworld/params.py` (a frozen dict) and hashed into `results/hashes.json`
before the grid runs.

## 3. What the analyst sees (data contract)

`World` (a plain dataclass or dict of numpy arrays), produced by `trapworld.world.simulate(cell, seed)`:

Observed (available to every analyzer):
- `y[N, T, p]`, `z[N, T, 3]`, `dose[N, T]`, `version[N, T]` (0/1), `switch[N]` (last occasion of v0),
  `jitter[N, T]` or None, `C_clean[N]`, `C_react[N]`, `W[N, T]`, `belief_logged[N, T]` (the policy's
  own person estimate; available ONLY to the forbidden `belief_partial` arm, which must be the only
  analyzer that reads it).

Hidden truth (metrics only): `theta[N]`, `theta_t[N, T]`, `delta[N, T]`, `psi_i[N]`, `belief[N, T]`,
`cell` (the resolved parameter dict), `implied_r`, `dW`.

Seeds: `SeedSequence(master).spawn(n_cells)[cell_index].spawn(R)[rep].spawn(9)` gives nine streams in
fixed order: theta, drift, eps, nuisance, read, jitter, criterion, outcome, bootstrap. The counterfactual
`kappa = 0` run for `dW` reuses the same streams.

## 4. Estimands and tolerances (B5.2)

- **Primary: E1 = Var(theta) on the pi_0 scale**, truth 1.0 (0 in the zero-person world). Tolerance
  ±0.10 absolute. Reported per analyzer: bias, RMSE, CI coverage where a CI exists.
  Why E1 and not beta: E1 is the project's own primary reverse estimand (round 1, 2.2) and the brief's
  "inflated person signal"; beta's truth under policy feedback is state-dependent (the W equation has an
  interaction), so beta is secondary with truth defined as `lam_C = 0.5` (slope of C_clean on theta).
- Secondary: `psi` (shadow responsiveness; truth per cell), `beta_clean` (slope of C_clean on the
  recovered theta; truth 0.5), person-level `corr(theta_hat, theta)` and `rmse(theta_hat, theta)` on the
  analyzer's claimed pi_0 scale.

## 5. Analyzers (all read only the observed contract; section 3)

| id | what it does | design it claims |
|---|---|---|
| `naive` | one-factor ML CFA on adaptive-phase person means of y; marker scaling; Bartlett scores; full validation battery (5.1) | none (conventional SEM) |
| `naive_pooled` | same on person means over all T | none |
| `epoch_fe` | subtract the cross-person mean of each indicator at each occasion (occasion fixed effects), then `naive_pooled` | S8 competitor (iii) |
| `belief_partial` | `naive` scores residualized on the policy's logged belief (mean over adaptive phase) | the forbidden rule (F13 positive control) |
| `dose_partial` | `naive` scores residualized on mean logged dose | the synthesis's original rule (K2) |
| `detrend` | per person, regress each indicator on occasion within the adaptive phase (after occasion FE) and take the fitted value at t*; CFA on those levels | S7 competent analyst |
| `anchor_only` | one-factor CFA on person means over `t <= switch_i` | D1 minimal |
| `agentsem` | D1 + D5: anchor CFA calibrates (mu, lambda, var_eps, Var(theta)); Bartlett scores for all occasions on the anchored calibration; state-space shadow model `eta_it = theta_i + d_it + psi * g_it + e_it` with `g` from the frozen surrogate set; certificate; Hessian CI and (audit cells) person bootstrap | D1 + D5 |
| `agentsem_d2` | when jitter is logged: within-person IV of `eta` on `g` with cumulative jitter as instrument; person scores `eta_bar - psi_hat * g_bar`; certificate with a weak-instrument gate | D2 |
| `misspecified` | as `agentsem` but with the wrong surrogate (recency-weighted dose, half-life 2) | deliberately misspecified alternative |

Frozen surrogate set S (hashed): `g1_it = (1/T_a) * sum_{s = t*..t} dose_is` (normalized cumulative logged
dose since the logged switch), 0 before the switch. Probe counts are NOT in S (S12). Version is design
(anchor), never a covariate.

Information the correction consumes beyond the logs (B5.1, stated in README): (1) the logged policy
version per occasion and the deployment's statement that version 0 applied the same dose to everyone
(exogenous anchor); (2) the logged dose path; (3) the parametric restriction that the shadow is a common
proportional shift linear in `g1` (D5). Nothing about the policy's belief.

### 5.1 Validation battery (naive analyst; all computed every replicate)

omega (McDonald), alpha, AVE, chi-square/df, CFI, RMSEA, SRMR; convergent r against C_react, C_clean and
mean W; incremental dR^2 of mean W on scores beyond nuisance scores; test-retest r (first vs second half
of the adaptive phase); discriminant |r| with nuisance scores; longitudinal invariance (two-occasion CFA,
anchor-window means vs adaptive-phase means, correlated factors, correlated same-indicator residuals):
configural, metric (loadings tied), scalar (intercepts tied, second latent mean free); report chi-square
difference p-values and dCFI. Both scaling conventions are reported from the same fit: marker-scale
Var(theta) and standardized loadings (F18).

Pass rule (pre-registered): omega >= .70 AND CFI >= .95 AND RMSEA <= .08 AND |r(C_react)| >= tau AND
test-retest >= .70 AND discriminant |r| <= .30, for tau in {.2, .3, .4}; reported with and without the
metric-invariance component (dCFI <= .01). Continuous statistics (mean, 5th, 50th, 95th percentiles) are
reported per cell alongside pass rates (S14).

### 5.2 Certificate rules (frozen as `trapworld/rules.json`, hashed; logic in `certificate.py`)

Two claims are certified separately.
- Claim A: "Var(theta) on the pi_0 scale". Claim B: "adaptive-phase person scores are on the pi_0 scale
  after correction".
- R1 provenance: version missing for any occasion -> both `unknown`.
- R2 anchor: an epoch with version 0 exists, dose constant across persons within it, length >= 7 for
  every person; else A `unsupported`, B `unsupported`. Switch-time exogeneity: if `switch_i` varies
  across persons and |corr(switch, anchor score)| > .10 with p < .01 -> A `bounded`, B `bounded`.
- R3 conditioning: condition number of the stacked design [1, 1[v=1], g1] > 30 -> B `bounded`. Always
  report: that condition number, the between-person variance share of `g1_bar`, and R^2 of `g1_bar` on the
  anchor score (informational; high values are expected under D1 and are not disqualifying).
- R4 unexplained shift: after the shadow model, person-clustered mean residual (adaptive minus anchor)
  exceeds 0.15 SD of the anchor score, or a change-point scan over occasions 3..T_a-3 of the adaptive
  phase finds a residual mean shift exceeding 0.15 SD -> B `unsupported`.
- R5 loading pattern: metric-invariance dCFI > .01 across anchor vs adaptive (same test as 5.1) -> B
  `unsupported` (the proportional-shift model is wrong).
- R6 heterogeneity: variance of person-level adaptive-mean residuals divided by the anchor analogue
  > 1.5 -> B `bounded`.
- D2 gate (agentsem_d2 only): jitter not logged -> `unsupported`; first-stage F < 10 -> `bounded`.
- Precedence: unknown > unsupported > bounded > identified. Output per claim: status, estimate (A:
  anchor CFA Var(theta) with CI when identified or bounded; B: shadow-model Var(theta) and psi with CI
  only when identified; `null` estimate when unsupported or unknown), the triggered rules, and the numbers
  behind every rule.

## 6. Grid (frozen as `trapworld/grid.py`, hashed; S13)

Base cell B: N 200, T 56, T0 14, kappa 1, psi 0.5, gamma_C 1, policy `ew`, generator G1 proportional,
sd_d 0.3, calendar trigger, no jitter, policy_reads_stream False.

| sweep | cells | what it tests |
|---|---|---|
| A headline | kappa {0,.5,1,2} x psi {0,.5,1} x gamma_C {0,1,2} = 36 | R1, R2, R3, R6 (innocence), false confidence in kappa and gamma_C |
| B misspecification | at B: G2; G3 rho {0,.3,.6}; G4; G4b = 5 | R4 (B4.1 grid) |
| C loading pattern | nonproportional at B and at psi 0 = 2 | R5 invariance (F10) |
| D policy | `kalman` at B and at kappa {.5, 2}; `heuristic` at B = 4 | S11, "learning" level |
| E trigger | performance trigger at B = 1 | S1/F15 |
| F specificity ROC | psi {0,.05,.1,.2,.3} x kappa {.5,1,2}, gamma_C 1 = 15 | R6 ROC (S4) |
| G special | uniform help (a = 1, psi .5); zero-person (var_theta 0) at B; no-drift at B; D2 jitter at B; feedback (policy_reads_stream) at psi {.25,.5}, kappa 1 = 6 | R7, R8, closed-form comparability, D2, closed loop |

Total 69 cells x R = 200 (500 for kappa 1 x psi {0,.5} x gamma_C {0,1}). Bootstrap coverage audit: B,
G2, G3 rho .3, 100 reps x B = 99. Runtime budget: 90 minutes on 12 cores; if a replicate exceeds 3 s the
runner halves R and records it in `run_manifest.json`. Monte Carlo SE is reported for every rate and for
every between-arm difference (paired by replicate seed where arms share draws).

## 7. Metrics (per replicate, aggregated per cell with mean, SD, MC SE, quantiles)

- Per analyzer: `var_theta_hat`, bias, CI lower/upper and covers-truth (where available), `psi_hat`,
  `beta_clean_hat`, person `corr`, person `rmse`, the battery statistics of 5.1, pass flags at each tau.
- Naive false confidence: battery passes AND |bias| > tolerance. Reported per tau and with/without the
  invariance component.
- Certificate: status per claim; false confidence = `identified` AND |bias| > tolerance; false refusal =
  status in {bounded, unsupported} AND the corrected estimate would have been within tolerance; coverage
  of the CI when identified. Rates reported as a pair on one axis.
- Specificity: TOST on psi_hat with 90% CI inside (-smd_psi, +smd_psi) = innocent. Innocent-kill rate at
  psi 0; detection rate (CI excludes 0) at psi > 0.
- Agent quality: dW with MC SE, per cell.
- Implied r per cell from hidden truth.

## 8. Falsification map (review F1 to F18 plus steering)

| test | where | kills | predicted (review) |
|---|---|---|---|
| F1 | closed_forms | K1 | both correlations fall in u |
| F2 | closed_forms + grid sweep A | R1 | omega up, corr down |
| F3 | closed_forms + sweep A (implied r) | R2 shape | hump, peak u* = r/(1-r), ceiling (1+r)/sqrt(2) |
| F4 | closed_forms | K2 | R^2 .50/.875/.966/.982 |
| F5 | closed_forms + `dose_partial` in sweep A | K2 | near-100% innocent kills |
| F6 | certificate R3 numbers, every cell | certificate honesty | numbers emitted; D1 route does not depend on positivity |
| F7 | sweep B G4 | certificate | claim B `unsupported`; a point estimate here kills R4 |
| F8 | tests/test_world.py permutation null | generator | Cov(delta, theta) vanishes when dose paths are permuted across persons |
| F9 | sweep G zero-person | nothing (demo) | naive omega >= .7 and r(C_react) > 0; certificate refuses or bounds |
| F10 | sweep C | "standard validation passes" | proportional: passes; nonproportional: metric invariance fails |
| F11 | `detrend` in sweeps A and B | trap robustness | repairs G1, not G2 |
| F12 | `epoch_fe` in sweeps A and B | framework value | does not repair; agentsem beats it beyond MC error under G2 and G3 too, else R4 narrows |
| F13 | `belief_partial` | rule demo | over-corrects toward zero |
| F14 | sweep B G3 | homogeneous form | bias grows with rho_psitheta; certificate `bounded` |
| F15 | sweep E | anchor exogeneity | R2 flags; anchor estimate biased when the trigger reads the stream |
| F16 | sweep D heuristic | closed form | belief error correlated with indicator noise; form differs |
| F17 | coverage audit | "recovers truth" | Hessian CI under-covers; bootstrap near nominal |
| F18 | standardized outputs, every cell | framing | correlations unchanged; Var inflation appears as standardized-loading inflation |

## 9. Outputs

- `results/closed_forms.json` (Task 1).
- `results/hashes.json`: sha256 of canonical JSON of params, grid, surrogate set, rules.json, and of
  `certificate.py` source; written before the grid runs and re-checked after.
- `results/grid/<cell_id>.json`: cell parameters, per-replicate records, aggregates.
- `results/summary.csv` and `results/summary.json`: one row per cell x analyzer.
- `results/coverage_audit.json`, `results/run_manifest.json` (seeds, versions, timing, R per cell).
- `plots/*.png` regenerated from results only (see plan): headline crossing, F3 hump, competitor bias
  by generator, certificate outcomes with the false-confidence/false-refusal pair, specificity ROC,
  zero-person world, F4 purity, coverage audit. Okabe-Ito palette, meaning also carried by marker
  shape and text labels (colour-blind reader).
- `README.md`, `HANDOFF.md` (four sections: original claims killed; replacement result supported;
  unresolved identification gaps; what must run next), `REVIEW-RESPONSE.md` (every B, S, F and
  checklist item: fixed where, or open why).

## 10. Success criteria

- SC1 `pytest` green; F1 to F5 JSON written with verdicts.
- SC2 Full grid runs end to end from one command, deterministic under reordering (test).
- SC3 Under G1 at B, `agentsem` |bias| <= .10 and bootstrap coverage in [.90, .99]; `naive` bias > .10.
- SC4 Under G4, `agentsem` claim B is `unsupported` in >= 95% of replicates.
- SC5 psi = 0 innocence >= 90% at every kappa; naive false confidence at psi 0 <= 5%.
- SC6 Every review item has a line in REVIEW-RESPONSE.md.
- SC7 HANDOFF.md separates killed / supported / open / next.

## Assumptions

- Linear-Gaussian world is sufficient to settle the claims (the review's own algebra is linear).
- Occasions are days; no missingness; no ESM compliance mechanism (S10 position 3 recorded as open).
- The analyst's naive arm analyses the adaptive phase (the deployed system); the pooled variant is secondary.
- Bootstrap only in three audit cells for runtime; Hessian CIs elsewhere, flagged as anti-conservative.

## 11. Amendments recorded before the grid (2026-09-16 evening, before `results/hashes.json` exists)

Each amendment was made on the basis of a synthetic test or a design constraint, not on grid results, and
is dated here so the pre-registration record is complete.

- **A1 Correlated random effects in the shadow model (section 5, `agentsem`).** `tests/test_shadow.py`
  showed the random-effects likelihood biased (psi +0.06, phi the mirror image) whenever the person-level
  mean of the surrogate is correlated with theta, which is the trap structure itself; on the real base
  replicate the fit gave phi 0.895, psi 0.576. The policy's dose is informative about theta beyond the
  stream (the read is private), so p(theta | g) is not N(0, phi) and the sequential-likelihood argument
  does not apply between persons. The model becomes `eta_it = theta_i + d_it + psi * g_it + b * gbar_i +
  e_it` with `gbar_i` the person's mean surrogate over included occasions and `theta_i ~ N(0, phi_u)`
  independent of `gbar_i` (Mundlak). psi is identified from within-person variation of the logged exposure
  path only; `Var(theta) = phi_u + b^2 Var(gbar)`; person scores add `b * gbar_i`. When exposure does not
  vary between persons (kappa 0, uniform help) the term is dropped; when g is identically zero psi is
  reported as not identified and the specificity statistics as not applicable. The within-person
  identification still rests on the drift not feeding back into the dose; `G_nodrift` isolates that
  channel and the base cell measures it.
- **A2 R4 gains an exposure-pattern test (section 5.2).** Every shadow in this world is signed by the
  person, so pooled mean shifts cannot see a mis-specified surrogate shape. D5's assumption
  `E[resid_it | exposure_i] = 0` is tested directly: adaptive occasions split into three equal windows per
  person; the correlation over persons between window-mean residual and `gbar_i` is computed per window;
  the smallest two-sided p below `exposure_p_max = 0.01` makes claim B `unsupported` under rule R4. The
  statistic is undefined (and the rule silent) when `gbar` has no variance. `rules.json` carries the two
  new numbers.
- **A3 Optional Jev policy arm (sections 2.2 and 6; JEV-CONSTRAINT-20260916.md).** A new policy value
  `external` lets the runner supply the dose from a callable that sees only past stream values, past doses,
  the switch and the adaptive mask. The Jev policy (`trapworld/jev.py`) asks one Score question per person
  every 7 occasions from a bounded 7-day state and maps the probability-weighted level to a dose on the
  ew policy's scale. Latent truth, shadow, criteria and outcome are untouched; the counterfactual for dW is
  the no-dose policy; `implied_r` uses `Cov(abar, theta)/Var(theta)`. Sweep J holds `J_fake` (fake
  transport, 50 reps, a plumbing check that is not a Jev result) and `J_live` (5 reps, runs only in a
  separately orchestrated live session with an ephemeral credential; recorded as not run otherwise). Jev
  cells are excluded from every headline aggregate. A failed call makes the replicate missing, never
  substituted.
- **A4 Bootstrap fits** warm-start from the point estimate and skip the Hessian; the percentile interval
  is the bootstrap CI (F17).
- **A5 Sweep J cells and the Jev policy configuration are hashed into `results/hashes.json`** alongside the
  frozen objects of section 9.
- **A6 Runtime halving rule relaxed (section 6), 2026-09-16 22:40.** The 3 s per-replicate rule halved
  cells during the first grid attempt because unrelated processes were loading the machine (replicates
  cost 1.3 s in one process and 2.0 s on an idle pool); the threshold is 8 s for the run that produced
  `results/`, recorded as `slow_seconds` in `run_manifest.json`. This changes how long the grid takes,
  not what it computes: replicate seeds are fixed by position, so no replicate's data depends on the
  rule.
