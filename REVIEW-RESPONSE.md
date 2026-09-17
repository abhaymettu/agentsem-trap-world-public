# Response to HARSH-REVIEW.md, item by item

Status legend: **fixed** (where, and the evidence), **open** (why it remains open and what would close it),
**by design** (resolved in the spec before code). Result-dependent entries cite `results/`; they are filled
after the grid run. Pre-registered design in `specs/001-trap-world/spec.md`.

## Blocking defects

| item | status | where / evidence |
|---|---|---|
| B1 inflation claim is scale algebra | fixed | F1 in `trapworld/closed_forms.py`, `results/closed_forms.json` (verdict field). K1 removed from the demonstration; R1 (B1.3) and R2 (B1.4) are the pre-registered replacements (spec 1.2). |
| B1.1 pre-register the derivation as a unit test | fixed | `tests/test_closed_forms.py` asserts corr decreasing and Var ratio increasing in u. |
| B1.2 both scaling conventions | fixed | every naive fit reports marker-scale Var(theta) and standardized loadings (spec 5.1, F18). |
| B1.3 reliability up, validity down | fixed | F2 closed form + sweep A; `P9_closed_forms.png`, `P1_headline.png`. RESULT: _filled after run_ |
| B1.4 criterion contamination with condition | fixed | F3 closed form; `gamma_C` swept with implied r recorded per cell; sign condition stated in spec 2.5 and README. RESULT: _filled after run_ |
| B2 partial cumulative dose deletes the person | fixed | F4/F5 closed forms; `dose_partial` kept only as a labelled competitor. K2 removed as the correction. |
| B2.1 name the design | fixed | D1 (exogenous version-0 anchor epoch) + D5 (frozen parametric surrogate) as `agentsem`; D2 (logged randomized dose component) as `agentsem_d2`. Spec 5. |
| B2.2 exogenous surrogate + overlap check, `unsupported` on failure | fixed | certificate R2/R3/R4/R5/R6 and D2 gate in `trapworld/rules.json`, `trapworld/certificate.py`. |
| B2.3 narrow statement of the rule | fixed | README states: condition only on action variation exogenous to the person, proven from the design (the anchor or the logged randomized component). |
| B3 not an instance of Theorem 1 | by design | README states this is a logged-action world and Theorem 1 does not bite; the finding is "ordinary validation does not notice a fully logged driver". |
| B3.2 heterogeneous psi nonidentification | partly | G3 cells with rho_psitheta in {0, .3, .6}. Rule R6 as pre-registered cannot fire at these parameters: its ratio sits below 1 in every test world (0.26 to 0.96) because the person effect is estimated from the residual the numerator uses, so heterogeneous responsiveness is not detectable from the residual variance at N = 200 and tau_psi = 0.5. The G3 cells therefore report the bias the identified-set problem produces, with the certificate silent on it. The formal identified-set derivation is open (HANDOFF). |
| B3.3 homogeneous vs person-specific psi | fixed | `tau_psi`, `rho_psitheta` are grid factors. |
| B4 circular validation | fixed | misspecification grid G1..G4c against one frozen surrogate (spec 6 sweep B); surrogate set hashed before the run (`results/hashes.json`). |
| B4.1 G4 must return `unsupported` | fixed | rule R4; kill test F7 = SC4 (>= 95% `unsupported` under B_G4). RESULT: _filled after run_ |
| B4.2 freeze the surrogate set | fixed | `surrogate_hash` in `results/hashes.json`, written before the grid and re-checked after. |
| B5 estimand drift | fixed | one primary estimand E1 = Var(theta) on the pi_0 scale, tolerance 0.10; secondary psi, beta_clean, person corr/rmse (spec 4). Deviation from the review's beta recommendation recorded there. |
| B5.1 name the extra information | fixed | spec 5 "Information the correction consumes"; README. |
| B5.2 bias, RMSE and bootstrap coverage | fixed (scoped) | Hessian CIs everywhere; full-pipeline person bootstrap in three audit cells (F17). RESULT: _filled after run_ |

## Serious defects

| item | status | where / evidence |
|---|---|---|
| S1 embargo exogeneity | fixed | `trigger` in {calendar, performance}; certificate R2 flags switch times correlated with anchor scores. RESULT: _filled_ |
| S2 positivity never checked | fixed (reframed) | R3 emits the condition number, between-person share of the surrogate and R^2 on the anchor score every run. Under D1 + D5 identification does not rest on positivity, so the threshold gates B to `bounded`, not `unsupported`; the D2 arm has the propensity-style gate (first-stage F). |
| S3 benign arm | fixed | benign = psi 0 with kappa > 0 and dose-theta coupling preserved (sweep A, psi 0 cells). The "invariant bank" wording is dropped. |
| S4 equivalence, ROC | fixed | TOST on psi_hat against smd 0.10; sweep F ROC over psi. RESULT: _filled_ |
| S5 false-refusal rate | fixed | reported as a pair with false confidence per cell (`P4_certificate.png`). |
| S6 invariance testing | fixed | two-occasion longitudinal CFA (configural/metric/scalar) in the battery; kill test F10 via sweep C; pass rates reported with and without the invariance component. RESULT: _filled_ |
| S7 competent preprocessing | fixed | `detrend` analyzer (occasion FE + person-specific linear trend, level at the switch); F11. RESULT: _filled_ |
| S8 competitor set | fixed | naive, naive_pooled, epoch_fe, belief_partial, dose_partial, detrend, anchor_only, agentsem, agentsem_d2, misspecified. F12 paired difference agentsem minus epoch_fe with MC SE. Settled by arithmetic before the grid: occasion fixed effects subtract the cross-person mean of each indicator at each occasion, and the person-signed shadow has zero cross-person mean, so `epoch_fe` returns exactly `naive_pooled` (`tests/test_analyzers.py` pins the identity). RESULT: _filled_ |
| S9 gamma and psi independent; equal-help arm | fixed | gamma_C x psi x kappa full sweep; `G_uniform` cell (R7). RESULT: _filled_ |
| S10 criterion position | fixed (2 of 3) | clean and policy-downstream criteria in every replicate, plus the true outcome W; the compliance/selection position is open (HANDOFF). |
| S11 heuristic and learning policies | fixed | `heuristic` and `kalman` policy cells (sweep D). RESULT: _filled_ |
| S12 probe counts | fixed | not in the surrogate set; version is design (anchor), never a covariate. |
| S13 MC design | fixed | 71 pre-registered cells, R per cell, runtime budget and halving rule in spec 6; MC SE on rates and on paired differences. Multiplicity: the five hypotheses are reported as estimates with MC SE, not as a family of significance tests. |
| S14 thresholds | fixed | continuous statistics with quantiles per cell alongside pass rates at tau .2/.3/.4. |
| S15 leakage: encoder | by design | the stream summary is a fixed Bartlett composite on the anchor-calibrated loadings, not learned on adaptive logs. |
| S15 leakage: seeds | fixed | nine spawned streams per replicate; reorder/subprocess determinism tests. |
| S16 parameter table | fixed | spec 2.6 with a justification per parameter; `trapworld/params.py`; hashed. |

## Amendments made before the grid (spec section 11, dated 2026-09-16)

| item | status | where / evidence |
|---|---|---|
| A1 correlated random effects in the shadow model | fixed | `tests/test_shadow.py` showed the random-effects likelihood biased whenever exposure correlates with theta (psi +0.06 with the mirror image in `Var(theta)`; on the real base replicate psi 0.576, `Var(theta)` 0.895). The policy reads the person privately, so its actions carry information about theta beyond the stream and the sequential-likelihood argument fails between persons. `fit_shadow(between=True)` adds `b * gbar_i`; base replicate now `Var(theta)` 0.960, psi 0.504. The remaining assumption is that drift does not feed back into the dose (`G_nodrift` isolates it). |
| A2 R4 exposure-pattern test | fixed | Pooled shift tests cannot see a mis-specified surrogate when every shadow is signed by the person. Window-mean residual vs `gbar_i`, p < .01, `rules.json`. Fires on G2 and on the wrong surrogate (p below 1e-40 at N = 1000); on the matched G1 world its signal is the drift-feedback channel (window correlations .06 to .08, vanishing with sd_d = 0), 0 of 20 refusals at N = 200. |
| A3 optional Jev policy arm | fixed | `trapworld/jev.py`, `JEV-LIVE-RUN.md`; fake transport only in this repo; no live call made. |
| A4 bootstrap warm start | fixed | `analyzers.py::_bootstrap`; 0.29 s per bootstrap fit. |
| A5 Jev objects hashed | fixed | `results/hashes.json` carries `jev_policy_hash`. |

## Falsification suite

| test | status | evidence |
|---|---|---|
| F1 | run, review confirmed | `results/closed_forms.json`: corr(eta,theta) 1.000/.981/.949/.894/.832/.747 and corr(eta,C) .447/.439/.424/.400/.372/.334 over u = 0/.25/.5/1/2/8, both strictly falling; only Var(eta)/Var(theta) rises. K1 killed. |
| F2 | run, review confirmed | omega .833/.926/.962 against corr 1.000/.949/.894 at u = 0/.5/1 (B1.3 table reproduced to 3 dp; MC alpha .833/.926/.962). Sweep A: _filled_ |
| F3 | run, review confirmed | inflates for all u exactly for r in {.4142, .6, 1}; monotone rise over [0, 2] only at r = 1; peaks at 0/.25/.71/1.5/8 = r/(1-r); ceiling 1.414. Seven cells of the review's B1.4 table are off by 1 in the third decimal against its own formula (e.g. 1.388 printed, 1.387 exact). |
| F4 | run, review confirmed | R^2(dose, theta) = .500/.875/.966/.982 at T = 1/7/28/56; attenuation .707/.354/.186/.132 (review printed .133). K2 killed. |
| F5 | run, review confirmed | benign arm (psi 0) killed by the dose-partial rule at .996/.992/.998 for kappa .5/1/2 at threshold .3 (MC SE about .003); conditional on the raw correlation clearing the threshold the kill rate is 1.000 in all nine cells. The synthesis's "<= 5%" is refuted. Grid `dose_partial`: _filled_ |
| F6 | run | R3 numbers in every certificate record. |
| F7 | run | B_G4 status_B: _filled_ |
| F8 | run | `tests/test_world.py` permutation null. |
| F9 | run | `G_zero_person`: _filled_ |
| F10 | run | sweep C: _filled_ |
| F11 | run | `detrend` G1 vs G2: _filled_ |
| F12 | run | agentsem minus epoch_fe under G1/G2/G3: _filled_ |
| F13 | run | `belief_partial`: _filled_ |
| F14 | run | G3 cells: _filled_ |
| F15 | run | `E_perf`: _filled_ |
| F16 | run | `D_heuristic`: _filled_ |
| F17 | run | `results/coverage_audit.json`: _filled_ |
| F18 | run | standardized loadings in every naive record. |

## Required spec edits (checklist 1 to 14)

1 fixed (spec 1.1/1.2). 2 fixed (spec 5). 3 fixed (README, spec 1.1 K3; heterogeneous psi as G3). 4 fixed
(spec 5). 5 fixed (spec 4). 6 fixed (spec 6; r reported per cell rather than set directly, closed-form F3
sweeps r directly). 7 fixed (spec 5.1, F10). 8 fixed (`detrend`, `epoch_fe`). 9 fixed (metrics). 10 fixed
(TOST + ROC). 11 fixed (psi 0 benign arm). 12 fixed (surrogate set). 13 fixed (spec 2.6). 14 fixed
(`results/hashes.json`).

## What the reviewer asked to see, in order

1. F1 to F5 before the generator: `results/closed_forms.json`, produced first.
2. F7 returning `unsupported`: _filled_
3. F10 reported honestly: _filled_
4. F12 margin under G2 and G3: _filled_
