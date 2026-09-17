# Response to HARSH-REVIEW.md, item by item

Status legend: **fixed** (where, and the evidence), **open** (why it remains open and what would close it),
**by design** (resolved in the spec before code). Numbers come from `results/summary.csv` (grid of
2026-09-17 00:14, commit `6a916ef`, 200 replicates per cell, 500 in the four heavy cells; Monte Carlo SE
of a rate at 0.5 is 0.035 and 0.022, of a bias mean about 0.005 to 0.013) and from
`results/closed_forms.json`. Pre-registered design: `specs/001-trap-world/spec.md`; amendments dated before
the grid in its section 11; the methods audit is `AUDIT.md`.

## Blocking defects

| item | status | where / evidence |
|---|---|---|
| B1 inflation claim is scale algebra | fixed | F1 in `trapworld/closed_forms.py`. K1 removed from the demonstration; R1 (B1.3) and R2 (B1.4) are the pre-registered replacements (spec 1.2). |
| B1.1 pre-register the derivation as a unit test | fixed | `tests/test_closed_forms.py` asserts corr decreasing and Var ratio increasing in u. |
| B1.2 both scaling conventions | fixed | every naive fit reports marker-scale Var(theta) and standardized loadings (spec 5.1, F18); AVE rises .961 / .975 / .983 at u_eff 0 / .26 / .51 while the person correlation stays .981 / .982 / .982 (`A_k1.0_p{0,0.5,1.0}_g1.0`). |
| B1.3 reliability up, validity down | fixed, supported | naive omega .993 / .994 / .995 / .997 / .998 across kappa x psi as u rises from 0 to 2, naive Var(theta) bias +.03 / +.30 / +.61 / +1.34 / +3.25, person corr .981 / .980 / .977: reliability and the reported variance rise, the rank of persons survives (Corollary 1.1). |
| B1.4 criterion contamination with condition | fixed, supported in direction, negligible in size | ratio corr(score, C_react) / corr(theta, C_react) = 1.005 to 1.008 at psi .5 and 1.011 to 1.020 at psi 1 (gamma_C 1 and 2; implied r .50 to .89), against 0.98 for the clean criterion (attenuation). The closed-form hump needs a belief error as large as the trait; the world's phase-averaged belief error is small, so the inflation is 1 to 2%. |
| B2 partial cumulative dose deletes the person | fixed | F4/F5 closed forms; in the grid `dose_partial` person corr .18 in every kappa > 0 cell (.98 for naive) and `belief_partial` Var(theta) bias -.95. |
| B2.1 name the design | fixed | D1 (exogenous version-0 anchor) + D5 (frozen surrogate, correlated-random-effects form per A1) as `agentsem`; D2 (logged randomized dose) as `agentsem_d2`. On `G_d2` the D2 arm is unbiased in the mean (+.088, MC SE .026) with interval coverage .89, but its estimate has SD .365 and its psi SD .51 at N = 200: valid, not useful at this size. |
| B2.2 exogenous surrogate + overlap check, `unsupported` on failure | fixed | certificate R2 to R6 and the D2 gate in `trapworld/rules.json`; `B_G4` `unsupported` in 200 of 200. |
| B2.3 narrow statement of the rule | fixed | README: condition only on action variation exogenous to the person, proven from the design. |
| B3 not an instance of Theorem 1 | by design | README states this is a logged-action world. |
| B3.2 heterogeneous psi nonidentification | partly | G3 cells: `agentsem` bias +.001 / +.020 / +.014 for rho_psitheta 0 / .3 / .6 (MC SE .008), certified identified in .94 / .99 / .92. R6 never fires (ratio below 1 everywhere, `task-04-report.md` 6.1); at tau_psi .5 the heterogeneity adds variance without a mean shift, so the identified-set problem is small here and invisible to the certificate. The formal identified-set derivation is open. |
| B3.3 homogeneous vs person-specific psi | fixed | `tau_psi`, `rho_psitheta` are grid factors (sweep B). |
| B4 circular validation | fixed | misspecification grid G1 to G4c against one frozen surrogate; surrogate hash in `results/hashes.json`, written before the run and re-checked after (`hashes_match` true). |
| B4.1 G4 must return `unsupported` | fixed, met | `B_G4` and `B_G4b`: claim B `unsupported` in 200 / 200 replicates each (R4). `B_G4c` (c0 = 0, the pre-registered undetectable limit): identified .86, refused .14, Var(theta) bias -.015 on the identified replicates. |
| B4.2 freeze the surrogate set | fixed | `surrogate_hash` in `results/hashes.json`. |
| B5 estimand drift | fixed | one primary estimand E1 = Var(theta) on the pi_0 scale, tolerance .10 (spec 4). |
| B5.1 name the extra information | fixed | spec 5 and README; amended by A1 (the correlated-random-effects assumption is a fourth piece of information and is named). |
| B5.2 bias, RMSE and bootstrap coverage | fixed (scoped) | Hessian interval coverage .55 to .61 in the headline cells (n 200 to 500), .39 under G2; the full-pipeline person bootstrap (`results/coverage_audit.json`, 100 replicates x 99 resamples per cell) covers .94 at the base cell and .93 under G3 (MC SE .025) at 2.4 times the width, .84 under G2 where claim B is refused anyway. |

## Serious defects

| item | status | where / evidence |
|---|---|---|
| S1 embargo exogeneity | fixed | `E_perf`: R2 flags the switch from the log alone in .94 of replicates (bounded); the anchor estimate it falls back to has bias +.064 against +.05 in calendar cells. |
| S2 positivity never checked | fixed (reframed) | R3 emits the condition number, the between-person share of the surrogate and its R^2 on the anchor score in every record; under D1 + D5 the threshold gates B to `bounded`; the D2 arm has the first-stage F gate. |
| S3 benign arm | fixed | benign = psi 0 with kappa > 0 and dose-theta coupling preserved (nine sweep A cells, five sweep F cells). |
| S4 equivalence, ROC | fixed, partly supported | innocence (TOST inside +-.10) at psi 0: .06 to .09 / .83 to .88 / 1.00 for kappa .5 / 1 / 2 across the psi-0 cells (the equivalence margin is tighter than psi's precision at kappa .5); detection at psi .05 / .1 / .2 / .3: kappa .5 .07 / .24 / .85 / .99, kappa 1 .17 / .82 / 1.00 / 1.00, kappa 2 .81 / 1.00 / 1.00 / 1.00; false detection at psi 0 .06 to .12. |
| S5 false-refusal rate | fixed | pre-registered pair per cell plus a strict variant (`false_refusal_strict`, `unsupported` only) and post-run `_sample` twins (A7). At the base cell: false confidence .38 pre-registered, .04 against the realized variance; false refusal .03. |
| S6 invariance testing | fixed, narrowed claim | proportional shadow: the two-occasion metric test passes (dCFI .0000 to .0001, p about .48) and the pass rate with the invariance component equals the pass rate without it (.93 to .98) in every proportional cell; nonproportional shadow (`C_nonprop`): dCFI .0135, p < .001, pass rate .94 without and .15 with the invariance component. Ordinary invariance testing catches the shadow only when its loading pattern differs from the trait's. |
| S7 competent preprocessing | fixed | `detrend` bias +.11 at the base cell and +.14 at psi 1 (repairs most of a linear ramp), +.58 under G2 (does not repair a saturating shadow), -.07 under G4. |
| S8 competitor set | fixed | ten analyzers. `epoch_fe` equals `naive_pooled` by arithmetic (occasion fixed effects remove only the cross-person mean, and the person-signed shadow has none): bias +.44 at the base cell, +.71 under G2, +.45 to +.47 under G3. Paired `agentsem` minus `epoch_fe` under G3: -.45 (MC SE .005). |
| S9 gamma and psi independent; equal-help arm | fixed | gamma_C x psi x kappa full sweep; `G_uniform`: naive bias +.036, agentsem +.006, psi recovered .498 (R7 supported: personalized help, not help, creates the trap). |
| S10 criterion position | fixed (2 of 3) | clean and policy-downstream criteria plus the true outcome W in every replicate; compliance / selection position open. |
| S11 heuristic and learning policies | fixed | kalman: naive +.28 / +.58 / +1.21 at kappa .5 / 1 / 2, agentsem +.028 / +.033 / +.025; heuristic (single channel, reads the stream): naive +.77, agentsem +.050 (identified .89). |
| S12 probe counts | fixed | not in the surrogate set; version is design. |
| S13 MC design | fixed | 73 pre-registered cells, R per cell, MC SE on every rate and on the paired differences; no pooled statistic across cells (`run_grid.py` writes one row per cell and analyzer). |
| S14 thresholds | fixed | continuous statistics with quantiles per cell; pass rates at tau .2 / .3 / .4 are .91 to .98 in every cell with a person and 0 in the zero-person world. |
| S15 leakage: encoder | by design | fixed Bartlett composite on the anchor-calibrated loadings. |
| S15 leakage: seeds | fixed | nine spawned streams per replicate; reorder, subprocess and master-seed tests (`tests/test_grid.py`, `tests/test_world.py`). |
| S16 parameter table | fixed | spec 2.6; `trapworld/params.py`; hashed. |

## Amendments made before the grid (spec section 11, dated 2026-09-16)

| item | status | where / evidence |
|---|---|---|
| A1 correlated random effects in the shadow model | fixed | `tests/test_shadow.py` showed the random-effects likelihood biased whenever exposure correlates with theta (psi +0.06 with the mirror image in Var(theta); on the real base replicate psi 0.576, Var(theta) 0.895). `fit_shadow(between=True)` adds `b * gbar_i`; on the grid `agentsem` bias is +.000 to +.021 in every G1 cell with a dose. In the base cell the fitted `b` is 2.45 with `phi_u` .046: the person effect is almost entirely a linear function of exposure, which is F4's point turned into an estimator. |
| A2 R4 exposure-pattern test | fixed | fires in 200 / 200 replicates under G2 and for the wrong surrogate; in matched G1 cells it refuses .04 to .09 of replicates, .04 without drift (`G_nodrift`). |
| A3 optional Jev policy arm | fixed | `trapworld/jev.py`, `JEV-LIVE-RUN.md`; `J_fake` ran (50 replicates), `J_live` recorded as not run; no live call made. |
| A4 bootstrap warm start | fixed | `analyzers.py::_bootstrap`. |
| A5 Jev objects hashed | fixed | `results/hashes.json` carries `jev_policy_hash`. |
| A6 runtime halving rule relaxed | fixed | `run_manifest.json` records `slow_seconds` 8; no cell was halved; 85.7 minutes on 10 workers. |
| A7 derived sample-truth flags (post-run, exploratory) | fixed | `trapworld/derive.py`; every `_sample` field is labelled exploratory. |

## Falsification suite

| test | status | evidence |
|---|---|---|
| F1 | run, review confirmed | corr(eta, theta) 1.000 / .981 / .949 / .894 / .832 / .747 and corr(eta, C) .447 / .439 / .424 / .400 / .372 / .334 over u = 0 / .25 / .5 / 1 / 2 / 8. K1 killed. |
| F2 | run, review confirmed | closed form omega .833 / .926 / .962 against corr 1.000 / .949 / .894 (B1.3 table to 3 dp). Sweep A: omega .993 to .998 and Var(theta) bias +.03 to +3.25 rising in u while person corr .981 to .977. |
| F3 | run, review confirmed | closed form: inflates for all u exactly for r in {.4142, .6, 1}; peaks at r / (1 - r); ceiling 1.414. Grid: ratio 1.005 to 1.020 for the downstream criterion, .98 for the clean one. |
| F4 | run, review confirmed | R^2(dose, theta) = .500 / .875 / .966 / .982 at T = 1 / 7 / 28 / 56. K2 killed. |
| F5 | run, review confirmed | closed form: benign arm killed at .996 / .992 / .998 for kappa .5 / 1 / 2. Grid `dose_partial`: person corr .18 wherever kappa > 0. |
| F6 | run | R3 numbers in every certificate record; condition number 4.8 to 5.8 with a dose, finite without one. |
| F7 | run, met | `B_G4` claim B `unsupported` 200 / 200; `B_G4b` 200 / 200; SC4 met. |
| F8 | run | `tests/test_world.py` permutation null. |
| F9 | run, narrowed | `G_zero_person`: naive omega .878, alpha .87, CFI .997, RMSEA .02, r(C_react) .235, r(C_clean) -.005, test-retest .42 (5th to 95th percentile .32 to .51) fails the .70 rule, so the pre-registered battery passes in 0 of 200; `agentsem` refuses in .88 and reports Var(theta) .010 (truth 0) in the rest. |
| F10 | run, narrowed | proportional shadow invisible to metric invariance (dCFI .0001); nonproportional caught (dCFI .0135, pass rate .94 to .15). |
| F11 | run | `detrend` +.11 (G1 base) against +.58 (G2). |
| F12 | run, met | agentsem minus epoch_fe -.45 (MC SE .005) under every G3 cell; under G2 agentsem refuses claim B (200 / 200) and returns claim A at +.056 against epoch_fe +.71. |
| F13 | run | `belief_partial` Var(theta) bias -.95 (over-corrects to zero). |
| F14 | run, not as predicted | G3 bias does not grow with rho_psitheta at tau_psi .5 (+.001 / +.020 / +.014); certificate identified, not bounded. |
| F15 | run | `E_perf`: R2 bounded .94; anchor estimate +.064. |
| F16 | run | `D_heuristic`: agentsem +.050 (base +.013), identified .89. |
| F17 | run, review confirmed | Hessian coverage .63 / .41 / .56 against bootstrap .94 / .84 / .93 in the base, G2 and G3 audit cells (100 replicates each); naive-SE coverage well below nominal, bootstrap near nominal, as predicted. |
| F18 | run | standardized loadings and AVE in every naive record; AVE .961 / .975 / .983 at u_eff 0 / .26 / .51. |

## Required spec edits (checklist 1 to 14)

1 fixed (spec 1.1/1.2). 2 fixed (spec 5). 3 fixed (README, spec 1.1 K3; heterogeneous psi as G3). 4 fixed
(spec 5, A1). 5 fixed (spec 4). 6 fixed (spec 6; r reported per cell). 7 fixed (spec 5.1, F10). 8 fixed
(`detrend`, `epoch_fe`). 9 fixed (metrics). 10 fixed (TOST + ROC). 11 fixed (psi 0 benign arm). 12 fixed
(surrogate set). 13 fixed (spec 2.6). 14 fixed (`results/hashes.json`).

## What the reviewer asked to see, in order

1. F1 to F5 before the generator: `results/closed_forms.json`, produced first; K1 and K2 killed.
2. F7 returning `unsupported`: 200 of 200 under G4 and under G4b; the undetectable limit G4c is refused in
   .14 and its identified Var(theta) is unbiased (-.015) while the person scores are not certified.
3. F10 reported honestly: invariance testing catches a shadow whose loading pattern differs from the trait's
   (.94 to .15) and not one proportional to it (no change in pass rate); the surviving claim is the narrower
   one.
4. F12 margin under G2 and G3: -.45 (MC SE .005) under G3; under G2 the correction refuses and its anchor
   claim beats epoch fixed effects by .65.
