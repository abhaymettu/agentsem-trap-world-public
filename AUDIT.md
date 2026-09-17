# AUDIT.md: methods audit of identification, certificate, and results

Hostile methods audit of the agentSEM trap-world build, against `task-08-brief.md` (A to F) and
`task-08b-brief.md` (G to M). Read: `HARSH-REVIEW.md`, `STEERING-20260916-1946.md`,
`specs/001-trap-world/spec.md` (incl. section 11), `JEV-CONSTRAINT-20260916.md`, `trapworld/{shadow,
certificate,analyzers,world,metrics,derive,jev,run_grid}.py`, `trapworld/rules.json`, `README.md`,
`MORNING-HANDOFF.md`, `REVIEW-RESPONSE.md`, and `results/{closed_forms,summary,hashes,run_manifest,
derived_manifest}.json` plus 73 files in `results/grid/`. Nothing under `results/` was written.

**Run provenance.** `results/run_manifest.json`: 72 cells, 15,450 replicates, 5142.4 s on 10 workers,
`slow_seconds` 8.0, zero halved cells, zero analyzer errors, `J_live` `not_run`.
`results/hashes.json`: `hashes_match` true, 8 frozen objects. Code at `6a916ef`.

**Coverage audit: absent.** `results/coverage_audit.json` does not exist as of this audit; PID 8455
(`run_grid --bootstrap --out results/bootstrap`) is still running and `results/bootstrap/` holds only
`hashes.json`. Every statement below about interval coverage uses the Hessian/delta intervals in the
grid, which are the thing the bootstrap exists to replace. **SC3's coverage clause is undecided.**

---

## 1. Verdict table

MC SE in parentheses. "Survives no-CRE" = does the result still stand for a reviewer who rejects
the correlated-random-effects (Mundlak) assumption added as spec 11 A1.

| # | claim | status | decisive numbers (MC SE) | survives no-CRE | evidence |
|---|---|---|---|---|---|
| K1 | validation corr inflated by (1+psi kappa) | **killed** | corr(eta,theta) 1.000 / .981 / .949 / .894 / .832 / .747 and corr(eta,C) .447 / .439 / .424 / .400 / .372 / .334 over u = 0 / .25 / .5 / 1 / 2 / 8, both strictly decreasing; 2e6-draw MC matches the closed form to 2.0e-4 | yes (no estimator) | `results/closed_forms.json` F1 |
| K2 | partial cumulative dose separates person from policy | **killed** | R2(theta ~ dose) .500 / .875 / .966 / .982 at T 1 / 7 / 28 / 56 (dev 7e-4); innocent-kill rate .996 to 1.000 conditional on the raw correlation clearing, at every kappa and tau | yes | `closed_forms.json` F4, F5 |
| K3 | mechanism is Theorem 1 aliasing | **killed by inspection** | actions logged, so delta is log-measurable; no simulation run | yes | spec 1.1, README:22 |
| R1 | reliability up, validity down | **split: closed form yes, grid leg not demonstrated** | closed form (F2): omega .833 to .962, corr 1.000 to .894. Grid, person means: omega .9928 (6e-5) to .9983 (1.4e-5); corr .9810 (1.8e-4) to .9768 (2.3e-4) at psi 1 over kappa 0 to 2, a fall of .0042 (3e-4), non-monotone (.9810/.9813/.9800/.9768). Naive corr never below .9748 in any of the 71 cells with a person | yes | `summary.json` A sweep |
| R2 | criterion contamination, hump with r >= .414 | **closed form yes, grid leg fails** | grid ratio r(score,C_react)/r(theta,C_react) 1.016 (4.1e-4) at kappa 2 psi .5, against **1.204** implied by B1.4 at that cell's own u_eff .512 and r .800 (1.008 against 1.113 at kappa 1). At psi 0 the same ratio is already 1.006 (4.2e-4), so the shadow's own share is about 1.0pp. Closed form reproduces the hump and the sqrt(2) ceiling | yes | `closed_forms.json` F3; `summary.json` `ratio_react_mean` |
| R3 | scale inflation of the person signal | **supported** | naive Var(theta) bias +.025 (.0073) / +.299 (.0094) / +.620 (.0075) / +1.363 (.0172) at psi .5 over kappa 0 / .5 / 1 / 2; +3.232 (.0298) at kappa 2 psi 1. u_eff .256 (3e-4) at base, exactly half of psi*kappa (ramp) | yes | `summary.json`, `grid/*.json` `cell.u_eff` |
| R4 | named identification recovers and refuses | **supported with three named limits** | base `agentsem` bias +.0132 (.0051) on n=478 of 500, `phi_B_raw` +.0134 (.0050) on all 500; identified .956; G2 and G4/G4b refuse 200/200 (SC4 met); G3 +.0012 (.0085) / +.0205 (.0080) / +.0142 (.0079). Limits: (a) 95.4% (5e-4) of the estimate is the CRE term b^2 Var(gbar); (b) the reported `bias` is a mixture of three estimators (MF1); (c) the interval covers .577 median (MF2) | **no** (see L1) | `grid/*.json` |
| R5 | competitors do not repair it | **supported, but `epoch_fe` is degenerate** | base: `epoch_fe` +.4524 (.0067) identical to `naive_pooled` to 1.5e-7 across all 72 cells; `detrend` +.1124 (.0056) at base and +.5796 (.0115) under G2; `belief_partial` -.9555 (2.2e-4); `dose_partial` corr .184. Paired `agentsem` minus `epoch_fe` abs-bias -.3618 (.0070) | yes | `grid/*.json` `paired` |
| R6 | specificity is an equivalence decision | **fails SC5** | innocence at psi 0: .060/.070/.070 (kappa .5), .870/.876/.865 (kappa 1), 1.000 (kappa 2); SC5 needs >= .90 at every kappa. Detection at psi 0 is .055 to .140 against nominal .05. ROC otherwise clean: detection .175/.825/1.00/1.00 at psi .05/.1/.2/.3, kappa 1 | no (uses the same Hessian SE) | `grid/F_*.json`, `grid/A_k*_p0.0_*.json` |
| R7 | personalized help, not help, makes the trap | **supported** | `G_uniform`: naive bias +.036 (.0074), `agentsem` +.0056 (.0075), psi .498 (.0024), certificate identified 200/200, no rule fires | **yes** (b is dropped: `var_gbar` 0, `between_dropped` true) | `grid/G_uniform.json` |
| R8 | zero-person world fools conventional SEM | **narrowed to "fools 5 of 6 battery clauses"** | omega .878, CFI .997, RMSEA .020, discriminant .057, r(C_react) .235 all pass or nearly; test-retest .416 fails the .70 rule, so the pre-registered battery passes **0 of 200** at every tau. Certificate refuses .875 (.023) | partly | `grid/G_zero_person.json` |
| A1 | Mundlak term (spec 11 A1) | **necessary, honestly dated, and now load-bearing** | pre-grid: committed `4b2d03c` 2026-09-16 22:17 CDT, grid started 22:48. Base cell `phi_u` .0455, `b` 2.452 (.0030), `var_gbar` .1608, so b^2 Var(gbar)/phi = .9545 (5e-4). Drift-feedback residual: base minus `G_nodrift` `phi_B_raw` +.0130 (.0093), 1.4 SE, unpaired. Closed-loop feedback cell minus base +.0470 (.0097), 4.9 SE. Coverage of phi: .566 (.022) at base, .577 median over 72 cells, .95 only where b is dropped | n/a (is the assumption) | `shadow.py:263`, `grid/*.json` |
| A2 | R4 exposure-pattern rule (spec 11 A2) | **it is the only thing that catches G2** | reconstructed per replicate against 0.15*sqrt(phi_A): under **G2** exposure fires 200/200, pooled shift 0/200. Under **G4** pooled fires 200/200, exposure 65/200; under G4b 200/200 and 61/200. Base-cell cost: .044 (.0092) at psi .5, .042 (.0090) at psi 0, .035 (.013) at `G_nodrift`, against ~.030 expected from three tests at alpha .01 | see L11 | `certificate.py:128-141`; reconstructed from `epoch_shift`, `max_changepoint_shift`, `exposure_p`, `anchor_only.var_theta_hat` |
| A3 | Jev arm (spec 11 A3) | **constraint met; `J_fake` is a plumbing check, not evidence about Jev** | `J_live` `not_run` in both `grid/J_live.json` and `run_manifest.json`; `jev` "fake"; returned model `fake-jev-stub`; 1200 calls/replicate, 0 failed, 50/50 `ok`; `HEADLINE_SWEEPS` excludes J (`grid.py:32`); hashes written before (`run_grid.py:399`) and rechecked after (`:469`), and `--jev live` refuses on mismatch (`:373`) | yes | `grid/J_fake.json`, `results/jev/` |
| A6 | halving threshold 3 s to 8 s | **accepted; runner-only** | `slow_seconds` 8.0 and `reps_halved` false for all 72 cells in `run_manifest.json`; seeds are position-fixed (`run_grid.py:156`) so no replicate's data moved. Spec text written at 22:50, two minutes after the run began (L8) | yes | `run_manifest.json` |
| A7 | `_sample` derived flags | **right measure, honestly labelled, fixes false confidence only** | base `agentsem` `false_confidence` .385 (.022) to `_sample` .038 (.0087); `abs_bias_ok` .615 to .962. `naive` `false_conf_0.3` at psi 0 .340/.365/.362/.275 to `_sample` .025/.005/.016/.015. But `false_refusal_sample` is **worse**: G4 .655 to .920, E_perf .625 to .940, C_nonprop .500 to .830 | yes | `derive.py:39-66`, `summary.json` |
| D2 gate | logged randomized dose | **valid, no useful precision at N=200** | `G_d2` `agentsem_d2`: psi .429 (.036) with SD .505 against truth .5; `phi_B_raw` 1.184 (.055) with SD .774; first-stage F median 87, q05 7.1; gate bounds .07 for F < 10; interval coverage .89 (.022). Instrument exogenous by construction (`world.py:143`, `:210`: jitter drawn from its own stream, independent of theta, enters eta only through the dose) | yes (no b term) | `grid/G_d2.json` |
| R1 rule (provenance) | version complete | fires 0/15,450; correct (version always logged) | | | `certificate.py:77-80` |
| R2 rule (anchor) | anchor + switch exogeneity | **fires when it should**: `E_perf` 200/200, bounding both claims; silent in all 71 calendar-trigger cells | | | `grid/E_perf.json` |
| R3 rule (conditioning) | cond number > 30 | **never fires**: cond 4.79 to 5.85 in every cell, q95 5.85 | | | `summary.json` `cond_number_*` |
| R4 rule | shift + exposure | fires G2 1.00, G4 1.00, G4b 1.00, G4c .140, base .044, psi-0 base .042, `G_uniform` .000, `G_nodrift` .035 | | | above |
| R5 rule (loading) | dCFI > .01 | **fires .835 under `C_nonprop`**, .000 under `C_nonprop_psi0`, .040 under `G_zero_person`; naive dCFI .0135 vs .00005 proportional | | | `grid/C_nonprop.json` |
| R6 rule (heterogeneity) | ratio > 1.5 | **cannot fire**: ratio mean .256 to .900, q95 <= 1.001 in all 72 cells; 0/15,450 | | | `summary.json` `hetero_ratio_*` |

### Success criteria
SC1 met (closed forms written with verdicts). SC2 met (`hashes_match` true, 0 errors, seeds
position-fixed). **SC3**: bias clause met (`agentsem` +.0132, .0051 <= .10; `naive` +.620 > .10);
coverage clause **undecided**, the audit file is absent. **SC4 met** (G4 unsupported 200/200 >= 95%).
**SC5 fails on both clauses as pre-registered**: innocence .06 to .88 at kappa <= 1 against >= .90, and
naive false confidence at psi 0 is .275 to .365 against <= .05. Under A7's `_sample` twin the second
clause passes (.005 to .025); the first is unaffected. SC6, SC7 met by inspection of
`REVIEW-RESPONSE.md` and `MORNING-HANDOFF.md`.

---

## 2. Identification (question A), circularity (E), and the amendments (G, H)

**The anchor does pin the scale but not the estimand.** `_anchor_pipeline` (`analyzers.py:307-319`)
fixes (mu, lambda, Theta) from version-0 person means, and marker scaling makes eta a pi_0-scale score;
the scale claim holds (`beta_clean_hat` .4859 (.0028) against truth .500 at base). But the anchor CFA's
`Var(theta)` is `Var(theta_i + d_bar_i)`, not `Var(theta_i)`: `anchor_only` bias is **+.0509 (.0079)** at
kappa 0 psi 0, against a closed form `Var(d_bar)` over 14 AR(1) occasions (rho .9, sd .3) of **.0584**,
and **+.0061 (.0078)** at `sd_d` = 0. That is 51% of the tolerance, present in 71 of 72 cells, and no
rule sees it: claim A is certified `identified` everywhere except `E_perf`.

**The shadow-model likelihood is not valid under feedback by the argument README:63-67 gives.** The
dose at t is `kappa * belief_prev` (`world.py:210`) and the belief is built from a **private** read
`r_it = theta_it + noise` (`world.py:234`) that is not in the analyst's data. The action path is
therefore not measurable with respect to the analyst's filtration, so the sequential-factorization /
ancillarity argument does not apply. That is precisely why A1 was needed, and the README sentence
should be corrected (MF10). With `policy_reads_stream = True` the loop closes through the stream and
the cost is measurable: `G_feedback_p05` `phi_B_raw` 1.0604 (.0082) against base 1.0134 (.0050), a
difference of **+.0470 (.0097)**.

**Within-person identification of psi (question G).** A1 buys the between-person channel by modelling
`E[theta | gbar]`. The within-person channel still needs strict exogeneity of `g` given the person
effect, and the dose responds to past drift, so it is violated. Size: base minus `G_nodrift` on
`phi_B_raw` is **+.0130 (.0093)**, 1.4 SE, unpaired (different cells, different seeds, R 500 vs 200).
The continuous statistic does separate: `max_exposure_corr` .0970 (.0021) at base against .0826 (.0033)
at `G_nodrift`, a difference of +.0144 (.0039), 3.7 SE. Caveat the handoff does not state: `G_nodrift`
removes both the feedback channel **and** drift's contribution to within-person variance, so +.013 is an
upper bound, not a clean estimate of the feedback channel.

**Nominal coverage of `phi = phi_u + b^2 Var(gbar)` (question G): no.** Median `covers_hessian` over
72 cells is **.577**; base .566 (.022); .390 (.034) at G2; .035 at `G_zero_person`. The only cells at
nominal are the ones where the b term is dropped (`G_uniform` .950 (.015), the kappa-0 cells .945 to
.965). Interval width .175 with b against .406 without. Two causes, both in the code:
`shadow.py:289-296` holds `var_gbar` fixed in the delta method by explicit choice, and the Hessian is
taken of a likelihood whose `eta` is a generated regressor from the anchor CFA (`analyzers.py:417-418`)
that the likelihood never sees.

**Where the correction consumes hidden truth: nowhere.** `analyzers.py` reads `world.theta` only in
`_result` (`:140`), for `corr` and `rmse`, and reads `world.params` only for `T`, `T0` (`:507`, `:528`)
and `smd_psi` (`:445`, `:597`), which are design constants, not generator truth. `belief_logged` is read
only by `analyzer_belief_partial` (`:256`).

**D2's instrument is genuinely exogenous.** `jitter` is drawn from its own RNG stream independent of
theta, drift and eps (`world.py:143`) and enters the dose additively (`world.py:210`); under the default
private-read policy it cannot re-enter the belief, so it affects eta only through `g`. Exclusion and
independence both hold by construction.

**Circularity (question B4/E): the mismatched cells do separate a consistent estimator from a fitted
generator, but weakly, and not where it matters most.** Three readings:

1. Under the matched generator G1 the surrogate `surrogate_g1` (`world.py:77-81`) is bit-for-bit the
   generator's own accumulator (`world.py:214, 218`). That cell is a unit test of the estimator.
2. The mismatched cells do bite. `misspecified` (recency, half-life 2) is refused 200/200 in **every**
   cell with a dose and psi > 0, and identified .94 to .97 at psi 0 where the wrong surrogate is
   harmless. G2 is refused 200/200 with a withheld bias of **+.127 (.0086)**. So the certificate is not
   a rubber stamp.
3. **But the incremental value of the D5 shadow model over the D1 anchor on the primary estimand is
   almost nil.** Paired, within replicate: `agentsem` minus `anchor_only` abs-bias is **-.0136 (.0021)**
   at base, -.0059 (.0030) / -.0106 (.0031) / -.0083 (.0030) across G3 rho 0/.3/.6, +.0078 (.0081) at
   `C_nonprop`, and **0.0000 (0.0000)** at `E_perf` (where `bounded` returns the anchor estimate
   verbatim). Against `epoch_fe` the anchor alone delivers -.3490 (.0056) of the -.3618 (.0070) margin
   at base; under G2 the anchor **beats** the shadow model's withheld estimate (-.613 (.0097) against
   -.568 (.0069)). The D5 leg's real contribution is the person scores: paired `agentsem` minus
   `anchor_only` rmse **-.0970 (.0007)** and corr **+.0202 (.0002)** at base.

**The review's fourth artifact (F12) is delivered under G3 and is undefined under G2.** Under G3 the
paired abs-bias margin over `epoch_fe` is -.3529 (.0120) / -.3780 (.0112) / -.3776 (.0116), far beyond
MC error. Under **G2 the paired margin has n = 0**: `agentsem` refuses 200/200 so there is no point
estimate (`grid/B_G2.json` `paired`). On the withheld estimate the margin is -.5678 (.0069), and on the
anchor fallback -.6128 (.0097). The paper must say which of the three it is quoting.

---

## 3. Estimand, multiplicity, and rates vs continuous statistics (question D)

One primary estimand is used consistently: `metrics.py:240-241` scores every arm as
`var_theta_hat - var_theta` against `tolerance` .10, and MC SE accompanies every mean and rate
(`metrics.py:295, 303`) including three paired within-replicate differences (`metrics.py:55, 433-440`).

Two failures. **(a)** `bias` for `agentsem` is not one estimand: `analyzers.py:462` reports the
certificate's output, which is the shadow model when `identified`, the **anchor CFA** when `bounded`
(`certificate.py:168`), and missing when `unsupported`. So `bias_mean` is conditioned on the
certificate and silently changes estimator between cells. **(b)** No multiplicity family was
pre-registered, contrary to S13's second half; `REVIEW-RESPONSE.md:49` answers only the cell-list half.
Every "<= 5%" clause is uncorrected across 72 cells x 10 arms x 3 taus.

**S14 bites, hard, and in the direction the review named.** Three places where a rate says the opposite
of the continuous statistic in the same cell:

| cell | rate | continuous statistic in the same cell |
|---|---|---|
| `A_k1.0_p0.5_g1.0` | `agentsem` false confidence .385 (.022) | mean bias +.0132 (.0051), an eighth of the tolerance |
| `B_G2` | false refusal .385 (.034) | withheld bias +.127 (.0086), 1.3 tolerances out, and `beta_clean_hat` .4656 (.0043) against .500 |
| `G_zero_person` | false refusal .875 (.023) | the refusal is the pre-registered success condition (R8) |

---

## 4. A7 and the derived sample-truth fields (question K)

**Is `_sample` the right per-replicate error measure? For false confidence, yes.** The pre-registered
flag asks whether a single estimate from 200 drawn persons is within .10 of the **population**
Var(theta), when `Var(var_theta_sample)` at N=200 is 2/199, i.e. an SD of .100. The flag therefore
fires at .27 to .41 for estimators whose mean bias is .007 to .025, which is not a property of the
estimator. Against `var_theta_sample` the same flag reads .038 (.0087) for `agentsem` and .016 (.0056)
for `naive` at `A_k1.0_p0.0_g1.0`. That is the correct per-replicate question for a per-replicate
estimate.

**It is honestly labelled.** `derive.py:1-20` and the spec 11 A7 text both say post hoc and
exploratory, the `_sample` suffix is carried on every field, `summary.json` carries
`derived_note`, and `results/derived_manifest.json` records the derivation separately. Commit
`6a916ef` 23:13 CDT postdates the grid start (22:48) and is labelled as such. Nothing hashed changed.

**Verdicts that change.** SC5's second clause ("naive false confidence at psi 0 <= 5%") **flips from
fail to pass**: .340/.365/.362/.275 pre-registered against .025/.005/.016/.015 under `_sample`. SC3
does not change: its bias clause is a cell-level mean, which is a property of the estimator and is
untouched (`agentsem` +.0132 (.0051) pre-registered, +.0120 (.0022) `_sample`). Mean bias, the quantity
behind R1, R3 and SC3, is unaffected, exactly as the brief states.

**Both must be reported**, and the paper must say which one each success criterion is judged on, because
one criterion flips. Report the pre-registered field first, as the build already does.

**One thing A7 does not fix: false refusal.** `derive.py:58` applies the same substitution to the
refusal flag, and the rate gets **worse** (G4 .655 to .920, E_perf .625 to .940, C_nonprop .500 to .830),
because the refusal flag's problem is an estimand mismatch, not sampling noise. See section 5.

---

## 5. The refusal-error pair (question L)

Claim B is "adaptive-phase person scores are on the pi_0 scale after correction". The pre-registered
flag (`metrics.py:167-172`) scores it against the **Var(theta)** tolerance. That is the wrong content,
and the two sweep-B cells show it pointing in opposite directions.

**`B_G2`: the refusal is right, and `false_refusal` .385 (.034) is an artifact.** Both legs of claim B
fail. Variance: withheld `phi_B_raw` bias **+.127 (.0086)**, i.e. 1.3 tolerances. Scale:
`beta_clean_hat` **.4656 (.0043)** against truth .500, so the scores are inflated by 7.4%. Person
recovery, paired against the base cell's own arm: rmse .2223 (against .1943 at base), corr .9783
(against .9822). The .385 is the fraction of replicates whose draw of a +.127-biased estimate happened
to land inside .10 of 1.0. It is sampling noise around a biased mean, not a refusal error.

**`B_G4`: the refusal is right by design, and the .655 (.034) is partly genuine.** The driver is
unlogged, so no D1+D5 argument exists and refusing is the pre-registered success condition (SC4, met
200/200). But the withheld estimate was in fact good: `phi_B_raw` bias **-.0157 (.0076)**,
`beta_clean_hat` **.4982 (.0047)**. The scores are degraded but not broken: rmse .2343 against .1943 at
base, corr .9756 against .9822 (paired against `anchor_only` in the same cell: rmse -.0529 (.0014),
corr +.0131 (.0005)). The accuracy is luck of the parameterization: the model fits psi .2817 (.0033)
against a truth of 0, absorbing the person-proportional part of the c1*theta step into the ramp
surrogate. Say that, rather than either "false refusal" or "correct estimate".

**What the paper should report instead of, or beside, the pre-registered flag**, per cell, never pooled:

> refusal rate, and for the withheld estimate: mean bias with MC SE, `beta_clean_hat` with MC SE, and
> the paired rmse and corr of the refused scores against the `anchor_only` arm in the same cell.

That pair reads, for G2: refuses 1.000, withheld bias +.127 (.0086), scale .4656 (.0043), rmse -.0684
(.0013) vs anchor. For G4: refuses 1.000, withheld bias -.0157 (.0076), scale .4982 (.0047), rmse
-.0529 (.0014) vs anchor. Those two rows say different things, which the single number .385 / .655
cannot.

---

## 6. F14 and Cov(psi, theta) (question M)

`agentsem` bias under G3 is **+.0012 (.0085) / +.0205 (.0080) / +.0142 (.0079)** at rho_psitheta
0 / .3 / .6. The rho 0 to rho .6 difference is +.0130 (.0116), 1.1 SE. F14's prediction that bias grows
with Cov(psi, theta) does not appear.

**Why.** Under G3 (`world.py:98-109`) the omitted term is
`(psi_i - psi) * D_it = psi * tau_psi * (rho * theta_i/sd + sqrt(1-rho^2) * w_i) * D_it`.
Its person mean is proportional to `(rho * theta_i/sd) * gbar_i + sqrt(1-rho^2) * w_i * gbar_i`, and
`gbar_i` is itself close to linear in theta. The first piece is therefore proportional to `theta^2`,
whose covariance with theta is `E[theta^3] = 0` for a Gaussian trait; the second has mean zero in
`w_i`. So the heterogeneity adds **variance orthogonal to theta** and contributes no first-order
covariance, which is what the variance estimator would have to pick up as bias. The certificate is
silent for the matching reason: R6's ratio is a variance ratio computed on a residual from which the
person effect has already been estimated, so it sits at .5366 (q95 .696) under G3 rho .6 against a
threshold of 1.5.

**What would make it bite.** Two routes, bounded from the records.
(i) **Break the symmetry**, not the covariance. `E[theta^3] = 0` is the whole reason the term cancels;
a skewed trait, or `psi_i` depending on `|theta_i|` rather than `theta_i`, would give a first-order
term immediately. This is the cheap decisive test and it costs one cell.
(ii) **Grow tau_psi.** The contribution to the person-score variance scales as
`(psi * tau_psi * rho * s)^2 * Var(theta^2)` where `s` is the slope of `gbar` on theta. At the base
cell `u_eff` = .2555 (3e-4) with psi .5 gives s = .511, so at tau_psi .5, rho .6 the term is
`(0.5*0.5*0.6*0.511)^2 * 2 = .0118`, which matches the observed +.0142 (.0079) inside one SE. Reaching
the .10 tolerance needs the coefficient about 2.9x larger, i.e. **tau_psi about 1.5 at rho .6, or
tau_psi 1.0 at rho about .9**, both of which put a large share of `psi_i` below zero. That bound is an
order of magnitude, not a measurement: it is an algebraic upper bound calibrated against one cell
(+.0142, .0079, against a predicted .0118), and the grid ran only tau_psi .5. The honest statement is
that the identified-set widening of B3.2 is real in theory, is below the tolerance at every
(tau_psi, rho) this grid ran, and that R6 could not have seen it if it were not.

---

## 7. Claims the documents make that the results do not support (question F)

1. `README.md:63-67`: "the likelihood factorizes sequentially and the estimator is valid under feedback
   (the likelihood principle)". False as written: the dose is a function of a private read
   (`world.py:210, 234`), not of logged data, so the action path is not analyst-measurable. This is the
   reason A1 exists and the sentence contradicts the paragraph directly below it.
2. `README.md:156-159` and `MORNING-HANDOFF.md:36` call R1 "supported" on a validity fall of .0042
   (3e-4) that is non-monotone across kappa, while the closed-form crossing it cites is a
   single-occasion result. Report the two levels separately.
3. `MORNING-HANDOFF.md:39` reports `E_perf` as "`agentsem` ... +.064" in the R4 row. That number is
   `anchor_only`'s, exactly: paired `agentsem` minus `anchor_only` bias is 0.0000 (0.0000) on n=189
   because `bounded` returns the anchor estimate (`certificate.py:168`). Same issue for any cell with
   `bounded` in it.
4. `MORNING-HANDOFF.md:41` gives false detection at psi 0 as ".06 to .12"; the measured range over the
   seven psi-0 cells is **.055 to .140** (`F_p0.0_k1.0` .140, MC SE .025).
5. `MORNING-HANDOFF.md:89-97` gives Hessian coverage as ".63 / .69" from the adversarial pass; the grid
   itself gives **.577 median over 72 cells**, .035 at `G_zero_person`, .39 at G2. Quote the grid.
6. `README.md:135` says 73 cells and `--workers 11`; the run was 72 cells on 10 workers
   (`run_manifest.json`).
7. `README.md:136` and `:166` point at `results/coverage_audit.json`, which does not exist yet.
8. `spec.md:3` says "Status: frozen before grid run" while A6 and A7 were written into it after the run
   began. Both are labelled and neither touches a hashed object, but the header is wrong.

---

## 8. Must-fix, ranked

Each item: what is wrong, the file to change, the test that would catch it.

**MF1. `agentsem`'s `var_theta_hat` silently switches estimator with the certificate's verdict.**
`analyzers.py:461-462` reports `cert["B"]["estimate"]`, which `certificate.py:165-170` sets to the
shadow model when `identified`, to `phi_A` (the anchor CFA) when `bounded`, and to `None` when
`unsupported`. Consequences in the results as they stand: `E_perf`'s "`agentsem` bias +.0644 (.0087)"
is `anchor_only`'s to 0 decimal places of difference; `B_G2`/`B_G4`/`B_G4b` have `bias_n` = 0 so they
drop out of every cross-cell bias comparison; `C_nonprop`'s mean bias is computed on the 31 of 200
replicates the certificate passed. **Fix:** store `var_theta_hat_shadow` (always `phi_B`),
`var_theta_hat_reported`, and `estimate_source` in {"shadow","anchor","none"}; aggregate bias on the
unconditional shadow estimate and report the certificate decision beside it. **Test:**
`tests/test_grid.py`, assert `bias_n == reps_run` for every arm in every cell, and
`tests/test_analyzers.py`, assert that on a world where R2 bounds claim B the reported estimate is not
recorded under the same key as `phi_B_raw`.

**MF2. Every `ci` in `results/` outside the audit cells is not a 95% interval.**
`shadow.py:289-296` holds `var_gbar` fixed in the delta method and the Hessian never sees the anchor
calibration error carried into `eta` (`analyzers.py:417-418`). Median `covers_hessian` .577 over 72
cells, base .566 (.022), .390 (.034) at G2, .035 at `G_zero_person`; nominal only where b is dropped
(`G_uniform` .950). **Fix:** run the bootstrap for every cell, or set `ci` to `None` outside the audit
cells and delete `covers`/`covers_when_identified` from the non-audit rows. **Test:**
`tests/test_shadow.py::test_reported_se_matches_sampling_sd` exists on synthetic eta; add the same over
>= 200 replicates of the real `_anchor_pipeline` eta at N=200 and assert coverage >= .90. It fails at
.57 today.

**MF3. `false_refusal` scores claim B against the wrong estimand and labels correct refusals as
errors.** `metrics.py:167-172`. Numbers in section 5. A7's `_sample` twin makes it worse, not better.
**Fix:** report the refusal-error pair of section 5 (refusal rate plus withheld bias, `beta_clean_hat`,
and paired person-score metrics, each with MC SE). Keep the pre-registered flag, never quote it alone.
**Test:** `tests/test_certificate.py`, a world whose `Var(theta)` is inside tolerance but whose person
scores are off-scale by 10% must not be counted as a false refusal.

**MF4. The claim card says D1 + D5; the estimator is D1 + D5 + Mundlak, and 95.4% (5e-4) of the
estimate is the Mundlak term.** `shadow.py:263`; `spec.md` 1.2 R4; `README.md:59-76`. At base
`phi_u` .0455, `b` 2.452 (.0030), `var_gbar` .1608; `R2(gbar on anchor score)` .885 (7e-4). The number
that "recovers truth" is, to 95%, `Var(b * gbar)`, read off a regressor that is 88.5% the anchor score.
**Fix:** name the correlated-random-effects assumption inside R4's claim text, and make the headline
number the person-score pair (paired rmse -.0970 (.0007), corr +.0202 (.0002) against the anchor), not
`Var(theta)`. **Test:** `tests/test_shadow.py`, assert `phi_u / phi < 0.2` on a base replicate, so the
fact is pinned and cannot drift out of the documentation silently.

**MF5. R6 cannot fire and R3 never fires, so the certificate is R2 + R4 + R5 and is silent on the
review's own nonidentification case.** `analyzers.py:375-377` computes `hetero_ratio` from a residual
the person effect has been estimated out of; `rules.json` R6 1.5, R3 30. `hetero_ratio` q95 <= 1.001 and
`cond_number` <= 5.85 in all 72 cells; both rules fire 0 of 15,450. Under G3, claim B is `identified`
in .92 to .995 with the certificate saying nothing about heterogeneity. **Fix:** redefine R6 against
the anchor-epoch analogue of the same ratio, or delete it and state the limit in `rules.json` and the
claim card. **Test:** `tests/test_analyzers.py::test_hetero_generator_bounds_claim_b_through_r6` is
already `xfail(strict=True)` and records exactly this; converting it to a passing test is the
acceptance criterion for whatever replaces R6.

**MF6. SC5 fails on both clauses.** Specificity: innocence at psi 0 is .060 to .070 at kappa .5 and
.865 to .876 at kappa 1 against >= .90, and detection at psi 0 runs .055 to .140 against nominal .05,
because `_psi_flags` (`analyzers.py:401-407`) uses the same Hessian SE as MF2. Naive false confidence at
psi 0 is .275 to .365 against <= .05 on the pre-registered definition (`metrics.py:138-139`).
**Fix:** use the bootstrap psi interval for the TOST or widen `smd_psi`, and state in the abstract which
definition SC5's second clause is judged on. **Test:** `tests/test_analyzers.py`, assert the psi-0
detection rate <= .08 over 200 replicates at kappa 1.

**MF7. R4's two sub-tests share one rule id, so the record cannot show which fired.**
`certificate.py:126` and `:141` both call `B.fire("R4", ...)`. This matters because A2 is doing all the
work under G2 (exposure 200/200, pooled 0/200) and none of the work under G4 (pooled 200/200, exposure
65/200), and the summary cannot tell you that; I had to reconstruct it from `epoch_shift`,
`max_changepoint_shift`, `exposure_p` and `anchor_only.var_theta_hat`. **Fix:** emit `R4_shift` and
`R4_exposure` as distinct ids in `rules.json` and in `triggered`. **Test:**
`tests/test_certificate.py::test_r4_either_half_fires_on_its_own` already exercises both halves; extend
it to assert the two ids are distinct.

**MF8. `epoch_fe` is `naive_pooled`, so F12's kill test is vacuous, and two of S8's seven competitors
were never built.** `analyzers.py:226-235`: occasion FE subtracts the cross-person mean, and a
person-signed shadow has none, so the two arms' bias differs by at most **1.5e-7** across all 72 cells.
S8(ii) "naive + person fixed effects" and S8(vi) "partial-invariance longitudinal model (D3)" are absent
from `ANALYZERS` (`analyzers.py:608-619`). **Fix:** say in the text that F12 is settled by arithmetic
rather than by simulation, and add the partial-invariance competitor, which is the one competitor that
could plausibly repair a nonproportional shadow. **Test:** `tests/test_analyzers.py`, assert
`epoch_fe` and `naive_pooled` agree to 1e-6 on a shadowed world, so the degeneracy is a documented fact
rather than an accident.

**MF9. The D2 arm has no misspecification detector and was exercised in one matched cell.**
`certificate.py:106` makes R3 to R6 optional under design "D2", and `analyzer_agentsem_d2` supplies none
of their numbers, so its certificate is R1 + R2(claim A) + the D2 gate. It also calibrates `eta` on the
**adaptive-phase** CFA (`analyzers.py:567-571`) rather than the anchor, which is safe only while the
shadow is proportional. `G_d2` is the only D2 cell and it is G1-proportional. **Fix:** feed the
exposure statistic and `dcfi_metric` into the D2 certificate, and run D2 under G2 and `C_nonprop`.
**Test:** `tests/test_certificate.py::test_d2_shadow_model_rules_apply_only_when_their_numbers_are_
supplied` documents the gap; add an analyzer-level test that `agentsem_d2` supplies `exposure_p` and
`dcfi_metric`.

**MF10. `README.md:63-67` gives an invalid likelihood argument.** "Actions depend only on past
information and the surrogate is a known function of the logged path, [so] the likelihood factorizes
sequentially and the estimator is valid under feedback (the likelihood principle)." The dose depends on
a private read of theta (`world.py:210, 234`), which is not logged, so the ancillarity of the action
path fails and the factorization does not give a theta-free term. **Fix:** replace with the true
statement: the action path is informative about theta, which is why A1 models `E[theta | gbar]`; within
person, identification of psi needs strict exogeneity of `g`, which the drift feedback violates at the
+.013 (.0093) level measured against `G_nodrift`. **Test:** none (prose). The closest mechanical guard
is `tests/test_shadow.py::test_between_false_reproduces_the_between_person_bias`, which already shows
the plain RE likelihood is biased in exactly this world.

---

## 9. Named limits that must appear in the handoff

`MORNING-HANDOFF.md` section 3 already carries nine. These are the ones it does not.

- **L1. 95.4% (5e-4) of the reported `Var(theta)` is `b^2 Var(gbar)`.** `phi_u` is .0455 at base. The
  estimator recovers the estimand as `Var(E[theta | logged exposure]) + residual`, with the first term
  dominant and its regressor 88.5% (7e-4) explained by the anchor score. A reviewer who rejects the
  correlated-random-effects assumption is rejecting 95% of the number. What survives that rejection is
  `anchor_only`, whose bias is **+.0573 (.0051)** at base.
- **L2. The D5 shadow model's incremental value over the D1 anchor on the primary estimand is
  -.0136 (.0021) in absolute bias at base**, -.0059 to -.0106 across G3, 0.0000 (0.0000) at `E_perf`,
  and negative (the anchor wins) under G2. Its real contribution is the person scores: paired rmse
  -.0970 (.0007) and corr +.0202 (.0002). The paper's contribution claim belongs on claim B, not on
  `Var(theta)`.
- **L3. Claim A carries a fixed drift bias of about +.05** from averaging `theta_i + d_it` over 14
  anchor occasions: `anchor_only` +.0509 (.0079) at kappa 0 psi 0, closed form .0584, +.0061 (.0078) at
  `sd_d` = 0. No rule detects it; claim A is `identified` in 71 of 72 cells. It is 51% of the tolerance
  and it is a property of the design (T0, rho_d, sd_d), not of the analysis.
- **L4. R1's validity leg is a .0042 (3e-4) fall, non-monotone in kappa**, and the person correlation
  never falls below .9748 anywhere in the grid. The B1.3 crossing that the figure shows is a
  single-occasion closed-form result (F2: 1.000 to .894); the 42-occasion person mean averages the
  belief error away and does not reproduce it. Say which level the figure is drawn at.
- **L5. R2's grid leg does not reproduce B1.4.** Ratio 1.016 (4.1e-4) at kappa 2 psi .5 against 1.204
  implied at that cell's own u_eff .512 and r .800; 1.008 against 1.113 at kappa 1; and at psi 0 the
  same ratio already reads 1.006 (4.2e-4). The reason is structural: the belief error averages over 42
  adaptive occasions, so the `u^2 s_e^2` term that drives B1.4's denominator is gone and the shadow acts
  as a near-pure rescaling of theta, which cancels out of every correlation. The hump and the ceiling
  are confirmed in closed form only.
- **L6. The certificate in this grid is R2 + R4 + R5.** R1, R3 and R6 fire 0 times in 15,450
  replicates. Do not describe a six-rule certificate.
- **L7. The bootstrap coverage audit had not landed when this audit was written**, so SC3's coverage
  clause and F17's "bootstrap near nominal" leg are undecided, and every interval currently in
  `results/` is the anti-conservative one.
- **L8. Spec 11 A6 (22:50) and A7 (23:13) were written into a document headed "frozen before grid run"
  after the run started at 22:48.** A1 to A5 (commit `4b2d03c`, 22:17) are genuinely pre-grid. Neither
  A6 nor A7 touches a hashed object; the header should say so.
- **L9. `belief_partial` is undefined in sweep J.** The external policy forms no belief
  (`world.py:243-244`), so the arm residualizes on a constant and equals `naive` (`J_fake`: corr .9806
  and rmse .3171 for both). The F13 positive control does not exist in the Jev cell.
- **L10. No multiplicity family was pre-registered** (S13, second half). Every "<= 5%" clause is
  uncorrected across 72 cells, 10 arms and 3 thresholds.
- **L11. A2 is load-bearing and its false-refusal cost is not separable from its null rate at this R.**
  It is the only detector of G2. Its base-cell fire rate .044 (.0092) against `G_nodrift` .035 (.013) is
  a difference of .009 (.016); the handoff's "named limit, not noise" holds for the continuous statistic
  (`max_exposure_corr` +.0144, .0039, 3.7 SE) but not for the rate.
- **L12. `J_fake` is a plumbing check, not evidence about Jev.** No live call was made; the returned
  model is `fake-jev-stub`; `J_live` is `not_run`. Its numbers (naive +.479 (.020), `agentsem` +.055
  (.015), dW .251 (.0031)) describe a deterministic stub policy and must never be read as a Jev result.
