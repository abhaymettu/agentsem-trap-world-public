# Crossover sweep: where the adaptive-phase estimators stop beating the anchor

2026-10-02. Lanes `agentsem-sweep-20261002` (registry, run, closed form, first scoring) and
`agentsem-sweep-r2-oct02` (final scoring, this file), branch
`abhaymettu/study3-methods-probe-20260927`. Registry committed at `a8b075e` before any cell was
scored; the first 11 cells and the closed form at `5e1bbe2`; the remaining 29 cells, the post-run
hash record, the scorer output, the table generator and this file in the commit that adds this file.
Private, methods-section input only.

## 0. Result

- The sign of "adaptive-phase estimator minus anchor" is set by the product psi x kappa and by
  the anchor length T0, and by nothing else that was varied. Matched products on the kappa 0.5,
  1 and 2 ladders agree in paired RMSE difference to within 0.009 (P1, bar 0.02).
- At the frozen anchor length (T0 = 14) the crossings are: `naive` 0.33, `naive_pooled` =
  `epoch_fe` 0.50, `misspecified` 0.60. `agentsem` never crosses (better than the anchor by
  0.094 to 0.099 at every product up to 2). `detrend` never wins at T0 = 14.
- A shorter anchor moves every crossing up. At T0 = 7: `naive` 0.50, pooled 0.58,
  `misspecified` 0.77, and `detrend` gains a crossing at 0.74 because it beats a 7-occasion
  anchor at low products. A longer anchor moves `naive` and the pooled pair down (T0 = 28:
  `naive` loses at every product from 0.2, pooled crosses at 0.44) but moves `misspecified` up,
  to 0.69: at product 0.6 the longer anchor cuts the anchor's own RMSE from 0.29 to 0.24 and
  `misspecified`'s from 0.29 to 0.22, so the corrected estimator gains more from the extra
  anchor occasions than the anchor does.
- The closed form for `anchor_only`, `naive` and `naive_pooled`
  (`results/crossover/DERIVATION.md`) predicts every measured crossing within 0.010 and the
  anchor's RMSE at each T0 within 0.002.
- Preregistration: P1 holds. P2 holds for `naive`, `epoch_fe` and the two no-crossing claims,
  and misses for `misspecified` at the band edge (measured 0.599 and 0.597 against a band
  starting at 0.60; both 0.6 cells tie). P3 holds for `naive` and the pooled pair at both anchor
  lengths and for `misspecified` at T0 = 7, and misses for `misspecified` at T0 = 28, where its
  crossing rises instead of falling.
- The brief's "kappa 1 null" and "kappa 2 reversal" are averages over the frozen psi set
  {0, 0.5, 1}. At kappa 1, `misspecified` minus `anchor_only` is -0.097, -0.044 and +0.096 at
  products 0, 0.5 and 1 (mean -0.015, the null). At kappa 2 it is -0.097, +0.104 and +0.465 at
  products 0, 1 and 2 (mean +0.157, the reversal). The null is a win, a smaller win and a loss
  that average to zero. The reversal is the product-2 cell. Neither is a property of kappa.

## 1. The quantity the brief is about

The brief's terms ("hindsight estimator", "kappa1", "kappa2") appear in no file of this repo,
the vault or the sibling repos, and the ask-queue to Instinct refused the question with a
client-side error, so the reading here is grounded in the frozen records. The frozen grid
(`results/summary.csv`, sweep A, the 2026-09-17 records) holds one quantity with the brief's
shape: the person-score RMSE against `theta`, on the raw scale, of the estimators that use the
adaptive phase, compared with `anchor_only`, which uses only the 14 pre-adaptivity occasions. At
gamma_C = 1.0 (the value this sweep holds fixed), averaged over psi 0, 0.5 and 1 at each kappa:

| estimator | kappa 0 | kappa 0.5 | kappa 1 | kappa 2 | minus anchor_only: k0 | k0.5 | k1 | k2 |
|---|---|---|---|---|---|---|---|---|
| anchor_only | 0.291 | 0.290 | 0.291 | 0.291 | +0.000 | +0.000 | +0.000 | +0.000 |
| naive | 0.209 | 0.276 | 0.387 | 0.639 | -0.083 | -0.014 | +0.096 | +0.348 |
| naive_pooled = epoch_fe | 0.188 | 0.235 | 0.314 | 0.499 | -0.103 | -0.055 | +0.023 | +0.208 |
| detrend | 0.339 | 0.346 | 0.353 | 0.373 | +0.048 | +0.056 | +0.062 | +0.082 |
| misspecified | 0.185 | 0.213 | 0.276 | 0.448 | -0.107 | -0.077 | -0.015 | +0.157 |
| agentsem | 0.185 | 0.195 | 0.194 | 0.195 | -0.107 | -0.096 | -0.097 | -0.096 |

The sweep maps where each of these sign changes sits and what moves it.

`epoch_fe` and `naive_pooled` coincide to every recorded decimal: with the calendar trigger
every person switches at the same occasion, so an occasion fixed effect removes a constant per
occasion and leaves the person means of the pooled scores unchanged.

What the estimators are (their docstrings in `trapworld/analyzers.py`): `anchor_only` is the D1
minimal, a conventional SEM on the anchor epoch alone; `naive` is a conventional SEM on the
adaptive phase with the full battery; `naive_pooled` is the same over all 56 occasions;
`epoch_fe` removes occasion fixed effects and then fits the pooled SEM; `detrend` removes
occasion fixed effects, then fits a person-specific linear detrend of the adaptive phase read
off at the switch; `misspecified` is the D1 + D5 correction with the deliberately wrong
surrogate (recency weighting, two-occasion half-life); `agentsem` is the same correction with
the cumulative surrogate.

## 2. What theory says controls the sign

The generator's adaptive-phase indicator mean is `eta = theta (1 + psi kappa) + psi kappa e`
(HARSH-REVIEW B1, from `world.py`): the agent reads the person with error `e` (`var_read` =
2.0), doses in proportion to its belief with gain `kappa`, and the shadow adds `psi` times the
cumulative dose to every indicator. Two things follow for a person score formed from adaptive
occasions.

- The error it inherits grows with the product `psi kappa` and with nothing else in the pair:
  `kappa` and `psi` enter only as their product. The frozen cells agree to the third decimal at
  matched products (`naive` RMSE 0.5979 at kappa 2, psi 0.5 against 0.5962 at kappa 1, psi 1;
  0.3581 at kappa 1, psi 0.5 against 0.3585 at kappa 0.5, psi 1).
- The anchor score's error does not depend on the product at all (0.2891 to 0.2922 across the
  12 frozen cells at gamma_C 1). It depends on the anchor length `T0`: a Bartlett score over
  the person means of `T0` occasions has error variance proportional to `1 / T0`.

So the sign of `rmse_X - rmse_anchor_only` is set by the product on one side and the anchor
length on the other, and those are the two axes of the sweep. `gamma_C` (criterion
contamination) does not enter either score and is held at 1.0; the generator is held at the
frozen `cumulative` one, because the generator-misspecification story is the one sweeps B and H
to T on the simulation branch already carry.

## 3. The grid

40 new cells, 200 replicates each (8,000 replicates), registered in
`results/crossover/registry.json` (cells hash `6709bb1f...`) before any was scored, plus the 10
frozen cells that sit on the same ladders (scored from `results/grid/*.json`, the 2026-09-17
records, never re-run).

| block | lever | cells |
|---|---|---|
| P, kappa 1, T0 14 | psi 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9 | 10 new + 3 frozen (psi 0.3, 0.5, 1.0) |
| P, kappa 2, T0 14 | psi 0.15, 0.175, 0.225, 0.25, 0.275, 0.325, 0.35, 0.375, 0.4, 0.45, 0.6, 0.75 | 12 new + 4 frozen (psi 0.2, 0.3, 0.5, 1.0) |
| P, kappa 0.5, T0 14 | psi 0.6, 0.7, 0.8, 0.9 | 4 new + 3 frozen (psi 0.3, 0.5, 1.0) |
| T, kappa 1, T0 7 | psi 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0 | 7 new |
| T, kappa 1, T0 28 | psi 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 | 7 new |

Everything else is the frozen `PARAMS` table. Each replicate is one world from
`trapworld/world.py` with seed `SeedSequence(20260916, spawn_key=(3000 + i, rep))`, a block
disjoint from the frozen grid's cell indices 0 to 72, run through the seven analyzers of
section 1 (the IV arm and the two partialling arms were left out, they report no usable score
here). Nothing under `trapworld/` was edited: all eight frozen hashes in `results/hashes.json`
match before and after the run (section 9).

## 4. Method

For each cell and each estimator X the statistic is the paired, within-replicate difference
`d_X = rmse_X - rmse_anchor_only` (same world, same persons), its mean over the replicates and
the Monte Carlo standard error of that mean (`trapworld.metrics._paired_difference`). A cell
**holds** (H) when the mean is below zero by more than two standard errors, **flips** (F) when
it is above zero by more than two, and **ties** (T) otherwise. A ladder's crossing is the
product at which the mean changes sign, linearly interpolated between the two neighbouring
cells; the tie band is the range of products whose cells tie.

## 5. Per-cell outcomes

Paired `d_X = rmse_X - rmse_anchor_only`, mean (mc_se) and verdict H/F/T, as written by the
scorer to `results/crossover/crossings.md`. Frozen cells carry their 2026-09-17 ids (`F_...`,
`A_...`); new cells are `X_...`.

### ladder P_k0.5_T014  (paired rmse_X - rmse_anchor_only, mean (mc_se) verdict H/F/T)
| cell | product | anchor rmse | naive | naive_pooled | epoch_fe | detrend | misspecified | agentsem |
|---|---|---|---|---|---|---|---|---|
| F_p0.3_k0.5 | 0.15 | 0.2919 | -0.0587 (0.0014) H | -0.0868 (0.0012) H | -0.0868 (0.0012) H | +0.0520 (0.0016) F | -0.0992 (0.0011) H | -0.0980 (0.0011) H |
| A_k0.5_p0.5_g1.0 | 0.25 | 0.2903 | -0.0284 (0.0014) H | -0.0654 (0.0012) H | -0.0654 (0.0012) H | +0.0558 (0.0017) F | -0.0914 (0.0013) H | -0.0956 (0.0012) H |
| X_P_k0.5_p0.6 | 0.3 | 0.2918 | -0.0117 (0.0015) H | -0.0541 (0.0012) H | -0.0541 (0.0012) H | +0.0543 (0.0016) F | -0.0884 (0.0013) H | -0.0970 (0.0011) H |
| X_P_k0.5_p0.7 | 0.35 | 0.2907 | +0.0081 (0.0015) F | -0.0405 (0.0013) H | -0.0405 (0.0013) H | +0.0568 (0.0016) F | -0.0804 (0.0015) H | -0.0968 (0.0011) H |
| X_P_k0.5_p0.8 | 0.4 | 0.2883 | +0.0262 (0.0016) F | -0.0272 (0.0013) H | -0.0272 (0.0013) H | +0.0593 (0.0016) F | -0.0705 (0.0015) H | -0.0952 (0.0011) H |
| X_P_k0.5_p0.9 | 0.45 | 0.2900 | +0.0492 (0.0015) F | -0.0119 (0.0012) H | -0.0119 (0.0012) H | +0.0599 (0.0015) F | -0.0591 (0.0015) H | -0.0957 (0.0010) H |
| A_k0.5_p1.0_g1.0 | 0.5 | 0.2891 | +0.0694 (0.0016) F | +0.0022 (0.0013) T | +0.0022 (0.0013) T | +0.0635 (0.0016) F | -0.0427 (0.0022) H | -0.0950 (0.0010) H |
crossings: naive 0.330; naive_pooled 0.492; epoch_fe 0.492; detrend none; misspecified none; agentsem none

### ladder P_k1_T014  (paired rmse_X - rmse_anchor_only, mean (mc_se) verdict H/F/T)
| cell | product | anchor rmse | naive | naive_pooled | epoch_fe | detrend | misspecified | agentsem |
|---|---|---|---|---|---|---|---|---|
| F_p0.3_k1.0 | 0.3 | 0.2924 | -0.0135 (0.0015) H | -0.0559 (0.0012) H | -0.0559 (0.0012) H | +0.0550 (0.0017) F | -0.0886 (0.0014) H | -0.0971 (0.0013) H |
| X_P_k1_p0.35 | 0.35 | 0.2925 | +0.0062 (0.0015) F | -0.0418 (0.0012) H | -0.0418 (0.0012) H | +0.0561 (0.0015) F | -0.0835 (0.0014) H | -0.0985 (0.0011) H |
| X_P_k1_p0.4 | 0.4 | 0.2935 | +0.0238 (0.0014) F | -0.0295 (0.0012) H | -0.0295 (0.0012) H | +0.0560 (0.0015) F | -0.0751 (0.0016) H | -0.0985 (0.0011) H |
| X_P_k1_p0.45 | 0.45 | 0.2929 | +0.0437 (0.0016) F | -0.0157 (0.0013) H | -0.0157 (0.0013) H | +0.0583 (0.0016) F | -0.0610 (0.0015) H | -0.0982 (0.0011) H |
| A_k1.0_p0.5_g1.0 | 0.5 | 0.2913 | +0.0669 (0.0011) F | +0.0010 (0.0008) T | +0.0010 (0.0008) T | +0.0610 (0.0010) F | -0.0437 (0.0013) H | -0.0970 (0.0007) H |
| X_P_k1_p0.55 | 0.55 | 0.2929 | +0.0909 (0.0017) F | +0.0181 (0.0014) F | +0.0181 (0.0014) F | +0.0625 (0.0017) F | -0.0238 (0.0022) H | -0.0974 (0.0012) H |
| X_P_k1_p0.6 | 0.6 | 0.2925 | +0.1107 (0.0018) F | +0.0323 (0.0014) F | +0.0323 (0.0014) F | +0.0644 (0.0015) F | +0.0003 (0.0020) T | -0.0967 (0.0011) H |
| X_P_k1_p0.65 | 0.65 | 0.2920 | +0.1311 (0.0020) F | +0.0465 (0.0015) F | +0.0465 (0.0015) F | +0.0653 (0.0016) F | +0.0158 (0.0020) F | -0.0967 (0.0011) H |
| X_P_k1_p0.7 | 0.7 | 0.2882 | +0.1590 (0.0020) F | +0.0670 (0.0015) F | +0.0670 (0.0015) F | +0.0676 (0.0015) F | +0.0352 (0.0018) F | -0.0943 (0.0010) H |
| X_P_k1_p0.75 | 0.75 | 0.2921 | +0.1817 (0.0021) F | +0.0834 (0.0016) F | +0.0834 (0.0016) F | +0.0677 (0.0016) F | +0.0399 (0.0017) F | -0.0966 (0.0010) H |
| X_P_k1_p0.8 | 0.8 | 0.2931 | +0.2069 (0.0021) F | +0.1012 (0.0016) F | +0.1012 (0.0016) F | +0.0669 (0.0015) F | +0.0514 (0.0019) F | -0.0972 (0.0010) H |
| X_P_k1_p0.9 | 0.9 | 0.2937 | +0.2506 (0.0024) F | +0.1333 (0.0018) F | +0.1333 (0.0018) F | +0.0717 (0.0016) F | +0.0715 (0.0019) F | -0.0984 (0.0011) H |
| A_k1.0_p1.0_g1.0 | 1 | 0.2922 | +0.3040 (0.0026) F | +0.1714 (0.0020) F | +0.1714 (0.0020) F | +0.0766 (0.0015) F | +0.0955 (0.0022) F | -0.0977 (0.0010) H |
crossings: naive 0.334; naive_pooled 0.497; epoch_fe 0.497; detrend none; misspecified 0.599; agentsem none

### ladder P_k2_T014  (paired rmse_X - rmse_anchor_only, mean (mc_se) verdict H/F/T)
| cell | product | anchor rmse | naive | naive_pooled | epoch_fe | detrend | misspecified | agentsem |
|---|---|---|---|---|---|---|---|---|
| X_P_k2_p0.15 | 0.3 | 0.2902 | -0.0138 (0.0015) H | -0.0557 (0.0013) H | -0.0557 (0.0013) H | +0.0553 (0.0017) F | -0.0881 (0.0013) H | -0.0962 (0.0012) H |
| X_P_k2_p0.175 | 0.35 | 0.2922 | +0.0032 (0.0015) F | -0.0440 (0.0013) H | -0.0440 (0.0013) H | +0.0560 (0.0016) F | -0.0812 (0.0015) H | -0.0981 (0.0012) H |
| F_p0.2_k2.0 | 0.4 | 0.2892 | +0.0273 (0.0015) F | -0.0268 (0.0013) H | -0.0268 (0.0013) H | +0.0585 (0.0014) F | -0.0706 (0.0015) H | -0.0947 (0.0010) H |
| X_P_k2_p0.225 | 0.45 | 0.2921 | +0.0466 (0.0015) F | -0.0142 (0.0012) H | -0.0142 (0.0012) H | +0.0587 (0.0014) F | -0.0595 (0.0017) H | -0.0969 (0.0010) H |
| X_P_k2_p0.25 | 0.5 | 0.2941 | +0.0653 (0.0017) F | -0.0002 (0.0013) T | -0.0002 (0.0013) T | +0.0605 (0.0015) F | -0.0447 (0.0021) H | -0.0976 (0.0011) H |
| X_P_k2_p0.275 | 0.55 | 0.2932 | +0.0910 (0.0017) F | +0.0181 (0.0014) F | +0.0181 (0.0014) F | +0.0608 (0.0015) F | -0.0227 (0.0023) H | -0.0968 (0.0011) H |
| F_p0.3_k2.0 | 0.6 | 0.2900 | +0.1147 (0.0018) F | +0.0345 (0.0014) F | +0.0345 (0.0014) F | +0.0629 (0.0017) F | +0.0013 (0.0020) T | -0.0962 (0.0011) H |
| X_P_k2_p0.325 | 0.65 | 0.2910 | +0.1353 (0.0019) F | +0.0493 (0.0016) F | +0.0493 (0.0016) F | +0.0633 (0.0015) F | +0.0175 (0.0020) F | -0.0962 (0.0011) H |
| X_P_k2_p0.35 | 0.7 | 0.2917 | +0.1595 (0.0019) F | +0.0672 (0.0015) F | +0.0672 (0.0015) F | +0.0680 (0.0017) F | +0.0301 (0.0018) F | -0.0972 (0.0011) H |
| X_P_k2_p0.375 | 0.75 | 0.2918 | +0.1870 (0.0021) F | +0.0865 (0.0016) F | +0.0865 (0.0016) F | +0.0672 (0.0016) F | +0.0429 (0.0019) F | -0.0966 (0.0011) H |
| X_P_k2_p0.4 | 0.8 | 0.2911 | +0.2073 (0.0022) F | +0.1022 (0.0017) F | +0.1022 (0.0017) F | +0.0693 (0.0015) F | +0.0524 (0.0020) F | -0.0960 (0.0011) H |
| X_P_k2_p0.45 | 0.9 | 0.2911 | +0.2582 (0.0023) F | +0.1381 (0.0018) F | +0.1381 (0.0018) F | +0.0719 (0.0016) F | +0.0749 (0.0019) F | -0.0974 (0.0011) H |
| A_k2.0_p0.5_g1.0 | 1 | 0.2900 | +0.3080 (0.0025) F | +0.1754 (0.0019) F | +0.1754 (0.0019) F | +0.0779 (0.0017) F | +0.1042 (0.0020) F | -0.0960 (0.0011) H |
| X_P_k2_p0.6 | 1.2 | 0.2912 | +0.4069 (0.0025) F | +0.2485 (0.0020) F | +0.2485 (0.0020) F | +0.0829 (0.0017) F | +0.1600 (0.0021) F | -0.0954 (0.0010) H |
| X_P_k2_p0.75 | 1.5 | 0.2886 | +0.5638 (0.0031) F | +0.3635 (0.0024) F | +0.3635 (0.0024) F | +0.1005 (0.0018) F | +0.2646 (0.0027) F | -0.0941 (0.0011) H |
| A_k2.0_p1.0_g1.0 | 2 | 0.2911 | +0.8192 (0.0043) F | +0.5537 (0.0033) F | +0.5537 (0.0033) F | +0.1219 (0.0018) F | +0.4648 (0.0037) F | -0.0963 (0.0011) H |
crossings: naive 0.341; naive_pooled 0.501; epoch_fe 0.501; detrend none; misspecified 0.597; agentsem none

### ladder T_k1_T07  (paired rmse_X - rmse_anchor_only, mean (mc_se) verdict H/F/T)
| cell | product | anchor rmse | naive | naive_pooled | epoch_fe | detrend | misspecified | agentsem |
|---|---|---|---|---|---|---|---|---|
| X_T_k1_p0.3_a7 | 0.3 | 0.3437 | -0.0780 (0.0016) H | -0.0982 (0.0014) H | -0.0982 (0.0014) H | -0.0104 (0.0016) H | -0.1346 (0.0014) H | -0.1467 (0.0013) H |
| X_T_k1_p0.4_a7 | 0.4 | 0.3491 | -0.0420 (0.0018) H | -0.0673 (0.0017) H | -0.0673 (0.0017) H | -0.0125 (0.0018) H | -0.1130 (0.0020) H | -0.1505 (0.0014) H |
| X_T_k1_p0.5_a7 | 0.5 | 0.3460 | +0.0001 (0.0018) T | -0.0318 (0.0017) H | -0.0318 (0.0017) H | -0.0101 (0.0018) H | -0.0762 (0.0020) H | -0.1484 (0.0013) H |
| X_T_k1_p0.6_a7 | 0.6 | 0.3466 | +0.0471 (0.0018) F | +0.0085 (0.0016) F | +0.0085 (0.0016) F | -0.0026 (0.0017) T | -0.0411 (0.0016) H | -0.1472 (0.0012) H |
| X_T_k1_p0.7_a7 | 0.7 | 0.3473 | +0.0924 (0.0023) F | +0.0476 (0.0020) F | +0.0476 (0.0020) F | -0.0021 (0.0017) T | -0.0166 (0.0021) H | -0.1492 (0.0013) H |
| X_T_k1_p0.8_a7 | 0.8 | 0.3472 | +0.1377 (0.0022) F | +0.0862 (0.0020) F | +0.0862 (0.0020) F | +0.0030 (0.0017) T | +0.0064 (0.0021) F | -0.1493 (0.0014) H |
| X_T_k1_p1_a7 | 1 | 0.3476 | +0.2388 (0.0024) F | +0.1742 (0.0021) F | +0.1742 (0.0021) F | +0.0096 (0.0016) F | +0.0710 (0.0022) F | -0.1500 (0.0012) H |
crossings: naive 0.500; naive_pooled 0.579; epoch_fe 0.579; detrend 0.740; misspecified 0.772; agentsem none

### ladder T_k1_T028  (paired rmse_X - rmse_anchor_only, mean (mc_se) verdict H/F/T)
| cell | product | anchor rmse | naive | naive_pooled | epoch_fe | detrend | misspecified | agentsem |
|---|---|---|---|---|---|---|---|---|
| X_T_k1_p0.2_a28 | 0.2 | 0.2382 | +0.0396 (0.0012) F | -0.0350 (0.0007) H | -0.0350 (0.0007) H | +0.1397 (0.0016) F | -0.0451 (0.0009) H | -0.0466 (0.0008) H |
| X_T_k1_p0.3_a28 | 0.3 | 0.2379 | +0.0721 (0.0015) F | -0.0221 (0.0008) H | -0.0221 (0.0008) H | +0.1381 (0.0016) F | -0.0414 (0.0009) H | -0.0458 (0.0008) H |
| X_T_k1_p0.4_a28 | 0.4 | 0.2363 | +0.1120 (0.0017) F | -0.0065 (0.0009) H | -0.0065 (0.0009) H | +0.1439 (0.0015) F | -0.0364 (0.0010) H | -0.0455 (0.0008) H |
| X_T_k1_p0.5_a28 | 0.5 | 0.2377 | +0.1508 (0.0017) F | +0.0100 (0.0010) F | +0.0100 (0.0010) F | +0.1447 (0.0017) F | -0.0299 (0.0012) H | -0.0463 (0.0009) H |
| X_T_k1_p0.6_a28 | 0.6 | 0.2385 | +0.1933 (0.0018) F | +0.0264 (0.0010) F | +0.0264 (0.0010) F | +0.1484 (0.0016) F | -0.0191 (0.0014) H | -0.0475 (0.0008) H |
| X_T_k1_p0.7_a28 | 0.7 | 0.2372 | +0.2452 (0.0021) F | +0.0494 (0.0012) F | +0.0494 (0.0012) F | +0.1524 (0.0018) F | +0.0017 (0.0022) T | -0.0467 (0.0008) H |
| X_T_k1_p0.8_a28 | 0.8 | 0.2378 | +0.2903 (0.0022) F | +0.0688 (0.0013) F | +0.0688 (0.0013) F | +0.1553 (0.0016) F | +0.0367 (0.0027) F | -0.0487 (0.0009) H |
crossings: naive none; naive_pooled 0.440; epoch_fe 0.440; detrend none; misspecified 0.692; agentsem none

## 6. The crossing region

The closed form (`results/crossover/closed_form.py`, derivation in `DERIVATION.md`) gives the
population RMSE of `anchor_only`, `naive` and `naive_pooled` as a quadratic in the product, with
no persons simulated; it was checked against the 12 frozen sweep-A cells (max absolute RMSE
error 0.0037) before any new cell was scored. `anchor_only`'s RMSE at each T0 is within 0.002 of it (the
line under the P3 table in section 7), and every estimator's measured crossing is on its ladder's
`crossings:` line in section 5. The closed form's crossings against the measured ones, with the
tie band (products whose cell ties), are below; every difference is within 0.010:

| ladder | estimator | measured | tie band | closed form | diff |
|---|---|---|---|---|---|
| P_k0.5_T014 | naive | 0.330 | none | 0.331 | -0.001 |
| P_k0.5_T014 | naive_pooled | 0.492 | 0.5 to 0.5 | 0.491 | +0.001 |
| P_k1_T014 | naive | 0.334 | none | 0.331 | +0.003 |
| P_k1_T014 | naive_pooled | 0.497 | 0.5 to 0.5 | 0.491 | +0.006 |
| P_k2_T014 | naive | 0.341 | none | 0.331 | +0.010 |
| P_k2_T014 | naive_pooled | 0.501 | 0.5 to 0.5 | 0.491 | +0.009 |
| T_k1_T028 | naive | none | none | 0.000 | - |
| T_k1_T028 | naive_pooled | 0.440 | none | 0.447 | -0.007 |
| T_k1_T07 | naive | 0.500 | 0.5 to 0.5 | 0.494 | +0.005 |
| T_k1_T07 | naive_pooled | 0.579 | none | 0.578 | +0.000 |

The three T0 = 14 ladders are one curve. At every matched product the paired differences of
the kappa 0.5, 1 and 2 ladders agree to within 0.009, with a median spread of 0.005, about two
standard errors of a between-cell difference (single-cell mc_se 0.0011 to 0.0025):

| product | cells | naive | epoch_fe | misspecified | agentsem | worst |
|---|---|---|---|---|---|---|
| 0.3 | X_P_k0.5_p0.6, F_p0.3_k1.0, X_P_k2_p0.15 | 0.0021 | 0.0018 | 0.0005 | 0.0009 | 0.0021 |
| 0.35 | X_P_k0.5_p0.7, X_P_k1_p0.35, X_P_k2_p0.175 | 0.0049 | 0.0035 | 0.0032 | 0.0018 | 0.0049 |
| 0.4 | X_P_k0.5_p0.8, X_P_k1_p0.4, F_p0.2_k2.0 | 0.0035 | 0.0027 | 0.0046 | 0.0039 | 0.0046 |
| 0.45 | X_P_k0.5_p0.9, X_P_k1_p0.45, X_P_k2_p0.225 | 0.0055 | 0.0038 | 0.0019 | 0.0025 | 0.0055 |
| 0.5 | A_k0.5_p1.0_g1.0, A_k1.0_p0.5_g1.0, X_P_k2_p0.25 | 0.0041 | 0.0024 | 0.0019 | 0.0026 | 0.0041 |
| 0.55 | X_P_k1_p0.55, X_P_k2_p0.275 | 0.0001 | 0.0000 | 0.0010 | 0.0006 | 0.0010 |
| 0.6 | X_P_k1_p0.6, F_p0.3_k2.0 | 0.0040 | 0.0022 | 0.0010 | 0.0005 | 0.0040 |
| 0.65 | X_P_k1_p0.65, X_P_k2_p0.325 | 0.0042 | 0.0028 | 0.0017 | 0.0005 | 0.0042 |
| 0.7 | X_P_k1_p0.7, X_P_k2_p0.35 | 0.0005 | 0.0002 | 0.0051 | 0.0029 | 0.0051 |
| 0.75 | X_P_k1_p0.75, X_P_k2_p0.375 | 0.0052 | 0.0032 | 0.0030 | 0.0000 | 0.0052 |
| 0.8 | X_P_k1_p0.8, X_P_k2_p0.4 | 0.0004 | 0.0009 | 0.0010 | 0.0012 | 0.0012 |
| 0.9 | X_P_k1_p0.9, X_P_k2_p0.45 | 0.0076 | 0.0048 | 0.0034 | 0.0010 | 0.0076 |
| 1 | A_k1.0_p1.0_g1.0, A_k2.0_p0.5_g1.0 | 0.0040 | 0.0040 | 0.0086 | 0.0017 | 0.0086 |

worst matched-product spread: 0.0086 (P1 bar 0.02: HOLDS)

Two features of the ladders are not in the crossing numbers. First, `agentsem`'s margin over
the anchor is flat in the product on every ladder but moves with T0: about -0.095 at T0 = 14,
-0.149 at T0 = 7 and -0.047 at T0 = 28, because its own RMSE stays near 0.19 while the
anchor's falls from 0.35 to 0.29 to 0.24. Second, `detrend`'s margin moves far less with the
product than any other estimator's (+0.052 at product 0.15 to +0.122 at product 2 on the
T0 = 14 ladders, against +0.819 for `naive` over the same span), so whether it wins is set
mostly by T0: it loses to a 14- or 28-occasion anchor at every product and beats a 7-occasion
anchor up to product 0.74.

## 7. Predictions against outcomes

The three predictions were written into the driver's docstring with the registry at `a8b075e`.

**P1, product invariance** (matched products agree within 0.02 for `naive`, `epoch_fe`,
`misspecified`, `agentsem` at T0 = 14): holds (section 6).

**P2, crossings at T0 = 14**:

| ladder | products spanned | estimator | band | measured | hit |
|---|---|---|---|---|---|
| P_k0.5_T014 | 0.15 to 0.5 | naive | [0.3, 0.4] | 0.330 | yes |
| P_k0.5_T014 | 0.15 to 0.5 | epoch_fe | [0.45, 0.55] | 0.492 | not spanned |
| P_k0.5_T014 | 0.15 to 0.5 | misspecified | [0.6, 0.8] | none | not spanned |
| P_k0.5_T014 | 0.15 to 0.5 | detrend | none | none | yes |
| P_k0.5_T014 | 0.15 to 0.5 | agentsem | none | none | yes |
| P_k1_T014 | 0.3 to 1 | naive | [0.3, 0.4] | 0.334 | yes |
| P_k1_T014 | 0.3 to 1 | epoch_fe | [0.45, 0.55] | 0.497 | yes |
| P_k1_T014 | 0.3 to 1 | misspecified | [0.6, 0.8] | 0.599 | NO |
| P_k1_T014 | 0.3 to 1 | detrend | none | none | yes |
| P_k1_T014 | 0.3 to 1 | agentsem | none | none | yes |
| P_k2_T014 | 0.3 to 2 | naive | [0.3, 0.4] | 0.341 | yes |
| P_k2_T014 | 0.3 to 2 | epoch_fe | [0.45, 0.55] | 0.501 | yes |
| P_k2_T014 | 0.3 to 2 | misspecified | [0.6, 0.8] | 0.597 | NO |
| P_k2_T014 | 0.3 to 2 | detrend | none | none | yes |
| P_k2_T014 | 0.3 to 2 | agentsem | none | none | yes |

**P3, anchor length** (every crossing higher at T0 = 7 and lower at T0 = 28 than at T0 = 14;
`anchor_only`'s RMSE falls with T0):

| estimator | T0=7 | T0=14 | T0=28 | higher at 7 | lower at 28 |
|---|---|---|---|---|---|
| naive | 0.500 | 0.334 | none | yes | n/a |
| naive_pooled = epoch_fe | 0.579 | 0.497 | 0.440 | yes | yes |
| detrend | 0.740 | none | none | n/a | n/a |
| misspecified | 0.772 | 0.599 | 0.692 | yes | NO |
| agentsem | none | none | none | n/a | n/a |

anchor_only rmse by T0, mean over the ladder's cells (closed form): T0=7: 0.3468 (0.3457), T0=14: 0.2923 (0.2909), T0=28: 0.2377 (0.2389)

`naive` has no crossing at T0 = 28 because it is already worse than the anchor at the lowest
rung (product 0.2, +0.040), which is the direction P3 predicts and what the closed form says
(P* = 0 at T0 = 28: with 28 anchor and 28 adaptive occasions the two scores tie at product 0 and
`naive` loses for any product above it).

Misses for `misspecified` between T0 = 14 and T0 = 28: the crossing rises from 0.599 to 0.692
(tie at the 0.7 rung, +0.0017 (0.0022)). The arithmetic is in the ladders: at product 0.6 the
anchor's RMSE is 0.2925 at T0 = 14 and 0.2385 at T0 = 28, while `misspecified`'s is 0.2928 and
0.2194, so the extra anchor occasions help the corrected estimator by 0.073 and the anchor by
0.054. P3 was written as if only the anchor's error moved with T0. That is approximately right
for `naive` and the pooled pair, whose closed-form RMSE gets slightly worse as T0 grows (fewer
adaptive occasions), and wrong for an estimator that also fits the anchor epoch. The closed form
does not cover `misspecified`, so the miss has no derivation here.

## 8. What this means for the methods section

1. Report the crossover in units of psi x kappa with the anchor length stated, not as a
   kappa-specific finding. "At kappa 2 the hindsight estimator is reliably worse" is true of the
   frozen psi set, where kappa 2 means products 1 and 2, and would be false at psi 0.15 (product
   0.3), where `naive` still beats the anchor. The product is the sufficient statistic to the
   precision of this sweep (0.009 in RMSE).
2. A methods section that reports the kappa 1 row as "no advantage" should say that the
   advantage changes sign inside that row, at product 0.60 for `misspecified`, 0.50 for the
   pooled estimators and 0.33 for `naive`.
3. The anchor length is the design lever, and it is not a free one. A 28-occasion anchor beats
   `naive` at any dose and leaves the pooled estimators a window only up to product 0.44, while
   widening `misspecified`'s to 0.69; a 7-occasion anchor loses to everything up to product
   0.50 and even to `detrend` up to 0.74.
   Whatever anchor length a study can afford sets where its adaptive-phase estimators stop
   helping, and the closed form gives that point without a simulation for `naive` and the
   pooled pair.
4. `agentsem` is the only estimator that never crosses, at any product up to 2 and at any anchor
   length tried. That is the result the correct-surrogate correction is supposed to deliver,
   and this sweep is the clean statement of it: the crossover is a property of estimators that
   do not correct for the dose, not of the design.
5. Limits. gamma_C is fixed at 1.0 and the generator at `cumulative`; the trigger is the
   calendar one (every person switches at T0), psi is homogeneous, N = 200, T = 56, 200
   replicates per cell. Crossings are linear interpolations between rungs 0.05 apart (0.1 on the
   T ladders), so they carry about 0.01 of interpolation error on top of Monte Carlo error. The
   closed form covers three of the six estimators.

## 9. Receipts

Run: `uv run python scripts/crossover_sweep.py run --workers 10`, started 21:20:43 CDT on
2026-10-02 in tmux `xover-sweep`, stdout in `results/crossover/run.out` (one done line per
cell). One `RuntimeWarning` (overflow in exp, `trapworld/shadow.py:281`, the Hessian interval)
is frozen-code behaviour and does not touch rmse. Each of the 40 cell files stores its registry entry and
200 error-free replicates (`test_stored_cells_match_the_registry`); `run.out` ends with
`RUN-EXIT 0`.

Scoring: `uv run python scripts/crossover_sweep.py score` writes
`results/crossover/crossings.{json,md}`; `uv run python results/crossover/report_tables.py`
prints every other table in this file, and the hash comparison, from the recorded files. Tests:
`uv run pytest -q tests/test_crossover.py` (4 tests: registry hash and cell validity, seed-block
disjointness, crossing interpolation and verdicts, stored records against the registry). One reproducibility caveat: the scorer reads
the 10 frozen ladder cells from `results/grid/`, which is gitignored and kept through
`results/FROZEN_GRID_SHA256.txt` and the Drive backup, so regenerating `crossings.json` from a
fresh clone needs that restore first; the 40 new cells are committed.

Hashes: every field of `hashes_after.json` (written at 23:56, right after the run's exit) equals
the registry's except `script_src_sha256` (`e969abc8...` registered, `e29ec82c...` after); the
generator prints the fields that differ. The registry's sha is the pre-review driver, which was not
kept and matches no committed file. The run loaded that driver: its last lines in `run.out`
("all hashes match", "summarized 40 cells") are that driver's summarize mode, which neither
committed driver has, and the `summary.{json,csv}` it wrote were removed (they carried the same
hash values). The after sha is the `5e1bbe2` driver, whose `hashes()` wrote `hashes_after.json`.
What binds the records to the registry: that test, plus matching cells, PARAMS, world and
analyzers hashes.
