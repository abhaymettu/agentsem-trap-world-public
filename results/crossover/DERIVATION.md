# Closed-form person-score RMSE: anchor_only vs naive vs psi*kappa

Script: `results/crossover/closed_form.py` (PARAMS only, deterministic, no persons simulated).
Output: `results/crossover/closed_form.json`. Every number below comes from that output or
from the frozen `results/summary.csv` (sweep A, gamma_C = 1.0, N = 200, 200 reps per cell, 500 in
the k1 p0 and k1 p0.5 cells).

## Model, read off `trapworld/world.py::_run`

Policy "ew", generator "cumulative", proportional shadow (nu = lam), calendar trigger
(switch = T0 for everyone), var_jitter = 0, policy_reads_stream False, homogeneous psi.

    theta_it = theta_i + d_it,            d AR(1), rho_d = 0.9, stationary sd 0.3
    b_t      = b_{t-1} + g (theta_it + r_it - b_{t-1}),  b_{-1} = 0,  r_it ~ N(0, var_read)
    dose_t   = kappa * b_{t-1}                 (t >= T0, else 0)
    delta_t  = psi * sum_{s=T0..t} dose_s / T_a   (t >= T0, else 0),  T_a = T - T0
    y_itj    = lam_j (theta_it + delta_it) + eps_itj

psi and kappa enter only through the product P = psi*kappa. Every quantity is linear in
(theta_i, d_i0..d_i55, r_i0..r_i55), so the script carries coefficient vectors through
the time loop and takes exact variances with the known covariance of that basis.

## From person means to the Bartlett score

For a set M of occasions, the person-mean battery is X_i = lam * eta_i + ebar_i with
eta_i = mean_{t in M}(theta_it + delta_it) and Var(ebar_ij) = var_eps_j / |M|, independent
of eta. That is an exact one-factor model, and `fit_one_factor` fixes the marker loading
lam_0 = 1 (which is the true lam_0), so in the population the fit recovers lam and the
residual variances, and the Bartlett score is

    s_i = eta_i + n_i,   Var(n_i) = 1 / (|M| * sum_j lam_j^2 / var_eps_j) = 1 / (3.3 |M|).

The latent metric is therefore the marker's metric, the raw theta scale, so the adaptive
inflation of eta (coefficient a = 1 + A*P on theta) enters the RMSE unshrunk.

The mean structure is saturated, so intercepts = sample means and scores are centered;
theta is not. With N persons:

    E[MSE] = (1 - 1/N) (Var(eta - theta) + Var(n)) + var_theta / N.

Since eta - theta = dbar_M + P*w exactly, Var(eta - theta) = c0 + 2 c1 P + c2 P^2.
anchor_only (M = t < T0): delta = 0, so its RMSE is flat in P. naive: M = t >= T0.
naive_pooled: M = all 56 occasions. epoch_fe equals it in every replicate, not only in the
population: on the balanced panel, removing occasion means and then averaging over all 56
occasions shifts each item's person-mean column by a constant, which the saturated mean
structure absorbs (frozen values agree to 1e-9, 0.1886 for both; the gap is optimizer tolerance).

## Crossing in closed form

anchor_only is flat in P, and the centering term is identical on both sides, so N cancels:

    c2 P^2 + 2 c1 P + D = 0,   D = (c0 + Var(n))_naive - (c0 + Var(n))_anchor
    P* = (-c1 + sqrt(c1^2 - c2 D)) / c2      (P* = 0 when D >= 0)

T0 = 14 coefficients: naive c0 0.03164, c1 0.01495, c2 0.28591 (theta part 0.26153,
drift 0.00952, read noise 0.01486), A 0.5114, Var(n) 0.00722; anchor c0 0.05840,
Var(n) 0.02165. The quadratic root matches a numerical root of the full RMSE to 1e-10.

## Against the frozen grid (T0 = 14)

One cell per product (the cells behind the naive reference list). The anchor_only list in
the brief mixes cells: 0.2921 is the k0 p0.5 cell (product 0); the k0.5 p0.5 cell
(product 0.25) is 0.2903.

| product | cell | anchor frozen / pred / err | naive frozen / pred / err | pooled frozen / pred / err |
|---|---|---|---|---|
| 0 | k0 p0 | 0.2916 / 0.2909 / 0.0007 | 0.2089 / 0.2090 / 0.0001 | 0.1886 / 0.1888 / 0.0002 |
| 0.25 | k0.5 p0.5 | 0.2903 / 0.2909 / 0.0007 | 0.2618 / 0.2624 / 0.0006 | 0.2249 / 0.2256 / 0.0008 |
| 0.5 | k1 p0.5 | 0.2913 / 0.2909 / 0.0003 | 0.3581 / 0.3601 / 0.0020 | 0.2923 / 0.2936 / 0.0013 |
| 1.0 | k1 p1 | 0.2922 / 0.2909 / 0.0013 | 0.5962 / 0.5982 / 0.0021 | 0.4635 / 0.4656 / 0.0020 |
| 2.0 | k2 p1 | 0.2911 / 0.2909 / 0.0001 | 1.1103 / 1.1140 / 0.0037 | 0.8448 / 0.8473 / 0.0025 |

Over all 12 sweep A cells at gamma_C = 1.0 and the three analyzers (36 rows), the max
absolute error is 0.0037, well inside the 0.02 bar, and the largest error is 2.08 Monte
Carlo SEs (naive, k1 p0.5). The prediction sits above the frozen mean in 25 of 36 rows,
which fits the two omitted O(1/N) terms: the Jensen gap (the grid averages sqrt(MSE) per
rep, which is below sqrt of the mean MSE) and sampling error in the fitted loadings.

## Predictions (T = 56, N = 200)

| T0 | T_a | anchor_only | naive P=0 / 0.25 / 0.5 / 1 / 2 | pooled P=0 / 0.25 / 0.5 / 1 / 2 |
|---|---|---|---|---|
| 7 | 49 | 0.3457 | 0.1980 / 0.2511 / 0.3482 / 0.5837 / 1.0917 | 0.1888 / 0.2337 / 0.3161 / 0.5190 / 0.9613 |
| 14 | 42 | 0.2909 | 0.2090 / 0.2624 / 0.3601 / 0.5982 / 1.1140 | 0.1888 / 0.2256 / 0.2936 / 0.4656 / 0.8473 |
| 28 | 28 | 0.2389 | 0.2389 / 0.2923 / 0.3890 / 0.6291 / 1.1555 | 0.1888 / 0.2096 / 0.2482 / 0.3537 / 0.6039 |

Crossing product P* (naive or pooled RMSE rises to anchor_only):

| T0 | naive P* | naive_pooled P* |
|---|---|---|
| 7 | 0.4943 | 0.5784 |
| 14 | 0.3308 | 0.4914 |
| 28 | 0 (tie at P = 0) | 0.4467 |

At T0 = 28 the naive and anchor epochs both have 28 occasions and the drift is stationary,
so they tie exactly at P = 0 and naive loses for every P > 0. The T0 = 14 crossings are
consistent with the frozen grid: naive 0.2618 at P = 0.25 and 0.3581 at P = 0.5 bracket the
anchor value, and pooled 0.2923 at P = 0.5 sits just above the anchor's 0.2913. The T0 = 7
and T0 = 28 rows have no frozen cells to check against yet.

## What controls the crossing

Among the generator knobs, the crossing is a function of the product psi*kappa alone:
psi and kappa never enter the equations separately (with zero jitter and homogeneous psi).
The P^2 term is dominated by the inflation of theta itself: at T0 = 14, c2 = 0.28591 of
which 0.26153 = A^2 var_theta, with A = 0.5114, the average over adaptive occasions of
(fraction of the adaptive phase elapsed) times (the belief's theta coefficient,
1 - (1 - g)^(t+1), already near 1 by T0). So the crossing is mostly a race between A*P
and the anchor's head start, D, which T0 sets: a short anchor has a large error (0.3457
at T0 = 7), giving naive more room (P* 0.4943); a long anchor removes the room (P* 0 at
T0 = 28). var_read and gain_ew act through the belief error only: they set the drift and
read-noise parts of c2 and the cross term c1, and the lag in the belief's theta
coefficient. Over var_read in {0, 2, 8} and gain_ew in {0.1, 0.3, 0.6} at T0 = 14 the naive
crossing moves from 0.3069 (var_read 8, gain 0.6) to 0.3763 (var_read 0, gain 0.1); more
read noise lowers P*, a slower gain raises it, because the belief carries less theta and
averages the read noise away. The crossing does not depend on N in this approximation, but
it does depend on var_theta, since the dominant term scales with it.
