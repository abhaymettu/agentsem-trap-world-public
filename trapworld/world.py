"""The trap-world generator: spec sections 2.1 to 2.5, 0-indexed occasions.

``simulate(cell, seed)`` returns a ``World`` holding both what the analyst sees (spec 3)
and the hidden truth used only by metrics. Nine RNG streams are spawned per replicate in
the fixed order theta, drift, eps, nuisance, read, jitter, criterion, outcome, bootstrap,
and every noise array is drawn up front, so the kappa = 0 counterfactual behind ``dW``
reuses exactly the same draws and the difference is the policy's doing and nothing else.

Conventions decided here (the spec leaves them open):

* ``switch[i]`` is the index of the person's first adaptive occasion; ``version[i, t] = 0``
  for ``t < switch[i]`` and 1 otherwise. With the calendar trigger every person has
  ``switch = T0 = 14`` and therefore ``T_a = T - T0 = 42`` adaptive occasions. (Spec 3
  calls ``switch`` "last occasion of v0", which is the same boundary written 1-indexed.)
* The performance trigger is resolved online inside the one time loop: ``switch`` starts at
  ``T0`` for everyone and is lowered to ``t + 1`` the first time a person's running mean
  crosses the threshold. This is exact because a person contributes no dose and no shadow
  before their own switch, so nothing the trigger reads can depend on a future switch, and
  persons do not interact. It also makes the kappa = 0 counterfactual use the same switch
  times; the counterfactual is handed the main run's ``switch`` array to keep that explicit.
* ``belief`` is the policy's own state, so it is zero throughout for the ``uniform`` and
  ``none`` arms, which form no belief. That keeps ``implied_r`` honest: a uniform dose
  carries no person information, so ``kappa_eff`` and the implied contamination share are 0.
* ``jitter`` is returned as None when ``var_jitter = 0``; the logged randomized component
  only exists in the D2 arm, and that is what the "or None" in the contract is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trapworld.params import (
    KALMAN_P0_FLOOR,
    NU_NONPROPORTIONAL,
    READ_STREAM_VAR_DROP,
    STREAM_ORDER,
    TRIGGER_MIN_T,
    TRIGGER_THRESHOLD,
    TRIGGER_WINDOW,
    resolve,
)


@dataclass
class World:
    """One replicate. Observed fields first, hidden truth after (spec 3)."""

    y: np.ndarray                 # [N, T, p] stream indicators
    z: np.ndarray                 # [N, T, n_nuisance] nuisance indicators
    dose: np.ndarray              # [N, T] logged action
    version: np.ndarray           # [N, T] policy version, 0 or 1
    switch: np.ndarray            # [N] index of the first adaptive occasion
    jitter: np.ndarray | None     # [N, T] logged randomized dose component, None if var_jitter = 0
    C_clean: np.ndarray           # [N]
    C_react: np.ndarray           # [N]
    W: np.ndarray                 # [N, T] true outcome
    belief_logged: np.ndarray     # [N, T] policy belief; only belief_partial may read it

    theta: np.ndarray             # [N] hidden trait
    theta_t: np.ndarray           # [N, T] trait plus drift
    delta: np.ndarray             # [N, T] shadow
    psi_i: np.ndarray             # [N] per-person shadow strength
    belief: np.ndarray            # [N, T] same array as belief_logged
    params: dict                  # resolved cell plus dW_se and bootstrap_seed
    implied_r: float
    dW: float
    bootstrap_ss: np.random.SeedSequence | None = None


# --------------------------------------------------------------------------- surrogates
def surrogate_g1(dose: np.ndarray, switch: np.ndarray, T_a: int) -> np.ndarray:
    """G1 exposure D_it: cumulative dose since the switch, normalized by T_a."""
    occ = np.arange(dose.shape[1])[None, :]
    D = np.cumsum(dose, axis=1) / T_a
    return np.where(occ >= np.asarray(switch)[:, None], D, 0.0)


def surrogate_recency(dose: np.ndarray, switch: np.ndarray, halflife: float) -> np.ndarray:
    """G2 exposure R_it: exponentially weighted dose with the given half-life."""
    n, T = dose.shape
    rho = 2.0 ** (-1.0 / halflife)
    R = np.empty((n, T))
    acc = np.zeros(n)
    for t in range(T):
        acc = rho * acc + (1.0 - rho) * dose[:, t]
        R[:, t] = acc
    occ = np.arange(T)[None, :]
    return np.where(occ >= np.asarray(switch)[:, None], R, 0.0)


# --------------------------------------------------------------------------- draws
def _psi_i(pr: dict, theta: np.ndarray, w_psi: np.ndarray) -> np.ndarray:
    """Per-person shadow strength. G3 spreads it; G4 and "none" switch the dose term off."""
    gen, psi = pr["generator"], float(pr["psi"])
    n = theta.shape[0]
    if gen in ("none", "unlogged"):
        return np.zeros(n)
    if gen != "hetero":
        return np.full(n, psi)
    sd_theta = np.sqrt(pr["var_theta"])
    z_theta = theta / sd_theta if sd_theta > 0 else np.zeros(n)
    rho = pr["rho_psitheta"]
    return psi * (1.0 + pr["tau_psi"] * (rho * z_theta + np.sqrt(1.0 - rho * rho) * w_psi))


def _draw_noise(pr: dict, kids: list) -> dict:
    """Every random draw of the replicate, from the nine streams, before any dynamics."""
    gens = {name: np.random.Generator(np.random.PCG64(kid))
            for name, kid in zip(STREAM_ORDER, kids)}
    N, T, p = pr["N"], pr["T"], pr["p"]

    g_theta = gens["theta"]
    theta = np.sqrt(pr["var_theta"]) * g_theta.standard_normal(N)
    w_psi = g_theta.standard_normal(N)  # G3 person-level draw, taken after theta

    zd = gens["drift"].standard_normal((N, T))
    d = np.empty((N, T))
    d[:, 0] = pr["sd_d"] * zd[:, 0]
    inn = np.sqrt(1.0 - pr["rho_d"] ** 2) * pr["sd_d"]
    for t in range(1, T):  # loop over occasions, vectorized over persons
        d[:, t] = pr["rho_d"] * d[:, t - 1] + inn * zd[:, t]

    eps = gens["eps"].standard_normal((N, T, p)) * np.sqrt(np.asarray(pr["var_eps"], float))

    g_nuis = gens["nuisance"]
    xi = g_nuis.standard_normal(N)
    z = (pr["nuisance_loading"] * xi[:, None, None]
         + g_nuis.standard_normal((N, T, pr["n_nuisance"])) * np.sqrt(pr["nuisance_var_eps"]))

    return dict(
        theta=theta,
        theta_t=theta[:, None] + d,
        psi_i=_psi_i(pr, theta, w_psi),
        eps=eps,
        z=z,
        read_noise=gens["read"].standard_normal((N, T)) * np.sqrt(pr["var_read"]),
        jitter=gens["jitter"].standard_normal((N, T)) * np.sqrt(pr["var_jitter"]),
        crit=gens["criterion"].standard_normal((N, 2)) * np.sqrt(pr["var_C"]),
        w_noise=gens["outcome"].standard_normal((N, T)) * np.sqrt(pr["var_W"]),
    )


# --------------------------------------------------------------------------- dynamics
def _run(pr: dict, nz: dict, kappa: float, policy: str,
         switch_fixed: np.ndarray | None = None, perm: np.ndarray | None = None) -> dict:
    """One pass of the time loop, vectorized over persons.

    ``kappa`` and ``policy`` are passed in rather than read from ``pr`` so the kappa = 0
    counterfactual can reuse the identical noise draws.
    """
    N, T, p = pr["N"], pr["T"], pr["p"]
    T0 = pr["T0"]
    T_a = T - T0
    lam = np.asarray(pr["loadings"], dtype=float)
    mu = np.asarray(pr["intercepts"], dtype=float)
    nu = (lam if pr["shadow_pattern"] == "proportional"
          else np.asarray(NU_NONPROPORTIONAL, dtype=float))

    theta, theta_t, psi_i = nz["theta"], nz["theta_t"], nz["psi_i"]
    gen = pr["generator"]
    use_g1 = gen in ("cumulative", "hetero", "unlogged_plus")
    use_rec = gen == "recency"
    use_step = gen in ("unlogged", "unlogged_plus")
    step = pr["c0"] + pr["c1"] * theta
    rho_r = 2.0 ** (-1.0 / pr["halflife"])
    read_scale = np.sqrt(max(pr["var_read"] - READ_STREAM_VAR_DROP, 0.0) / pr["var_read"]) \
        if pr["var_read"] > 0 else 0.0

    y = np.empty((N, T, p))
    dose = np.zeros((N, T))
    delta = np.zeros((N, T))
    belief = np.zeros((N, T))
    W = np.empty((N, T))

    online_trigger = pr["trigger"] == "performance" and switch_fixed is None
    m_bar = np.zeros((N, T0)) if online_trigger else None
    switch = (np.full(N, T0, dtype=np.int64) if switch_fixed is None
              else switch_fixed.astype(np.int64, copy=True))
    pending = np.ones(N, dtype=bool)

    belief_prev = np.zeros(N)
    P = max(pr["var_theta"], KALMAN_P0_FLOOR)  # Kalman state variance, floored for Var(theta) = 0
    cum = np.zeros(N)   # G1 accumulator
    rec = np.zeros(N)   # G2 accumulator

    for t in range(T):
        adaptive = t >= switch

        # 1. dose: actions at t use information through t - 1 only.
        if policy == "uniform":
            dose[:, t] = np.where(adaptive, 1.0, 0.0)
        elif policy != "none":
            dose[:, t] = np.where(adaptive, kappa * belief_prev + nz["jitter"][:, t], 0.0)

        # 2. shadow. Under the F8 null person i is fed person perm[i]'s dose path.
        a_sh = dose[:, t] if perm is None else dose[perm, t]
        cum = cum + a_sh
        rec = rho_r * rec + (1.0 - rho_r) * a_sh
        d_t = np.zeros(N)
        if use_g1:
            d_t = d_t + psi_i * (cum / T_a)
        if use_rec:
            d_t = d_t + psi_i * rec
        if use_step and t >= pr["t_cb"]:
            d_t = d_t + step
        delta[:, t] = np.where(adaptive, d_t, 0.0)

        # 3. stream indicators.
        y[:, t, :] = mu + lam * theta_t[:, t, None] + nu * delta[:, t, None] + nz["eps"][:, t, :]

        # 4. belief, formed at every occasion including the anchor.
        if policy in ("ew", "kalman"):
            if pr["policy_reads_stream"]:
                r = ((y[:, t, :] - mu).mean(axis=1) / lam.mean()
                     + nz["read_noise"][:, t] * read_scale)
            else:
                r = theta_t[:, t] + nz["read_noise"][:, t]
            if policy == "ew":
                b = belief_prev + pr["gain_ew"] * (r - belief_prev)
            else:
                K = P / (P + pr["var_read"])
                b = belief_prev + K * (r - belief_prev)
                P = (1.0 - K) * P
        elif policy == "heuristic":
            b = (y[:, t, 0] - mu[0]) / lam[0]
        else:
            b = np.zeros(N)  # uniform and none form no belief
        belief[:, t] = b
        belief_prev = b

        # 5. true outcome.
        W[:, t] = (pr["beta_W"] * theta_t[:, t] + pr["gamma_W1"] * dose[:, t]
                   + pr["gamma_W2"] * dose[:, t] * theta_t[:, t] + nz["w_noise"][:, t])

        # 6. performance trigger, resolved online and only for persons still on v0.
        if online_trigger and t < T0:
            m_bar[:, t] = y[:, t, :].mean(axis=1)
            if t >= TRIGGER_MIN_T - 1:
                run = m_bar[:, t - TRIGGER_WINDOW + 1: t + 1].mean(axis=1)
                fire = pending & (run > TRIGGER_THRESHOLD)
                switch[fire] = t + 1
                pending &= ~fire

    return dict(y=y, dose=dose, delta=delta, belief=belief, W=W, switch=switch)


# --------------------------------------------------------------------------- entry point
def simulate(cell: dict, seed: int | np.random.SeedSequence) -> World:
    """Generate one replicate of the trap world."""
    pr = resolve(cell)
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    # SeedSequence.spawn mutates its parent's child counter, so spawning straight off the
    # caller's object would hand a second call to simulate a different set of nine streams.
    # Rebuilding from (entropy, spawn_key) is the same sequence every time, so a replicate is
    # reproducible no matter how many times or in what order it is run.
    kids = np.random.SeedSequence(entropy=ss.entropy, spawn_key=ss.spawn_key,
                                  pool_size=ss.pool_size).spawn(9)
    nz = _draw_noise(pr, kids)

    policy = pr["policy"]
    kappa = 0.0 if policy == "none" else float(pr["kappa"])  # the none arm doses nobody
    perm = None
    if pr["permute_dose"]:
        # F8 null. It uses the otherwise idle bootstrap stream, so no substantive stream moves.
        perm = np.random.Generator(np.random.PCG64(kids[8])).permutation(pr["N"])

    run = _run(pr, nz, kappa, policy, perm=perm)

    # kappa = 0 counterfactual on the same draws; a uniform arm's counterfactual is no help.
    cf_policy = "none" if policy == "uniform" else policy
    cf = _run(pr, nz, 0.0, cf_policy, switch_fixed=run["switch"], perm=perm)

    N, T = pr["N"], pr["T"]
    switch = run["switch"]
    version = (np.arange(T)[None, :] >= switch[:, None])
    n_ad = version.sum(axis=1)

    theta = nz["theta"]
    abar = (run["dose"] * version).sum(axis=1) / n_ad
    belief_mean = (run["belief"] * version).sum(axis=1) / n_ad
    C_clean = pr["lam_C"] * theta + nz["crit"][:, 0]
    C_react = pr["lam_C"] * theta + pr["gamma_C"] * abar + nz["crit"][:, 1]

    var_theta_hidden = float(np.var(theta, ddof=1)) if N > 1 else 0.0
    if kappa == 0.0 or var_theta_hidden <= 0.0:
        kappa_eff = 0.0
    else:
        kappa_eff = kappa * float(np.cov(belief_mean, theta, ddof=1)[0, 1]) / var_theta_hidden
    denom = pr["lam_C"] + pr["gamma_C"] * kappa_eff
    implied_r = float(pr["gamma_C"] * kappa_eff / denom) if denom != 0 else 0.0

    per_person = ((run["W"] - cf["W"]) * version).sum(axis=1) / n_ad
    dW = float(per_person.mean())
    dW_se = float(per_person.std(ddof=1) / np.sqrt(N)) if N > 1 else float("nan")

    params = dict(pr)
    params["dW_se"] = dW_se
    params["kappa_eff"] = float(kappa_eff)
    params["bootstrap_seed"] = [int(v) for v in kids[8].generate_state(4)]

    belief = run["belief"]
    return World(
        y=run["y"], z=nz["z"], dose=run["dose"], version=version.astype(np.int8), switch=switch,
        jitter=nz["jitter"] if pr["var_jitter"] > 0 else None,
        C_clean=C_clean, C_react=C_react, W=run["W"], belief_logged=belief,
        theta=theta, theta_t=nz["theta_t"], delta=run["delta"], psi_i=nz["psi_i"], belief=belief,
        params=params, implied_r=implied_r, dW=dW, bootstrap_ss=kids[8],
    )
