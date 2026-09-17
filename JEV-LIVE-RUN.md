# Running the live Jev arm (`J_live`)

No live Jev grid has been run. Everything in the repo today is the fake transport, which
answers offline from a seeded stub named `fake-jev-stub` and is a plumbing check, not a Jev
result. This file is the procedure for the separately orchestrated session that would run the
real thing, and the boundaries it may not cross. The authority above it is
`JEV-CONSTRAINT-20260916.md`, then spec section 11 A3.

The Jev arm is an alternative adaptive policy and nothing else. The deterministic `ew` arm
stays the central result, and `J_live` is excluded from every headline aggregate.

## (a) Preconditions

1. The deterministic grid has already run to completion. The Jev arm is a robustness arm on
   top of a finished central result, never a thing to run first.
2. `results/hashes.json` exists and the runner's re-check passes:
   `.venv/bin/python -m trapworld.run_grid --check-hashes`
   (the runner and that flag are being built by the grid task; the live run does not start
   until the check exits clean). The hashed set includes `JEV_POLICY`, whose hash is
   `trapworld.jev.JEV_POLICY_HASH`, so the question text and the cadence are pinned before a
   single call is paid for.
3. `.venv/bin/pytest` is green on the commit being run.
4. The fake arm has already been run on the same cell, so the plumbing, the log schema and
   the call count are known before any money is spent.

## (b) The credential

The key lives in the vault. It is injected into the runner's process environment for that one
process, read from the vault's read command inline, with a leading space so the shell keeps
the line out of history:

```
  TYPESAFE_API_KEY="$(<vault read command>)" .venv/bin/python -m trapworld.run_grid --cells J_live --jev live
```

The leading space is load bearing (`HISTCONTROL`/`setopt histignorespace` must be on in the
shell doing this; check before, not after).

It is never written to a file, a `.env`, a log, a fixture, a results file, a shell alias or
any source file, and it is never echoed. `HttpJevTransport` reads it from `os.environ` inside
the call, puts it in one outgoing header, and keeps no copy: it is not stored on the
instance, not returned, and not in any exception message. If it ever does land somewhere,
rotate it in the provider console before anything else.

Nothing in the test suite reads the environment variable, and the one test that covers the
live transport asserts it raises `JevCredentialMissing` with `urlopen` monkeypatched, so a
stray key in a developer shell cannot cause a test to call the API.

## (c) What the run writes

- `results/jev/J_live_rep<k>.jsonl`, one record per call, written by `JevLog`. Keys: `arm`,
  `cell_id`, `rep`, `person`, `occasion`, `timestamp_utc`, `requested_model`,
  `returned_model`, `question` (the verbatim instructions and criteria), `state` (the bounded
  seven day state, all numbers rounded to 2 dp), `answer` (`score`, `probabilities`,
  `confidence`, `legend`, or null on a failure), `latency_ms`, `http_status`,
  `failure_class`; plus `request_id` only when the service returned one and `usage` only when
  the response body carried one. No credential, no header dump, no person identity beyond the
  replicate's own index. The live arm logs every person (the fake arm samples persons 0 and
  1); a failed call is always written whatever the sampling.
- `results/grid/J_live.json`, the cell's results plus the per replicate Jev record
  (`arm`, `status`, `failure`, `n_calls`, `n_failed`, `failure_classes`, `mean_latency_ms`,
  `returned_models`, `n_logged`) returned by `run_jev_replicate`.
- `astra-usage-jev-live.json` beside the outputs: the provider's `usage` objects verbatim,
  per call, plus their totals. House rule for every non-Claude model API call. Copy it to the
  vault's `05-Bridge/exec/files/` in the same session. The per call side of this is already in
  the JSONL; assembling the aggregate file is the runner's job.

`returned_model` is whatever the response's `model` field says, and it is null if the
response omits it. Nothing invents a model name, a request id or a seed the service did not
return. The two public doc pages do not promise a request or trace id at all, so most or all
records may legitimately have no `request_id`.

## (d) What the live run cannot change

`params.py`, `grid.py`, `rules.json`, the surrogate, `certificate.py` and the analyzers are
hashed before the run and re-checked after. A mismatch fails the run and the Jev outputs are
discarded, not patched.

The Jev arm's outputs are one extra results cell. No DGP code and no analysis code reads
them. Inside the generator the policy's only effect is the logged dose: the trait, the drift,
the shadow, the criteria and the outcome are produced by the same code paths as every other
arm, the policy sees only past occasions through the `external_policy` view, and its belief
is zero throughout. No Jev output may move a frozen parameter, a certificate threshold or an
analysis choice, and none can, because nothing downstream reads anything but `dose`.

## (e) Call budget

One call per adaptive person per decision occasion. At `cadence = 7` with `T0 = 14` and
`T = 56`, decisions land on occasions 14, 21, 28, 35, 42 and 49, so `ceil(T_a / cadence) = 6`
per person:

```
200 persons x 6 decisions x 5 reps = 6000 calls for J_live
```

That is the whole budget for the arm. The counterfactual run behind `dW` uses the no dose
policy, so it costs nothing. The fake arm was measured at 1200 calls per replicate, which is
the same arithmetic at one rep, so a live run that reports anything other than 1200 calls per
completed replicate is a bug, not a surprise. Do not spend calls to decorate the build: one
`J_live` grid, five reps, and stop.

## (f) Failure policy

Any call that comes back with a failure class (`http_401`, `http_422`, `http_429`,
`http_529`, `http_other`, `timeout`, `connection`, `schema`) raises `JevArmFailed` after the
failure has been logged. `run_jev_replicate` then returns `(None, record)` with
`status: "failed"` and the person, occasion and class of the failure.

That replicate is missing. It is never filled in from the deterministic arm, never retried
silently inside the run, and never averaged over as if it had succeeded. Re-run failed
replicates by hand, one at a time, after looking at the failure class: a 429 means wait, a
401 means the credential is wrong or rotated, a 422 means the request body drifted from the
contract and the run should stop until that is understood. Re-running a replicate replaces
its JSONL file, because a `JevLog` truncates its path when it is constructed.

The count of completed replicates is reported next to the results. Four of five is a
four of five result, reported as such.

## (g) Reproducing it offline

The JSONL log is a complete record of every answer, so the whole arm replays with no
credential and no network:

```
.venv/bin/python -m trapworld.run_grid --cells J_live --jev recorded --recorded-dir results/jev
```

`RecordedJevTransport` serves each recorded answer by `(person, occasion)` and raises
`JevLogExhausted` if the run asks for a call the log does not have, so a divergence is loud
rather than silent. The test suite proves the replay reproduces the dose array bit for bit on
the fake arm. Replay is the way anyone else checks the Jev arm, and the way it stays checkable
after the credential is rotated.

## (h) Reporting

The morning handoff keeps the two apart:

- Deterministic central results first, on their own, with no Jev number in any headline
  aggregate.
- A separate Jev section that states whether a live grid ran. If it did not, it says so
  plainly and reports the fake arm as a plumbing check with the stub model name attached. If
  it did, it gives the returned model string, the number of completed replicates out of five,
  the failure classes seen, the mean latency, and the total usage, and it points at both
  `results/jev/` and `astra-usage-jev-live.json`.
- Any Jev-specific difference from the deterministic adaptive arm is reported as a policy
  difference, since that is the only channel through which it can act.
