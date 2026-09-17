"""Tests for trapworld.jev (Task 9): the optional Jev policy arm.

Nothing here touches a network or a credential. Every test runs the fake transport, which is
deterministic given its SeedSequence, and the one test that covers the live transport proves
it refuses before any socket is opened.

The two properties these tests exist to protect, from JEV-CONSTRAINT-20260916.md: the Jev arm
cannot move the data-generating process (test 4), and a failed call makes the replicate
explicitly missing instead of quietly borrowing another policy's numbers (test 6).
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import pytest

from trapworld import jev
from trapworld import world as tw

SEED = 20260916
CADENCE = jev.JEV_POLICY["cadence"]
WINDOW = jev.JEV_POLICY["window"]
QID = jev.JEV_POLICY["question_id"]

RECORD_KEYS = ["arm", "cell_id", "rep", "person", "occasion", "timestamp_utc",
               "requested_model", "returned_model", "question", "state", "answer",
               "latency_ms", "http_status", "failure_class"]


def fake_parts(seed=SEED, *, sample_persons=None, path=None, arm="jev_fake",
               cell_id="J_test", rep=0, kappa=1.0, **fake_kw):
    """The (SeedSequence, log, policy) for one fake replicate, not yet run.

    The transport sits on child 9 of the replicate's SeedSequence, exactly where
    ``make_transport`` puts it, so the nine DGP streams (children 0 to 8) are untouched.
    """
    ss = np.random.SeedSequence(seed)
    child = np.random.SeedSequence(entropy=ss.entropy, spawn_key=tuple(ss.spawn_key) + (9,),
                                   pool_size=ss.pool_size)
    log = jev.JevLog(path=path, sample_persons=sample_persons, arm=arm)
    policy = jev.JevPolicy(jev.FakeJevTransport(child, **fake_kw), log, kappa=kappa,
                           cadence=CADENCE, window=WINDOW, cell_id=cell_id, rep=rep)
    return ss, log, policy


def run_fake(seed=SEED, cell=None, **kwargs):
    ss, log, policy = fake_parts(seed, **kwargs)
    world = tw.simulate({**(cell or {}), "policy": "external"}, ss, external_policy=policy)
    return world, log, policy


@pytest.fixture(scope="module")
def fake_world():
    world, log, policy = run_fake()
    return world, log, policy


class StrictView(dict):
    """A world view that refuses to hand out anything the policy is not allowed to see."""

    ALLOWED = {"t", "y", "dose", "switch", "adaptive"}

    def __getitem__(self, key):
        if key not in self.ALLOWED:
            raise AssertionError(f"build_state read the forbidden view key {key!r}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        return self[key]


# --------------------------------------------------------------------------- 1. contract
def test_build_request_matches_the_api_contract(fake_world):
    """https://docs.typesafe.ai/api.md: model, state, questions; one Score question."""
    world = fake_world[0]
    state = jev.build_state(
        {"t": 30, "y": world.y[:, :30], "dose": world.dose[:, :30],
         "switch": world.switch, "adaptive": world.version[:, 30] == 1}, 0, WINDOW)
    request = jev.build_request(state)

    assert set(request) == {"model", "state", "questions"}
    assert request["model"] == "jev-latest"
    assert request["state"] is state
    assert list(request["questions"]) == [QID]

    question = request["questions"][QID]
    assert set(question) == {"type", "instructions", "criteria"}
    assert question["type"] == "score"
    assert isinstance(question["instructions"], str) and question["instructions"].strip()
    assert isinstance(question["criteria"], list) and len(question["criteria"]) == 4
    assert 2 <= len(question["criteria"]) <= 10  # /primitives/score.md allows 2 to 10 levels
    assert all(isinstance(level, str) and level.strip() for level in question["criteria"])
    for name in jev.CHANNELS:
        assert name in question["instructions"]

    json.dumps(request)  # the body has to be plain JSON, with no numpy types in it


# --------------------------------------------------------------------------- 2. the state
def test_state_is_bounded_and_reads_nothing_it_should_not(fake_world):
    world = fake_world[0]
    view = StrictView(t=30, y=world.y[:, :30], dose=world.dose[:, :30], switch=world.switch,
                      adaptive=world.version[:, 30] == 1,
                      theta=world.theta, belief=world.belief, W=world.W, delta=world.delta)

    state = jev.build_state(view, 7, WINDOW)
    assert set(state) == {"day", "days_on_adaptive_policy", "recent_days"}
    assert state["day"] == 31
    assert state["days_on_adaptive_policy"] == 30 - world.params["T0"]
    assert len(state["recent_days"]) == WINDOW
    assert [d["day"] for d in state["recent_days"]] == list(range(24, 31))

    allowed = ["day", *jev.CHANNELS, "reminder_dose"]
    for day in state["recent_days"]:
        assert list(day) == allowed
        for key, value in day.items():
            if key == "day":
                assert isinstance(value, int)
            else:
                assert isinstance(value, float)
                assert value == round(value, jev.JEV_POLICY["round_dp"])

    short = jev.build_state(StrictView(t=3, y=world.y[:, :3], dose=world.dose[:, :3],
                                       switch=world.switch,
                                       adaptive=world.version[:, 3] == 1), 7, WINDOW)
    assert len(short["recent_days"]) == 3  # at most window days, fewer early on
    json.dumps(state)


# --------------------------------------------------------------------------- 3. determinism
def test_fake_arm_is_deterministic_and_seed_sensitive():
    a = run_fake(SEED)[0]
    b = run_fake(SEED)[0]
    np.testing.assert_array_equal(a.dose, b.dose)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.dW == b.dW
    assert a.implied_r == b.implied_r

    c = run_fake(SEED + 1)[0]
    assert not np.array_equal(a.dose, c.dose)
    assert not np.array_equal(a.y, c.y)


# --------------------------------------------------------------------------- 4. DGP intact
def test_jev_arm_cannot_move_the_data_generating_process():
    """The binding constraint: Jev is a policy, never a source of truth."""
    before = tw.simulate({}, SEED)
    world = run_fake(SEED)[0]
    none = tw.simulate({"policy": "none"}, SEED)

    for name in ("theta", "theta_t", "z", "C_clean"):
        np.testing.assert_array_equal(getattr(world, name), getattr(none, name), err_msg=name)

    # and the deterministic arm is untouched by having run a Jev replicate in this process
    # (dW was 0.3142273928429936 on the base cell both before and after the world.py hook)
    after = tw.simulate({}, SEED)
    for name in ("y", "z", "dose", "W", "belief", "theta", "theta_t", "delta", "C_react"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name), err_msg=name)
    assert after.dW == before.dW
    assert after.implied_r == before.implied_r

    with pytest.raises(ValueError):
        tw.simulate({"policy": "external"}, SEED)  # no callable, no silent fallback


# --------------------------------------------------------------------------- 5. log schema
def test_log_schema_and_summary():
    world, log, policy = run_fake(sample_persons=(0, 1))
    n_decisions = len(range(world.params["T0"], world.params["T"], CADENCE))

    assert log.records
    for record in log.records:
        assert list(record) == RECORD_KEYS  # exact keys, in order; no request_id, no usage
        assert record["arm"] == "jev_fake"
        assert record["cell_id"] == "J_test" and record["rep"] == 0
        assert record["requested_model"] == "jev-latest"
        assert record["returned_model"] == jev.FAKE_MODEL_NAME
        assert record["question"] == jev.question_payload()
        assert set(record["state"]) == {"day", "days_on_adaptive_policy", "recent_days"}
        assert set(record["answer"]) == {"score", "probabilities", "confidence", "legend"}
        assert record["http_status"] == 200
        assert record["failure_class"] is None
        assert record["latency_ms"] >= 0.0
        json.dumps(record)

    assert {r["person"] for r in log.records} == {0, 1}  # sample_persons respected

    summary = log.summary()
    assert summary["n_calls"] == policy.n_calls == world.params["N"] * n_decisions
    assert summary["n_logged"] == len(log.records) == 2 * n_decisions
    assert summary["n_failed"] == 0 and summary["failure_classes"] == {}
    assert summary["returned_models"] == [jev.FAKE_MODEL_NAME]
    assert summary["mean_latency_ms"] >= 0.0

    with_id = run_fake(sample_persons=(0,), request_id="trace-abc")[1]
    assert all(list(r) == RECORD_KEYS + ["request_id"] for r in with_id.records)
    assert {r["request_id"] for r in with_id.records} == {"trace-abc"}

    everyone = run_fake()[1]
    assert everyone.summary()["n_logged"] == everyone.summary()["n_calls"]


# --------------------------------------------------------------------------- 6. failure
def test_a_failed_call_makes_the_replicate_missing(monkeypatch, tmp_path):
    # Decisions land on occasions 14, 21, 28, 35, 42 and 49 under the frozen cadence of 7
    # with T0 = 14, so occasion 20 is never a call and cannot fail.
    assert run_fake(fail_at=(3, 20))[0] is not None

    ss, log, policy = fake_parts(sample_persons=(0, 1), fail_at=(3, 21))
    with pytest.raises(jev.JevArmFailed) as excinfo:
        tw.simulate({"policy": "external"}, ss, external_policy=policy)
    assert (excinfo.value.person, excinfo.value.occasion) == (3, 21)
    assert excinfo.value.failure_class == "http_529"

    last = log.records[-1]  # a failure is written even for a person outside the sample
    assert (last["person"], last["occasion"]) == (3, 21)
    assert last["failure_class"] == "http_529"
    assert last["http_status"] == 529
    assert last["answer"] is None
    assert last["returned_model"] is None
    assert policy.n_failed == 1

    real_make = jev.make_transport

    def failing(kind, ss_, cell_id, rep, recorded_dir=None):
        transport = real_make(kind, ss_, cell_id, rep, recorded_dir)
        transport.fail_at = (3, 21)
        return transport

    monkeypatch.setattr(jev, "make_transport", failing)
    world, record = jev.run_jev_replicate({}, np.random.SeedSequence(SEED), "fake",
                                          tmp_path, "J_fake", 0)
    assert world is None  # no world at all, never one stitched from another policy
    assert record["status"] == "failed"
    assert record["failure"] == {"person": 3, "occasion": 21, "failure_class": "http_529"}
    assert record["arm"] == "jev_fake"
    assert record["n_failed"] == 1 and record["failure_classes"] == {"http_529": 1}

    lines = (tmp_path / "J_fake_rep0.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["failure_class"] == "http_529"


# --------------------------------------------------------------------------- 7. live path
def test_live_transport_refuses_without_a_credential(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    opened = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *args, **kwargs: opened.append(args))

    transport = jev.HttpJevTransport()
    request = jev.build_request({"day": 1, "days_on_adaptive_policy": 0, "recent_days": []})
    with pytest.raises(jev.JevCredentialMissing):
        transport(request)
    assert opened == []  # it never got as far as a socket

    assert jev.classify_status(200) is None
    assert jev.classify_status(401) == "http_401"
    assert jev.classify_status(422) == "http_422"
    assert jev.classify_status(429) == "http_429"
    assert jev.classify_status(529) == "http_529"
    assert jev.classify_status(500) == "http_other"
    assert jev.classify_status(None) == "http_other"

    assert jev.extract_request_id({"nothing": 1}, {"content-type": "application/json"}) is None
    assert jev.extract_request_id({"request_id": "r1"}, None) == "r1"
    assert jev.extract_request_id(None, {"X-Request-Id": "r2"}) == "r2"


# --------------------------------------------------------------------------- 8. replay
def test_recorded_transport_replays_a_run(tmp_path):
    path = tmp_path / "J_fake_rep0.jsonl"
    original = run_fake(path=path)[0]

    transport = jev.make_transport("recorded", np.random.SeedSequence(SEED), "J_fake", 0,
                                   recorded_dir=tmp_path)
    log = jev.JevLog(arm="jev_recorded")
    policy = jev.JevPolicy(transport, log, kappa=1.0, cadence=CADENCE, window=WINDOW,
                           cell_id="J_fake", rep=0)
    replayed = tw.simulate({"policy": "external"}, np.random.SeedSequence(SEED),
                           external_policy=policy)

    np.testing.assert_array_equal(original.dose, replayed.dose)
    np.testing.assert_array_equal(original.y, replayed.y)
    np.testing.assert_array_equal(original.W, replayed.W)
    assert original.dW == replayed.dW
    assert original.implied_r == replayed.implied_r
    assert log.summary()["returned_models"] == [jev.FAKE_MODEL_NAME]

    with pytest.raises(jev.JevLogExhausted):
        transport(jev.build_request({"day": 1}), person=0, occasion=14)


# --------------------------------------------------------------------------- 9. cadence
def test_dose_is_held_between_decisions(fake_world):
    world = fake_world[0]
    T0, T = world.params["T0"], world.params["T"]
    for start in range(T0, T, CADENCE):
        block = world.dose[:, start:start + CADENCE]
        np.testing.assert_array_equal(block, np.repeat(block[:, :1], block.shape[1], axis=1))
    assert np.any(world.dose[:, T0] != world.dose[:, T0 + CADENCE])  # but it does move
    assert np.all(world.dose[:, :T0] == 0.0)


# --------------------------------------------------------------------------- 10. dW
def test_fake_arm_dW_is_finite(fake_world):
    """dW <= 0 would be a finding about the fake policy, not a reason to fail the suite."""
    world = fake_world[0]
    assert np.isfinite(world.dW)
    assert np.isfinite(world.params["dW_se"])
    assert np.isfinite(world.implied_r)
    print(f"\nfake Jev arm: dW = {world.dW:.6f}, dW_se = {world.params['dW_se']:.6f}, "
          f"dW/dW_se = {world.dW / world.params['dW_se']:.2f}, "
          f"implied_r = {world.implied_r:.6f}")


# --------------------------------------------------------- 11. no socket, ever (A3)
def test_nothing_in_the_fake_arm_can_reach_the_network(monkeypatch):
    """Hard guard: every outbound socket entry point raises for the whole call."""
    import socket

    class Blocked(RuntimeError):
        pass

    def block(*args, **kwargs):
        raise Blocked("a network call was attempted")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, block, raising=False)
    for name in ("create_connection", "getaddrinfo", "gethostbyname"):
        monkeypatch.setattr(socket, name, block, raising=False)

    world, log, policy = run_fake(cell={"N": 30})
    assert policy.n_calls > 0 and log.summary()["n_failed"] == 0
    assert np.all(np.isfinite(world.dose))
    assert log.summary()["returned_models"] == [jev.FAKE_MODEL_NAME]

    # and the guard really is armed
    with pytest.raises(Blocked):
        socket.getaddrinfo("example.invalid", 443)


def test_the_fake_transport_sits_off_the_nine_dgp_streams():
    """`make_transport` uses child 9 of the replicate seed; children 0 to 8 are the DGP."""
    ss = np.random.SeedSequence(SEED)
    before = [list(np.random.SeedSequence(entropy=ss.entropy,
                                          spawn_key=tuple(ss.spawn_key) + (k,),
                                          pool_size=ss.pool_size).generate_state(4))
              for k in range(9)]
    transport = jev.make_transport("fake", ss, "J_fake", 0)
    after = [list(np.random.SeedSequence(entropy=ss.entropy,
                                         spawn_key=tuple(ss.spawn_key) + (k,),
                                         pool_size=ss.pool_size).generate_state(4))
             for k in range(9)]
    assert before == after
    assert list(transport.ss.spawn_key) == list(ss.spawn_key) + [9]
    assert list(transport.ss.generate_state(4)) not in before


# ----------------------------------------------- 12. missingness through the runner
def test_a_failed_call_leaves_the_replicate_out_of_every_rate(monkeypatch, tmp_path):
    """spec 11 A3: a failed call makes the replicate missing, counted in `n_failed`
    and in no mean, no rate and no status distribution."""
    from trapworld import grid, metrics, run_grid

    real_make = jev.make_transport
    fail_rep = 1

    def patched(kind, ss, cell_id, rep, recorded_dir=None):
        transport = real_make(kind, ss, cell_id, rep, recorded_dir)
        if rep == fail_rep:
            transport.fail_at = (3, 21)     # person 3, a real decision occasion
        return transport

    monkeypatch.setattr(jev, "make_transport", patched)
    index = grid.CELL_INDEX["J_fake"]
    records = [run_grid.run_replicate(index, rep, log_dir=str(tmp_path),
                                      analyzer_names=("agentsem", "naive"))
               for rep in range(3)]

    failed = records[fail_rep]
    assert failed["jev_status"] == "failed"
    assert "analyzers" not in failed
    assert set(failed) == {"cell_id", "rep", "policy", "jev_status", "jev_failure", "jev"}
    assert failed["jev_failure"] == {"person": 3, "occasion": 21,
                                     "failure_class": "http_529"}
    assert failed["jev"]["n_failed"] == 1
    for ok in (records[0], records[2]):
        assert ok["jev_status"] == "ok" and "analyzers" in ok

    agg = metrics.aggregate(records)
    assert agg["n_reps"] == 3 and agg["n_scored"] == 2 and agg["n_failed"] == 1
    assert agg["jev"]["status"]["counts"] == {"failed": 1, "ok": 2}
    for arm in ("agentsem", "naive"):
        for key, summary in agg["analyzers"][arm].items():
            if isinstance(summary, dict) and "n" in summary:
                assert summary["n"] in (0, 2), (arm, key, summary["n"])
    assert agg["analyzers"]["agentsem"]["status_B"]["n"] == 2
