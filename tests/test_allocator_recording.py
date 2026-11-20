"""What the allocator records for the evaluation (the research side scores and compares from these files): the slim
decider answers, the full-input decision log on its schedule, the daily rotation, and the paired executable policies.
None of it may change the live weights."""

from __future__ import annotations

import asyncio
import json

import pytest

from ctb import allocator as A
from test_allocator_computed import DAY, SLOT, T0, Clock, FakeDecider, answers, ctl, make, root  # noqa: F401  (fixture)


def decide(a, **kw):
    return asyncio.run(a.decide(**kw))


def jl(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------- answers.jsonl
def test_every_decider_answer_is_one_slim_line_in_answers_jsonl(root):
    a = make(root, FakeDecider(answers({"carry": 0.61, "trend": 0.48, "xsmom": 0.57})))
    decide(a, trigger="schedule")
    decide(a, trigger="schedule")
    rows = jl(root / "allocator" / "answers.jsonl")
    t = int(T0 * 1000)
    assert rows[0] == {"issued_ms": t, "horizon_start_ms": t, "maturity_ms": t + 24 * 3600_000, "question_version": "alloc-v1.1",
                       "model": "ordecide / test", "p": {"carry": 0.61, "trend": 0.48, "xsmom": 0.57}}
    assert len(rows) == 2 and len((root / "allocator" / "answers.jsonl").read_text().splitlines()[0]) < 400  # no model input


def test_a_dry_run_or_a_failed_poll_writes_no_answer(root):
    from ctb.kairos.client import KairosError

    decide(make(root, FakeDecider(answers({"carry": 0.6, "trend": 0.6, "xsmom": 0.6}))), dry_run=True)
    decide(make(root, FakeDecider(KairosError("rate_limited", "slow down"))))
    assert not (root / "allocator" / "answers.jsonl").exists()


# ---------------------------------------------------------------- the full model input in decisions.jsonl
def test_routine_polls_keep_the_full_input_only_every_nth_time_weights_runs_and_errors_always(root):
    from dataclasses import replace

    from ctb.kairos.client import KairosError

    put_state = {"peaks": {}, "history": {}, "last_weights_ms": int((SLOT - DAY + 60) * 1000)}  # weights set: polls are refreshes
    (root / "allocator").mkdir(exist_ok=True)
    (root / "allocator" / "state.json").write_text(json.dumps(put_state))
    a = make(root, FakeDecider(answers({"carry": 0.6, "trend": 0.6, "xsmom": 0.6})))
    a.calc = replace(a.calc, full_input_every=3)
    for _ in range(4):
        decide(a, trigger="schedule")
    lines = jl(root / "allocator" / "decisions.jsonl")
    assert [("inputs" in x, x.get("inputs_omitted", False)) for x in lines] == [(False, True), (False, True), (True, False), (False, True)]
    assert all(x["weights_mode"] == "refresh" and x["state_bytes"]["total"] > 0 and x["answers"]["p"] for x in lines)
    decide(a, trigger="decide_now")  # a weights run
    assert "inputs" in jl(root / "allocator" / "decisions.jsonl")[-1]
    b = make(root, FakeDecider(KairosError("rate_limited", "slow down")))  # an error: always
    b.calc = replace(b.calc, full_input_every=3)
    decide(b, trigger="schedule")
    last = jl(root / "allocator" / "decisions.jsonl")[-1]
    assert "slow down" in last["error"] and "inputs" in last


# ---------------------------------------------------------------- the daily rotation of decisions.jsonl
def test_decisions_rotate_daily_with_gzip_and_keep_90_days(tmp_path):
    from ctb import decision_log as DL

    d = tmp_path / "allocator"
    day0 = DAY * (int(T0) // DAY)
    for k in range(95):  # one line a day at 12:00 UTC, 95 days
        DL.append(d, {"t_ms": (day0 + k * DAY + 12 * 3600) * 1000, "k": k}, (day0 + k * DAY + 12 * 3600) * 1000)
    names = sorted(p.name for p in d.iterdir())
    days = [m.group(1) for m in map(DL.ARCHIVE_RE.match, names) if m]
    assert "decisions.jsonl" in names and len(days) == 90  # 94 days archived, the 4 older than 90 days deleted
    plain = [n for n in names if n.startswith("decisions-") and not n.endswith(".gz")]
    assert plain == [f"decisions-{days[-1]}.jsonl"]  # every older day is gzipped, the newest archive stays plain
    assert [r["k"] for r in DL.tail(d, 3)] == [92, 93, 94]  # the feed reads across the rotation
    assert [r["k"] for r in DL.iter_all(d)] == list(range(4, 95))


def test_the_allocator_feed_and_cost_read_across_midnight(root):
    clock = Clock(SLOT - 60)  # 23:49 UTC
    a = make(root, FakeDecider(answers({"carry": 0.6, "trend": 0.6, "xsmom": 0.6})), clock=clock)
    decide(a, trigger="decide_now")
    clock.t = SLOT + 30 * 60  # 00:20 UTC the next day
    decide(a, trigger="schedule")
    d = root / "allocator"
    assert sorted(p.name for p in d.iterdir() if p.name.startswith("decisions")) == ["decisions-" + A.time.strftime("%Y-%m-%d", A.time.gmtime(SLOT - 60)) + ".jsonl", "decisions.jsonl"]
    assert len(a.decisions(10)) == 2 and len(jl(d / "decisions.jsonl")) == 1


def test_the_predictions_panel_finds_yesterdays_weights_line_after_the_rotation(tmp_path):
    from ctb import predictions as P

    d = tmp_path / "allocator"
    d.mkdir()
    (d / "decisions-2027-01-14.jsonl").write_text(json.dumps({"t_ms": 1, "core": {"trend": 0.4}}) + "\n")
    (d / "decisions.jsonl").write_text(json.dumps({"t_ms": 2, "weights_mode": "refresh"}) + "\n")
    assert P.newest_with(d, "core")["core"] == {"trend": 0.4}


# ---------------------------------------------------------------- paired executable policies
from ctb.config import Settings  # noqa: E402
from test_allocator_computed import forecast, put  # noqa: E402


def _low_trend(root, hours_ago=3):
    t = int(T0 * 1000) - hours_ago * 3600_000
    put(root / "allocator" / "state.json", {"peaks": {}, "history": {}, "p_hist": {"trend": [[t, 0.2]] * 23}})


def _alloc(root, clock=None, cls=A.Allocator, venues=None):
    return cls(root, FakeDecider(answers({"carry": 0.5, "trend": 0.2, "xsmom": 0.5})), _market,
               model="ordecide / test", clock=clock or Clock(T0), budget=A.RateBudget(50), calc=A.Calc.from_settings(Settings()),
               venues=venues)


async def _market():
    return {"major": {"BTC": {"ret_1d_pct": 1.0}}}


def test_each_weights_run_records_the_four_paired_policies(root):
    put(root / "forecast" / "latest.json", forecast(T0 - 7200, kappa=0.0))  # no tilt room: live is the core plus the veto
    _low_trend(root)
    rec = decide(_alloc(root), trigger="decide_now")
    vr = rec["venues"]["bybit-demo"]
    pol = vr["policies"]
    assert list(pol) == ["core", "live", "tilt", "core_veto"]
    assert (vr["t_ms"], vr["venue"], vr["policy_state_version"]) == (int(T0 * 1000), "bybit-demo", 1)
    assert pol["live"] == {i: ctl(root, i)["weight"] for i in ("carry", "trend", "xsmom")}
    assert pol["core_veto"] == pol["live"]  # kappa 0 and the veto live: live is exactly core plus the veto
    assert pol["live"]["trend"] == pytest.approx(0.1) and pol["core"]["trend"] > 0.3  # the veto halved live trend, not the core
    assert pol["core"]["carry"] == pol["live"]["carry"] and pol["tilt"]["trend"] > pol["core"]["trend"]  # tilt at kappa 1
    assert jl(root / "allocator" / "policies.jsonl") == [{**{k: vr[k] for k in ("policies", "t_ms", "venue", "policy_state_version")},
                                                          "weights_mode": "computed"}]


def test_a_shadow_policy_steps_from_its_own_weights_not_from_live(root):
    put(root / "forecast" / "latest.json", forecast(T0 - 7200, kappa=0.0))
    _low_trend(root)
    clock = Clock(T0)
    a = _alloc(root, clock)
    first = decide(a, trigger="decide_now")["venues"]["bybit-demo"]["policies"]
    clock.t = T0 + DAY
    put(root / "forecast" / "latest.json", forecast(clock.t - 7200, kappa=0.0))
    second = decide(a, trigger="decide_now")["venues"]["bybit-demo"]["policies"]
    assert second["live"]["trend"] == pytest.approx(0.1)  # vetoed again (the exploration floor keeps 0.1)
    # the core held its own 0.37 overnight; stepping from live's 0.1 it could reach 0.35 at most
    assert second["core"]["trend"] == first["core"]["trend"] > 0.35


def test_a_dry_run_records_policies_but_keeps_no_policy_state(root):
    _low_trend(root)
    rec = decide(_alloc(root), trigger="decide_now", dry_run=True)
    assert rec["venues"]["bybit-demo"]["policies"]["core"] and not (root / "allocator" / "policies.jsonl").exists()
    assert "policy_state" not in json.loads((root / "allocator" / "state.json").read_text())


def test_the_recording_never_changes_the_live_weights(tmp_path):
    """Two identical worlds (Bybit with a vetoed and a tilted book, OKX with a strategy Bybit does not list), three weights
    runs over three days: every control.json written with the recording equals the one written without it."""

    class Unrecorded(A.Allocator):
        def _record_policies(self, *a, **k):
            self._policy_jobs = []
            return None

    def world(name):
        base = tmp_path / name
        bybit, okx = base / "live", base / "live-okx"
        put(bybit / "strategies.json", [{"id": i, "label": i} for i in ("carry", "trend", "xsmom")])
        put(okx / "strategies.json", [{"id": i, "label": i} for i in ("trend", "xsmom", "longvol")])
        for r, ids in ((bybit, ("carry", "trend", "xsmom")), (okx, ("trend", "xsmom", "longvol"))):
            for i, w in zip(ids, (0.5, 0.2, 0.3)):
                put(r / i / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "pnl_pct": 0.0, "positions": []})
                put(r / i / "control.json", {"enabled": True, "weight": w, "owner_lock": False})
        _low_trend(bybit)
        return bybit, okx

    runs = []
    for name, cls in (("with", A.Allocator), ("without", Unrecorded)):
        bybit, okx = world(name)
        clock = Clock(T0)
        a = _alloc(bybit, clock, cls, [A.AllocVenue("bybit-demo", bybit, "BYBIT·DEMO"), A.AllocVenue("okx-demo", okx, "OKX·DEMO")])
        seen = []
        for day in range(3):
            clock.t = T0 + day * DAY
            put(bybit / "forecast" / "latest.json", forecast(clock.t - 7200))  # kappa 1: the live tilt is on
            decide(a, trigger="decide_now")
            seen.append({str(p.relative_to(tmp_path / name)): json.loads(p.read_text())
                         for r in (bybit, okx) for p in sorted(r.glob("*/control.json")) if p.parent.name != "allocator"})
        runs.append(seen)
        assert (bybit / "allocator" / "policies.jsonl").exists() is (cls is A.Allocator)
    assert runs[0] == runs[1] and runs[0][0] != runs[0][2]  # identical, and the weights did move over the three runs


# ---------------------------------------------------------------- a recording failure never reaches live
FIVE = (("bybit-demo", ("carry", "trend", "xsmom")), ("okx-demo", ("trend", "xsmom")), ("deribit-test", ("trend", "xsmom")),
        ("bitget-demo", ("trend", "xsmom")), ("extra-demo", ("trend", "xsmom")))


class Unrecorded(A.Allocator):
    def _record_policies(self, *a, **k):
        self._policy_jobs = []
        return None


def five_venues(base, cls=A.Allocator):
    roots = {}
    for vid, ids in FIVE:
        r = base / vid
        put(r / "strategies.json", [{"id": i, "label": i} for i in ids])
        for i, w in zip(ids, (0.5, 0.2, 0.3)):
            put(r / i / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "pnl_pct": 0.0, "positions": []})
            put(r / i / "control.json", {"enabled": True, "weight": w, "owner_lock": False})
        roots[vid] = r
    bybit = roots["bybit-demo"]
    put(bybit / "forecast" / "latest.json", forecast(T0 - 7200))
    put(bybit / "allocator" / "control.json", {"enabled": True, "interval_min": 60})
    llm = FakeDecider(answers({"carry": 0.5, "trend": 0.6, "xsmom": 0.5}))
    a = cls(bybit, llm, _market, model="ordecide / test", clock=Clock(T0), budget=A.RateBudget(50), calc=A.Calc.from_settings(Settings()),
            venues=[A.AllocVenue(vid, roots[vid], vid.upper()) for vid, _ in FIVE])
    return a, llm, roots


def controls(roots):
    return {f"{vid}/{p.parent.name}": json.loads(p.read_text()) for vid, r in roots.items() for p in sorted(r.glob("*/control.json"))
            if p.parent.name != "allocator"}


@pytest.mark.parametrize("fault", ["corrupt_state", "policies_jsonl_unwritable", "recording_raises"])
def test_a_recording_failure_never_stops_or_changes_a_weights_run(tmp_path, monkeypatch, fault):
    ref, _, ref_roots = five_venues(tmp_path / "reference", Unrecorded)
    asyncio.run(ref._tick())
    a, llm, roots = five_venues(tmp_path / "faulty")
    bybit = roots["bybit-demo"]
    if fault == "corrupt_state":
        put(bybit / "allocator" / "state.json", {"peaks": {}, "history": {}, "policy_state": {"version": 1, "books": ["damaged"]}})
    elif fault == "policies_jsonl_unwritable":
        real = A.append_jsonl
        monkeypatch.setattr(A, "append_jsonl", lambda path, obj: (_ for _ in ()).throw(OSError(28, "No space left on device"))
                            if path.name == "policies.jsonl" else real(path, obj))
    else:
        monkeypatch.setattr(A.Allocator, "_policy_variant", lambda self, *x: (_ for _ in ()).throw(RuntimeError("shadow bug")))
    asyncio.run(a._tick())  # the 23:50-style weights run of a fresh install: due at once
    asyncio.run(a._tick())  # the next tick: nothing due
    assert controls(roots) == controls(ref_roots) and controls(roots) != {}  # all five venues written exactly as without recording
    lines = jl(bybit / "allocator" / "decisions.jsonl")
    assert len(lines) == 1 and lines[0]["weights_mode"] == "computed" and llm.calls == 1  # one weights run, no repeat
    st = json.loads((bybit / "allocator" / "state.json").read_text())
    assert st["last_weights_ms"] == int(T0 * 1000)
    assert len(a.budget.calls) == 1 and len(A.Allocator(bybit, llm, _market, model="m", clock=Clock(T0), budget=A.RateBudget(50)).budget.calls) == 1
    status = json.loads((bybit / "allocator" / "status.json").read_text())
    if fault == "corrupt_state":
        assert "unusable" in lines[0]["venues"]["bybit-demo"]["policy_state_reset"] and lines[0]["venues"]["okx-demo"]["policies"]["core"]
    else:
        assert "evaluation recording failed, live weights unaffected" in lines[0]["error"] and "evaluation recording failed" in status["last_error"]
        assert all("policies_error" in lines[0]["venues"][vid] for vid, _ in FIVE)
