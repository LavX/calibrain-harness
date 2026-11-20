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
    assert {k: rows[0][k] for k in ("issued_ms", "horizon_start_ms", "maturity_ms", "question_version", "model", "p")} == {
        "issued_ms": t, "horizon_start_ms": t, "maturity_ms": t + 24 * 3600_000, "question_version": "alloc-v1.1",
        "model": "ordecide / test", "p": {"carry": 0.61, "trend": 0.48, "xsmom": 0.57}}
    # what the label needs: the allocator venue's books, their equity at the poll and the weight they hold after it
    assert rows[0]["venue"] == "bybit-demo" and rows[0]["e0"] == {"carry": 1000.0, "trend": 1000.0, "xsmom": 1000.0}
    assert rows[0]["w0"] == {i: ctl(root, i)["weight"] for i in ("carry", "trend", "xsmom")} and rows[0]["best"]["trend"] == 0.3
    assert len(rows) == 2 and len((root / "allocator" / "answers.jsonl").read_text().splitlines()[0]) < 600  # no model input


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
    assert jl(root / "allocator" / "policies.jsonl") == [{**{k: vr[k] for k in ("policies", "policy_enabled", "t_ms", "venue", "policy_state_version")},
                                                          "weights_mode": "computed"}]


def test_a_shadow_policy_steps_from_its_own_weights_not_from_live(root):
    put(root / "forecast" / "latest.json", forecast(T0 - 7200, kappa=0.0))
    _low_trend(root)
    clock = Clock(T0)
    a = _alloc(root, clock)
    first = decide(a, trigger="decide_now")["venues"]["bybit-demo"]["policies"]
    clock.t = T0 + DAY
    put(root / "forecast" / "latest.json", forecast(clock.t - 7200, kappa=0.0))
    a._state()["p_hist"]["trend"] = [[int(clock.t * 1000) - 3600_000, 0.2]] * 23  # the p window is 24 h: still low today
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


def test_the_live_policy_is_what_the_books_hold_after_the_writes(root, monkeypatch):
    put(root / "forecast" / "latest.json", forecast(T0 - 7200, kappa=0.0))
    real = A.write_json_atomic
    monkeypatch.setattr(A, "write_json_atomic", lambda path, obj: (_ for _ in ()).throw(OSError(13, "Permission denied"))
                        if path == root / "trend" / "control.json" else real(path, obj))
    rec = decide(_alloc(root), trigger="decide_now")
    vr = rec["venues"]["bybit-demo"]
    assert "trend" in vr["write_failed"] and rec["weights"]["trend"] != 0.2  # the plan moved trend, the write failed
    assert vr["policies"]["live"]["trend"] == 0.2 == ctl(root, "trend")["weight"]  # live records what the book holds
    assert vr["policies"]["live"]["carry"] == ctl(root, "carry")["weight"] == rec["weights"]["carry"]


def test_routine_polls_without_their_input_are_still_scored(root):
    """59 of 60 routine polls drop the model input; the scorer reads answers.jsonl, so they are labelled all the same."""
    t = int(T0 * 1000)
    put(root / "allocator" / "state.json", {"peaks": {}, "history": {}, "last_weights_ms": int((SLOT - DAY + 60) * 1000)})
    a = make(root, FakeDecider(answers({"carry": 0.6, "trend": 0.6, "xsmom": 0.6})))
    decide(a, trigger="schedule")
    line = jl(root / "allocator" / "decisions.jsonl")[-1]
    assert line["weights_mode"] == "refresh" and line["inputs_omitted"] is True
    st = json.loads((root / "allocator" / "state.json").read_text())
    later = t + 24 * 3600_000
    st["history"] = {"bybit-demo/carry": [[later, 1010.0]], "bybit-demo/trend": [[later, 990.0]], "bybit-demo/xsmom": [[later, 1003.0]]}
    (root / "allocator" / "state.json").write_text(json.dumps(st))
    out = A.score_decisions(root, now_ms=later + 3600_000)
    assert out["unlabelled"] == {"flat": 0, "no_equity": 0} and {k: v["n"] for k, v in out["strategies"].items()} == {"carry": 1, "trend": 1, "xsmom": 1}
    assert out["strategies"]["trend"]["base_rate"] == 0.0 and out["strategies"]["carry"]["base_rate"] == 1.0


def test_the_scorer_still_reads_old_decision_lines_once(tmp_path):
    """Lines from before answers.jsonl carry their input and are scored from it; from the first answer on, answers.jsonl
    is the only source, so nothing counts twice."""
    alloc = tmp_path / "allocator"
    alloc.mkdir()
    h = 24 * 3600_000
    t = 1_000_000_000_000
    old = {"t_ms": t, "qset": "alloc-v1.1", "error": None, "dry_run": False, "weights": {"carry": 1.0}, "answers": {"p": {"carry": 0.7}, "best": None},
           "inputs": {"strategies": {"carry": {"equity_usdt": 1000.0}}}}
    dup = {**old, "t_ms": t + 2 * h}
    (alloc / "decisions.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(dup) + "\n")
    (alloc / "answers.jsonl").write_text(json.dumps({"issued_ms": t + 2 * h, "horizon_start_ms": t + 2 * h, "maturity_ms": t + 3 * h,
                                                     "question_version": "alloc-v1.1", "model": "m", "p": {"carry": 0.4}, "venue": "bybit-demo",
                                                     "e0": {"carry": 1000.0}, "w0": {"carry": 1.0}, "best": None}) + "\n")
    (alloc / "state.json").write_text(json.dumps({"history": {"carry": [[t + h, 1020.0], [t + 3 * h, 990.0]]}}))
    c = A.score_decisions(tmp_path, now_ms=t + 10 * h)["strategies"]["carry"]
    assert c["n"] == 2 and c["brier"] == pytest.approx(((0.7 - 1) ** 2 + (0.4 - 0) ** 2) / 2, abs=1e-4)


def test_the_first_rotation_splits_a_legacy_log_by_day_before_retention(tmp_path):
    """A log from before rotation spans many days. Each line goes to its own day's archive; only days past the 90-day
    retention are deleted, never recent ones for the date of the file's first line."""
    import calendar
    import datetime as dt

    from ctb import decision_log as DL

    def ms(m, d, h=12):
        return calendar.timegm(dt.datetime(2026, m, d, h).timetuple()) * 1000

    d = tmp_path / "allocator"
    d.mkdir()
    legacy = [{"t_ms": ms(7, 1), "k": "jul1"}, {"t_ms": ms(11, 18, 9), "k": "nov18a"}, {"t_ms": ms(11, 18, 23), "k": "nov18b"},
              {"t_ms": ms(11, 19), "k": "nov19"}]
    (d / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in legacy))
    DL.append(d, {"t_ms": ms(11, 20), "k": "nov20"}, ms(11, 20))
    names = sorted(p.name for p in d.iterdir())
    assert names == ["decisions-2026-11-18.jsonl.gz", "decisions-2026-11-19.jsonl", "decisions.jsonl"]  # July 1 is past 90 days
    assert [r["k"] for r in DL.iter_all(d)] == ["nov18a", "nov18b", "nov19", "nov20"]
    assert [r["k"] for r in DL.tail(d, 2)] == ["nov19", "nov20"]


def test_a_split_cut_short_is_finished_at_the_next_rotation(tmp_path):
    from ctb import decision_log as DL

    d = tmp_path / "allocator"
    d.mkdir()
    day = 86_400_000
    t0 = (int(T0 * 1000) // day) * day
    (d / "decisions.jsonl.rotating").write_text(json.dumps({"t_ms": t0 + 1000, "k": "a"}) + "\n")  # a crash left it
    (d / "decisions.jsonl").write_text(json.dumps({"t_ms": t0 + day + 1000, "k": "b"}) + "\n")
    DL.append(d, {"t_ms": t0 + 2 * day + 1000, "k": "c"}, t0 + 2 * day + 1000)
    assert [r["k"] for r in DL.iter_all(d)] == ["a", "b", "c"] and not (d / "decisions.jsonl.rotating").exists()


# ---------------------------------------------------------------- archiving never stands in the way of a decision
@pytest.mark.parametrize("step", ["replace", "split", "gzip", "fsync", "unlink"])
def test_a_failing_rotation_never_stops_a_weights_run(root, monkeypatch, step):
    from ctb import decision_log as DL

    DL._failures.clear()
    d = root / "allocator"
    d.mkdir(exist_ok=True)
    put(d / "control.json", {"enabled": True, "interval_min": 60})
    now = int(T0 * 1000)
    three = now - 3 * DAY * 1000
    (d / f"decisions-{DL.utc_day(three)}.jsonl").write_text(json.dumps({"t_ms": three, "k": "old"}) + "\n")  # something to gzip
    (d / "decisions.jsonl").write_text(json.dumps({"t_ms": now - DAY * 1000, "k": "yesterday"}) + "\n")  # due for rotation
    boom = OSError(5, f"{step} failed")

    def fail(*a, **k):
        raise boom

    target = {"replace": "_replace", "split": "_split", "gzip": "_gzip", "fsync": "_fsync", "unlink": "_unlink"}[step]
    monkeypatch.setattr(DL, target, fail)
    llm = FakeDecider(answers({"carry": 0.6, "trend": 0.6, "xsmom": 0.6}))
    a = make(root, llm)
    asyncio.run(a._tick())  # a weights run (none was ever set)
    asyncio.run(a._tick())  # nothing due
    mine = [r for r in DL.iter_all(d) if r.get("t_ms") == now]
    assert len(mine) == 1 and mine[0]["weights_mode"] == "computed" and llm.calls == 1  # written once, decided once
    assert json.loads((d / "state.json").read_text())["last_weights_ms"] == now
    status = json.loads((d / "status.json").read_text())
    assert status["last_decision_ms"] == now and "decision log rotation failed" in status["last_error"] and f"{step} failed" in status["last_error"]
    b = make(root, llm)
    assert len(b.budget.calls) == 1  # the restart's budget seed counts the one attempt once
    DL._failures.clear()


@pytest.mark.parametrize("damage", ["truncated", "corrupt"])
def test_a_damaged_archive_is_skipped_by_every_reader(root, damage):
    import gzip as gz

    from ctb import decision_log as DL
    from ctb import predictions as P

    d = root / "allocator"
    d.mkdir(exist_ok=True)
    now = int(T0 * 1000)
    rows = [{"t_ms": now - 2 * DAY * 1000 + i, "attempts": 0, "k": i} for i in range(200)]
    blob = gz.compress("".join(json.dumps(r) + "\n" for r in rows).encode())
    if damage == "truncated":
        bad = blob[: len(blob) // 2]  # cut in half: BadGzipFile or EOFError
    else:
        bad = blob[:20] + b"\xff\x00\xff\x00" + blob[24:]  # garbage in the deflate stream: zlib.error
    (d / "decisions-2027-01-13.jsonl.gz").write_bytes(bad)
    (d / "decisions.jsonl").write_text(json.dumps({"t_ms": now - 60_000, "attempts": 1, "core": {"carry": 0.4}}) + "\n")
    got = DL.tail(d, 500)
    assert got[-1]["t_ms"] == now - 60_000 and all(r["k"] == i for i, r in enumerate(got[:-1]))  # any readable part, in order
    assert [r["t_ms"] for r in DL.iter_all(d)][-1] == now - 60_000
    a = make(root, FakeDecider(answers({"carry": 0.6, "trend": 0.6, "xsmom": 0.6})))  # the budget seed reads it
    assert len(a.budget.calls) == 1 and a.cost_usd_today(now) == 0.0 and len(a.decisions(50)) >= 1
    A.score_decisions(root, now_ms=now)
    assert P.newest_with(d, "core")["core"] == {"carry": 0.4}


def test_a_gzip_cut_short_leaves_the_archive_and_its_source_intact(tmp_path, monkeypatch):
    import gzip as gz

    from ctb import decision_log as DL

    d = tmp_path / "allocator"
    d.mkdir()
    (d / "decisions-2027-01-10.jsonl.gz").write_bytes(gz.compress(b'{"t_ms":1,"k":"a"}\n'))
    src = d / "decisions-2027-01-10.jsonl"
    src.write_text('{"t_ms":2,"k":"b"}\n')

    def fail(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(DL, "_fsync", fail)
    with pytest.raises(OSError):
        DL._gzip(src)
    assert src.exists() and [r["k"] for r in DL._lines(d / "decisions-2027-01-10.jsonl.gz")] == ["a"] and not list(d.glob(".*.tmp"))
    monkeypatch.undo()
    DL._gzip(src)
    assert not src.exists() and [r["k"] for r in DL._lines(d / "decisions-2027-01-10.jsonl.gz")] == ["a", "b"]


def test_a_switched_off_flat_book_is_recorded_without_exposure(root):
    """The owner switched xsmom off and locked it; the control keeps its weight 0.3. Flat, it holds nothing: every policy
    records 0 for it (and says it is off). Switched off but still holding positions, it keeps its exposure."""
    put(root / "forecast" / "latest.json", forecast(T0 - 7200, kappa=0.0))
    put(root / "xsmom" / "control.json", {"enabled": False, "weight": 0.3, "owner_lock": True})
    put(root / "xsmom" / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "positions": []})
    vr = decide(_alloc(root), trigger="decide_now")["venues"]["bybit-demo"]
    assert ctl(root, "xsmom")["weight"] == 0.3
    assert {k: v["xsmom"] for k, v in vr["policies"].items()} == {"core": 0.0, "live": 0.0, "tilt": 0.0, "core_veto": 0.0}
    assert {k: v["xsmom"] for k, v in vr["policy_enabled"].items()} == {"core": False, "live": False, "tilt": False, "core_veto": False}
    put(root / "xsmom" / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0,
                                         "positions": [{"symbol": "BTCUSDT", "qty": 0.01, "mark_price": 60000.0}]})
    vr = decide(_alloc(root), trigger="decide_now")["venues"]["bybit-demo"]
    assert vr["policies"]["live"]["xsmom"] == 0.3  # off, not yet flat: still exposed


def test_the_predictions_panel_finds_a_weights_line_in_a_compressed_archive(tmp_path):
    import gzip as gz

    from ctb import predictions as P

    d = tmp_path / "allocator"
    d.mkdir()
    (d / "decisions-2027-01-11.jsonl.gz").write_bytes(gz.compress((json.dumps({"t_ms": 1, "core": {"trend": 0.4}}) + "\n").encode()))
    (d / "decisions-2027-01-12.jsonl.gz").write_bytes(gz.compress((json.dumps({"t_ms": 2, "weights_mode": "held"}) + "\n").encode()))
    (d / "decisions-2027-01-13.jsonl").write_text(json.dumps({"t_ms": 3, "weights_mode": "held"}) + "\n")
    (d / "decisions.jsonl").write_text(json.dumps({"t_ms": 4, "weights_mode": "refresh"}) + "\n")
    assert P.newest_with(d, "core")["core"] == {"trend": 0.4}  # two days of held runs in between
