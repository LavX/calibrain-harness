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
