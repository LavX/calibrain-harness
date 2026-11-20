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
