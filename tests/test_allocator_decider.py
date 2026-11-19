"""Decision-model allocator (question set alloc-v1.0): questions, probability mapping, scoring, model
choice. No network: a fake decision client stands in for the model."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ctb import allocator as A
from ctb.config import Settings
from ctb.kairos.allocator_questions import (BEST_QID, QUESTION_SET_VERSION, MappingParams, batches, build_questions, map_answers,
                                            target_weight)
from ctb.kairos.client import KairosError, KairosResult, RateBudget
from ctb.kairos.parse import validate

T0 = 1_800_000_000.0
IDS = ["carry", "trend", "xsmom"]

@pytest.fixture(autouse=True)
def _no_explore_floor(monkeypatch):
    """These tests pin the decider-only mapping itself; the demo minimum is tested in test_allocator_explore_floor.py."""
    monkeypatch.setenv("CTB_ALLOCATOR_EXPLORE_FLOOR", "0")



def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def answers(p, best=None):
    out = {f"net_positive_{i}": {"type": "noul", "noul": v} for i, v in p.items()}
    if best is not None:
        top = max(best, key=lambda k: best[k])
        out[BEST_QID] = {"type": "choice", "choice": top, "probabilities": best, "confidence": best[top]}
    return out


BEST = {"carry": 0.2, "trend": 0.1, "xsmom": 0.5, "none": 0.2}
P = {"carry": 0.61, "trend": 0.48, "xsmom": 0.57}


class FakeDecider:
    def __init__(self, *replies, max_questions=None):
        self.replies, self.calls = list(replies), []
        self.spec = SimpleNamespace(max_questions=max_questions, provider="ordecide")

    async def ask(self, state, questions, **kw):
        self.calls.append((state, questions))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        ans = {q: a for q, a in r.items() if q in questions}  # a batch only gets answers to its own questions
        return KairosResult(answers=ans, usage={}, metadata={"request_id": f"req-{len(self.calls)}"}, latency_ms=1.0,
                            attempts=1, request_bytes=10)


async def market():
    return {"major": {"BTC": {"ret_1d_pct": 1.0}}}


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "live"
    put(r / "strategies.json", [{"id": i, "label": i.title()} for i in IDS])
    for i in IDS:
        put(r / i / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "pnl_pct": 0.0})
        put(r / i / "control.json", {"enabled": True, "weight": 1.0, "owner_lock": False})
    return r


def make(root, llm, **kw):
    return A.Allocator(root, llm, kw.pop("market", market), model="ordecide / test", clock=lambda: T0, budget=RateBudget(20), **kw)


def decide(a, **kw):
    return asyncio.run(a.decide(**kw))


def ctl(root, sid):
    return json.loads((root / sid / "control.json").read_text())


def test_question_set_is_observable_versioned_and_batched():
    qs = build_questions(IDS)
    assert QUESTION_SET_VERSION == "alloc-v1.1" and len(qs) == 4
    assert [q["type"] for q in qs.values()] == ["noul", "noul", "noul", "choice"]
    assert "net return" in qs["net_positive_carry"]["instructions"] and "24 hours" in qs["net_positive_carry"]["instructions"]
    assert set(qs[BEST_QID]["criteria"]) == {*IDS, "none"}
    ok, issues = validate(answers(P, BEST), qs)  # the harness's own validator reads these answers
    assert not issues and ok[BEST_QID]["choice"] == "xsmom"
    got = batches(qs, 3)
    assert [len(b) for b in got] == [3, 1] and {k for b in got for k in b} == set(qs)
    assert len(batches(qs, None)) == 1


def test_probability_to_weight_mapping():
    prm = MappingParams()
    assert target_weight(0.65, prm) == 1 and target_weight(0.9, prm) == 1 and target_weight(0.5, prm) == 0 and target_weight(0.3, prm) == 0
    assert target_weight(0.575, prm) == pytest.approx(0.5)
    hold = {i: (True, 1.0) for i in IDS}
    w, e, rows, summary = map_answers(IDS, P, BEST, hold, prm)
    # carry 0.61 -> 0.733; trend 0.48 -> 0; xsmom is the winner with p 0.57 > 0.55: 0.467 lifted to 0.5
    assert w == {"carry": pytest.approx(0.7333, abs=1e-3), "trend": 0, "xsmom": 0.5} and e == {"carry": True, "trend": False, "xsmom": True}
    assert rows["xsmom"]["winner_boost"] is True and rows["carry"]["winner_boost"] is False
    assert summary.startswith("carry p=0.61 -> 0.73, trend p=0.48 -> 0, xsmom winner p=0.57 -> 0.5")
    w2, _, rows2, _ = map_answers(IDS, {**P, "xsmom": 0.54}, BEST, hold, prm)  # winner but p <= 0.55: no boost
    assert w2["xsmom"] == pytest.approx(0.2667, abs=1e-3) and not rows2["xsmom"]["winner_boost"]
    with pytest.raises(ValueError):
        MappingParams(p_zero=0.6, p_full=0.6)


def test_constants_live_in_config():
    s = Settings()
    assert (s.allocator_p_zero, s.allocator_p_full, s.allocator_winner_min_p, s.allocator_winner_weight) == (0.5, 0.65, 0.55, 0.5)


def test_decision_applies_rules_and_logs_everything_needed_for_scoring(root):
    llm = FakeDecider(answers(P, BEST))
    a = make(root, llm)
    rec = decide(a)
    assert rec["error"] is None and rec["mode"] == "decider" and len(llm.calls) == 1
    assert llm.calls[0][0]["strategies"]["carry"]["weight"] == 1.0  # the same context is the decision state
    # current weight 1.0: carry 0.73 is inside the step; trend wants 0 but is limited to 0.5; xsmom 0.5
    assert ctl(root, "carry")["weight"] == pytest.approx(0.7333, abs=1e-3) and ctl(root, "trend")["weight"] == 0.5
    assert ctl(root, "trend")["enabled"] is True and ctl(root, "xsmom")["weight"] == 0.5
    assert rec["step_limited"] == ["trend"] and ctl(root, "carry")["reason"].startswith("carry p=0.61 -> 0.73")
    line = json.loads((root / "allocator" / "decisions.jsonl").read_text().splitlines()[-1])
    assert line["qset"] == "alloc-v1.1" and line["model"] == "ordecide / test" and line["request_ids"] == ["req-1"]
    assert line["answers"]["p"] == P and line["answers"]["best"]["xsmom"] == 0.5
    assert line["mapping"]["trend"]["target"] == 0 and line["weights"]["trend"] == 0.5 and line["inputs"]["strategies"]["carry"]
    assert a.status()["model"] == "ordecide / test"


def test_target_zero_disables_once_the_step_limit_allows(root):
    for i in IDS:
        put(root / i / "control.json", {"enabled": True, "weight": 0.4, "owner_lock": False})
    decide(make(root, FakeDecider(answers({"carry": 0.7, "trend": 0.45, "xsmom": 0.4}, BEST))))
    assert ctl(root, "trend")["weight"] == 0 and ctl(root, "trend")["enabled"] is False and ctl(root, "carry")["weight"] == 0.9


def test_floor_and_owner_lock_still_apply_after_the_mapping(root):
    put(root / "carry" / "status.json", {"state": "running", "equity_usdt": 800.0, "start_equity_usdt": 1000.0})
    put(root / "trend" / "control.json", {"enabled": False, "weight": 0.3, "owner_lock": True})
    before = (root / "trend" / "control.json").read_bytes()
    rec = decide(make(root, FakeDecider(answers({"carry": 0.9, "trend": 0.9, "xsmom": 0.55}, BEST))))
    assert ctl(root, "carry")["weight"] == 0 and rec["risk_floor"] == ["carry"]
    assert (root / "trend" / "control.json").read_bytes() == before and rec["skipped_owner_lock"] == ["trend"]


def test_missing_answers_are_retried_once_then_that_strategy_keeps_its_weight(root):
    partial = answers({"carry": 0.62, "xsmom": 0.9}, BEST)  # no answer for trend
    llm = FakeDecider(partial, partial)
    rec = decide(make(root, llm))
    assert len(llm.calls) == 2 and rec["attempts"] == 2 and rec["unanswered"] == ["net_positive_trend"]
    assert rec["mapping"]["trend"]["held"] is True and ctl(root, "trend")["weight"] == 1.0  # untouched
    assert ctl(root, "xsmom")["weight"] == 1.0 and ctl(root, "carry")["weight"] == pytest.approx(0.8, abs=1e-3)


def test_questions_are_batched_to_the_model_limit(root):
    llm = FakeDecider(answers(P), answers(P, BEST), max_questions=3)
    rec = decide(make(root, llm))
    assert [len(c[1]) for c in llm.calls] == [3, 1] and rec["request_ids"] == ["req-1", "req-2"] and rec["error"] is None


def test_model_error_keeps_the_allocation(root):
    before = {i: (root / i / "control.json").read_bytes() for i in IDS}
    rec = decide(make(root, FakeDecider(KairosError("rate_limited", "slow down"))))
    assert "slow down" in rec["error"] and {i: (root / i / "control.json").read_bytes() for i in IDS} == before


def test_dry_run_writes_nothing(root):
    snap = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    rec = decide(make(root, FakeDecider(answers(P, BEST))), dry_run=True)
    assert rec["mapping"]["carry"]["target"] > 0
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == snap and not (root / "allocator").exists()


def test_model_choice_default_follows_the_configured_decision_provider(monkeypatch):
    monkeypatch.delenv("CTB_ALLOCATOR_MODEL", raising=False)
    for provider, want in (("jev", "jev"), ("kairos", "kairos"), ("or:google/gemini-decision-1", "or:google/gemini-decision-1")):
        monkeypatch.setenv("CTB_MODEL_PROVIDER", provider)
        assert A.allocator_spec().provider == want and A.allocator_spec().kind == "systemone"
    monkeypatch.setenv("CTB_MODEL_PROVIDER", "llm")  # a chat provider is not a decision spec
    assert A.allocator_spec().provider == "ordecide"
    assert A.allocator_spec("kairos").provider == "kairos" and A.allocator_spec("ordecide:google/gemini-decision-1").kind == "systemone"
    assert A.allocator_spec("llm").kind == "chat" and A.allocator_spec("or:deepseek/deepseek-v4.1-flash-1017").kind == "chat"
    assert A.model_display(A.allocator_spec("jev")).startswith("jev / ")
    assert A.model_display(A.allocator_spec("ordecide:google/gemini-decision-1")) == "ordecide / google/gemini-decision-1"
    assert A.model_display(A.allocator_spec("llm")).startswith("llm / ")


def test_score_compares_answers_with_realised_book_returns(root):
    h = 24 * 3600_000
    t = 1_000_000_000_000
    lines = []
    for k, (p_carry, e1) in enumerate([(0.8, 1020.0), (0.7, 980.0), (0.4, 1010.0)]):  # hit, miss (p .7, fell), miss (p .4, rose)
        lines.append({"t_ms": t + k * h * 3, "qset": "alloc-v1.0", "error": None, "dry_run": False,
                      "weights": {"carry": 1.0, "trend": 0.0}, "answers": {"p": {"carry": p_carry, "trend": 0.5}, "best": None},
                      "inputs": {"strategies": {"carry": {"equity_usdt": 1000.0}, "trend": {"equity_usdt": 1000.0}}}})
    (root / "allocator").mkdir()
    (root / "allocator" / "decisions.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    put(root / "allocator" / "state.json", {"peaks": {}, "history": {"carry": [[t + h, 1020.0], [t + 3 * h + h, 980.0], [t + 6 * h + h, 1010.0]]}})
    out = A.score_decisions(root, now_ms=t + 20 * h)
    c = out["strategies"]["carry"]
    assert c["n"] == 3 and c["base_rate"] == pytest.approx(0.667, abs=1e-3) and c["hit_rate"] == pytest.approx(0.333, abs=1e-3)
    assert c["brier"] == pytest.approx(((0.2) ** 2 + (0.7) ** 2 + (0.6) ** 2) / 3, abs=1e-3)
    assert "trend" not in out["strategies"] and out["unlabelled"]["flat"] == 3  # weight 0: cannot be labelled
    assert out["pending"] == 0
    assert A.score_decisions(root, now_ms=t + 10)["pending"] == 3
