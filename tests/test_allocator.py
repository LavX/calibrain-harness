"""LLM allocator: validation, retry, hard rules, atomic writes, logs. No network: a fake model
client and a fake market stand in for the model and Bybit."""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from ctb import allocator as A
from ctb.kairos.client import KairosError, RateBudget

T0 = 1_800_000_000.0  # fixed clock, seconds
IDS = ["carry", "trend", "xsmom"]


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def good(**over):
    d = {"weights": {"carry": 0.8, "trend": 0.6, "xsmom": 0.4}, "enabled": {"carry": True, "trend": True, "xsmom": True},
         "reasoning": "funding is positive and breadth is wide", "confidence": 0.7}
    d.update(over)
    return d


class FakeLLM:
    """Replays scripted replies; an Exception instance is raised instead of returned."""

    def __init__(self, *replies, cost=0.002):
        self.replies, self.calls, self.cost = list(replies), [], cost
        self.spec = None

    async def ask_json(self, messages, schema):
        self.calls.append((messages, schema))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r, {"cost": self.cost}


async def fake_market():
    return {"major": {"BTC": {"ret_1d_pct": 1.2}}, "trend_universe": {"share_above_ema50": 0.7}}


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "live"
    put(r / "strategies.json", [{"id": i, "label": i.title()} for i in IDS])
    for i in IDS:
        put(r / i / "status.json", {"id": i, "state": "running", "enabled": True, "updated_ms": int(T0 * 1000),
                                    "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "pnl_pct": 0.0,
                                    "positions": [], "targets": [], "trades_total": 0})
        put(r / i / "control.json", {"enabled": True, "weight": 1.0, "flatten": False, "set_by": "owner", "owner_lock": False})
    return r


def make(root, llm, **kw):
    return A.Allocator(root, llm, kw.pop("market", fake_market), model="test/model", clock=lambda: T0,
                       budget=RateBudget(kw.pop("rph", 12)), mode="chat", **kw)


def ctl(root, sid):
    return json.loads((root / sid / "control.json").read_text())


def decide(a, **kw):
    return asyncio.run(a.decide(**kw))


def test_validate_decision_contract():
    d = A.validate_decision(good(reasoning="x" * 900), IDS)
    assert len(d.reasoning) == 600 and d.weights["carry"] == 0.8
    for bad in (good(weights={"carry": 0.8, "trend": 0.6, "xsmom": 1.2}),          # out of range
                good(weights={"carry": 0.8, "trend": 0.6, "xsmom": 0.4, "ghost": 0.1}),  # unknown id
                good(weights={"carry": 0.8, "trend": 0.6}),                         # missing id
                good(enabled={"carry": "yes", "trend": True, "xsmom": True}),       # not a bool
                good(confidence=1.5), good(reasoning=" "), "nope"):
        with pytest.raises(A.DecisionInvalid):
            A.validate_decision(bad, IDS)


def test_invalid_output_is_retried_once_then_applied(root):
    llm = FakeLLM({"weights": {"carry": 2}}, good())
    rec = decide(make(root, llm))
    assert rec["error"] is None and rec["attempts"] == 2 and len(llm.calls) == 2
    assert "rejected" in llm.calls[1][0][-1]["content"]  # the retry says what was wrong
    assert ctl(root, "carry")["weight"] == 0.8 and ctl(root, "carry")["set_by"] == "llm"


def test_two_invalid_answers_keep_the_allocation_and_log_the_error(root):
    before = {i: snapshot(root / i) for i in IDS}
    a = make(root, FakeLLM("garbage", {"weights": {}}))
    rec = decide(a)
    assert rec["error"].startswith("invalid model output after one retry")
    assert {i: snapshot(root / i) for i in IDS} == before  # controls untouched
    line = json.loads((root / "allocator" / "decisions.jsonl").read_text().splitlines()[-1])
    assert line["error"] and line["applied"] == {i: False for i in IDS}
    st = a.status()
    assert st["state"] == "error" and "invalid model output" in st["last_error"]


def test_malformed_json_from_the_client_also_retries_but_transport_errors_do_not(root):
    llm = FakeLLM(KairosError("malformed_response", "reply is not JSON"), good())
    assert decide(make(root, llm))["attempts"] == 2
    llm = FakeLLM(KairosError("auth", "bad key"), good())
    rec = decide(make(root, llm))
    assert "bad key" in rec["error"] and len(llm.calls) == 1


def test_owner_lock_is_never_overridden(root):
    put(root / "trend" / "control.json", {"enabled": False, "weight": 0.3, "owner_lock": True, "set_by": "owner", "reason": "owner"})
    before = snapshot(root / "trend")
    rec = decide(make(root, FakeLLM(good())))
    assert rec["skipped_owner_lock"] == ["trend"] and rec["applied"]["trend"] is False
    assert snapshot(root / "trend") == before
    assert ctl(root, "carry")["weight"] == 0.8  # the others still move


def test_owner_lock_set_between_read_and_write_still_wins(root):
    a = make(root, FakeLLM(good()))
    cur = {"carry": A.Current(True, 1.0, False, 0.0)}
    put(root / "carry" / "control.json", {"enabled": True, "weight": 0.1, "owner_lock": True})
    plan = A.Plan({"carry": False}, {"carry": 0.0}, [], [], [])
    assert a._apply(plan, cur, {"carry": "x"}, 1)["carry"] is False
    assert ctl(root, "carry")["weight"] == 0.1


def test_drawdown_floor_zeroes_a_book_regardless_of_the_model(root):
    put(root / "xsmom" / "status.json", {"state": "running", "equity_usdt": 840.0, "start_equity_usdt": 1000.0, "updated_ms": 1})
    rec = decide(make(root, FakeLLM(good(weights={"carry": 0.8, "trend": 0.6, "xsmom": 1.0}))))
    c = ctl(root, "xsmom")
    assert c["weight"] == 0 and c["enabled"] is False and c["flatten_on_zero"] is True and "risk floor" in c["reason"]
    assert rec["risk_floor"] == ["xsmom"] and rec["weights"]["xsmom"] == 0
    assert ctl(root, "trend")["weight"] == 0.6


def test_floor_applies_even_when_the_model_is_down_and_without_a_model_call(root):
    put(root / "carry" / "status.json", {"state": "running", "equity_usdt": 800.0, "start_equity_usdt": 1000.0})
    llm = FakeLLM()
    a = make(root, llm)
    assert asyncio.run(a.risk_check())["risk_floor"] == ["carry"] and not llm.calls
    assert ctl(root, "carry")["weight"] == 0
    assert asyncio.run(a.risk_check()) is None  # already cut: nothing more to write
    llm = FakeLLM(KairosError("upstream_unavailable", "down"))
    put(root / "trend" / "status.json", {"state": "running", "equity_usdt": 700.0, "start_equity_usdt": 1000.0})
    rec = decide(make(root, llm))
    assert rec["error"] and "trend" in rec["risk_floor"] and ctl(root, "trend")["weight"] == 0


def test_step_limit_and_floor_bypass():
    cur = {"a": A.Current(True, 1.0, False, 2.0), "b": A.Current(True, 0.2, False, 2.0), "c": A.Current(True, 1.0, False, 20.0)}
    d = A.Decision({"a": 0.0, "b": 1.0, "c": 1.0}, {"a": True, "b": True, "c": True}, "r", 0.5)
    p = A.resolve(cur, d)
    assert p.weights == {"a": 0.5, "b": 0.7, "c": 0.0}
    assert p.step_limited == ["a", "b"] and p.floored == ["c"]  # the floor is not step-limited


def test_control_write_is_atomic(root, monkeypatch):
    path = root / "carry" / "control.json"
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(A.os, "replace", boom)
    with pytest.raises(OSError):
        A.write_json_atomic(path, {"enabled": False})
    assert path.read_bytes() == before  # a failed write leaves the old file whole
    monkeypatch.undo()
    A.write_json_atomic(path, {"enabled": False, "weight": 0.5})
    assert json.loads(path.read_text())["weight"] == 0.5
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_decision_log_and_status(root):
    a = make(root, FakeLLM(good()))
    decide(a)
    line = json.loads((root / "allocator" / "decisions.jsonl").read_text().splitlines()[-1])
    assert {"t_ms", "model", "weights", "enabled", "reasoning", "inputs", "applied", "skipped_owner_lock"} <= set(line)
    assert line["model"] == "test/model" and line["inputs"]["strategies"]["carry"]["drawdown_pct"] == 0.0
    assert line["inputs"]["market"]["trend_universe"]["share_above_ema50"] == 0.7
    st = a.status()
    assert st["state"] == "running" and st["model"] == "test/model" and st["cost_usd_today"] == 0.002
    assert st["last_decision_ms"] == int(T0 * 1000) and st["next_decision_ms"] == int(T0 * 1000) + 3600_000
    assert a.decisions(5)[0]["t_ms"] == line["t_ms"]


def test_dry_run_writes_nothing(root):
    before = snapshot(root)
    rec = decide(make(root, FakeLLM(good())), dry_run=True)
    assert rec["dry_run"] is True and rec["requested"]["weights"]["carry"] == 0.8 and rec["inputs"]["strategies"]
    assert snapshot(root) == before and not (root / "allocator").exists()


def test_pnl_windows_come_from_equity_samples(root):
    a = make(root, FakeLLM(good()))
    st = a._state()
    st["history"]["carry"] = [[int(T0 * 1000) - 8 * 86_400_000, 900.0], [int(T0 * 1000) - 25 * 3600_000, 950.0]]
    put(root / "carry" / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 900.0, "pnl_pct": 11.1})
    ctx = a.strategy_context({"id": "carry"}, int(T0 * 1000))[0]
    assert ctx["pnl_1d_pct"] == 5.26 and ctx["pnl_7d_pct"] == 11.11 and ctx["pnl_since_start_pct"] == 11.1
    assert a.strategy_context({"id": "trend"}, int(T0 * 1000))[0]["pnl_1d_pct"] is None  # not enough history yet


def test_loop_respects_allocator_control(root):
    llm = FakeLLM(good(), good())
    a = make(root, llm)
    put(root / "allocator" / "control.json", {"enabled": False, "interval_min": 60})
    asyncio.run(a._tick())
    assert not llm.calls and a.status()["state"] == "paused"
    put(root / "allocator" / "control.json", {"enabled": True, "interval_min": 60, "decide_now": True})
    asyncio.run(a._tick())
    assert len(llm.calls) == 1 and json.loads((root / "allocator" / "control.json").read_text())["decide_now"] is False
    asyncio.run(a._tick())  # interval has not passed yet
    assert len(llm.calls) == 1


def test_request_budget_is_enforced(root):
    rec = decide(make(root, FakeLLM(good()), rph=0))
    assert "budget" in rec["error"]


def test_market_summary_and_model_choice():
    up = [100 * 1.01 ** i for i in range(120)]
    m = A.summarize_market({"BTCUSDT": up, "ETHUSDT": up, "SOLUSDT": up[::-1]}, {"BTCUSDT": up, "ETHUSDT": up},
                           {"BTCUSDT": {"fundingRate": "0.0001", "markPrice": "101", "indexPrice": "100", "lastPrice": "101"}},
                           {"BTCUSDT": [0.0001, 0.0002]}, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert m["major"]["BTC"]["funding_bp_8h"] == 1.0 and m["major"]["BTC"]["basis_pct"] == 1.0
    assert m["major"]["BTC"]["vol_30d_ann_pct"] is not None
    assert m["trend_universe"]["share_above_ema50"] == pytest.approx(0.67, abs=0.01)
    assert m["trend_universe"]["dispersion_7d_std_pct"] > 0
    assert A.allocator_spec("or:deepseek/deepseek-v4.1-flash-1017").kind == "chat"  # the chat fallback
    with pytest.raises(ValueError):
        A.allocator_spec("nonsense")


def test_missing_weight_counts_as_zero(root):
    put(root / "carry" / "control.json", {"enabled": True})
    a = make(root, FakeLLM(good()))
    assert a.strategy_context({"id": "carry"}, int(T0 * 1000))[1].weight == 0.0
    decide(a)
    assert ctl(root, "carry")["weight"] == 0.5  # model wanted 0.8, step limit from 0
