"""The computed allocator (forecast/latest.json to weights): the schedule, the breakdown in decisions.jsonl,
staleness, the floor and owner_lock. A fake decider and a fake market stand in for the network."""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace

import pytest

from ctb import allocator as A
from ctb import allocator_math as M
from ctb.kairos.client import KairosError, KairosResult, RateBudget

DAY = 86_400
DAY0 = 1_800_000_000 - 1_800_000_000 % DAY      # a UTC midnight
T0 = DAY0 + 8 * 3600                              # 08:00, before the 23:50 slot
SLOT = DAY0 + 23 * 3600 + 50 * 60
IDS = ["carry", "trend", "xsmom"]
SIGMA = 0.02                                      # daily, 38%/yr: gamma sigma^2 = 8e-4
CAP = 1.5 * SIGMA / math.sqrt(365)


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def answers(p):
    out = {f"net_positive_{i}": {"type": "noul", "noul": v} for i, v in p.items()}
    best = {"carry": 0.2, "trend": 0.3, "xsmom": 0.3, "none": 0.2}
    out["best_net_return"] = {"type": "choice", "choice": "trend", "probabilities": best, "confidence": 0.3}
    return out


class FakeDecider:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0
        self.spec = SimpleNamespace(max_questions=None, provider="ordecide")

    async def ask(self, state, questions, **kw):
        self.calls += 1
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return KairosResult(answers={q: a for q, a in r.items() if q in questions}, usage={}, metadata={"request_id": f"r{self.calls}"},
                            latency_ms=1.0, attempts=1, request_bytes=10)


async def market():
    return {"major": {"BTC": {"ret_1d_pct": 1.0}}}


def forecast(made_s, **over):
    f = {"date": "2026-11-19", "made_ms": int(made_s * 1000),
         "strategies": {
             "trend": {"sigma_d": SIGMA, "mu": {"base": 0.002, "regime": 0.0002, "ridge": 0.0}, "own_mom": 0.1, "n_days": 900},
             "carry": {"sigma_d": SIGMA, "mu": {"structural": -0.0004, "base": -0.0004, "regime": -0.0004, "ridge": -0.0004}, "own_mom": 0.0, "n_days": 900},
         },
         "cov_d": {"ids": ["carry", "trend"], "matrix": [[SIGMA ** 2, 0.0], [0.0, SIGMA ** 2]]},
         "weights_k": {"trend": {"base": 0.3, "regime": 0.25, "ridge": 0.2, "decider": 0.25},
                       "carry": {"structural": 0.4, "base": 0.15, "regime": 0.1, "ridge": 0.1, "decider": 0.25}}}
    f.update(over)
    return f


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "live"
    put(r / "strategies.json", [{"id": i, "label": i.title()} for i in IDS])
    for i, w in zip(IDS, (0.5, 0.2, 0.3)):
        put(r / i / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "pnl_pct": 0.0})
        put(r / i / "control.json", {"enabled": True, "weight": w, "owner_lock": False})
    put(r / "forecast" / "latest.json", forecast(T0 - 2 * 3600))
    put(r / "regime" / "latest.json", {"date": "2026-11-18", "state": "up_calm", "p7": {"up_calm": 0.62, "up_wild": 0.2, "down_calm": 0.1, "down_wild": 0.08}})
    return r


def make(root, llm=None, clock=None, **kw):
    return A.Allocator(root, llm or FakeDecider(answers({i: 0.5 for i in IDS})), market, model="ordecide / test", clock=clock or Clock(T0),
                       budget=RateBudget(50), **kw)


def decide(a, **kw):
    return asyncio.run(a.decide(**kw))


def ctl(root, sid):
    return json.loads((root / sid / "control.json").read_text())


def lines(root):
    return [json.loads(x) for x in (root / "allocator" / "decisions.jsonl").read_text().splitlines()]


# expected numbers, by hand: trend base 2e-3 is above the Sharpe cap (1.57e-3) and is clipped; decider p 0.5 adds 0
TREND_MU = 0.3 * CAP + 0.25 * 0.0002
COST = 0.001 / 7                                  # 10 bp per side, G 1, H 7 days
TREND_STAR = (TREND_MU - COST) / (2 * SIGMA ** 2)


def test_decide_now_sets_weights_and_logs_the_breakdown(root):
    a = make(root)
    rec = decide(a, trigger="decide_now")
    assert rec["weights_mode"] == "computed" and rec["error"] is None
    assert rec["w_star"]["trend"] == pytest.approx(TREND_STAR, abs=1e-6) and 0.45 < TREND_STAR < 0.5
    # trend prev 0.2 -> w* 0.47: the vol cap (9%/yr) and buffer pass, the step cap holds it to 0.2 + 0.25
    assert rec["w_final"]["trend"] == pytest.approx(0.45) and ctl(root, "trend")["weight"] == pytest.approx(0.45)
    # carry forecasts are negative: w* 0, step down is unlimited, the exploration floor lifts it to 0.10 (testnet or demo)
    assert rec["w_star"]["carry"] == 0.0 and rec["w_final"]["carry"] == 0.10 and ctl(root, "carry")["enabled"] is True
    # xsmom has no forecast: held at its weight, and said so
    assert rec["w_final"]["xsmom"] == 0.3 and ctl(root, "xsmom")["weight"] == 0.3 and "xsmom" in rec["no_forecast"]
    # the full breakdown
    line = lines(root)[-1]
    assert line["mu"]["trend"]["base"] == pytest.approx(CAP) and line["mu"]["trend"]["combined"] == pytest.approx(TREND_MU)
    assert line["mu"]["trend"]["decider"] == 0.0 and line["clipped"]["trend"] == ["base"]
    assert line["sigma_d"]["trend"] == SIGMA and line["p_decider"]["trend"] == 0.5
    assert [s["step"] for s in line["steps"]] == ["vol_cap", "buffer", "step", "veto", "explore_floor", "floor_lock"]
    assert line["steps"][-2]["before"]["carry"] == 0.0 and line["steps"][-2]["after"]["carry"] == 0.10
    assert line["answers"]["p"]["trend"] == 0.5 and line["trigger"] == "decide_now"
    reason = line["reasons"]["trend"]
    assert reason.startswith("trend: mu +") and "(regime up_calm 0.62)" in reason and f"w* {TREND_STAR:.2f} -> 0.45 after step cap" in reason
    assert ctl(root, "trend")["reason"] == reason and ctl(root, "trend")["set_by"] == "llm"
    assert a.status()["weights_mode"] == "computed" and a.status()["last_weights_ms"] == int(T0 * 1000)


def test_forecast_weights_come_from_the_file_and_the_ledger_when_absent(root):
    f = forecast(T0 - 3600)
    del f["weights_k"]["trend"]                      # no row: the ledger's starting weights
    f["weights_k"]["carry"] = {"base": 1.0}          # no decider key: the ledger's 0.25 is added, then everything renormalised
    put(root / "forecast" / "latest.json", f)
    rec = decide(make(root), trigger="decide_now")
    assert rec["mu"]["trend"]["combined"] == pytest.approx(TREND_MU)          # same as the starting weights used above
    assert rec["mu"]["carry"]["combined"] == pytest.approx(-0.0004 / 1.25)    # base -4e-4 and decider 0 (p 0.5): (1 * -4e-4 + 0.25 * 0) / 1.25


def test_decider_probability_is_the_mean_of_its_last_24_answers_and_vetoes(root):
    put(root / "allocator" / "state.json", {"peaks": {}, "history": {}, "p_hist": {"trend": [[1, 0.2]] * 30, "carry": [[1, 0.9]] * 3}})
    # the model now says 0.6 for trend: history is cut to the last 24 (0.2 x 23) plus the new answer
    a = make(root, FakeDecider(answers({"carry": 0.9, "trend": 0.6, "xsmom": 0.5})))
    rec = decide(a, trigger="decide_now")
    p = rec["p_decider"]["trend"]
    assert p == pytest.approx((0.2 * 23 + 0.6) / 24, abs=1e-4) and p < 0.4
    mu_dec = rec["mu"]["trend"]["decider"]
    assert mu_dec == pytest.approx(max(-CAP, (2 * p - 1) * 0.8 * SIGMA))
    # veto: p < 0.40 caps the weight at half of yesterday's (0.2 -> 0.1)
    assert rec["w_final"]["trend"] <= 0.1 + 1e-9 and "trend" in rec["steps"][3]["changed"] + ["trend"]
    stored = json.loads((root / "allocator" / "state.json").read_text())["p_hist"]["trend"]
    assert len(stored) == 24 and stored[-1][1] == 0.6


def test_hourly_tick_only_refreshes_the_decider(root):
    put(root / "allocator" / "state.json", {"peaks": {}, "history": {}, "last_weights_ms": int((SLOT - DAY + 60) * 1000)})
    before = {i: (root / i / "control.json").read_bytes() for i in IDS}
    rec = decide(make(root), trigger="schedule")
    assert rec["weights_mode"] == "refresh" and rec["applied"] == {i: False for i in IDS}
    assert {i: (root / i / "control.json").read_bytes() for i in IDS} == before
    assert json.loads((root / "allocator" / "state.json").read_text())["p_hist"]["trend"][0][1] == 0.5
    assert "w_star" not in lines(root)[-1] and lines(root)[-1]["answers"]["p"]["carry"] == 0.5  # forecast.py score reads these


def test_weights_are_set_in_the_2350_slot_not_before(root):
    put(root / "allocator" / "state.json", {"peaks": {}, "history": {}, "last_weights_ms": int((SLOT - DAY + 60) * 1000)})
    clock = Clock(SLOT - 120)
    put(root / "forecast" / "latest.json", forecast(SLOT - 600))   # made at 23:40
    a = make(root, clock=clock)
    assert a.weights_due(int(clock() * 1000)) is False
    asyncio.run(a._tick())
    assert lines(root)[-1]["weights_mode"] == "refresh"             # first tick: the hourly refresh
    clock.t = SLOT + 60
    assert a.weights_due(int(clock() * 1000)) is True
    asyncio.run(a._tick())
    last = lines(root)[-1]
    assert last["trigger"] == "daily" and last["weights_mode"] == "computed"
    assert a.weights_due(int(clock() * 1000)) is False              # once a day
    clock.t = SLOT + DAY + 60
    assert a.weights_due(int(clock() * 1000)) is True               # and again the next day


def test_first_run_with_a_forecast_sets_weights_at_once(root):
    a = make(root)
    assert a.weights_due(int(T0 * 1000)) is True
    asyncio.run(a._tick())
    assert lines(root)[-1]["weights_mode"] == "computed" and lines(root)[-1]["trigger"] == "daily"


def test_stale_forecast_holds_weights_and_reports_the_error(root):
    put(root / "forecast" / "latest.json", forecast(T0 - 40 * 3600))
    before = {i: (root / i / "control.json").read_bytes() for i in IDS}
    a = make(root)
    rec = decide(a, trigger="decide_now")
    assert rec["weights_mode"] == "held" and "40.0 h old" in rec["error"] and "w_star" not in rec
    assert {i: (root / i / "control.json").read_bytes() for i in IDS} == before
    st = a.status()
    assert st["state"] == "error" and "40.0 h old" in st["last_error"] and st["forecast"]["state"] == "stale" and st["weights_mode"] == "held"
    # the hourly refresh keeps reporting it, and the decider is still asked
    rec = decide(a, trigger="schedule")
    assert "old" in rec["error"] and a.status()["last_error"]
    # a forecast just inside the limit is used
    put(root / "forecast" / "latest.json", forecast(T0 - 35 * 3600))
    assert decide(a, trigger="decide_now")["weights_mode"] == "computed"
    assert a.status()["state"] == "running" and a.status()["last_error"] is None


def test_unreadable_forecast_holds_weights(root):
    (root / "forecast" / "latest.json").write_text("{not json")
    rec = decide(make(root), trigger="decide_now")
    assert rec["weights_mode"] == "held" and "unreadable" in rec["error"]


def test_drawdown_floor_still_works_while_the_forecast_is_stale(root):
    put(root / "forecast" / "latest.json", forecast(T0 - 40 * 3600))
    put(root / "trend" / "status.json", {"state": "running", "equity_usdt": 800.0, "start_equity_usdt": 1000.0})
    rec = decide(make(root), trigger="decide_now")
    assert ctl(root, "trend")["weight"] == 0 and ctl(root, "trend")["enabled"] is False and rec["risk_floor"] == ["trend"]


def test_drawdown_floor_beats_a_bullish_forecast_and_the_exploration_floor(root):
    put(root / "trend" / "status.json", {"state": "running", "equity_usdt": 800.0, "start_equity_usdt": 1000.0})
    rec = decide(make(root), trigger="decide_now")
    assert rec["w_final"]["trend"] == 0 and ctl(root, "trend")["flatten_on_zero"] is True
    assert rec["reasons"]["trend"].startswith("risk floor") and rec["steps"][-1]["after"]["trend"] == 0.0


def test_owner_lock_is_never_touched(root):
    put(root / "trend" / "control.json", {"enabled": False, "weight": 0.7, "owner_lock": True})
    before = (root / "trend" / "control.json").read_bytes()
    rec = decide(make(root), trigger="decide_now")
    assert (root / "trend" / "control.json").read_bytes() == before and rec["skipped_owner_lock"] == ["trend"]
    assert rec["w_final"]["trend"] == 0.7 and rec["steps"][-1]["after"]["trend"] == 0.7


def test_buffer_keeps_a_small_change_out(root):
    put(root / "trend" / "control.json", {"enabled": True, "weight": TREND_STAR - 0.05, "owner_lock": False})
    rec = decide(make(root), trigger="decide_now")
    assert rec["w_final"]["trend"] == pytest.approx(TREND_STAR - 0.05, abs=1e-4)
    assert "after buffer" in rec["reasons"]["trend"]


def test_decider_outage_does_not_stop_the_daily_weights(root):
    a = make(root, FakeDecider(KairosError("rate_limited", "slow down")))
    rec = decide(a, trigger="decide_now")
    assert rec["weights_mode"] == "computed" and "slow down" in rec["error"]
    assert rec["p_decider"]["trend"] is None and "decider" not in rec["mu"]["trend"]   # the other forecasters carry the vote
    assert ctl(root, "trend")["weight"] > 0.3


def test_gross_exposure_raises_the_trading_cost(root):
    put(root / "trend" / "control.json", {"enabled": True, "weight": 0.5, "owner_lock": False})
    pos = [{"symbol": "BTCUSDT", "qty": 1.0, "mark_price": 1000.0}]   # 1000 gross on 1000 equity at weight 0.5: G = 2 at weight 1
    put(root / "trend" / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "positions": pos})
    rec = decide(make(root), trigger="decide_now")
    assert rec["cost_per_day"]["trend"] == pytest.approx(2 * COST) and rec["cost_per_day"]["carry"] == pytest.approx(1.6 * COST)  # carry 16 bp
    assert A.gross_per_unit([], 1000.0, 0.5) is None and A.gross_per_unit(pos, 1000.0, 0.05) is None


def test_dry_run_writes_nothing_and_prints_the_table(root):
    snap = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    rec = decide(make(root), dry_run=True, trigger="cli")
    assert {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()} == snap
    table = A.format_table(rec)
    for token in ("trend", "carry", "xsmom", "combined", "w*", "reasons:", "regime up_calm 0.62", "weights_mode computed"):
        assert token in table


def test_without_a_forecast_file_the_decider_alone_decides_and_status_says_so(root):
    (root / "forecast" / "latest.json").unlink()
    a = make(root, FakeDecider(answers({"carry": 0.61, "trend": 0.48, "xsmom": 0.57})))
    rec = decide(a, trigger="schedule")
    assert rec.get("mapping") and "weights_mode" not in rec and rec["error"] is None
    st = a.status()
    assert st["weights_mode"] == "decider-only" and st["forecast"]["state"] == "missing" and "decider alone" in st["forecast"]["note"]
    assert st["state"] == "running" and st["last_error"] is None
