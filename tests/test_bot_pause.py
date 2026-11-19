"""While the live strategies own the demo account the per-symbol bot must not ask its model, check its budget or log budget
warnings. In mode "bot" a zero budget is logged at most once an hour, not once per symbol per bar."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from ctb.config import load_settings
from ctb.engine import Engine


@pytest.fixture
def eng(tmp_path):
    s = load_settings()
    s.db_path = tmp_path / "bot.db"
    s.kairos_max_requests_per_hour = 0  # the aurora4 setting: the old bot's budget is deliberately zero
    e = Engine(s)
    e.kairos = SimpleNamespace(configured=True)
    return e


def events(e, kind):
    return [r for r in e.db.query("SELECT level, kind, message FROM events") if r["kind"] == kind]


def test_strategies_mode_makes_no_decisions_and_logs_one_info_event(eng):
    eng.strategies_own_account = lambda: True
    taken = []
    eng.budget.try_take = lambda: taken.append(1) or False
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        for _ in range(3):  # three bars each
            assert asyncio.run(eng.decide(sym)) is None
    assert taken == []  # the budget was never consulted
    assert events(eng, "budget") == []
    paused = events(eng, "bot_paused")
    assert len(paused) == 1 and paused[0]["level"] == "info"
    assert paused[0]["message"] == "per-symbol bot paused: strategies mode"


def test_bar_close_in_strategies_mode_skips_the_settle_wait(eng):
    eng.strategies_own_account = lambda: True
    eng.s.bar_settle_s = 60  # would hang the test if the bar still waited
    asyncio.run(asyncio.wait_for(eng.on_bar_close("BTCUSDT", eng.s.decision_tf), 2))
    assert len(events(eng, "bot_paused")) == 1


def test_entering_strategies_mode_again_logs_again(eng):
    mode = {"own": True}
    eng.strategies_own_account = lambda: mode["own"]
    asyncio.run(eng.decide("BTCUSDT"))
    mode["own"] = False
    asyncio.run(eng.decide("BTCUSDT"))  # back in bot mode: zero budget, one warning
    mode["own"] = True
    asyncio.run(eng.decide("BTCUSDT"))
    assert len(events(eng, "bot_paused")) == 2


def test_bot_mode_zero_budget_warns_at_most_once_per_hour(eng):
    eng.strategies_own_account = lambda: False
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        for _ in range(3):
            assert asyncio.run(eng.decide(sym)) is None
    warns = events(eng, "budget")
    assert len(warns) == 1 and warns[0]["level"] == "warn" and "exhausted" in warns[0]["message"]
    assert events(eng, "bot_paused") == []
    eng._budget_warn_at = time.time() - 3601  # an hour later the warning may return, once
    asyncio.run(eng.decide("BTCUSDT"))
    asyncio.run(eng.decide("ETHUSDT"))
    assert len(events(eng, "budget")) == 2


def test_default_engine_is_never_paused(eng):
    assert eng._bot_paused() is False


def test_server_wires_the_strategy_mode_check(eng, tmp_path):
    from ctb import server
    from ctb import strategies as st

    svc = st.StrategyService(tmp_path / "live", extra_units=(), dashboard_account="demo-main", allocator_available=True)
    server.make_app(eng, svc)
    assert eng.strategies_own_account == svc.owns_account
