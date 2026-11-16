"""Managed stops are checked on every mainnet ticker update, not only on the 0.5 s pass."""

import asyncio
from types import SimpleNamespace

from ctb.engine import Engine
from ctb.market.bybit_public import BybitPublicFeed
from ctb.market.store import MarketStore


def test_ticker_update_notifies_tick_callbacks():
    feed = BybitPublicFeed(MarketStore(["BTCUSDT"]), None)
    seen = []
    feed.on_tick.append(seen.append)
    asyncio.run(feed._handle({"topic": "tickers.BTCUSDT", "type": "delta", "data": {"symbol": "BTCUSDT", "ask1Price": "65000"}}))
    assert seen == ["BTCUSDT"]
    assert feed.store["BTCUSDT"].ticker["ask1Price"] == "65000"


class _Executor:
    def __init__(self):
        self._managed = {"BTCUSDT": {}}
        self.calls = 0

    async def check_stop(self, symbol):
        self.calls += 1
        await asyncio.sleep(0.01)


def _engine(executor):
    eng = SimpleNamespace(executor=executor, _tick_checks={})
    eng._tick_stop = lambda ex, sym: Engine._tick_stop(eng, ex, sym)
    return eng


def test_tick_checks_managed_stops_once_at_a_time():
    async def run():
        ex = _Executor()
        eng = _engine(ex)
        for _ in range(5):  # a burst of ticks while one check is still running
            Engine._on_mainnet_tick(eng, "BTCUSDT")
        Engine._on_mainnet_tick(eng, "ETHUSDT")  # no managed stop on ETH
        await asyncio.sleep(0.05)
        assert ex.calls == 1 and not eng._tick_checks
        Engine._on_mainnet_tick(eng, "BTCUSDT")  # next tick after the check finished
        await asyncio.sleep(0.05)
        assert ex.calls == 2
        Engine._on_mainnet_tick(_engine(None), "BTCUSDT")  # no executor (no keys): nothing to do
    asyncio.run(run())
