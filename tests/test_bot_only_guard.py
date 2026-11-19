"""The per-symbol bot's settings (EXPLORE, position cap, risk, trade timeframe, model) change only in mode "bot".
In LLM DECIDES and MANUAL the routes answer 409 with the dashboard's tooltip text, so a stale page cannot edit them."""

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from ctb import server
from ctb import strategies as st
from ctb.config import load_settings

FIX = Path(__file__).parent / "fixtures" / "live"
MSG = "per-symbol bot only: switch the header mode to PER-SYMBOL BOT"


class FakeEngine:
    """Just enough of Engine for the routes under test. No network, no database."""

    def __init__(self):
        self.s = load_settings()
        self.probe_rate = 0.0
        self.budget = SimpleNamespace(per_hour=0)
        self.models: list[str] = []
        self.events: list = []
        self.db = SimpleNamespace(event=lambda *a, **k: a, one=lambda *a, **k: None)

    def emit(self, *a, **k):
        self.events.append(a)

    async def set_model(self, provider):
        self.models.append(provider)
        return {"ok": True, "model_provider": provider}


@pytest.fixture
def saved(monkeypatch):
    """Never let a test write the developer's .env."""
    calls: list[dict] = []
    monkeypatch.setattr(server, "save_env", lambda updates, *a, **k: calls.append(dict(updates)))
    return calls


def call(tmp_path, mode, method, path, body):
    root = tmp_path / "live"
    shutil.copytree(FIX, root, dirs_exist_ok=True)
    svc = st.StrategyService(root, clock=lambda: 1_800_000_030.0, extra_units=(), dashboard_account="demo-main", allocator_available=True)
    svc._mode = mode
    engine = FakeEngine()
    app = server.make_app(engine, svc)
    app.on_startup.clear()
    app.on_cleanup.clear()

    async def go():
        async with TestClient(TestServer(app)) as c:
            r = await c.request(method, path, json=body)
            return r.status, await r.json()

    return asyncio.run(go()), engine


BOT_ONLY_CALLS = [
    ("/api/settings", {"exploration_mode": True}),
    ("/api/settings", {"max_positions": 5}),
    ("/api/settings", {"risk_per_trade_pct": 0.5}),
    ("/api/decision-tf", {"tf": "15"}),
    ("/api/model", {"provider": "jev"}),
]


@pytest.mark.parametrize("mode", ["llm", "manual"])
@pytest.mark.parametrize("path,body", BOT_ONLY_CALLS)
def test_refused_when_the_strategies_own_the_account(tmp_path, saved, mode, path, body):
    (status, doc), engine = call(tmp_path, mode, "POST", path, body)
    assert status == 409 and doc == {"error": MSG}
    assert saved == [] and engine.models == []  # nothing was changed or persisted


def test_refused_settings_change_nothing(tmp_path, saved):
    before = load_settings().max_positions
    (status, _), engine = call(tmp_path, "llm", "POST", "/api/settings", {"max_positions": before + 1, "exploration_mode": True})
    assert status == 409 and engine.s.max_positions == before and not engine.s.exploration_mode


def test_allowed_in_bot_mode(tmp_path, saved):
    (status, _), engine = call(tmp_path, "bot", "POST", "/api/settings", {"exploration_mode": True, "max_positions": 4})
    assert status == 200 and engine.s.exploration_mode is True and engine.s.max_positions == 4
    assert {"CTB_MAX_POSITIONS": "4"} in saved and {"CTB_EXPLORATION": "1"} in saved
    (status, doc), engine = call(tmp_path, "bot", "POST", "/api/model", {"provider": "jev"})
    assert status == 200 and doc["ok"] and engine.models == ["jev"]
    (status, _), engine = call(tmp_path, "bot", "POST", "/api/decision-tf", {"tf": "15"})
    assert status == 200 and engine.s.decision_tf == "15"


def test_other_settings_still_change_in_any_mode(tmp_path, saved):
    (status, _), engine = call(tmp_path, "llm", "POST", "/api/settings", {"min_edge_r": 0.123})
    assert status == 200 and engine.s.min_edge_r == 0.123
