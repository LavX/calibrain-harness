"""The LLM allocator on every venue: one decision per strategy, applied to each venue the allocator controls whose runner
lists that strategy, under that venue's own risk rules (vol cap over its strategies, drawdown floor per book, day anchor
per (venue, strategy), exploration floor) and the per-write safety checks (owner_lock, pending kill, pause). Also the
state-key migration, the CTB_ALLOCATOR_VENUES opt-out, the header switch on every venue and "hand to allocator".
No network: fake models, fake market, fake systemd."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from ctb import allocator as A
from ctb import strategies as st
from ctb import venues as V
from ctb.kairos.client import RateBudget

T0 = 1_800_000_000.0
BYBIT_IDS = ["carry", "trend", "xsmom"]
OKX_IDS = ["trend", "xsmom", "longvol"]  # carry is not on OKX; longvol is not on Bybit


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def ctl(root, sid):
    return json.loads((root / sid / "control.json").read_text())


def book(root, sid, *, weight=0.0, enabled=False, equity=1000.0, owner_lock=False, **ctl_extra):
    put(root / sid / "status.json", {"id": sid, "state": "running", "enabled": enabled, "updated_ms": int(T0 * 1000),
                                     "equity_usdt": equity, "start_equity_usdt": 1000.0, "pnl_pct": 0.0,
                                     "positions": [], "targets": [], "trades_total": 0})
    put(root / sid / "control.json", {"enabled": enabled, "weight": weight, "set_by": "llm", "owner_lock": owner_lock, **ctl_extra})


@pytest.fixture
def roots(tmp_path):
    bybit, okx = tmp_path / "live", tmp_path / "live-okx"
    put(bybit / "strategies.json", [{"id": i, "label": i.title(), "unit": "calibrain-live.service"} for i in BYBIT_IDS])
    put(okx / "strategies.json", [{"id": i, "label": i.title(), "unit": "calibrain-live-okx.service"} for i in OKX_IDS])
    for i in BYBIT_IDS:
        book(bybit, i)
    for i in OKX_IDS:
        book(okx, i)
    return bybit, okx


def venues(bybit, okx, **over):
    return [A.AllocVenue("bybit-demo", bybit, "BYBIT·DEMO"), A.AllocVenue("okx-demo", okx, "OKX·DEMO", **over)]


class ChatModel:
    """The chat fallback: every strategy it is asked about on, at 0.8."""

    def __init__(self, gate: asyncio.Event | None = None):
        self.gate, self.asked, self.spec = gate, asyncio.Event(), None

    async def ask_json(self, messages, schema):
        self.asked.set()
        if self.gate is not None:
            await self.gate.wait()
        ids = list(schema["properties"]["weights"]["properties"])
        return ({"weights": {i: 0.8 for i in ids}, "enabled": {i: True for i in ids}, "reasoning": "everything on",
                 "confidence": 0.6}, {"cost": 0.0})


async def market():
    return {"major": {"BTC": {"ret_1d_pct": 1.0}}}


def chat_alloc(bybit, okx, model=None, **over):
    return A.Allocator(bybit, model or ChatModel(), market, model="chat/test", clock=lambda: T0, budget=RateBudget(50), mode="chat",
                       venues=venues(bybit, okx, **over))


def decide(a, **kw):
    return asyncio.run(a.decide(**kw))


# ---------------------------------------------------------------- one decision, applied per venue
def test_the_decision_is_applied_to_every_venue_that_lists_the_strategy(roots):
    bybit, okx = roots
    rec = decide(chat_alloc(bybit, okx))
    assert {i: ctl(bybit, i)["weight"] for i in BYBIT_IDS} == {"carry": 0.5, "trend": 0.5, "xsmom": 0.5}  # step limit 0.5 from 0
    assert ctl(okx, "trend")["weight"] == 0.5 and ctl(okx, "xsmom")["weight"] == 0.5 and ctl(okx, "trend")["set_by"] == "llm"
    assert rec["venues"]["okx-demo"]["applied"] == {"trend": True, "xsmom": True, "longvol": False}
    assert "no decision: strategy not on the decision venue" in rec["venues"]["okx-demo"]["reasons"]["longvol"]  # the model sees Bybit's books
    assert ctl(okx, "longvol")["weight"] == 0.0 and ctl(okx, "longvol")["enabled"] is False  # held, never switched on


def test_a_strategy_a_venue_does_not_list_is_never_written_there(roots):
    bybit, okx = roots
    rec = decide(chat_alloc(bybit, okx))
    assert ctl(bybit, "carry")["enabled"] is True
    assert not (okx / "carry").exists() and "carry" not in rec["venues"]["okx-demo"]["weights"]


def test_the_drawdown_floor_is_per_venue_book(roots):
    bybit, okx = roots
    book(okx, "trend", weight=0.5, enabled=True, equity=800.0)  # OKX's trend book is 20% below its start
    rec = decide(chat_alloc(bybit, okx))
    assert ctl(okx, "trend")["weight"] == 0.0 and ctl(okx, "trend")["enabled"] is False
    assert "OKX·DEMO book 20.0% below its peak" in ctl(okx, "trend")["reason"]
    assert ctl(bybit, "trend")["weight"] == 0.5  # Bybit's trend book is fine
    assert rec["venues"]["okx-demo"]["risk_floor"] == ["trend"] and rec["risk_floor"] == []
    state = json.loads((bybit / "allocator" / "state.json").read_text())
    assert state["peaks"]["okx-demo/trend"] == 1000.0 and "bybit-demo/trend" in state["peaks"]  # one peak per (venue, strategy)


def test_the_risk_check_cuts_a_venue_book_between_decisions(roots):
    bybit, okx = roots
    book(okx, "xsmom", weight=0.5, enabled=True, equity=800.0)
    a = chat_alloc(bybit, okx)
    rec = asyncio.run(a.risk_check())
    assert rec["venue"] == "okx-demo" and rec["risk_floor"] == ["xsmom"] and ctl(okx, "xsmom")["enabled"] is False


# ---------------------------------------------------------------- per-write safety on every venue
def test_owner_lock_and_a_pending_kill_are_respected_per_venue(roots):
    bybit, okx = roots
    book(okx, "trend", owner_lock=True)  # the owner's
    book(okx, "xsmom", flatten=True)     # a kill is pending
    before_trend, before_xsmom = ctl(okx, "trend"), ctl(okx, "xsmom")
    rec = decide(chat_alloc(bybit, okx))
    v = rec["venues"]["okx-demo"]
    assert v["skipped_owner_lock"] == ["trend"] and ctl(okx, "trend") == before_trend
    assert v["skipped_kill_pending"] == ["xsmom"] and ctl(okx, "xsmom") == before_xsmom
    assert ctl(bybit, "trend")["enabled"] is True and ctl(bybit, "xsmom")["enabled"] is True  # Bybit's books go ahead


def test_a_paused_allocator_only_reduces_risk_on_every_venue(roots):
    bybit, okx = roots
    book(okx, "trend", weight=0.3, enabled=True)
    gate = asyncio.Event()
    model = ChatModel(gate)
    a = chat_alloc(bybit, okx, model)
    put(bybit / "allocator" / "control.json", {"enabled": True, "interval_min": 60})

    async def go():
        task = asyncio.create_task(a.decide(trigger="decide_now"))
        await asyncio.wait_for(model.asked.wait(), 5)
        put(bybit / "allocator" / "control.json", {"enabled": False, "interval_min": 60})  # the owner pauses it mid-call
        gate.set()
        return await asyncio.wait_for(task, 5)

    rec = asyncio.run(go())
    assert ctl(okx, "trend")["weight"] == 0.3 and ctl(okx, "xsmom")["enabled"] is False  # nothing raised or switched on
    assert sorted(rec["venues"]["okx-demo"]["skipped_paused"]) == ["trend", "xsmom"]
    assert all(ctl(bybit, i)["enabled"] is False for i in BYBIT_IDS)


# ---------------------------------------------------------------- computed weights: the vol cap per venue
def test_the_vol_cap_is_computed_over_each_venues_own_strategies(roots):
    from test_allocator_computed import FakeDecider, answers, forecast

    bybit, okx = roots
    day0 = T0 - T0 % 86_400
    t = day0 + 8 * 3600
    put(bybit / "forecast" / "latest.json", forecast(t - 7200))  # carry and trend, independent, the same sigma
    a = A.Allocator(bybit, FakeDecider(answers({i: 0.5 for i in BYBIT_IDS})), market, model="ordecide / test", clock=lambda: t,
                    budget=RateBudget(50), venues=venues(bybit, okx))
    rec = decide(a, trigger="decide_now")
    one = 0.10 / (0.02 * math.sqrt(365))  # one book alone at the 10% cap (OKX has trend but not carry)
    assert rec["venues"]["okx-demo"]["core"]["trend"] == pytest.approx(one, rel=1e-3)
    assert rec["core"]["trend"] != pytest.approx(one, rel=1e-3)  # Bybit's cap is over carry and trend together
    assert set(rec["venues"]["okx-demo"]["core"]) == {"trend"}  # xsmom and longvol have no forecast: held


# ---------------------------------------------------------------- state keys
def legacy_state(bybit):
    """A state.json from before venues: keys are the allocator venue's strategy ids."""
    put(bybit / "allocator" / "state.json", {"peaks": {"trend": 1250.0}, "history": {"trend": [[int(T0 * 1000) - 3600_000, 1240.0]]},
                                              "p_hist": {"trend": [[int(T0 * 1000), 0.6]]},
                                              "day_anchor": {"day": "2027-01-15", "w": {"carry": 0.4, "trend": 0.2, "xsmom": 0.3}}})


def test_old_state_keys_are_migrated_without_moving_bybit(roots, tmp_path):
    import shutil

    bybit, okx = roots
    for i, w in zip(BYBIT_IDS, (0.4, 0.2, 0.3)):
        book(bybit, i, weight=w, enabled=True)
    twin = tmp_path / "twin"  # the same Bybit books, the allocator alone, its state already in the new keys
    shutil.copytree(bybit, twin)
    legacy_state(bybit)
    put(twin / "allocator" / "state.json", {"peaks": {"bybit-demo/trend": 1250.0},
                                             "history": {"bybit-demo/trend": [[int(T0 * 1000) - 3600_000, 1240.0]]},
                                             "p_hist": {"trend": [[int(T0 * 1000), 0.6]]}, "keys_version": 2, "primary_venue": "bybit-demo",
                                             "day_anchor": {"day": "2027-01-15", "w": {"bybit-demo/carry": 0.4, "bybit-demo/trend": 0.2,
                                                                                       "bybit-demo/xsmom": 0.3}}})
    migrated = decide(chat_alloc(bybit, okx))
    reference = decide(A.Allocator(twin, ChatModel(), market, model="chat/test", clock=lambda: T0, budget=RateBudget(50), mode="chat"))
    assert migrated["weights"] == reference["weights"] and migrated["risk_floor"] == reference["risk_floor"] == ["trend"]  # 1000 vs peak 1250
    state = json.loads((bybit / "allocator" / "state.json").read_text())
    assert state["keys_version"] == 2 and state["peaks"]["bybit-demo/trend"] == 1250.0 and "trend" not in state["peaks"]
    assert state["p_hist"]["trend"] == [[int(T0 * 1000), 0.6]]  # the decider's answers stay per strategy


def test_score_reads_the_migrated_history(roots):
    bybit, okx = roots
    t_dec = int(T0 * 1000) - 25 * 3600_000  # a decision 25 h ago, its 24 h horizon is over
    put(bybit / "allocator" / "state.json", {"peaks": {"trend": 1100.0},
                                              "history": {"trend": [[t_dec + 24 * 3600_000, 1050.0]]}})  # legacy key, the outcome sample
    (bybit / "allocator" / "decisions.jsonl").write_text(json.dumps({
        "t_ms": t_dec, "qset": "alloc-v1.1", "answers": {"p": {"trend": 0.7}}, "weights": {"trend": 1.0},
        "inputs": {"strategies": {"trend": {"equity_usdt": 1000.0}}}}) + "\n")
    a = chat_alloc(bybit, okx)
    a._state()
    a._save_state()  # the migration writes "bybit-demo/trend"
    assert "bybit-demo/trend" in json.loads((bybit / "allocator" / "state.json").read_text())["history"]
    out = A.score_decisions(bybit, now_ms=int(T0 * 1000))
    assert out["strategies"]["trend"]["n"] == 1 and out["strategies"]["trend"]["base_rate"] == 1.0  # labelled from the new key


def test_an_old_day_anchor_is_migrated_and_still_limits_the_step(roots, tmp_path):
    """The computed mode steps up from the day's anchor: a migrated anchor must bind exactly as before."""
    import shutil
    import time

    from test_allocator_computed import FakeDecider, answers, forecast

    bybit, okx = roots
    t = (T0 - T0 % 86_400) + 8 * 3600
    put(bybit / "forecast" / "latest.json", forecast(t - 7200))
    for i, w in zip(BYBIT_IDS, (0.5, 0.6, 0.3)):
        book(bybit, i, weight=w, enabled=True)
    twin = tmp_path / "twin"
    shutil.copytree(bybit, twin)
    day = time.strftime("%Y-%m-%d", time.gmtime(t))
    anchor = {"carry": 0.1, "trend": 0.1, "xsmom": 0.3}  # this morning's weights: trend may rise to 0.35 at most today
    put(bybit / "allocator" / "state.json", {"day_anchor": {"day": day, "w": anchor}})
    put(twin / "allocator" / "state.json", {"keys_version": 2, "primary_venue": "bybit-demo",
                                             "day_anchor": {"day": day, "w": {f"bybit-demo/{k}": v for k, v in anchor.items()}}})

    def run_on(root, vs=None):
        a = A.Allocator(root, FakeDecider(answers({i: 0.5 for i in BYBIT_IDS})), market, model="ordecide / test", clock=lambda: t,
                        budget=RateBudget(50), venues=vs)
        return decide(a, trigger="decide_now")

    migrated, reference = run_on(bybit, venues(bybit, okx)), run_on(twin)
    assert migrated["weights"] == reference["weights"]
    assert migrated["weights"]["trend"] <= 0.1 + 0.25 + 1e-9  # the migrated anchor bound the step (without it: from 0.6)


# ---------------------------------------------------------------- the opt-out
def test_ctb_allocator_venues_leaves_a_venue_to_the_owner(roots, monkeypatch):
    bybit, okx = roots
    monkeypatch.setenv("CTB_LIVE_ROOTS", f"bybit-demo={bybit},okx-demo={okx}")
    monkeypatch.setenv("CTB_ALLOCATOR_VENUES", "bybit-demo")
    h = V.VenueHub.from_env()
    assert {v.id: v.controlled_by for v in h.venues} == {"bybit-demo": "allocator", "okx-demo": "owner"}
    assert [v.id for v in h.allocator_venues()] == ["bybit-demo"] and h.services["okx-demo"].has_allocator is False
    a = A.Allocator(bybit, ChatModel(), market, model="chat/test", clock=lambda: T0, budget=RateBudget(50), mode="chat",
                    venues=h.allocator_venues())
    before = ctl(okx, "trend")
    rec = decide(a)
    assert ctl(okx, "trend") == before and "okx-demo" not in rec["venues"]
    monkeypatch.delenv("CTB_ALLOCATOR_VENUES")
    h = V.VenueHub.from_env()
    assert [v.id for v in h.allocator_venues()] == ["bybit-demo", "okx-demo"]  # default: every configured venue
    monkeypatch.setenv("CTB_ALLOCATOR_VENUES", "bybit-demo,ftx")
    with pytest.raises(ValueError):
        V.VenueHub.from_env()


def test_an_unmanaged_allocator_venue_is_computed_but_not_written(roots):
    bybit, okx = roots
    a = A.Allocator(bybit, ChatModel(), market, model="chat/test", clock=lambda: T0, budget=RateBudget(50), mode="chat",
                    venues=[A.AllocVenue("bybit-demo", bybit, "BYBIT·DEMO", managed=False), A.AllocVenue("okx-demo", okx, "OKX·DEMO")])
    rec = decide(a)
    assert all(ctl(bybit, i)["enabled"] is False for i in BYBIT_IDS) and rec["applied"] == {}
    assert ctl(okx, "trend")["enabled"] is True


# ---------------------------------------------------------------- the dashboard side
class FakeSystemd:
    def __init__(self, active=()):
        self.active = set(active)

    async def __call__(self, argv, **_):
        unit = argv[-1]
        if argv[2] == "show":
            return 0, f"LoadState=loaded\nActiveState={'active' if unit in self.active else 'inactive'}\n", ""
        if argv[2] == "start":
            self.active.add(unit)
        return 0, "", ""


def hub_of(bybit, okx, tmp_path, sd):
    vs = [V.venue_config("bybit-demo", bybit, unit=""), V.venue_config("okx-demo", okx)]
    services = {"bybit-demo": st.StrategyService(bybit, run=sd, clock=lambda: T0, extra_units=(), dashboard_account="", allocator_available=True),
                "okx-demo": st.StrategyService(okx, run=sd, clock=lambda: T0, extra_units=("calibrain-live-okx.service",),
                                               dashboard_account="", shares_bot_account=False, has_allocator=True)}
    services["bybit-demo"]._mode = "llm"
    return V.VenueHub(vs, services, state=tmp_path / "state", clock=lambda: T0)


def test_hand_to_allocator_clears_owner_locks_but_not_over_a_pending_kill(roots, tmp_path):
    bybit, okx = roots
    for i in OKX_IDS:  # the deploy state: the owner's, weight 1
        put(okx / i / "control.json", {"enabled": True, "weight": 1.0, "set_by": "owner", "owner_lock": True})
    c = ctl(okx, "longvol")
    put(okx / "longvol" / "control.json", {**c, "flatten": True})  # a kill is pending there
    svc = hub_of(bybit, okx, tmp_path, FakeSystemd()).service("okx-demo")
    with pytest.raises(st.StrategyError) as e:
        svc.hand_to_allocator()
    assert e.value.status == 409 and sorted(e.value.detail["handed"]) == ["trend", "xsmom"] and "longvol" in e.value.detail["refused"]
    assert ctl(okx, "trend")["owner_lock"] is False and ctl(okx, "trend")["set_by"] == "llm" and ctl(okx, "trend")["weight"] == 1.0
    assert ctl(okx, "longvol")["owner_lock"] is True
    out = svc.hand_to_allocator(cancel_kill=True)
    assert out["handed"] == ["longvol"] and sorted(out["already"]) == ["trend", "xsmom"] and ctl(okx, "longvol")["flatten"] is False
    owner_svc = st.StrategyService(okx, run=FakeSystemd(), clock=lambda: T0, extra_units=(), dashboard_account="",
                                   shares_bot_account=False, has_allocator=False)
    with pytest.raises(st.StrategyError):
        owner_svc.hand_to_allocator()  # a venue left to the owner


def test_the_header_switch_reaches_every_allocator_venue(roots, tmp_path):
    bybit, okx = roots
    for i in OKX_IDS:
        book(okx, i, weight=0.5, enabled=True)
    sd = FakeSystemd()
    h = hub_of(bybit, okx, tmp_path, sd)
    out = asyncio.run(h.auto_others(True, "llm"))
    assert "calibrain-live-okx.service" in sd.active and "allocator decides" in out["okx-demo"]
    out = asyncio.run(h.auto_others(False, "llm"))
    assert out["okx-demo"] == "paused" and all(ctl(okx, i)["enabled"] is False for i in OKX_IDS)


def test_the_drawdown_floor_keeps_running_on_every_venue_while_the_allocator_is_paused(roots, tmp_path):
    """A KILL on bybit-demo pauses the allocator; OKX's books are still open and must still be cut by the floor."""
    bybit, okx = roots
    for i in BYBIT_IDS:
        book(bybit, i, weight=0.5, enabled=True)
    book(okx, "trend", weight=0.6, enabled=True)
    put(bybit / "allocator" / "control.json", {"enabled": True, "interval_min": 60})
    sd = FakeSystemd(active=["calibrain-live.service", "calibrain-live-okx.service"])
    h = hub_of(bybit, okx, tmp_path, sd)
    asyncio.run(h.kill("bybit-demo"))  # pauses the allocator (and flags bybit's books)
    assert json.loads((bybit / "allocator" / "control.json").read_text())["enabled"] is False
    book(okx, "trend", weight=0.6, enabled=True, equity=750.0)  # OKX trend falls 25%
    a = chat_alloc(bybit, okx)
    asyncio.run(a._tick())
    c = ctl(okx, "trend")
    assert c["weight"] == 0.0 and c["enabled"] is False and c["flatten_on_zero"] is True
    assert a.status()["state"] == "paused"  # no decision ran


def test_an_opted_out_allocator_venue_refuses_unlock_and_hand_over(roots, monkeypatch):
    bybit, okx = roots
    book(bybit, "trend", owner_lock=True)
    book(okx, "trend", owner_lock=True)
    monkeypatch.setenv("CTB_LIVE_ROOTS", f"bybit-demo={bybit},okx-demo={okx}")
    monkeypatch.setenv("CTB_ALLOCATOR_VENUES", "okx-demo")  # bybit-demo is the owner's
    h = V.VenueHub.from_env()
    assert h.services["bybit-demo"].has_allocator is False and h.services["okx-demo"].has_allocator is True
    assert [(v.id, v.managed) for v in h.allocator_venues()] == [("bybit-demo", False), ("okx-demo", True)]
    for svc in (h.services["bybit-demo"],):
        svc.run = FakeSystemd()
        with pytest.raises(st.StrategyError) as e:
            svc.hand_to_allocator()
        assert e.value.status == 409 and "not controlled by the LLM allocator" in e.value.message
        with pytest.raises(st.StrategyError) as e:
            asyncio.run(svc.act("trend", "unlock", llm_trading=False, venue="demo"))
        assert e.value.status == 409
    assert ctl(bybit, "trend")["owner_lock"] is True
    assert h.services["okx-demo"].hand_to_allocator()["handed"] == ["trend"]  # the managed venue still can


def test_a_strategy_only_a_secondary_venue_lists_is_held_in_the_computed_mode(roots):
    from test_allocator_computed import FakeDecider, answers, forecast

    bybit, okx = roots
    book(okx, "longvol", weight=0.3, enabled=True)  # OKX lists longvol, Bybit does not
    day0 = T0 - T0 % 86_400
    t = day0 + 8 * 3600
    f = forecast(t - 7200)
    f["strategies"]["longvol"] = dict(f["strategies"]["trend"])  # the forecast file still has a row for it
    put(bybit / "forecast" / "latest.json", f)
    a = A.Allocator(bybit, FakeDecider(answers({i: 0.5 for i in BYBIT_IDS})), market, model="ordecide / test", clock=lambda: t,
                    budget=RateBudget(50), venues=venues(bybit, okx))
    rec = decide(a, trigger="decide_now")
    okx_rec = rec["venues"]["okx-demo"]
    assert "longvol" not in okx_rec["core"] and okx_rec["applied"]["longvol"] is False
    assert ctl(okx, "longvol")["weight"] == 0.3 and "no decision: strategy not on the decision venue" in okx_rec["reasons"]["longvol"]


def test_an_unusable_venue_strategies_json_writes_nothing_there(roots):
    bybit, okx = roots
    put(okx / "strategies.json", [{"id": "../live/carry", "label": "escape"}])  # an id that would leave the venue's root
    before = {i: ctl(bybit, i) for i in BYBIT_IDS}
    a = chat_alloc(bybit, okx)
    assert a.strategies(a.venues[1]) == []
    rec = decide(a)
    assert "no usable strategies.json" in rec["venues"]["okx-demo"]["note"]
    assert ctl(bybit, "carry")["weight"] == 0.5 and before["carry"]["weight"] == 0.0  # only the allocator's own Bybit write


def test_the_header_switch_reports_venues_it_could_not_reach(roots, tmp_path):
    from types import SimpleNamespace

    from aiohttp.test_utils import TestClient, TestServer

    from ctb.server import make_app

    bybit, okx = roots
    put(okx / "strategies.json", [{"id": "trend", "unit": "bad unit; rm"}])  # unusable: its strategies cannot be paused

    class Engine:
        s = SimpleNamespace(trading_enabled=False, public_view=lambda: {})
        db = SimpleNamespace(event=lambda *a: a, one=lambda *a: None)
        executor = None

        async def start(self): ...
        async def stop(self): ...
        def emit(self, *a): ...

    h = hub_of(bybit, okx, tmp_path, FakeSystemd())
    app = make_app(Engine(), hub=h)
    app.on_startup.clear()
    app.on_cleanup.clear()

    async def go():
        async with TestClient(TestServer(app)) as c:
            r = await c.post("/api/strategies/auto", json={"enabled": False})
            return r.status, await r.json()

    status, body = asyncio.run(go())
    assert status == 409 and "OKX·DEMO" in body["error"] and "unusable" in body["error"]
    assert all(ctl(bybit, i)["enabled"] is False for i in BYBIT_IDS)  # the venues it could reach were switched off
