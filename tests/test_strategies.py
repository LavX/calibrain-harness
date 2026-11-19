"""Strategies tab backend: registry parsing, status/trades reading on fixture files, the unit allowlist,
control.json writes and the two-way guard with the LLM auto-trade. No systemd, no network: the runner is a fake."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from ctb import strategies as st

FIX = Path(__file__).parent / "fixtures" / "live"
NOW = 1_800_000_030.0  # seconds; the fixtures' updated_ms is 1_800_000_000_000


class FakeSystemd:
    """Stands in for `systemctl --user`: records every argument list, answers `show` from a state map."""

    def __init__(self, active=()):
        self.active = set(active)
        self.calls: list[list[str]] = []

    async def __call__(self, argv, **_):
        self.calls.append(list(argv))
        assert isinstance(argv, list) and argv[:2] == ["systemctl", "--user"]
        unit = argv[-1]
        if argv[2] == "show":
            state = "active" if unit in self.active else "inactive"
            return 0, f"LoadState=loaded\nActiveState={state}\nSubState=x\n", ""
        if argv[2] == "start":
            self.active.add(unit)
        if argv[2] == "stop":
            self.active.discard(unit)
        return 0, "", ""


@pytest.fixture
def root(tmp_path):
    dst = tmp_path / "live"
    shutil.copytree(FIX, dst)
    return dst


def svc(root, systemd, account="demo-main", extra=(), mode="manual", allocator=True):
    s = st.StrategyService(root, run=systemd, clock=lambda: NOW, extra_units=extra, dashboard_account=account, allocator_available=allocator)
    s._mode = mode
    return s


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- registry
def test_registry_parses_fixture_and_collects_units(root):
    reg = st.load_registry(root)
    assert [e.id for e in reg.entries] == ["carry", "trend", "xsmom"]
    assert reg.units == ("calibrain-live@carry.service", "calibrain-live@trend.service", "calibrain-live@xsmom.service")
    assert reg.error is None


@pytest.mark.parametrize("bad", [
    [{"id": "carry", "unit": "x; rm -rf ~.service"}],
    [{"id": "carry", "unit": "../../etc/passwd"}],
    [{"id": "carry", "unit": "nosuffix"}],
    [{"id": "Carry"}],
    [{"id": "allocator"}],
    [{"id": "a"}, {"id": "a"}],
    {"id": "a"},
])
def test_registry_rejects_bad_entries(bad):
    with pytest.raises(ValueError):
        st.parse_registry(bad)


def test_missing_registry_is_empty_and_broken_one_is_an_error(tmp_path):
    assert st.load_registry(tmp_path).entries == ()
    (tmp_path / "strategies.json").write_text("{not json")
    assert st.load_registry(tmp_path).error


# ---------------------------------------------------------------- reading
def test_snapshot_reads_status_trades_and_pills(root):
    doc = run(svc(root, FakeSystemd(active=["calibrain-live@carry.service"])).snapshot())
    by = {s["id"]: s for s in doc["strategies"]}
    assert by["carry"]["pill"] == "running" and by["trend"]["pill"] == "paused" and by["xsmom"]["pill"] == "latched"
    carry = by["carry"]["status"]
    assert carry["equity_usdt"] == 10112.5 and carry["pnl_pct"] == 1.125 and len(carry["positions"]) == 2
    assert carry["positions"][0]["upnl_usdt"] == -5.0 and carry["targets"][0]["target_qty"] == 0.05
    assert [t["order_link_id"] for t in by["carry"]["trades"]][:2] == ["carry-1", "carry-2"]  # newest first
    assert by["xsmom"]["status"]["equity_usdt"] is None and by["xsmom"]["trades"] == []
    assert by["trend"]["trades"] == []  # the one broken line is skipped, not fatal
    assert doc["runners"][0]["active"] == "active"


def test_pill_stale_and_no_data(root):
    status = st.clean_status(st.read_json(root / "carry" / "status.json"))
    now_ms = status["updated_ms"]
    assert st.pill(status, now_ms + 60_000, True) == "running"
    assert st.pill(status, now_ms + 600_000, True) == "stale"
    assert st.pill(status, now_ms + 600_000, False) == "runner_stopped"
    assert st.pill(None, now_ms, True) == "no_data"


def test_snapshot_is_cached_for_15_seconds(root):
    sd = FakeSystemd()
    s = svc(root, sd)
    run(s.snapshot())
    n = len(sd.calls)
    run(s.snapshot())
    assert len(sd.calls) == n


# ---------------------------------------------------------------- control.json
def read_ctl(root, sid):
    return json.loads((root / sid / "control.json").read_text())


def test_pause_and_enable_write_control_atomically(root):
    sd = FakeSystemd()
    s = svc(root, sd)
    run(s.act("carry", "pause", llm_trading=False, venue="demo"))
    assert read_ctl(root, "carry")["enabled"] is False
    run(s.act("carry", "enable", llm_trading=False, venue="demo"))
    ctl = read_ctl(root, "carry")
    assert ctl["enabled"] is True and ctl["owner_lock"] is True and ctl["set_by"] == "owner" and ctl["reason"] == "owner"
    assert ctl["weight"] == 0.6  # the LLM's weight is kept
    assert "calibrain-live@carry.service" in sd.active  # enabling makes sure the runner runs
    assert not list((root / "carry").glob("*.tmp"))


def test_flatten_and_reset_need_the_runner_running(root):
    s = svc(root, FakeSystemd())
    for action, sid in (("flatten", "carry"), ("reset", "xsmom")):
        with pytest.raises(st.StrategyError) as e:
            run(s.act(sid, action, llm_trading=False, venue="demo"))
        assert e.value.status == 409 and "not running" in e.value.message
    assert read_ctl(root, "carry")["flatten"] is False  # nothing was written


def test_flatten_sets_flag_and_keeps_enabled(root):
    s = svc(root, FakeSystemd(active=["calibrain-live@carry.service"]))
    run(s.act("carry", "flatten", llm_trading=False, venue="demo"))
    ctl = read_ctl(root, "carry")
    assert ctl["flatten"] is True and ctl["enabled"] is True


def test_reset_only_when_latched(root):
    s = svc(root, FakeSystemd(active=["calibrain-live@carry.service", "calibrain-live@xsmom.service"]))
    with pytest.raises(st.StrategyError):
        run(s.act("carry", "reset", llm_trading=False, venue="demo"))
    run(s.act("xsmom", "reset", llm_trading=False, venue="demo"))
    assert read_ctl(root, "xsmom")["reset_latch"] is True


def test_unknown_strategy_or_action_is_404(root):
    s = svc(root, FakeSystemd())
    for sid, action in (("nope", "pause"), ("carry", "rm"), ("../carry", "pause")):
        with pytest.raises(st.StrategyError) as e:
            run(s.act(sid, action, llm_trading=False, venue="demo"))
        assert e.value.status == 404


# ---------------------------------------------------------------- unit allowlist
def test_runner_act_only_touches_listed_units_with_argument_lists(root):
    sd = FakeSystemd()
    s = svc(root, sd)
    run(s.runner_act("calibrain-live@carry.service", "start", llm_trading=False, venue="demo"))
    assert ["systemctl", "--user", "start", "--", "calibrain-live@carry.service"] in sd.calls
    n = len(sd.calls)
    for unit in ("sshd.service", "calibrain-live@carry.service; reboot", "calibrain-live@other.service", ""):
        with pytest.raises(st.StrategyError) as e:
            run(s.runner_act(unit, "stop", llm_trading=False, venue="demo"))
        assert e.value.status == 404
    with pytest.raises(st.StrategyError):
        run(s.runner_act("calibrain-live@carry.service", "restart", llm_trading=False, venue="demo"))
    assert len(sd.calls) == n  # nothing else reached systemctl


def test_env_units_are_allowed_before_the_runner_writes_strategies_json(tmp_path):
    sd = FakeSystemd()
    s = svc(tmp_path, sd, extra=("calibrain-live@carry.service",))
    run(s.runner_act("calibrain-live@carry.service", "start", llm_trading=False, venue="demo"))
    assert "calibrain-live@carry.service" in sd.active


# ---------------------------------------------------------------- owner lock
def test_owner_flip_locks_the_strategy_and_unlock_hands_it_back(root):
    s = svc(root, FakeSystemd(), mode="llm")
    run(s.act("trend", "enable", llm_trading=False, venue="demo"))  # trend had weight 0: the owner's "on" means full weight
    ctl = read_ctl(root, "trend")
    assert ctl["enabled"] is True and ctl["weight"] == 1.0 and ctl["owner_lock"] is True
    run(s.act("trend", "unlock", llm_trading=False, venue="demo"))
    ctl = read_ctl(root, "trend")
    assert ctl["owner_lock"] is False and ctl["set_by"] == "llm" and ctl["enabled"] is True


def test_control_merge_keeps_keys_the_runner_uses(root):
    p = root / "carry" / "control.json"
    doc = json.loads(p.read_text())
    doc["flatten_on_zero"] = True
    p.write_text(json.dumps(doc))
    run(svc(root, FakeSystemd()).act("carry", "pause", llm_trading=False, venue="demo"))
    assert read_ctl(root, "carry")["flatten_on_zero"] is True


def test_snapshot_shows_weight_reason_and_lock(root):
    doc = run(svc(root, FakeSystemd()).snapshot())
    by = {x["id"]: x["control"] for x in doc["strategies"]}
    assert by["carry"]["weight"] == 0.6 and by["carry"]["set_by"] == "llm" and by["carry"]["reason"].startswith("funding")
    assert by["trend"]["owner_lock"] is True and by["trend"]["set_by"] == "owner"


# ---------------------------------------------------------------- allocator panel
def test_allocator_panel_has_status_control_and_feed_without_inputs(root):
    doc = run(svc(root, FakeSystemd(), mode="llm").snapshot())
    a = doc["allocator"]
    assert a["available"] and a["status"]["model"] == "example-model" and a["status"]["cost_usd_today"] == 0.0421
    assert a["control"] == {"enabled": True, "interval_min": 60, "decide_now_pending": False}
    assert [d["reasoning"][:10] for d in a["decisions"]] == ["decision 1", "decision 2"]  # newest first
    assert all("inputs" not in d for d in a["decisions"]) and a["decisions"][0]["skipped_owner_lock"] == ["trend"]
    assert doc["auto_on"] is True  # in LLM DECIDES the header switch shows the allocator


def test_decide_now_and_interval(root):
    s = svc(root, FakeSystemd(), mode="llm")
    run(s.allocator_act("decide_now"))
    ctl = json.loads((root / "allocator" / "control.json").read_text())
    assert ctl["decide_now"] is True and ctl["enabled"] is True and ctl["interval_min"] == 60
    run(s.allocator_act("interval", 15))
    assert json.loads((root / "allocator" / "control.json").read_text())["interval_min"] == 15
    for bad in (0, 1441, "5", True, None):
        with pytest.raises(st.StrategyError):
            run(s.allocator_act("interval", bad))
    with pytest.raises(st.StrategyError):
        run(svc(root, FakeSystemd(), mode="manual").allocator_act("decide_now"))  # only in LLM DECIDES
    with pytest.raises(st.StrategyError):
        run(svc(root, FakeSystemd(), mode="llm", allocator=False).allocator_act("decide_now"))


def test_decide_now_refused_while_the_allocator_is_paused(root):
    (root / "allocator" / "control.json").write_text('{"enabled": false, "interval_min": 60}')
    with pytest.raises(st.StrategyError):
        run(svc(root, FakeSystemd(), mode="llm").allocator_act("decide_now"))


# ---------------------------------------------------------------- the mode guard (one mode trades the demo account)
def test_bot_auto_trade_refused_unless_mode_is_bot_and_runner_stopped(root):
    s = svc(root, FakeSystemd(active=["calibrain-live@trend.service"]), mode="bot")
    why = run(s.trading_block("demo"))
    assert why and "calibrain-live@trend.service" in why  # runner active
    assert run(s.trading_block("testnet")) is None  # another host, no shared account
    assert run(svc(root, FakeSystemd(), mode="bot").trading_block("demo")) is None
    for mode in ("manual", "llm"):
        why = run(svc(root, FakeSystemd(), mode=mode).trading_block("demo"))
        assert why and mode.upper() in why
    assert run(svc(root, FakeSystemd(active=["calibrain-live@carry.service"]), account="demo-other", mode="bot").trading_block("demo")) is None


def test_unknown_dashboard_account_counts_as_shared(root):
    assert run(svc(root, FakeSystemd(active=["calibrain-live@carry.service"]), account=None, mode="bot").trading_block("demo"))


def test_unusable_registry_blocks_bot_trading_on_demo(tmp_path):
    (tmp_path / "strategies.json").write_text("[1]")
    assert run(svc(tmp_path, FakeSystemd(), mode="bot").trading_block("demo"))


def test_mode_defaults_after_restart(root, tmp_path):
    def fresh(r):
        return st.StrategyService(r, run=FakeSystemd(), clock=lambda: NOW, extra_units=(), dashboard_account="demo-main")
    s = fresh(root)
    assert s.mode(s.registry()) == "llm"  # the allocator is switched on in the fixture
    (root / "allocator" / "control.json").write_text('{"enabled": false}')
    s = fresh(root)
    assert s.mode(s.registry()) == "manual"
    s = fresh(tmp_path)
    assert s.mode(s.registry()) == "bot"  # no strategies configured: the dashboard behaves as before


def test_cannot_start_runner_or_enable_strategy_in_bot_mode_or_while_bot_trading(root):
    sd = FakeSystemd()
    s = svc(root, sd, mode="bot")
    with pytest.raises(st.StrategyError) as e:
        run(s.runner_act("calibrain-live@carry.service", "start", llm_trading=False, venue="demo"))
    assert e.value.status == 409 and "PER-SYMBOL BOT" in e.value.message and not sd.active
    with pytest.raises(st.StrategyError):
        run(s.act("carry", "enable", llm_trading=False, venue="demo"))
    s2 = svc(root, sd, mode="manual")
    with pytest.raises(st.StrategyError) as e:
        run(s2.runner_act("calibrain-live@carry.service", "start", llm_trading=True, venue="demo"))
    assert "AUTO-TRADE" in e.value.message
    with pytest.raises(st.StrategyError):
        run(s2.act("carry", "enable", llm_trading=True, venue="demo"))
    run(s.runner_act("calibrain-live@carry.service", "start", llm_trading=True, venue="testnet"))  # no shared account on testnet
    assert "calibrain-live@carry.service" in sd.active


def test_switch_away_from_bot_needs_flat_bot_and_turns_it_off(root):
    s = svc(root, FakeSystemd(), mode="bot")
    with pytest.raises(st.StrategyError) as e:
        run(s.set_mode("llm", venue="demo", bot_open=True))
    assert e.value.status == 409 and s._mode == "bot"
    assert run(s.set_mode("manual", venue="demo", bot_open=False)) == {"mode": "manual", "changed": True, "llm_off": True}


def test_switch_to_bot_pauses_everything_and_stops_the_runner(root):
    sd = FakeSystemd(active=["calibrain-live@carry.service", "calibrain-live@trend.service"])
    s = svc(root, sd, mode="llm")
    with pytest.raises(st.StrategyError) as e:  # carry holds positions in the fixture
        run(s.set_mode("bot", venue="demo", bot_open=False))
    assert e.value.status == 409 and "FLATTEN" in e.value.message and s._mode == "llm"
    p = root / "carry" / "status.json"
    doc = json.loads(p.read_text())
    doc["positions"] = []
    p.write_text(json.dumps(doc))
    run(s.set_mode("bot", venue="demo", bot_open=False))
    assert s._mode == "bot" and not sd.active
    assert all(read_ctl(root, i)["enabled"] is False for i in ("carry", "trend", "xsmom"))
    assert json.loads((root / "allocator" / "control.json").read_text())["enabled"] is False
    assert run(s.trading_block("demo")) is None


def test_llm_to_manual_pauses_the_allocator_only(root):
    sd = FakeSystemd(active=["calibrain-live@carry.service"])
    s = svc(root, sd, mode="llm")
    run(s.set_mode("manual", venue="demo", bot_open=False))
    assert json.loads((root / "allocator" / "control.json").read_text())["enabled"] is False
    assert read_ctl(root, "carry")["enabled"] is True and sd.active  # strategies and runner untouched


def test_mode_change_on_testnet_has_no_side_effects(root):
    sd = FakeSystemd(active=["calibrain-live@carry.service"])
    s = svc(root, sd, mode="manual")
    run(s.set_mode("bot", venue="testnet", bot_open=True))
    assert s._mode == "bot" and sd.active and read_ctl(root, "carry")["enabled"] is True


# ---------------------------------------------------------------- the header switch
def test_header_in_manual_mode_turns_selected_on_owner_locked_and_starts_runner(root):
    sd = FakeSystemd()
    s = svc(root, sd, mode="manual")
    out = run(s.auto(True, ["trend"], llm_trading=False, venue="demo"))
    assert out["enabled"] == ["trend"]
    assert read_ctl(root, "trend")["enabled"] is True and read_ctl(root, "trend")["owner_lock"] is True
    assert "calibrain-live@trend.service" in sd.active and "calibrain-live@xsmom.service" not in sd.active
    run(s.auto(False, None, llm_trading=False, venue="demo"))
    assert all(read_ctl(root, i)["enabled"] is False for i in ("carry", "trend", "xsmom"))


def test_header_in_llm_mode_switches_the_allocator_and_starts_every_runner(root):
    sd = FakeSystemd()
    (root / "allocator" / "control.json").write_text('{"enabled": false, "interval_min": 30}')
    s = svc(root, sd, mode="llm")
    run(s.auto(True, None, llm_trading=False, venue="demo"))
    ctl = json.loads((root / "allocator" / "control.json").read_text())
    assert ctl["enabled"] is True and ctl["interval_min"] == 30
    assert len(sd.active) == 3
    assert read_ctl(root, "trend")["owner_lock"] is True  # the LLM does not touch what the owner set
    run(s.auto(False, None, llm_trading=False, venue="demo"))
    assert json.loads((root / "allocator" / "control.json").read_text())["enabled"] is False
    assert all(read_ctl(root, i)["enabled"] is False for i in ("carry", "trend", "xsmom"))


def test_header_refused_in_bot_mode_without_allocator_or_with_bot_trading_on(root):
    for call, mode, kw in ((lambda s: s.auto(True, None, llm_trading=False, venue="demo"), "bot", {}),
                           (lambda s: s.auto(True, None, llm_trading=False, venue="demo"), "llm", {"allocator": False}),
                           (lambda s: s.auto(True, None, llm_trading=True, venue="demo"), "manual", {})):
        with pytest.raises(st.StrategyError) as e:
            run(call(svc(root, FakeSystemd(), mode=mode, **kw)))
        assert e.value.status == 409


# ---------------------------------------------------------------- the routes in the dashboard server
def test_routes_and_trading_toggle_guard(root):
    from types import SimpleNamespace

    from aiohttp.test_utils import TestClient, TestServer

    from ctb.server import make_app

    class Engine:
        s = SimpleNamespace(trading_enabled=False, public_view=lambda: {})
        db = SimpleNamespace(event=lambda *a: a, one=lambda *a: None)

        async def start(self): ...
        async def stop(self): ...
        def emit(self, *a): ...

    eng = Engine()
    eng.executor = SimpleNamespace(halted_reason=None)
    sd = FakeSystemd(active=["calibrain-live@carry.service"])

    async def go():
        import os
        os.environ["CTB_VENUE"] = "demo"
        async with TestClient(TestServer(make_app(eng, svc(root, sd, mode="bot")))) as c:
            doc = await (await c.get("/api/strategies")).json()
            assert doc["venue"] == "demo" and doc["mode"] == "bot" and len(doc["strategies"]) == 3
            r = await c.post("/api/trading", json={"enabled": True})  # the runner is active
            assert r.status == 409 and "calibrain-live@carry.service" in (await r.json())["error"]
            assert (await c.post("/api/runner/stop", json={"unit": "calibrain-live@carry.service"})).status == 200
            r = await c.post("/api/trading", json={"enabled": True})
            assert r.status == 200 and eng.s.trading_enabled is True
            assert (await c.post("/api/runner/start", json={"unit": "calibrain-live@carry.service"})).status == 409
            assert (await c.post("/api/runner/start", json={"unit": "evil.service"})).status == 404
            assert (await c.post("/api/strategies/carry/pause", data="x")).status == 415  # not JSON: refused by the middleware
            # switching away from the bot turns its auto-trade off
            r = await c.post("/api/mode", json={"mode": "llm"})
            assert r.status == 200 and (await r.json())["llm_off"] and eng.s.trading_enabled is False
            assert (await c.post("/api/mode", json={"mode": "nonsense"})).status == 400
            r = await c.post("/api/strategies/auto", json={"enabled": True})
            assert r.status == 200 and "calibrain-live@carry.service" in sd.active
            assert (await c.post("/api/allocator/decide_now", json={})).status == 200
            assert (await c.post("/api/allocator/interval", json={"interval_min": 5})).status == 200
            assert (await c.post("/api/strategies/trend/unlock", json={})).status == 200
            r = await c.post("/api/trading", json={"enabled": True})  # bot trading refused again in LLM DECIDES
            assert r.status == 409

    run(go())


# ---------------------------------------------------------------- the real allocator on the shared root
class FakeModel:
    async def ask_json(self, messages, schema):
        ids = list(schema["properties"]["weights"]["properties"])
        return ({"weights": {i: (0.8 if i == "carry" else 0.0) for i in ids}, "enabled": {i: i == "carry" for i in ids},
                 "reasoning": "fake: carry only", "confidence": 0.5}, {"cost": 0.01})


def make_alloc(root):
    from ctb import allocator as alloc_mod

    async def market():
        return {}

    clock = lambda: NOW  # noqa: E731
    return alloc_mod.Allocator(root, FakeModel(), market, model="fake-model", clock=clock, tick_s=0.01)


def alloc_ctl(root):
    return json.loads((root / "allocator" / "control.json").read_text())


def test_attach_writes_the_allocator_control_from_the_mode(tmp_path, root):
    fresh = tmp_path / "fresh"
    (fresh / "carry").mkdir(parents=True)
    (fresh / "strategies.json").write_text('[{"id": "carry", "unit": "calibrain-live@carry.service", "account": "demo-main"}]')
    s = st.StrategyService(fresh, run=FakeSystemd(), clock=lambda: NOW, extra_units=(), dashboard_account="demo-main")
    s.attach_allocator(make_alloc(fresh))  # no allocator dir yet: MANUAL, so the loop must start paused, not on its default
    assert alloc_ctl(fresh)["enabled"] is False and s.allocator_available
    s2 = svc(root, FakeSystemd(), mode="llm", allocator=False)
    s2._mode = None  # the fixture has the allocator switched on: derived mode is LLM DECIDES
    s2.attach_allocator(make_alloc(root))
    assert alloc_ctl(root)["enabled"] is True


def test_mode_switches_always_write_the_allocator_control(tmp_path):
    fresh = tmp_path / "fresh"
    (fresh / "carry").mkdir(parents=True)
    (fresh / "strategies.json").write_text('[{"id": "carry", "unit": "calibrain-live@carry.service", "account": "demo-main"}]')
    s = svc(fresh, FakeSystemd(), mode="llm")
    run(s.set_mode("manual", venue="demo", bot_open=False))
    assert alloc_ctl(fresh)["enabled"] is False


def test_allocator_loop_writes_strategy_controls_only_in_llm_mode(root):
    s = svc(root, FakeSystemd(), mode="llm")
    alloc = make_alloc(root)
    s.attach_allocator(alloc)
    before = read_ctl(root, "carry")
    run(s.set_mode("manual", venue="demo", bot_open=False))  # allocator paused
    run(alloc._tick())
    assert read_ctl(root, "carry") == before  # untouched while paused
    run(s.set_mode("llm", venue="demo", bot_open=False))
    run(s.auto(True, None, llm_trading=False, venue="demo"))  # allocator on, runner started
    run(s.allocator_act("decide_now"))
    assert alloc_ctl(root)["decide_now"] is True
    run(alloc._tick())  # the loop: decide_now handled
    assert alloc_ctl(root)["decide_now"] is False
    ctl = read_ctl(root, "carry")
    assert ctl["weight"] == 0.8 and ctl["set_by"] == "llm" and ctl["reason"] == "fake: carry only"
    assert read_ctl(root, "trend")["owner_lock"] is True and read_ctl(root, "trend")["weight"] == 0.0  # the owner's choice stands
    doc = run(s.snapshot(force=True))
    first = doc["allocator"]["decisions"][0]
    assert first["trigger"] == "decide_now" and first["reasoning"] == "fake: carry only" and "inputs" not in first
    assert doc["allocator"]["status"]["model"] == "fake-model" and doc["allocator"]["status"]["state"] == "running"
