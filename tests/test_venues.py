"""Venues: the registry from CTB_LIVE_ROOTS (never a glob), presence per venue, KILL refused unless the runner is live,
the old routes as bybit-demo aliases, the persisted 00:00 UTC baseline, IDLE and the one short-name map. No systemd, no
network: the runner is a fake and the roots are fixture copies."""

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ctb import strategies as st
from ctb import venues as V

FIX = Path(__file__).parent / "fixtures" / "live"
NOW = 1_800_000_030.0  # the fixtures' updated_ms is 1_800_000_000_000 (30 s earlier)
BYBIT_UNITS = ["calibrain-live@carry.service", "calibrain-live@trend.service", "calibrain-live@xsmom.service"]
OKX_UNIT = "calibrain-live-okx.service"


class FakeSystemd:
    def __init__(self, active=()):
        self.active = set(active)
        self.calls: list[list[str]] = []

    async def __call__(self, argv, **_):
        self.calls.append(list(argv))
        unit = argv[-1]
        if argv[2] == "show":
            return 0, f"LoadState=loaded\nActiveState={'active' if unit in self.active else 'inactive'}\nSubState=x\n", ""
        if argv[2] == "start":
            self.active.add(unit)
        if argv[2] == "stop":
            self.active.discard(unit)
        return 0, "", ""


def run(coro):
    return asyncio.run(coro)


def okx_root(tmp_path: Path) -> Path:
    """An OKX-like root: trend and longvol, one unit for both, fresh status files."""
    root = tmp_path / "live-okx"
    for sid, positions in (("trend", [{"symbol": "BTCUSDT", "side": "short", "qty": 0.01, "entry_price": 61729.1, "mark_price": 61664.6,
                                       "upnl_usdt": 0.67, "market": "linear"}]), ("longvol", [])):
        (root / sid).mkdir(parents=True)
        (root / sid / "status.json").write_text(json.dumps({
            "id": sid, "enabled": True, "state": "running", "updated_ms": 1_800_000_000_000, "equity_usdt": 10004.0 if sid == "trend" else 10000.0,
            "start_equity_usdt": 10000.0, "pnl_usdt": 4.0 if sid == "trend" else 0.0, "positions": positions, "targets": [],
            "weight": 1.0, "set_by": "owner", "owner_lock": True, "reason": "OKX demo plumbing run"}))
        (root / sid / "control.json").write_text(json.dumps({"enabled": True, "weight": 1.0, "set_by": "owner", "owner_lock": True}))
    (root / "strategies.json").write_text(json.dumps([{"id": i, "label": i, "unit": OKX_UNIT, "account": "okx-demo"} for i in ("trend", "longvol")]))
    return root


@pytest.fixture
def roots(tmp_path):
    bybit = tmp_path / "live"
    shutil.copytree(FIX, bybit)
    return bybit, okx_root(tmp_path), tmp_path / "live-deribit"


def hub(roots, systemd, tmp_path, *, now=NOW):
    bybit, okx, deribit = roots
    venues = [V.venue_config("bybit-demo", bybit, unit=""), V.venue_config("okx-demo", okx), V.venue_config("deribit-test", deribit)]
    services = {
        "bybit-demo": st.StrategyService(bybit, run=systemd, clock=lambda: now, extra_units=(), dashboard_account="demo-main",
                                         allocator_available=True, auto_set_path=tmp_path / "state" / "auto-set-bybit-demo.json"),
        "okx-demo": st.StrategyService(okx, run=systemd, clock=lambda: now, extra_units=(OKX_UNIT,), dashboard_account="",
                                       shares_bot_account=False, has_allocator=False),
        "deribit-test": st.StrategyService(deribit, run=systemd, clock=lambda: now, extra_units=("calibrain-live-deribit.service",),
                                           dashboard_account="", shares_bot_account=False, has_allocator=False),
    }
    services["bybit-demo"]._mode = "manual"
    return V.VenueHub(venues, services, state=tmp_path / "state", clock=lambda: now)


def ctl(root, sid):
    return json.loads((root / sid / "control.json").read_text())


# ---------------------------------------------------------------- the registry
def test_unset_means_one_bybit_venue_at_the_old_root(tmp_path):
    vs = V.parse_live_roots(None, tmp_path / "live")
    assert [(v.id, v.label, v.root, v.controlled_by) for v in vs] == [("bybit-demo", "BYBIT·DEMO", tmp_path / "live", "allocator")]
    assert not vs[0].unit  # the unit comes from strategies.json, as before


def test_known_venues_get_label_unit_currency_and_controller(tmp_path):
    vs = V.parse_live_roots(f"bybit-demo={tmp_path}/live, okx-demo={tmp_path}/live-okx,deribit-test={tmp_path}/d,bitget-demo={tmp_path}/b", tmp_path)
    got = {v.id: (v.label, v.unit, v.ccy, v.controlled_by) for v in vs}
    assert got == {"bybit-demo": ("BYBIT·DEMO", "calibrain-live.service", "USDT", "allocator"),
                   "okx-demo": ("OKX·DEMO", "calibrain-live-okx.service", "USDT", "owner"),
                   "deribit-test": ("DERIBIT·TEST", "calibrain-live-deribit.service", "USDC", "owner"),
                   "bitget-demo": ("BITGET·DEMO", "calibrain-live-bitget.service", "USDT", "owner")}


@pytest.mark.parametrize("bad", ["okx-demo=/x", "bybit-demo=/a,bybit-demo=/b", "bybit-demo", "Bybit Demo=/x", "bybit-demo=/a,../x=/b"])
def test_malformed_roots_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        V.parse_live_roots(bad, tmp_path)


def test_venues_come_from_config_never_a_glob(tmp_path, monkeypatch, roots):
    bybit, okx, _ = roots
    shutil.copytree(okx, tmp_path / "live-okx-rejected-20261120")  # a quarantined root next to the real one
    monkeypatch.setenv("CTB_LIVE_ROOTS", f"bybit-demo={bybit},okx-demo={okx}")
    h = V.VenueHub.from_env()
    assert [v.id for v in h.venues] == ["bybit-demo", "okx-demo"] and h.primary.root == bybit
    assert not h.services["okx-demo"].shares_bot_account and h.services["bybit-demo"].shares_bot_account


# ---------------------------------------------------------------- presence
def test_presence_live_stopped_absent(roots, tmp_path):
    sd = FakeSystemd(active=BYBIT_UNITS)
    doc = run(hub(roots, sd, tmp_path).summary())
    by = {v["id"]: v for v in doc["venues"]}
    assert by["bybit-demo"]["presence"] == "live"
    assert by["okx-demo"]["presence"] == "stopped" and OKX_UNIT in by["okx-demo"]["presence_text"]
    assert by["deribit-test"]["presence"] == "absent" and "not started" in by["deribit-test"]["presence_text"]
    assert by["deribit-test"]["strategies"] == [] and by["deribit-test"]["kill"]["ok"] is False
    assert any(a["kind"] == "unit_down" and a["venue"] == "okx-demo" for a in doc["alerts"])
    assert not any(a["venue"] == "deribit-test" for a in doc["alerts"])  # not started is not an alarm


def test_presence_stale_when_every_status_is_old(roots, tmp_path):
    sd = FakeSystemd(active=[OKX_UNIT])
    doc = run(hub(roots, sd, tmp_path, now=NOW + 600).summary())
    okx = next(v for v in doc["venues"] if v["id"] == "okx-demo")
    assert okx["presence"] == "stale" and "older than 3 min" in okx["presence_text"]


def test_presence_error_fails_closed(roots, tmp_path):
    (roots[1] / "strategies.json").write_text("{broken")
    doc = run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).summary())
    okx = next(v for v in doc["venues"] if v["id"] == "okx-demo")
    assert okx["presence"] == "error" and okx["kill"]["ok"] is False


# ---------------------------------------------------------------- kill
def alloc_ctl(root):
    return json.loads((root / "allocator" / "control.json").read_text())


def test_kill_with_the_runner_down_still_writes_and_says_not_confirmed(roots, tmp_path):
    h = hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path)
    with pytest.raises(st.StrategyError) as e:
        run(h.kill("okx-demo"))
    assert e.value.status == 409 and "NOT confirmed" in e.value.message and OKX_UNIT in e.value.message
    assert sorted(e.value.detail["written"]) == ["longvol", "trend"] and e.value.detail["confirmed"] == []
    for sid in ("trend", "longvol"):  # the owner's intent is on disk for when the runner comes back
        c = ctl(roots[1], sid)
        assert c["flatten"] is True and c["enabled"] is False and c["owner_lock"] is True
    assert not (roots[1] / "allocator").exists()


def test_kill_with_stale_status_writes_and_says_how_old(roots, tmp_path):
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path, now=NOW + 600)
    with pytest.raises(st.StrategyError) as e:
        run(h.kill("okx-demo"))
    assert e.value.status == 409 and "min old" in e.value.message and ctl(roots[1], "trend")["flatten"] is True


def test_refused_bybit_kill_still_pauses_the_allocator(roots, tmp_path):
    assert alloc_ctl(roots[0])["enabled"] is True
    with pytest.raises(st.StrategyError) as e:
        run(hub(roots, FakeSystemd(), tmp_path).kill("bybit-demo"))  # every runner unit down
    assert e.value.status == 409 and e.value.detail["allocator_paused"] is True
    assert alloc_ctl(roots[0])["enabled"] is False and ctl(roots[0], "carry")["flatten"] is True


def test_kill_liveness_is_per_strategy_when_units_are(roots, tmp_path):
    h = hub(roots, FakeSystemd(active=["calibrain-live@carry.service"]), tmp_path)  # one unit per strategy in the fixture
    with pytest.raises(st.StrategyError) as e:
        run(h.kill("bybit-demo"))
    d = e.value.detail
    assert d["confirmed"] == ["carry"] and sorted(d["unconfirmed"]) == ["trend", "xsmom"]
    assert "calibrain-live@trend.service is inactive" in d["unconfirmed"]["trend"] and "Confirmed: carry" in e.value.message


def test_kill_on_absent_or_broken_roots_writes_nothing(roots, tmp_path):
    bybit, okx, deribit = roots
    h = hub(roots, FakeSystemd(active=BYBIT_UNITS + [OKX_UNIT]), tmp_path)
    with pytest.raises(st.StrategyError) as e:
        run(h.kill("deribit-test"))
    assert e.value.status == 409 and "not started" in e.value.message and not deribit.exists() and e.value.detail == {}
    before = {sid: (okx / sid / "control.json").read_bytes() for sid in ("trend", "longvol")}
    (okx / "strategies.json").write_text("{broken")
    with pytest.raises(st.StrategyError) as e:
        run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).kill("okx-demo"))
    assert e.value.status == 409 and "Nothing was written" in e.value.message
    assert {sid: (okx / sid / "control.json").read_bytes() for sid in ("trend", "longvol")} == before


def test_kill_live_venue_flattens_its_strategies_only(roots, tmp_path):
    bybit, okx, _ = roots
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path)
    out = run(h.kill("okx-demo"))
    assert out["ok"] and sorted(out["flattening"]) == ["longvol", "trend"] and sorted(out["confirmed"]) == ["longvol", "trend"]
    for sid in ("trend", "longvol"):
        c = ctl(okx, sid)
        assert c["flatten"] is True and c["enabled"] is False and c["owner_lock"] is True
    assert not (okx / "allocator").exists()  # no allocator on an owner venue
    assert ctl(bybit, "carry")["flatten"] is False and alloc_ctl(bybit)["enabled"] is True  # another venue is untouched


def test_partial_kill_write_reports_what_was_written(roots, tmp_path):
    okx = roots[1]
    (okx / "trend").chmod(0o500)  # control.json there cannot be replaced
    try:
        with pytest.raises(st.StrategyError) as e:
            run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).kill("okx-demo"))
    finally:
        (okx / "trend").chmod(0o700)
    assert e.value.status == 500 and e.value.detail["written"] == ["longvol"] and "trend" in e.value.detail["failed"]
    assert "partially written" in e.value.message and ctl(okx, "longvol")["flatten"] is True


def test_kill_all_reports_per_venue_and_skips_not_started(roots, tmp_path):
    out = run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).kill_all())
    by = {r["venue"]: r for r in out["venues"]}
    assert by["bybit-demo"]["ok"] and by["okx-demo"]["ok"] is False and "NOT confirmed" in by["okx-demo"]["error"]
    assert sorted(by["okx-demo"]["written"]) == ["longvol", "trend"]  # written although not confirmed
    assert by["deribit-test"]["skipped"] is True
    assert out["ok"] is False


def test_enable_after_a_kill_clears_the_flatten_flag(roots, tmp_path):
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path)
    run(h.kill("okx-demo"))
    run(h.service("okx-demo").act("trend", "enable", llm_trading=False, venue="demo"))
    c = ctl(roots[1], "trend")
    assert c["enabled"] is True and c["flatten"] is False and c["owner_lock"] is True


def test_unlock_is_refused_on_an_owner_venue(roots, tmp_path):
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path)
    with pytest.raises(st.StrategyError) as e:
        run(h.service("okx-demo").act("trend", "unlock", llm_trading=False, venue="demo"))
    assert e.value.status == 409 and ctl(roots[1], "trend")["set_by"] == "owner"


def test_unknown_venue_is_404(roots, tmp_path):
    with pytest.raises(st.StrategyError) as e:
        run(hub(roots, FakeSystemd(), tmp_path).kill("ftx"))
    assert e.value.status == 404


# ---------------------------------------------------------------- baseline and P&L today
def test_baseline_is_persisted_and_survives_a_restart(roots, tmp_path):
    sd = FakeSystemd(active=BYBIT_UNITS)
    run(hub(roots, sd, tmp_path).summary())
    p = tmp_path / "state" / "baseline-bybit-demo.json"
    first = json.loads(p.read_text())
    assert first["day"] == V.utc_day(NOW) and first["strategies"]["carry"]["equity"] == 10112.5
    status = roots[0] / "carry" / "status.json"
    doc = json.loads(status.read_text())
    doc["equity_usdt"] = 10120.0
    status.write_text(json.dumps(doc))
    later = run(hub(roots, sd, tmp_path, now=NOW + 120).summary())  # a new hub: the restart
    carry = next(s for s in later["venues"][0]["strategies"] if s["id"] == "carry")
    assert carry["pnl_today"] == pytest.approx(7.5)
    assert json.loads(p.read_text()) == first  # the day's baseline is not overwritten


def test_baseline_resets_on_the_next_utc_day(roots, tmp_path):
    sd = FakeSystemd(active=BYBIT_UNITS)
    run(hub(roots, sd, tmp_path).summary())
    nxt = NOW + 86400
    run(hub(roots, sd, tmp_path, now=nxt).summary())
    assert json.loads((tmp_path / "state" / "baseline-bybit-demo.json").read_text())["day"] == V.utc_day(nxt)


def test_since_label_says_when_the_baseline_was_taken():
    midnight = 1_800_057_600  # a UTC midnight
    assert V.since_label(midnight * 1000 + 60_000, midnight + 3600) == "since 00:00 UTC"
    assert V.since_label(midnight * 1000 + 3 * 3600_000, midnight + 4 * 3600).startswith("since 03:00 UTC")


ACCOUNT = {"wallet": {"equity": 163245.78, "usdt_equity": 49911.05, "available": 99007.9, "unrealized_pnl": 24.4},
           "positions": [{"symbol": "BTCUSDT", "side": "short", "size": 0.05, "entry": 90000.0, "mark": 90100.0, "upnl": -5.0}]}


def test_bybit_account_block_compares_like_with_like(roots, tmp_path):
    doc = run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).summary(ACCOUNT))
    b = doc["venues"][0]
    a = b["account"]
    assert a["equity"] == 49911.05 and a["collateral_usd"] == 163245.78
    assert a["pnl_today"] is None  # the USDT coin's equity moves on spot buys: no account-level P&L today
    # carry holds a linear leg (uPnL -5.0) and a spot leg (+4.5): only the linear one is in the exchange's unrealised
    assert a["books_unrealised"] == pytest.approx(-5.0) and a["unattributed"] == pytest.approx(24.4 + 5.0)
    assert b["books"]["unrealised"] == pytest.approx(-0.5)  # the books' own figure keeps both legs
    assert b["positions"][0]["held_by"] == ["Carry"]  # the exchange's netted row, with who holds a leg
    okx = doc["venues"][1]
    assert okx["account"] is None and okx["books"]["equity"] == 20004.0  # other venues: books only


def test_a_spot_purchase_does_not_move_any_pnl_today(roots, tmp_path):
    sd = FakeSystemd(active=BYBIT_UNITS)
    run(hub(roots, sd, tmp_path).summary(ACCOUNT))
    bought = {**ACCOUNT, "wallet": {**ACCOUNT["wallet"], "usdt_equity": 49911.05 - 4500.0}}  # 0.05 BTC bought with USDT
    v = run(hub(roots, sd, tmp_path, now=NOW + 60).summary(bought))["venues"][0]
    assert v["account"]["pnl_today"] is None and v["pnl_today"] == pytest.approx(0.0)


def test_unknown_values_stay_unknown(roots, tmp_path):
    okx = roots[1]
    p = okx / "trend" / "status.json"
    doc = json.loads(p.read_text())
    doc["positions"][0]["upnl_usdt"] = None
    doc.pop("pnl_usdt")
    p.write_text(json.dumps(doc))
    v = run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).summary())["venues"][1]
    assert v["books"]["unrealised"] is None  # one unvalued leg: unknown, not zero
    assert v["books"]["pnl"] == 0.0 and v["books"]["incomplete"] == ["Trend"]  # longvol's 0.0 is known; trend's is not
    for sid in ("trend", "longvol"):
        q = okx / sid / "status.json"
        d = json.loads(q.read_text())
        d.pop("pnl_usdt", None)
        q.write_text(json.dumps(d))
    v = run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).summary())["venues"][1]
    assert v["books"]["pnl"] is None  # nothing known: unknown, never "+0.00"


# ---------------------------------------------------------------- strategies: names, IDLE, applied values, netting
def test_short_names_come_from_one_map():
    assert [st.short_name(i) for i in ("carry", "dated_carry", "trend", "xsmom", "basis", "longvol", "events")] == \
        ["Carry", "Dated carry", "Trend", "Momentum", "Basis", "Long-only", "Events"]
    assert st.short_name("newone", "Pairs (BTC, ETH)") == "Pairs"


def test_idle_is_distinct_from_running_and_applied_values_are_read(roots, tmp_path):
    doc = run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).venue_doc("okx-demo"))
    by = {s["id"]: s for s in doc["strategies"]}
    assert by["longvol"]["pill"] == "idle" and by["trend"]["pill"] == "running"
    assert by["trend"]["status"]["weight"] == 1.0 and by["trend"]["status"]["owner_lock"] is True
    assert by["trend"]["name"] == "Trend" and doc["venue_label"] == "OKX·DEMO" and doc["allocator_venue"] is False


def test_net_positions_sum_legs_per_symbol_and_market():
    legs = [{"name": "Trend", "status": {"positions": [{"symbol": "BTCUSDT", "market": "linear", "side": "short", "qty": 2, "entry_price": 100, "upnl_usdt": 1}]}},
            {"name": "Momentum", "status": {"positions": [{"symbol": "BTCUSDT", "market": "linear", "side": "long", "qty": 0.5, "entry_price": 90, "upnl_usdt": -1}]}},
            {"name": "Carry", "status": {"positions": [{"symbol": "BTCUSDT", "market": "spot", "side": "long", "qty": 1, "entry_price": 95, "upnl_usdt": 2}]}}]
    rows = {(r["symbol"], r["market"]): r for r in V.net_positions(legs)}
    lin = rows[("BTCUSDT", "linear")]
    assert lin["side"] == "short" and lin["size"] == 1.5 and lin["upnl"] == 0 and lin["entry"] is None  # legs on both sides
    assert lin["held_by"] == ["Trend", "Momentum"] and lin["legs"] == 2
    assert rows[("BTCUSDT", "spot")]["entry"] == 95


def test_header_set_is_kept_by_the_harness(roots, tmp_path):
    sd = FakeSystemd()
    h = hub(roots, sd, tmp_path)
    svc = h.primary
    with pytest.raises(st.StrategyError):
        svc.set_auto_set(["carry", "nope"])
    svc.set_auto_set(["trend"])
    assert svc.auto_set() == ["trend"] and run(svc.snapshot(force=True))["auto_set"] == ["trend"]
    run(svc.auto(True, None, llm_trading=False, venue="demo"))  # no ids from the page: the saved set
    assert ctl(roots[0], "trend")["enabled"] is True and "calibrain-live@carry.service" not in sd.active


# ---------------------------------------------------------------- routes
def test_routes_aliases_and_kill_refusal(roots, tmp_path, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    from ctb.server import make_app

    class Engine:
        s = SimpleNamespace(trading_enabled=False, public_view=lambda: {})
        db = SimpleNamespace(event=lambda *a: a, one=lambda *a: None)
        executor = None
        events: list = []

        async def start(self): ...
        async def stop(self): ...
        def emit(self, *a): self.events.append(a)

    monkeypatch.setenv("CTB_VENUE", "demo")
    sd = FakeSystemd(active=[OKX_UNIT])  # bybit's runner is down, OKX's is up
    h = hub(roots, sd, tmp_path)
    app = make_app(Engine(), hub=h)
    app.on_startup.clear()
    app.on_cleanup.clear()

    async def go():
        async with TestClient(TestServer(app)) as c:
            old = await (await c.get("/api/strategies")).json()
            new = await (await c.get("/api/venues/bybit-demo/strategies")).json()
            assert [s["id"] for s in old["strategies"]] == [s["id"] for s in new["strategies"]] == ["carry", "trend", "xsmom"]
            assert old["venue_id"] == "bybit-demo" and old["venue"] == "demo"
            okx = await (await c.get("/api/venues/okx-demo/strategies")).json()
            assert okx["venue_label"] == "OKX·DEMO" and [s["name"] for s in okx["strategies"]] == ["Trend", "Long-only"]
            assert (await c.get("/api/venues/nope/strategies")).status == 404
            summary = await (await c.get("/api/venues")).json()
            assert [v["presence"] for v in summary["venues"]] == ["stopped", "live", "absent"]
            # per-venue action on OKX: no bot guard there
            assert (await c.post("/api/venues/okx-demo/strategies/longvol/disable", json={})).status == 200
            assert ctl(roots[1], "longvol")["enabled"] is False
            assert (await c.post("/api/strategies/carry/pause", json={})).status == 200  # the old route: bybit-demo
            assert ctl(roots[0], "carry")["enabled"] is False
            # kill: bybit's runner is down, so its flags are written but not confirmed (409 with the detail); OKX confirms
            r = await c.post("/api/venues/bybit-demo/kill", json={})
            body = await r.json()
            assert r.status == 409 and "NOT confirmed" in body["error"] and sorted(body["written"]) == ["carry", "trend", "xsmom"]
            assert ctl(roots[0], "trend")["flatten"] is True and alloc_ctl(roots[0])["enabled"] is False
            r = await c.post("/api/kill", json={})  # the old alias answers the same way and says the bot's auto-trade is off
            body = await r.json()
            assert r.status == 409 and body["bot_auto_trade"] == "off"
            r = await c.post("/api/kill-all", json={})
            body = await r.json()
            assert r.status == 200 and body["ok"] is False and body["bot_auto_trade"] == "off"
            assert {x["venue"]: x["ok"] for x in body["venues"]} == {"bybit-demo": False, "okx-demo": True, "deribit-test": False}
            assert ctl(roots[1], "trend")["flatten"] is True
            assert (await c.post("/api/venues/okx-demo/strategies/trend/unlock", json={})).status == 409
            r = await c.post("/api/strategies/auto-set", json={"ids": ["carry"]})
            assert r.status == 200 and (await r.json())["ids"] == ["carry"]

    run(go())
