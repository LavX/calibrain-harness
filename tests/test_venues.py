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
    # the shared fixture's xsmom is latched; here it runs, so a kill can be confirmed (the latched case has its own test)
    xs = json.loads((bybit / "xsmom" / "status.json").read_text())
    (bybit / "xsmom" / "status.json").write_text(json.dumps({**xs, "state": "running", "last_error": None}))
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


@pytest.mark.parametrize("bad", ["okx-demo=/x", "bybit-demo=/a,bybit-demo=/b", "bybit-demo", "Bybit Demo=/x", "bybit-demo=/a,../x=/b",
                                 "bybit-demo=/a,okx-demo=/a/", "bybit-demo=/a,okx-demo=/a/../a"])
def test_malformed_roots_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        V.parse_live_roots(bad, tmp_path)


def files_under(*dirs):
    return {f: f.read_bytes() for d in dirs for f in sorted(Path(d).rglob("*")) if f.is_file()}


@pytest.mark.parametrize("configured", ["unset", "listed"])
def test_a_quarantined_sibling_root_is_never_read_or_written(tmp_path, monkeypatch, roots, configured):
    """data/live-okx-rejected-<date> sits next to data/live and data/live-okx: a glob over data/live* would pick it up."""
    bybit, okx, _ = roots
    rejected = tmp_path / "live-okx-rejected-20261120"
    shutil.copytree(okx, rejected)
    if configured == "unset":
        monkeypatch.setenv("CTB_LIVE_ROOT", str(bybit))
        want = ["bybit-demo"]
    else:
        monkeypatch.setenv("CTB_LIVE_ROOTS", f"bybit-demo={bybit},okx-demo={okx}")
        want = ["bybit-demo", "okx-demo"]
    h = V.VenueHub.from_env()
    for svc in h.services.values():
        svc.run = FakeSystemd(active=BYBIT_UNITS + [OKX_UNIT])  # never the real systemctl
        svc.clock = lambda: NOW
    h.clock = lambda: NOW
    before = files_under(rejected)
    doc = run(h.summary())
    run(h.kill_all())
    assert [v["id"] for v in doc["venues"]] == want and not any(str(rejected) in v["root"] for v in doc["venues"])
    assert files_under(rejected) == before  # neither read into the page nor written by KILL ALL
    assert not any("rejected" in f.name for f in (tmp_path / "state").rglob("*"))


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


def test_unknown_venue_with_no_unit_and_no_status_is_not_started(tmp_path):
    root = tmp_path / "mynew"
    (root / "trend").mkdir(parents=True)
    (root / "strategies.json").write_text(json.dumps([{"id": "trend", "label": "Trend"}]))  # no unit named
    cfg = V.venue_config("mynew-demo", root)
    assert cfg.unit is None and cfg.label == "MYNEW·DEMO"
    svc = st.StrategyService(root, run=FakeSystemd(), clock=lambda: NOW, extra_units=(), dashboard_account="", shares_bot_account=False)
    h = V.VenueHub([V.venue_config("bybit-demo", tmp_path / "none", unit=""), cfg],
                   {"bybit-demo": st.StrategyService(tmp_path / "none", run=FakeSystemd(), clock=lambda: NOW, extra_units=()), "mynew-demo": svc},
                   state=tmp_path / "state", clock=lambda: NOW)
    v = run(h.summary())["venues"][1]
    assert v["presence"] == "absent" and "not started" in v["presence_text"] and v["kill"]["available"] is False
    with pytest.raises(st.StrategyError) as e:  # the kill agrees with the page: not started, nothing written
        run(h.kill("mynew-demo"))
    assert "has not started" in e.value.message and not (root / "trend" / "control.json").exists()


def test_a_strategy_with_no_known_unit_is_never_confirmed(tmp_path):
    root = tmp_path / "mynew"
    (root / "trend").mkdir(parents=True)
    (root / "strategies.json").write_text(json.dumps([{"id": "trend", "label": "Trend"}]))
    (root / "trend" / "status.json").write_text(json.dumps({"enabled": True, "state": "running", "updated_ms": int(NOW * 1000) - 5000,
                                                           "equity_usdt": 100.0, "positions": [], "targets": []}))
    svc = st.StrategyService(root, run=FakeSystemd(), clock=lambda: NOW, extra_units=(), dashboard_account="", shares_bot_account=False)
    h = V.VenueHub([V.venue_config("bybit-demo", tmp_path / "none", unit=""), V.venue_config("mynew-demo", root)],
                   {"bybit-demo": st.StrategyService(tmp_path / "none", run=FakeSystemd(), clock=lambda: NOW, extra_units=()), "mynew-demo": svc},
                   state=tmp_path / "state", clock=lambda: NOW)
    v = run(h.summary())["venues"][1]
    assert v["kill"]["ok"] is False and "no runner unit is known" in v["kill"]["why"]  # the preview says it too
    with pytest.raises(st.StrategyError) as e:
        run(h.kill("mynew-demo"))
    assert e.value.status == 409 and "no runner unit is known" in e.value.detail["unconfirmed"]["trend"]
    assert ctl(root, "trend")["flatten"] is True  # written all the same


def test_a_listed_strategy_without_a_directory_is_named(roots, tmp_path):
    okx = roots[1]
    reg = json.loads((okx / "strategies.json").read_text()) + [{"id": "basis", "label": "basis", "unit": OKX_UNIT, "account": "okx-demo"}]
    (okx / "strategies.json").write_text(json.dumps(reg))
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path)
    assert "basis" in run(h.summary())["venues"][1]["kill"]["why"]
    with pytest.raises(st.StrategyError) as e:
        run(h.kill("okx-demo"))
    assert e.value.detail["confirmed"] == ["trend", "longvol"] and "no directory" in e.value.detail["unconfirmed"]["basis"]


def test_a_broken_status_file_does_not_stop_the_page_or_kill_all(roots, tmp_path):
    set_status(roots[1], "trend", positions=1, targets="x")  # malformed: treated as no positions, never a crash
    h = hub(roots, FakeSystemd(active=BYBIT_UNITS + [OKX_UNIT]), tmp_path)
    assert run(h.summary())["venues"][1]["presence"] == "live"
    out = run(h.kill_all())
    assert {r["venue"]: r["ok"] for r in out["venues"]}["bybit-demo"] is True and ctl(roots[1], "trend")["flatten"] is True


def test_a_stopped_unit_of_one_strategy_raises_an_alert(roots, tmp_path):
    later = NOW + 600  # every fixture status is old now; carry's runner wrote a fresh one
    p = roots[0] / "carry" / "status.json"
    doc = json.loads(p.read_text())
    doc["updated_ms"] = int((later - 5) * 1000)
    p.write_text(json.dumps(doc))
    out = run(hub(roots, FakeSystemd(active=["calibrain-live@carry.service"]), tmp_path, now=later).summary())
    assert out["venues"][0]["presence"] == "live"
    downs = {a["strategy"]: a for a in out["alerts"] if a["kind"] == "unit_down" and a["venue"] == "bybit-demo"}
    assert sorted(downs) == ["trend", "xsmom"] and "calibrain-live@trend.service" in downs["trend"]["text"]
    k = out["venues"][0]["kill"]  # the preview agrees with what KILL would answer: not confirmable for trend and xsmom
    assert k["available"] is True and k["ok"] is False and "xsmom" in k["why"]
    assert "trend" not in k["why"]  # paused and flat: the kill only locks it, no runner has to confirm anything


def test_unit_times_come_as_unix_seconds_with_a_fallback():
    calls = []

    async def new_systemd(argv, **_):
        calls.append(argv)
        return 0, "LoadState=loaded\nActiveState=active\nActiveEnterTimestamp=@1795170550\n", ""

    async def old_systemd(argv, **_):
        calls.append(argv)
        if "--timestamp=unix" in argv:
            return 1, "", "unknown option"
        return 0, "LoadState=loaded\nActiveState=active\nActiveEnterTimestamp=Fri 2026-11-20 01:33:05 CET\n", ""

    u = run(st.unit_state(new_systemd, "calibrain-live.service"))
    assert u["since_s"] == 1795170550 and "--timestamp=unix" in calls[0]
    u = run(st.unit_state(old_systemd, "calibrain-live.service"))
    assert u["active"] == "active" and u["since_s"] is None and u["since"].startswith("Fri")


def test_an_empty_strategies_json_is_not_started(roots, tmp_path):
    (roots[1] / "strategies.json").write_text("[]")
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path)
    v = run(h.summary())["venues"][1]
    assert v["presence"] == "absent" and "lists no strategy" in v["presence_text"] and v["kill"]["available"] is False
    with pytest.raises(st.StrategyError) as e:
        run(h.kill("okx-demo"))
    assert "has not started" in e.value.message and e.value.detail == {}


def test_a_never_traded_idle_strategy_is_flagged_after_a_day_not_at_once(roots, tmp_path):
    p = roots[1] / "longvol" / "control.json"  # idle: no position, no target, never traded
    for age_h, flagged in ((1, False), (25, True)):
        p.write_text(json.dumps({"enabled": True, "weight": 1.0, "owner_lock": True, "updated_ms": int((NOW - age_h * 3600) * 1000)}))
        alerts = run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).summary())["alerts"]
        assert any(a["strategy"] == "longvol" and a["kind"] == "idle" for a in alerts) is flagged, age_h


def test_exchange_position_size_is_unsigned(roots, tmp_path):
    acct = {**ACCOUNT, "positions": [{**ACCOUNT["positions"][0], "size": -0.05}]}
    row = next(r for r in run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).summary(acct))["venues"][0]["positions"]
               if r["source"] == "exchange")
    assert row["size"] == 0.05 and row["side"] == "short"


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
    assert d["confirmed"] == ["carry"] and sorted(d["unconfirmed"]) == ["xsmom"] and d["already_flat"] == ["trend"]
    assert "calibrain-live@xsmom.service is inactive" in d["unconfirmed"]["xsmom"] and "Confirmed: carry" in e.value.message


def test_a_latched_strategy_is_never_confirmed(roots, tmp_path):
    set_status(roots[0], "xsmom", state="latched", last_error="venue position differs from the book: SOLUSDT")
    h = hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path)
    v = run(h.summary())["venues"][0]
    assert v["kill"]["ok"] is False and "latched" in v["kill"]["why"]  # the preview says it before
    with pytest.raises(st.StrategyError) as e:
        run(h.kill("bybit-demo"))
    d = e.value.detail
    assert e.value.status == 409 and d["confirmed"] == ["carry"] and d["already_flat"] == ["trend"] and "latched" in d["unconfirmed"]["xsmom"]
    assert "flatten runs after the reset" in e.value.message and ctl(roots[0], "xsmom")["flatten"] is True  # written all the same
    set_status(roots[0], "xsmom", positions=[{"symbol": "SOLUSDT", "side": "long", "qty": 1, "entry_price": 1, "mark_price": 1,
                                              "upnl_usdt": 0.0, "market": "linear"}])
    alerts = run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).summary())["alerts"]  # a fresh hub: no snapshot cache
    assert any(a["strategy"] == "xsmom" and a["kind"] == "kill_pending" and a["level"] == "crit" and "latched" in a["text"] for a in alerts)


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


def test_a_pending_kill_is_not_cancelled_by_enable_unless_the_owner_says_so(roots, tmp_path):
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path)
    run(h.kill("okx-demo"))
    svc = h.service("okx-demo")
    with pytest.raises(st.StrategyError) as e:
        run(svc.act("trend", "enable", llm_trading=False, venue="demo"))
    assert e.value.status == 409 and "kill is pending" in e.value.message and e.value.detail == {"kill_pending": ["trend"]}
    assert ctl(roots[1], "trend")["flatten"] is True and ctl(roots[1], "trend")["enabled"] is False  # untouched
    run(svc.act("trend", "enable", llm_trading=False, venue="demo", cancel_kill=True))  # the explicit owner choice
    c = ctl(roots[1], "trend")
    assert c["enabled"] is True and c["flatten"] is False and c["owner_lock"] is True


def test_an_enable_racing_a_kill_does_not_switch_on_over_it(roots, tmp_path):
    """The enable awaits systemctl (start the runner); a KILL written during that await must win."""
    okx = roots[1]

    class KillDuringStart(FakeSystemd):
        async def __call__(self, argv, **kw):
            if argv[2] == "start":  # the event loop yields here in real life; the kill lands meanwhile
                c = json.loads((okx / "trend" / "control.json").read_text())
                (okx / "trend" / "control.json").write_text(json.dumps({**c, "enabled": False, "owner_lock": True, "flatten": True}))
            return await super().__call__(argv, **kw)

    h = hub(roots, KillDuringStart(), tmp_path)  # the OKX unit is down, so enable starts it
    with pytest.raises(st.StrategyError) as e:
        run(h.service("okx-demo").act("trend", "enable", llm_trading=False, venue="demo"))
    assert e.value.status == 409 and "kill is pending" in e.value.message
    c = ctl(okx, "trend")
    assert c["flatten"] is True and c["enabled"] is False


def test_manual_header_switch_does_not_cancel_a_pending_kill(roots, tmp_path):
    sd = FakeSystemd(active=BYBIT_UNITS)
    set_status(roots[0], "trend", positions=[{"symbol": "ETHUSDT", "side": "long", "qty": 1, "entry_price": 1, "mark_price": 1,
                                              "upnl_usdt": 0.0, "market": "linear"}])  # trend holds something: its kill has work
    h = hub(roots, sd, tmp_path)
    run(h.kill("bybit-demo"))
    svc = h.primary  # MANUAL in the test hub
    with pytest.raises(st.StrategyError) as e:
        run(svc.auto(True, ["carry", "trend"], llm_trading=False, venue="demo"))
    assert e.value.status == 409 and sorted(e.value.detail["kill_pending"]) == ["carry", "trend"]
    assert all(ctl(roots[0], i)["flatten"] is True and ctl(roots[0], i)["enabled"] is False for i in ("carry", "trend"))
    with pytest.raises(st.StrategyError) as e:  # the confirm named carry only: trend's kill is not cancelled with it
        run(svc.auto(True, ["carry", "trend"], llm_trading=False, venue="demo", cancel_kill=["carry"]))
    assert e.value.detail["kill_pending"] == ["trend"] and ctl(roots[0], "carry")["flatten"] is True
    run(svc.auto(True, ["carry"], llm_trading=False, venue="demo", cancel_kill=["carry"]))
    assert ctl(roots[0], "carry")["enabled"] is True and ctl(roots[0], "carry")["flatten"] is False
    assert ctl(roots[0], "trend")["flatten"] is True  # not selected: its kill goes on


def test_unlock_and_cancel_kill(roots, tmp_path):
    h = hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path)
    run(h.kill("bybit-demo"))
    svc = h.primary
    with pytest.raises(st.StrategyError) as e:
        run(svc.act("carry", "unlock", llm_trading=False, venue="demo"))
    assert e.value.status == 409 and ctl(roots[0], "carry")["owner_lock"] is True
    run(svc.act("trend", "unlock", llm_trading=False, venue="demo", cancel_kill=True))
    c = ctl(roots[0], "trend")
    assert c["owner_lock"] is False and c["flatten"] is False
    run(svc.act("carry", "cancel_kill", llm_trading=False, venue="demo"))
    c = ctl(roots[0], "carry")
    assert c["flatten"] is False and c["enabled"] is False and c["owner_lock"] is True  # positions stay open, still paused
    with pytest.raises(st.StrategyError):
        run(svc.act("carry", "cancel_kill", llm_trading=False, venue="demo"))  # nothing pending any more


def test_cancelling_a_flatten_pauses_and_locks_the_strategy(roots, tmp_path):
    okx = roots[1]
    svc = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).service("okx-demo")
    (okx / "trend" / "control.json").write_text(json.dumps({"enabled": True, "weight": 1.0, "owner_lock": False, "set_by": "llm"}))
    run(svc.act("trend", "flatten", llm_trading=False, venue="demo"))  # an enabled strategy with a FLATTEN request
    run(svc.act("trend", "cancel_kill", llm_trading=False, venue="demo"))
    c = ctl(okx, "trend")
    assert c["flatten"] is False and c["enabled"] is False and c["owner_lock"] is True  # as the confirm says


def test_the_allocator_never_switches_on_a_strategy_with_a_pending_kill(roots):
    from ctb import allocator as alloc_mod

    async def market():
        return {}

    a = alloc_mod.Allocator(roots[0], object(), market, model="fake", clock=lambda: NOW, tick_s=0.01)
    p = roots[0] / "carry" / "control.json"
    p.write_text(json.dumps({"enabled": False, "flatten": True, "owner_lock": False, "weight": 0.6}))
    before = p.read_bytes()
    assert a._write_control("carry", True, 0.8, "the LLM wants it on", False, int(NOW * 1000)) is False
    assert p.read_bytes() == before and a._kill_skips == ["carry"]  # the decision feed shows it next to owner-lock skips
    assert a._write_control("carry", False, 0.0, "off", False, int(NOW * 1000)) is True  # switching off is fine


def test_no_lasting_kill_alert_once_flat_and_paused(roots, tmp_path):
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path)
    run(h.kill("okx-demo"))  # trend holds a position, longvol none
    kinds = {(a["strategy"], a["kind"]) for a in run(h.summary())["alerts"]}
    assert ("trend", "kill_pending") in kinds and ("longvol", "kill_pending") not in kinds


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
    freshen(roots[0], nxt - 10)  # the runner kept writing overnight
    run(hub(roots, sd, tmp_path, now=nxt).summary())
    assert json.loads((tmp_path / "state" / "baseline-bybit-demo.json").read_text())["day"] == V.utc_day(nxt)
    status = roots[0] / "carry" / "status.json"
    doc = json.loads(status.read_text())
    doc["equity_usdt"] += 5.0
    status.write_text(json.dumps(doc))
    carry = next(r for r in run(hub(roots, sd, tmp_path, now=nxt + 60).summary())["venues"][0]["strategies"] if r["id"] == "carry")
    assert carry["pnl_today"] == pytest.approx(5.0) and carry["pnl_today_since"] == "since 08:00 UTC"  # from the new day's sample


MIDNIGHT = 1_800_057_600  # a UTC midnight; NOW is 08:00:30 UTC the day before


def test_a_stale_status_never_seeds_the_baseline(roots, tmp_path):
    okx = roots[1]
    sd = FakeSystemd(active=[OKX_UNIT])
    set_status(okx, "trend", updated_ms=MIDNIGHT * 1000)  # alive at midnight
    set_status(okx, "longvol", updated_ms=(MIDNIGHT - 3600) * 1000, equity_usdt=9000.0)  # hung since 23:00 yesterday
    v = run(hub(roots, sd, tmp_path, now=MIDNIGHT + 30).summary())["venues"][1]
    rows = {r["id"]: r for r in v["strategies"]}
    assert rows["trend"]["pnl_today_since"] == "since 00:00 UTC" and rows["longvol"]["pnl_today"] is None
    assert v["pnl_today_partial"] is True and "no fresh status yet today for Long-only" in v["pnl_today_note"]
    later = MIDNIGHT + 45 * 60  # the runner comes back at 00:45 and writes 10,000
    set_status(okx, "longvol", updated_ms=later * 1000, equity_usdt=10000.0)
    set_status(okx, "trend", updated_ms=later * 1000)
    v = run(hub(roots, sd, tmp_path, now=later + 5).summary())["venues"][1]
    rows = {r["id"]: r for r in v["strategies"]}
    assert rows["longvol"]["pnl_today"] == 0.0 and rows["longvol"]["pnl_today_since"] == "since 00:45 UTC"  # not +1,000 from yesterday
    assert "Long-only counted from 00:45 UTC (its status was stale until then" in v["pnl_today_note"]
    assert "harness took" not in v["pnl_today_note"]  # the harness was sampling at midnight


def test_since_label_says_when_the_baseline_was_taken():
    assert V.since_label(MIDNIGHT * 1000 + 60_000) == "since 00:00 UTC"  # the sampler's first minute
    assert V.since_label(MIDNIGHT * 1000 + 4 * 60_000) == "since 00:04 UTC"  # never rounded down to midnight
    assert V.since_label(MIDNIGHT * 1000 + 3 * 3600_000) == "since 03:00 UTC"


def set_status(root, sid, **changes):
    p = root / sid / "status.json"
    doc = json.loads(p.read_text())
    doc.update(changes)
    p.write_text(json.dumps(doc))


def freshen(root, t_s):
    """Every status.json under root written at t_s (the runner is alive then)."""
    for p in root.glob("*/status.json"):
        set_status(root, p.parent.name, updated_ms=int(t_s * 1000))


def test_venue_label_at_midnight_is_plain(roots, tmp_path):
    freshen(roots[1], MIDNIGHT)
    v = run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path, now=MIDNIGHT + 30).summary())["venues"][1]
    assert v["pnl_today_since"] == "since 00:00 UTC" and v["pnl_today_partial"] is False and v["pnl_today_note"] is None


def test_venue_label_says_when_and_why_baselines_differ(roots, tmp_path):
    okx = roots[1]
    sd = FakeSystemd(active=[OKX_UNIT])
    first = run(hub(roots, sd, tmp_path).summary())["venues"][1]
    assert first["pnl_today_since"] == "since 08:00 UTC" and first["pnl_today_partial"] is False
    assert "first sample at 08:00 UTC" in first["pnl_today_note"]  # the harness was not sampling at midnight
    (okx / "xsmom").mkdir()  # a strategy that appears an hour later
    (okx / "xsmom" / "status.json").write_text(json.dumps({"enabled": True, "state": "running", "updated_ms": (NOW + 3590) * 1000,
                                                          "equity_usdt": 10000.0, "start_equity_usdt": 10000.0, "pnl_usdt": 0.0,
                                                          "positions": [], "targets": []}))
    reg = json.loads((okx / "strategies.json").read_text()) + [{"id": "xsmom", "label": "xsmom", "unit": OKX_UNIT, "account": "okx-demo"}]
    (okx / "strategies.json").write_text(json.dumps(reg))
    v = run(hub(roots, sd, tmp_path, now=NOW + 3600).summary())["venues"][1]
    rows = {r["id"]: r for r in v["strategies"]}
    assert rows["xsmom"]["pnl_today_since"] == "since 09:00 UTC" and rows["trend"]["pnl_today_since"] == "since 08:00 UTC"
    assert v["pnl_today_since"] is None and v["pnl_today_partial"] is True
    assert "Momentum first seen 09:00 UTC, counted from then" in v["pnl_today_note"]


ACCOUNT = {"wallet": {"equity": 163245.78, "usdt_equity": 49911.05, "available": 99007.9, "unrealized_pnl": 24.4},
           "positions": [{"symbol": "BTCUSDT", "side": "short", "size": 0.05, "entry": 90000.0, "mark": 90100.0, "upnl": -5.0}],
           "last_sync": NOW + 50, "sync_error": None}  # a good read seconds before every NOW used with it below


def test_bybit_account_block_compares_like_with_like(roots, tmp_path):
    doc = run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).summary(ACCOUNT))
    b = doc["venues"][0]
    a = b["account"]
    assert a["equity"] == 49911.05 and a["collateral_usd"] == 163245.78
    assert a["pnl_today"] is None  # the USDT coin's equity moves on spot buys: no account-level P&L today
    # carry holds a linear leg (uPnL -5.0) and a spot leg (+4.5): only the linear one is in the exchange's unrealised
    assert a["books_unrealised"] == pytest.approx(-5.0) and a["unattributed"] == pytest.approx(24.4 + 5.0)
    assert b["books"]["unrealised"] == pytest.approx(-0.5)  # the books' own figure keeps both legs
    assert b["positions"][0]["held_by"] == ["Carry"] and b["positions_source"] == "exchange"  # the exchange's netted row, with who holds a leg
    rows = {(r["symbol"], r["market"]): r["source"] for r in b["positions"]}  # the exchange feed is linear only: the books' spot leg stays
    assert rows == {("BTCUSDT", "linear"): "exchange", ("BTCUSDT", "spot"): "books"} and "spot legs" in b["positions_note"]
    assert b["pnl_today_partial"] is True and "equity unknown for Momentum" in b["pnl_today_note"]  # xsmom has no equity
    assert a["stale"] is False and not any(x["kind"] == "account_stale" for x in doc["alerts"])
    okx = doc["venues"][1]
    assert okx["account"] is None and okx["books"]["equity"] == 20004.0  # other venues: books only


def test_an_unknown_market_makes_unattributed_unknown(roots, tmp_path):
    doc = json.loads((roots[0] / "carry" / "status.json").read_text())
    doc["positions"][1]["market"] = ""  # the spot leg without a market: not guessed to be a derivative
    (roots[0] / "carry" / "status.json").write_text(json.dumps(doc))
    a = run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).summary(ACCOUNT))["venues"][0]["account"]
    assert a["books_unrealised"] is None and a["unattributed"] is None


def test_unattributed_counts_only_the_strategies_on_the_account(roots, tmp_path):
    reg = json.loads((roots[0] / "strategies.json").read_text())
    for e in reg:
        if e["id"] == "trend":
            e["account"] = "demo-other"  # trades another account
    (roots[0] / "strategies.json").write_text(json.dumps(reg))
    set_status(roots[0], "trend", positions=[{"symbol": "ETHUSDT", "side": "long", "qty": 1, "entry_price": 3000, "mark_price": 3007,
                                              "upnl_usdt": 7.0, "market": "linear"}])
    v = run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).summary(ACCOUNT))["venues"][0]
    a = v["account"]
    assert a["books_unrealised"] == pytest.approx(-5.0) and a["unattributed"] == pytest.approx(29.4)  # trend's +7 is not this account's
    other = [r for r in v["positions"] if r.get("account") == "demo-other"]  # but its position stays on the page, labelled
    assert [(r["symbol"], r["source"], r["held_by"]) for r in other] == [("ETHUSDT", "books", ["Trend"])] and "demo-other" in v["positions_note"]


def test_a_spot_purchase_does_not_move_any_pnl_today(roots, tmp_path):
    sd = FakeSystemd(active=BYBIT_UNITS)
    run(hub(roots, sd, tmp_path).summary(ACCOUNT))
    bought = {**ACCOUNT, "wallet": {**ACCOUNT["wallet"], "usdt_equity": 49911.05 - 4500.0}}  # 0.05 BTC bought with USDT
    v = run(hub(roots, sd, tmp_path, now=NOW + 60).summary(bought))["venues"][0]
    assert v["account"]["pnl_today"] is None and v["pnl_today"] == pytest.approx(0.0)


@pytest.mark.parametrize("failure", [{"sync_error": "HTTP 503 from api-demo.bybit.com"}, {"last_sync": NOW - 300}])
def test_a_failing_exchange_read_is_stale_and_positions_come_from_the_books(roots, tmp_path, failure):
    acct = {**ACCOUNT, "positions": [], **failure}  # the executor kept an old, empty snapshot
    doc = run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).summary(acct))
    b = doc["venues"][0]
    a = b["account"]
    assert a["stale"] is True and a["unattributed"] is None  # never compare fresh books with a stale account
    assert b["positions_source"] == "books" and "stale" in b["positions_note"]
    assert {(p["symbol"], p["market"]) for p in b["positions"]} == {("BTCUSDT", "linear"), ("BTCUSDT", "spot")}  # carry's legs, not "flat"
    alert = next(x for x in doc["alerts"] if x["kind"] == "account_stale")
    assert alert["venue"] == "bybit-demo" and ("503" in alert["text"] or "min ago" in alert["text"])


def test_an_account_never_synced_is_unknown(roots, tmp_path):
    acct = {**ACCOUNT, "last_sync": 0.0, "sync_error": "invalid api key"}
    a = run(hub(roots, FakeSystemd(active=BYBIT_UNITS), tmp_path).summary(acct))["venues"][0]["account"]
    assert a["stale"] is True and a["equity"] is None and a["unrealised"] is None and a["sync_age_s"] is None


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
            assert r.status == 409 and "NOT confirmed" in body["error"] and sorted(body["written"]) == ["carry", "xsmom"]
            assert body["already_flat"] == ["trend"] and ctl(roots[0], "trend")["flatten"] is False  # paused and flat: only locked
            assert ctl(roots[0], "carry")["flatten"] is True and alloc_ctl(roots[0])["enabled"] is False
            r = await c.post("/api/kill", json={})  # the old alias answers the same way and says the bot's auto-trade is off
            body = await r.json()
            assert r.status == 409 and body["bot_auto_trade"] == "off"
            r = await c.post("/api/kill-all", json={})
            body = await r.json()
            assert r.status == 200 and body["ok"] is False and body["bot_auto_trade"] == "off"
            assert {x["venue"]: x["ok"] for x in body["venues"]} == {"bybit-demo": False, "okx-demo": True, "deribit-test": False}
            assert ctl(roots[1], "trend")["flatten"] is True
            assert (await c.post("/api/venues/okx-demo/strategies/trend/unlock", json={})).status == 409
            r = await c.post("/api/venues/okx-demo/strategies/trend/enable", data="{oops", headers={"Content-Type": "application/json"})
            assert r.status == 400  # a malformed cancel_kill body is not read as "no"
            for bad in ("{not json", ""):  # a malformed or empty body is the client's error: 400, not 500
                r = await c.post("/api/venues/okx-demo/runner/start", data=bad, headers={"Content-Type": "application/json"})
                assert r.status == 400 and "not valid JSON" in (await r.json())["error"]
            r = await c.post("/api/venues/okx-demo/strategies/trend/enable", json={})  # its kill is pending
            assert r.status == 409 and (await r.json())["kill_pending"] == ["trend"]
            assert (await c.post("/api/venues/okx-demo/strategies/trend/enable", json={"cancel_kill": True})).status == 200
            assert ctl(roots[1], "trend")["flatten"] is False
            # crafted ids never reach the file system: an unknown venue, a sid or unit with a path in it
            before = files_under(*roots[:2])
            for path, body in (("/api/venues/..%2F..%2Fetc/strategies/carry/pause", {}), ("/api/venues/bybit-demo/strategies/..%2Fcarry/pause", {}),
                               ("/api/venues/bybit-demo/strategies/%2E%2E/pause", {}), ("/api/venues/okx-demo/strategies/..%2F..%2Flive%2Fcarry/disable", {}),
                               ("/api/venues/bybit-demo/runner/start", {"unit": "../../evil.service"}),
                               ("/api/venues/..%2Flive/kill", {}), ("/api/venues/okx-demo/runner/stop", {"unit": "calibrain-live@carry.service"})):
                r = await c.post(path, json=body)
                assert r.status == 404, (path, r.status)
            assert files_under(*roots[:2]) == before
            r = await c.post("/api/strategies/auto-set", json={"ids": ["carry"]})
            assert r.status == 200 and (await r.json())["ids"] == ["carry"]

    run(go())


class FakeExecutor:
    """Executor.kill's contract: closes each position (a failed close is only logged), cancels orders, never raises."""

    def __init__(self, closes_work=True):
        self.positions = [{"symbol": "BTCUSDT"}]
        self.open_orders: list = []
        self.sync_error = None
        self.closes_work = closes_work
        self.kills: list[str] = []

    async def kill(self, reason):
        self.kills.append(reason)
        if self.closes_work:
            self.positions = []

    async def sync(self):
        pass


@pytest.mark.parametrize("mode,closes_work", [("bot", True), ("bot", False), ("manual", True)])
def test_kill_all_kills_the_per_symbol_bot_only_when_it_owns_the_account(roots, tmp_path, monkeypatch, mode, closes_work):
    from aiohttp.test_utils import TestClient, TestServer

    from ctb.server import make_app

    ex = FakeExecutor(closes_work)

    class Engine:
        s = SimpleNamespace(trading_enabled=mode == "bot", public_view=lambda: {})
        db = SimpleNamespace(event=lambda *a: a, one=lambda *a: None)
        executor = ex

        async def start(self): ...
        async def stop(self): ...
        def emit(self, *a): ...

    monkeypatch.setenv("CTB_VENUE", "demo")
    h = hub(roots, FakeSystemd(active=BYBIT_UNITS + [OKX_UNIT]), tmp_path)
    h.primary._mode = mode
    eng = Engine()
    app = make_app(eng, hub=h)
    app.on_startup.clear()
    app.on_cleanup.clear()

    async def go():
        async with TestClient(TestServer(app)) as c:
            return await (await c.post("/api/kill-all", json={})).json()

    body = run(go())
    by = {x["venue"]: x for x in body["venues"]}
    assert by["okx-demo"]["ok"] is True and eng.s.trading_enabled is False and body["bot"]["auto_trade"] == "off"
    # bybit-demo is killed in every mode: nothing assumes the bot mode paused it (it does only when they share the account)
    assert by["bybit-demo"]["ok"] is True and ctl(roots[0], "carry")["flatten"] is True
    if mode == "manual":  # the strategies own it: Executor.kill would close their netted legs, so the bot is not touched on the exchange
        assert ex.kills == [] and body["bot"]["killed"] is False
    elif closes_work:  # verified on a fresh read
        assert ex.kills == ["dashboard KILL ALL"] and body["bot"]["killed"] is True and "verified" in body["bot"]["why"]
    else:  # a close failed (Executor.kill only logs it): never reported as closed
        assert body["bot"]["killed"] is False and "still open" in body["bot"]["why"] and "BTCUSDT" in body["bot"]["why"]
        assert body["ok"] is False


@pytest.mark.parametrize("pause", ["auto_off", "to_manual"])
def test_a_decision_in_flight_does_not_switch_strategies_back_on_after_a_pause(roots, tmp_path, pause):
    """The owner presses AUTO-TRADE OFF (or switches to MANUAL) while the allocator's model call is still running."""
    from ctb import allocator as alloc_mod

    class SlowModel:
        def __init__(self):
            self.asked = asyncio.Event()
            self.gate = asyncio.Event()

        async def ask_json(self, messages, schema):
            self.asked.set()
            await self.gate.wait()
            ids = list(schema["properties"]["weights"]["properties"])
            return ({"weights": {i: 0.8 for i in ids}, "enabled": {i: True for i in ids}, "reasoning": "slow: everything on",
                     "confidence": 0.5}, {"cost": 0.0})

    async def market():
        return {}

    bybit = roots[0]
    for sid in ("carry", "trend", "xsmom"):  # no owner locks: the allocator may write every strategy
        (bybit / sid / "control.json").write_text(json.dumps({"enabled": False, "weight": 0.0, "set_by": "llm"}))
    sd = FakeSystemd(active=BYBIT_UNITS)
    svc = st.StrategyService(bybit, run=sd, clock=lambda: NOW, extra_units=(), dashboard_account="demo-main", allocator_available=True)
    svc._mode = "llm"
    model = SlowModel()
    alloc = alloc_mod.Allocator(bybit, model, market, model="slow", clock=lambda: NOW, tick_s=0.01)
    svc.attach_allocator(alloc)

    async def go():
        await svc.auto(True, None, llm_trading=False, venue="demo")  # allocator on
        await svc.allocator_act("decide_now")
        tick = asyncio.create_task(alloc._tick())
        await asyncio.wait_for(model.asked.wait(), 5)  # the model call is in flight
        if pause == "auto_off":
            await svc.auto(False, None, llm_trading=False, venue="demo")
        else:
            await svc.set_mode("manual", venue="demo", bot_open=False)
        model.gate.set()
        await asyncio.wait_for(tick, 5)

    run(go())
    assert {sid: ctl(bybit, sid)["enabled"] for sid in ("carry", "trend", "xsmom")} == {"carry": False, "trend": False, "xsmom": False}
    last = json.loads((bybit / "allocator" / "decisions.jsonl").read_text().splitlines()[-1])
    assert sorted(last["skipped_paused"]) == ["carry", "trend", "xsmom"]  # the feed says why nothing was applied


def test_bot_mode_kill_all_then_back_to_manual_and_auto_trade_works(roots, tmp_path):
    """PER-SYMBOL BOT pauses the bybit strategies and stops their runner (they share the bot's account). KILL ALL there must not
    leave flags nobody clears: back in MANUAL, AUTO-TRADE on works."""
    bybit = roots[0]
    set_status(bybit, "carry", positions=[])  # set_mode("bot") requires the strategies flat
    sd = FakeSystemd(active=BYBIT_UNITS)
    h = hub(roots, sd, tmp_path)
    svc = h.primary
    run(svc.set_mode("bot", venue="demo", bot_open=False))
    assert not sd.active & set(BYBIT_UNITS)  # the runner is stopped
    out = run(h.kill_all())
    by = {r["venue"]: r for r in out["venues"]}
    assert by["bybit-demo"]["ok"] is True and sorted(by["bybit-demo"]["already_flat"]) == ["carry", "trend", "xsmom"]
    assert all(ctl(bybit, i).get("flatten") is not True and ctl(bybit, i)["owner_lock"] is True for i in ("carry", "trend", "xsmom"))
    run(svc.set_mode("manual", venue="demo", bot_open=False))
    run(svc.auto(True, None, llm_trading=False, venue="demo"))
    assert all(ctl(bybit, i)["enabled"] is True for i in ("carry", "trend", "xsmom"))


def test_enable_clears_a_stale_flag_on_a_flat_book_without_cancel(roots, tmp_path):
    okx = roots[1]
    c = ctl(okx, "longvol")  # longvol is flat
    (okx / "longvol" / "control.json").write_text(json.dumps({**c, "enabled": False, "flatten": True}))
    h = hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path)
    run(h.service("okx-demo").act("longvol", "enable", llm_trading=False, venue="demo"))  # no cancel_kill needed: nothing to keep open
    c = ctl(okx, "longvol")
    assert c["enabled"] is True and c["flatten"] is False


def test_api_kill_turns_the_bots_trading_off_even_when_the_venue_has_not_started(tmp_path, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    from ctb.server import make_app

    class Engine:
        s = SimpleNamespace(trading_enabled=True, public_view=lambda: {})
        db = SimpleNamespace(event=lambda *a: a, one=lambda *a: None)
        executor = None

        async def start(self): ...
        async def stop(self): ...
        def emit(self, *a): ...

    monkeypatch.setenv("CTB_VENUE", "demo")
    svc = st.StrategyService(tmp_path / "nothing", run=FakeSystemd(), clock=lambda: NOW, extra_units=(), dashboard_account="demo-main")
    svc._mode = "manual"  # the strategies own the account, but bybit-demo has not started
    eng = Engine()
    app = make_app(eng, hub=V.VenueHub.single(svc, state=tmp_path / "state"))
    app.on_startup.clear()
    app.on_cleanup.clear()

    async def go():
        async with TestClient(TestServer(app)) as c:
            r = await c.post("/api/kill", json={})
            return r.status, await r.json()

    status, body = run(go())
    assert status == 409 and "has not started" in body["error"] and body["bot_auto_trade"] == "off" and eng.s.trading_enabled is False


def test_liveness_falls_back_to_the_venues_units_when_strategies_json_names_none(tmp_path):
    root = tmp_path / "mynew"
    (root / "trend").mkdir(parents=True)
    (root / "strategies.json").write_text(json.dumps([{"id": "trend", "label": "Trend"}]))  # no unit per strategy
    (root / "trend" / "status.json").write_text(json.dumps({"enabled": True, "state": "running", "updated_ms": int(NOW * 1000) - 5000,
                                                           "equity_usdt": 100.0, "positions": [], "targets": []}))
    unit = "calibrain-live-mynew.service"  # known only from CTB_LIVE_UNITS (extra units)
    svc = st.StrategyService(root, run=FakeSystemd(active=[unit]), clock=lambda: NOW, extra_units=(unit,), dashboard_account="",
                             shares_bot_account=False)
    h = V.VenueHub([V.venue_config("bybit-demo", tmp_path / "none", unit=""), V.venue_config("mynew-demo", root)],
                   {"bybit-demo": st.StrategyService(tmp_path / "none", run=FakeSystemd(), clock=lambda: NOW, extra_units=()), "mynew-demo": svc},
                   state=tmp_path / "state", clock=lambda: NOW)
    v = run(h.summary())["venues"][1]
    assert v["presence"] == "live" and v["kill"]["ok"] is True
    assert run(h.kill("mynew-demo"))["confirmed"] == ["trend"]


def test_control_writes_never_share_the_runners_temp_file(roots, tmp_path):
    seen = []
    real = st.os.replace

    def spy(src, dst):
        seen.append(Path(src).name)
        return real(src, dst)

    st.os.replace = spy
    try:
        run(hub(roots, FakeSystemd(active=[OKX_UNIT]), tmp_path).service("okx-demo").act("trend", "pause", llm_trading=False, venue="demo"))
    finally:
        st.os.replace = real
    assert seen and all(n != "control.json.tmp" and n.startswith(".control.json.") for n in seen)
