"""Predictions panel backend: /api/predictions on fixture files, missing and damaged files, formats, the 30 s cache.
No network, no runner: only files under a temp copy of the fixtures."""

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

from ctb import allocator_math as M
from ctb import predictions as pr
from ctb import strategies as st
from ctb.server import make_app

FIX = Path(__file__).parent / "fixtures"
NOW = 1_800_000_030.0  # seconds; the fixtures' ms fields are around 1_800_000_000_000


@pytest.fixture
def root(tmp_path):
    dst = tmp_path / "live"
    shutil.copytree(FIX / "live", dst)
    shutil.copytree(FIX / "predictions", dst, dirs_exist_ok=True)
    return dst


def by_id(rows):
    return {r["id"]: r for r in rows}


# ---------------------------------------------------------------- the document
def test_full_document_from_fixture_files(root):
    doc = pr.build(root, NOW, "2026-11-19")
    assert doc["missing"] == [] and doc["go_live"] == "2026-11-19"
    # regime: state, the 8 features with a meaning each, the 4 probabilities
    rg = doc["regime"]
    assert rg["state"] == "up_calm" and rg["date"] == "2026-11-20"
    assert [f["key"] for f in rg["features"]] == list(pr.FEATURES_KEYS) and len(rg["features"]) == 8
    assert all(f["meaning"] and f["value"] is not None for f in rg["features"])
    assert {f["key"]: f["unit"] for f in rg["features"]}["funding_level"] == "pct"
    assert [(p["state"], p["p"]) for p in rg["p7"]] == [("up_calm", 0.62), ("up_wild", 0.2), ("down_calm", 0.1), ("down_wild", 0.08)]
    # forecast age and staleness
    assert doc["forecast"]["date"] == "2026-11-20" and doc["forecast"]["age_h"] == pytest.approx(0.28, abs=0.01) and not doc["forecast"]["stale"]
    # the allocator line used is the newest one that carries the breakdown, not the hourly refresh after it
    assert doc["allocator"]["t_ms"] == 1799999400000 and doc["allocator"]["weights_mode"] == "computed"
    rows = by_id(doc["strategies"])
    assert list(rows) == ["carry", "trend", "xsmom"]
    t = rows["trend"]
    assert t["label"] == "Trend (BTC, ETH)" and 0 < t["core"] < 1 and t["shadow_tau"] > 0 and t["sr"] > 0 and t["tau"] == 0.0
    assert t["mu_ann_pct"] == {"base": 10.95, "regime": 14.6, "ridge": -3.65, "structural": None, "decider": -2.1, "combined": 8.4}
    assert t["sigma_ann_pct"] == pytest.approx(0.03 * 365 ** 0.5 * 100, abs=0.01)
    assert t["p_decider"] == 0.47 and t["live_weight"] == 0.0 and t["reason"].startswith("trend: core 0.")
    assert rows["carry"]["live_weight"] == 0.6 and rows["xsmom"]["live_weight"] is None  # xsmom has no control.json
    assert rows["carry"]["mu_ann_pct"]["structural"] == 4.38


def test_scoreboard_and_the_tilt_line(root):
    sb = pr.build(root, NOW)["scoreboard"]
    c = by_id(sb["per_strategy"])["carry"]
    assert c["loss"]["structural"] == 0.8 and c["weight"]["decider"] == 0.35
    assert "structural" not in by_id(sb["per_strategy"])["trend"]["loss"]
    assert c["brier"] == 0.24 and c["brier_base"] == 0.25 and c["n_brier"] == 2 and c["n_scored"] == 2 and c["decider_no_skill"] is False
    t = sb["tilt"]
    assert t["kappa"] == 0.0 and t["live"] is False and t["n_days"] == 2 and t["days_needed"] == 90
    assert t["line"] == "tilt live control: off until the shadow tilt beats the core over 90 days (2/90 days so far)"
    assert t["skill_90_ann_pct"] is None


def test_tilt_line_when_earned_and_when_empty():
    assert pr.tilt_line(0.5, 120).startswith("tilt live control: on at room 0.50")
    assert pr.tilt_line(None, 0) == "tilt live control: off until the shadow tilt beats the core over 90 days (0/90 days so far)"


def test_live_pnl_against_the_shadow_return_since_go_live(root):
    rows = by_id(pr.build(root, NOW, "2026-11-19")["pnl"])
    c = rows["carry"]
    # carry returns from 2026-11-19: 0.0002, -0.0001, 0.0003 (2026-11-18 is before go-live and is left out)
    assert c["days"] == 3 and c["last_day"] == "2026-11-21"
    assert c["shadow_pct"] == pytest.approx(((1.0002 * 0.9999 * 1.0003) - 1) * 100, abs=1e-3)
    assert c["live_pnl_pct"] == 1.125 and c["live_pnl_usdt"] == 112.5 and c["weight"] == 0.6
    # each day at the live weight recorded the evening before: 11-20 at 0.6, 11-21 at 0.4; 11-19 has no record: missing
    assert c["shadow_weighted_pct"] == pytest.approx(((1 - 0.0001 * 0.6) * (1 + 0.0003 * 0.4) - 1) * 100, abs=5e-4)  # 3 decimals
    assert (c["weighted_days"], c["missing_days"]) == (2, 1)
    x = rows["xsmom"]
    assert x["live_pnl_pct"] is not None and x["weight"] is None and x["days"] == 3
    assert x["shadow_weighted_pct"] == pytest.approx(((1 + 0.002 * 0.2) * (1 + 0.005 * 0.2) - 1) * 100, abs=5e-4)
    # an earlier go-live day adds the earlier row
    assert by_id(pr.build(root, NOW, "2026-11-18")["pnl"])["carry"]["days"] == 4


def test_without_recorded_weights_no_day_is_weighted_with_todays_weight(root):
    (root / "allocator" / "policies.jsonl").unlink()
    c = by_id(pr.build(root, NOW, "2026-11-19")["pnl"])["carry"]
    assert c["weight"] == 0.6 and c["shadow_weighted_pct"] is None and (c["weighted_days"], c["missing_days"]) == (0, 3)


def test_history_compares_the_recorded_policies_day_by_day(root):
    h = pr.build(root, NOW, "2026-11-19")["history"]
    # the weights runs of 11-19 and 11-20 govern 11-20 and 11-21; 11-19 has no record and is missing, not a zero
    assert h["since"] == "2026-11-19" and h["days"] == ["2026-11-20", "2026-11-21"] and h["missing_days"] == ["2026-11-19"]
    r20 = {"carry": -0.0001, "trend": 0.004, "xsmom": 0.002}
    r21 = {"carry": 0.0003, "trend": -0.002, "xsmom": 0.005}
    w20 = {"core": {"carry": 0.5, "trend": 0.3, "xsmom": 0.2}, "tilt": {"carry": 0.4, "trend": 0.45, "xsmom": 0.25},
           "live": {"carry": 0.6, "trend": 0.0, "xsmom": 0.2}, "core_veto": {"carry": 0.5, "trend": 0.15, "xsmom": 0.2}}
    w21 = {"core": {"carry": 0.5, "trend": 0.35, "xsmom": 0.2}, "tilt": {"carry": 0.35, "trend": 0.5, "xsmom": 0.3},
           "live": {"carry": 0.4, "trend": 0.1, "xsmom": 0.2}, "core_veto": {"carry": 0.5, "trend": 0.2, "xsmom": 0.2}}
    for k, out in pr.POLICY_KEYS:
        d1 = sum(w20[k][s] * r20[s] for s in r20) / 3
        d2 = sum(w21[k][s] * r21[s] for s in r21) / 3
        assert h[out] == [pytest.approx(d1 * 100, abs=5e-5), pytest.approx(((1 + d1) * (1 + d2) - 1) * 100, abs=5e-5)]  # 4 decimals
    assert h["core_pct"][0] != h["tilted_pct"][0]  # the policies differ


def test_history_is_null_before_the_first_governed_day(root):
    assert pr.build(root, NOW, "2026-12-01")["history"] is None


# ---------------------------------------------------------------- missing and damaged files
def test_empty_root_is_not_computed_yet(tmp_path):
    doc = pr.build(tmp_path / "nothing", NOW)
    assert doc["regime"] is None and doc["forecast"] is None and doc["allocator"] is None and doc["scoreboard"] is None
    assert doc["strategies"] == [] and doc["pnl"] == [] and doc["history"] is None
    assert doc["missing"] == ["regime", "forecast", "scores", "allocator"]
    json.dumps(doc)


def test_each_file_missing_on_its_own_only_blanks_its_section(root):
    (root / "regime" / "latest.json").unlink()
    (root / "forecast" / "scores.json").unlink()
    doc = pr.build(root, NOW)
    assert doc["regime"] is None and doc["scoreboard"] is None and doc["missing"] == ["regime", "scores"]
    assert doc["strategies"] and doc["forecast"] and doc["history"]


def test_without_an_allocator_line_the_stored_forecasters_are_shown(root):
    (root / "allocator" / "decisions.jsonl").write_text(json.dumps({"t_ms": 1, "weights_mode": "refresh", "answers": {"p": {"carry": 0.5}}}) + "\n")
    doc = pr.build(root, NOW)
    assert doc["allocator"] is None and "allocator" in doc["missing"]
    c = by_id(doc["strategies"])["carry"]
    assert c["core"] is None and c["shadow_tau"] is None and c["reason"] is None and c["p_decider"] is None
    assert c["mu_ann_pct"]["structural"] == pytest.approx(0.00012 * 365 * 100, abs=0.01) and c["mu_ann_pct"]["decider"] is None
    assert c["mu_ann_pct"]["combined"] == pytest.approx(0.00008 * 365 * 100, abs=0.01)
    assert c["sigma_ann_pct"] is not None


def test_damaged_files_never_break_the_answer(root):
    (root / "regime" / "latest.json").write_text("{not json")
    (root / "forecast" / "latest.json").write_text("[]")
    (root / "forecast" / "scores.json").write_text('{"loss": {"carry": "x"}, "tilt_skill": 5, "n_scored": []}')
    (root / "returns" / "carry.csv").write_text("date,ret_net\nbad,row\n2026-11-20,nan\n2026-11-21,0.01\n")
    (root / "forecast" / "archive" / "2026-11-20.json").write_text("{")
    doc = pr.build(root, NOW)
    assert doc["regime"] is None and doc["forecast"] is None
    assert doc["scoreboard"]["tilt"]["n_days"] == 0 and doc["scoreboard"]["tilt"]["line"].endswith("(0/90 days so far)")
    assert by_id(doc["pnl"])["carry"]["days"] == 1
    json.dumps(doc)


def test_stale_forecast_is_flagged(root):
    doc = pr.build(root, NOW + 40 * 3600)
    assert doc["forecast"]["stale"] is True and doc["forecast"]["age_h"] > 36


def test_a_huge_decision_log_is_read_from_its_end(root):
    p = root / "allocator" / "decisions.jsonl"
    old = p.read_text()
    filler = json.dumps({"t_ms": 5, "weights_mode": "refresh", "inputs": {"x": "z" * 100_000}}) + "\n"
    p.write_text(filler * 100 + old)  # 10 MB of lines before the real ones
    assert pr.build(root, NOW)["allocator"]["t_ms"] == 1799999400000


def test_a_half_written_last_line_is_skipped(root):
    p = root / "allocator" / "decisions.jsonl"
    p.write_text(p.read_text() + '{"t_ms": 9, "core": {"carry": 1.0}, "weights_mo')
    assert pr.build(root, NOW)["allocator"]["t_ms"] == 1799999400000


# ---------------------------------------------------------------- the service and the route
def test_service_caches_for_30_seconds(root):
    clock = [NOW]
    svc = pr.PredictionsService(root, clock=lambda: clock[0])
    a = asyncio.run(svc.snapshot())
    (root / "regime" / "latest.json").unlink()
    assert asyncio.run(svc.snapshot()) is a            # cached
    clock[0] += pr.CACHE_S + 1
    assert asyncio.run(svc.snapshot())["regime"] is None  # re-read
    assert asyncio.run(svc.snapshot(force=True))["generated_at"] == clock[0]


def serve(root):
    app = make_app(SimpleNamespace(s=SimpleNamespace(trading_enabled=False)),
                   st.StrategyService(root, clock=lambda: NOW, extra_units=(), dashboard_account="demo-main"))
    app.on_startup.clear()   # no engine, no allocator, no runner: only the route
    app.on_cleanup.clear()
    return TestServer(app)


def test_route_serves_the_document_and_is_read_only(root):
    async def run():
        async with serve(root) as srv, aiohttp.ClientSession() as session:
            r = await session.get(srv.make_url("/api/predictions"))
            body = await r.json()
            r2 = await session.post(srv.make_url("/api/predictions"), json={})
            return r.status, r.content_type, body, r2.status

    status, ctype, body, post_status = asyncio.run(run())
    assert status == 200 and ctype == "application/json" and post_status == 405
    assert body["regime"]["state"] == "up_calm" and [r["id"] for r in body["strategies"]] == ["carry", "trend", "xsmom"]
    assert body["scoreboard"]["tilt"]["line"].startswith("tilt live control: off until")


def test_route_on_an_empty_root_answers_not_computed_yet(tmp_path):
    async def run():
        async with serve(tmp_path / "empty") as srv, aiohttp.ClientSession() as session:
            r = await session.get(srv.make_url("/api/predictions"))
            return r.status, await r.json()

    status, body = asyncio.run(run())
    assert status == 200 and body["regime"] is None and body["missing"] == ["regime", "forecast", "scores", "allocator"]


def test_no_em_dashes_in_the_ui_text():
    dash = chr(8212)
    base = Path(__file__).parents[1]
    for f in [base / "ctb" / "predictions.py", base / "web-next" / "components" / "Predictions.tsx"]:
        if f.exists():
            assert dash not in f.read_text(), f


def test_history_keeps_a_retired_strategy_from_its_recorded_weights(root):
    """basis was held in the recorded policies and has its returns file, but is no longer in the registry, the forecast,
    the latest weights line or the scores: its days are still computed, not missing."""
    lines = [json.loads(x) for x in (root / "allocator" / "policies.jsonl").read_text().splitlines()]
    for rec in lines:
        if rec["venue"] == "bybit-demo":
            for w in rec["policies"].values():
                w["basis"] = 0.2
    (root / "allocator" / "policies.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (root / "returns" / "basis.csv").write_text("date,ret_net\n2026-11-20,0.01\n2026-11-21,-0.02\n")
    h = pr.build(root, NOW, "2026-11-19")["history"]
    assert h["days"] == ["2026-11-20", "2026-11-21"] and h["missing_days"] == ["2026-11-19"]
    d1 = (0.5 * -0.0001 + 0.3 * 0.004 + 0.2 * 0.002 + 0.2 * 0.01) / 4  # core on 11-20, four books now
    assert h["core_pct"][0] == pytest.approx(d1 * 100, abs=1e-4)  # stored at 4 decimals
