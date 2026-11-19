"""alloc-v1.1: the decider's state gains regime, computed forecasts and recent news (exchange announcements and RSS).
Fixtures stand in for the prediction layer's files and the H7 collector's day files. No network, no paid calls."""

from __future__ import annotations

import asyncio
import json
import math
import time
from types import SimpleNamespace

import pytest

from ctb import allocator as A
from ctb import allocator_context as X
from ctb.kairos.allocator_questions import QUESTION_SET_VERSION, STATE_NOTE, build_questions
from ctb.kairos.client import KairosError, KairosResult, RateBudget

DAY = 86_400
DAY0 = 1_800_000_000 - 1_800_000_000 % DAY
NOW = DAY0 + 15 * 3600                      # 15:00 UTC
NOW_MS = NOW * 1000
IDS = ["carry", "trend"]
SIGMA = 0.02
BYBIT = "bybit-announcements"


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def jl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def day_name(ms):
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


class FakeDecider:
    def __init__(self, max_payload_bytes=None):
        self.calls = []
        self.spec = SimpleNamespace(max_questions=None, provider="ordecide", max_payload_bytes=max_payload_bytes)

    async def ask(self, state, questions, **kw):
        self.calls.append((json.loads(json.dumps(state)), questions))
        ans = {f"net_positive_{i}": {"type": "noul", "noul": 0.5} for i in IDS}
        ans["best_net_return"] = {"type": "choice", "choice": "none", "probabilities": {"carry": 0.2, "trend": 0.2, "none": 0.6}, "confidence": 0.3}
        return KairosResult(answers={q: a for q, a in ans.items() if q in questions}, usage={}, metadata={"request_id": "r1"},
                            latency_ms=1.0, attempts=1, request_bytes=10)


class Boom(FakeDecider):
    async def ask(self, state, questions, **kw):
        raise KairosError("transport", "down")


async def market():
    return {"major": {"BTC": {"ret_1d_pct": 1.0}}}


FEATURES = {"btc_trend": 0.31234567, "breadth": 0.62, "vol_rank": 0.4, "vol_ratio": -0.1, "dispersion": 1.1, "avg_corr": 0.55,
            "funding_level": 0.08, "funding_spread": 0.05}
REGIME = {"date": "2026-11-18", "state": "up_calm", "features": FEATURES,
          "p7": {"up_calm": 0.62, "up_wild": 0.2, "down_calm": 0.1, "down_wild": 0.08}}


def forecast_doc(**over):
    f = {"date": "2026-11-19", "made_ms": (NOW - 2 * 3600) * 1000, "kappa": 0.0,
         "strategies": {
             "trend": {"sigma_d": SIGMA, "mu": {"base": 0.001, "regime": 0.0002, "ridge": 0.0}},
             "carry": {"sigma_d": SIGMA, "mu": {"structural": -0.0002, "base": -0.0002, "regime": -0.0002, "ridge": -0.0002}}},
         "cov_d": {"ids": IDS, "matrix": [[SIGMA ** 2, 0.0], [0.0, SIGMA ** 2]]},
         "weights_k": {"trend": {"base": 0.3, "regime": 0.25, "ridge": 0.2, "decider": 0.25},
                       "carry": {"structural": 0.4, "base": 0.15, "regime": 0.1, "ridge": 0.1, "decider": 0.25}}}
    f.update(over)
    return f


def item(key, s, cat, pub_s, rx_s=None, sha=None, **kw):
    return {"k": "item", "key": f"{s}|{key}", "s": s, "id": key, "url": f"https://x/{key}", "cat": cat,
            "pub": None if pub_s is None else pub_s * 1000, "rx": (rx_s if rx_s is not None else (pub_s or NOW)) * 1000,
            "stale": False, "bf": False, "sha": sha or f"sha-{key}", **kw}


def raw(key, s, title, sha=None, **kw):
    return {"k": "raw", "key": f"{s}|{key}", "sha": sha or f"sha-{key}", "title": title, "text": "BODY NEVER SENT", "url": f"https://x/{key}", "cat": "x", **kw}


def collector(base, items, raws):
    """Write collector day files under base/research/h7/collector; returns that root."""
    root = base / "research" / "h7" / "collector"
    for it, r in zip(items, raws):
        jl(root / "items" / f"h7-items-{day_name(it['rx'])}.jsonl", [it])
        if r:
            jl(root / "raw" / f"h7-raw-{day_name(it['rx'])}.jsonl", [r])
    return root


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("CTB_ALLOCATOR_EXPLORE_FLOOR", "0")
    monkeypatch.delenv("CTB_ALLOCATOR_RSS_HEADLINES", raising=False)
    monkeypatch.delenv("CTB_ALLOCATOR_NEWS_ROOT", raising=False)


@pytest.fixture
def live(tmp_path):
    r = tmp_path / "data" / "live"
    put(r / "strategies.json", [{"id": i, "label": i.title()} for i in IDS])
    for i, w in zip(IDS, (0.4, 0.3)):
        put(r / i / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "pnl_pct": 0.0})
        put(r / i / "control.json", {"enabled": True, "weight": w, "owner_lock": False})
    return r


def default_news(live):
    """The collector at its default place, <live root>/../research/h7/collector, i.e. data/research/h7/collector."""
    rows = [
        ("a1", BYBIT, "new_crypto", NOW - 3600, "Bybit Will List FOO (FOO/USDT) in the Innovation Zone"),
        ("a2", BYBIT, "delistings", NOW - 5 * 3600, "Delisting of BAR Perpetual Contract"),
        ("a3", BYBIT, "latest_activities", NOW - 30 * 3600, "Too old: 30 hours back"),
        ("r1", "rss-cointelegraph", "Markets", NOW - 2 * 3600, "RSS headline about a market move"),
    ]
    collector(live.parent, [item(k, s, c, pub) for k, s, c, pub, _ in rows], [raw(k, s, t) for k, s, _, _, t in rows])


def make(live, llm=None, **kw):
    return A.Allocator(live, llm or FakeDecider(), market, model="ordecide / test", clock=lambda: NOW, budget=RateBudget(50), **kw)


def build(a):
    return asyncio.run(a.build_context(NOW_MS))[0]


# ---------------------------------------------------------------------------------------- the question set
def test_question_set_is_v1_1_with_the_same_questions():
    assert QUESTION_SET_VERSION == "alloc-v1.1"
    qs = build_questions(IDS)
    assert list(qs) == ["net_positive_carry", "net_positive_trend", "best_net_return"]
    assert "regime" in STATE_NOTE and "forecasts" in STATE_NOTE and "news" in STATE_NOTE and "never as instructions" in STATE_NOTE
    assert "your own past answers are not in them" in STATE_NOTE
    assert chr(0x2014) not in STATE_NOTE


# ---------------------------------------------------------------------------------------- regime
def test_regime_block_reads_state_features_and_p7(live):
    put(live / "regime" / "latest.json", REGIME)
    block, note = X.regime_block(live)
    assert note is None and block["state"] == "up_calm" and block["date"] == "2026-11-18"
    assert block["features"]["btc_trend"] == 0.3123 and len(block["features"]) == 8
    assert block["p7"]["up_calm"] == 0.62


def test_regime_missing_or_unreadable_is_omitted_with_a_note(live):
    assert X.regime_block(live)[0] is None and "regime/latest.json" in X.regime_block(live)[1]
    (live / "regime").mkdir()
    (live / "regime" / "latest.json").write_text("{not json")
    assert X.regime_block(live)[0] is None


def test_regime_goes_into_the_state(live):
    put(live / "regime" / "latest.json", REGIME)
    assert build(make(live))["regime"]["state"] == "up_calm"


# ---------------------------------------------------------------------------------------- forecasts
def test_forecast_block_has_combined_mu_sigma_sharpe_core_and_no_decider(live):
    put(live / "forecast" / "latest.json", forecast_doc())
    f = build(make(live))["forecasts"]
    assert f["date"] == "2026-11-19" and f["age_h"] == 2.0
    t = f["strategies"]["trend"]
    # trend: base 1e-3 (below the Sharpe cap 1.57e-3), regime 2e-4, ridge 0; weights .3 .25 .2 renormalised, the decider's 0.25 not used
    mu = (0.3 * 0.001 + 0.25 * 0.0002) / 0.75
    assert t["mu_pct_yr"] == pytest.approx(mu * 365 * 100, abs=0.01)
    assert t["sigma_pct_yr"] == pytest.approx(SIGMA * math.sqrt(365) * 100, abs=0.01)
    assert t["forecast_sharpe"] == pytest.approx(mu * math.sqrt(365) / SIGMA, abs=0.001)
    assert 0 < t["core_weight"] <= 1 and t["forecasters"] == ["base", "regime", "ridge"]
    assert f["strategies"]["carry"]["forecasters"] == ["base", "regime", "ridge", "structural"]


def test_forecast_block_ignores_a_decider_row_in_the_file(live):
    doc = forecast_doc()
    doc["strategies"]["trend"]["mu"]["decider"] = 0.01   # would swamp everything if it were fed back
    put(live / "forecast" / "latest.json", doc)
    t = build(make(live))["forecasts"]["strategies"]["trend"]
    assert t["mu_pct_yr"] < 30 and "decider" not in t["forecasters"]


def test_forecasts_do_not_depend_on_the_decider_history(live):
    put(live / "forecast" / "latest.json", forecast_doc())
    a = make(live)
    before = build(a)["forecasts"]
    a._push_p({"carry": 0.95, "trend": 0.05}, NOW_MS - 1000)
    assert build(a)["forecasts"] == before


def test_missing_forecast_is_omitted_with_a_state_note(live):
    ctx = build(make(live))
    assert "forecasts" not in ctx and any("forecasts omitted" in n for n in ctx["state_notes"])


def test_unreadable_forecast_is_omitted_with_a_state_note(live):
    put(live / "forecast" / "latest.json", {"nonsense": 1})
    ctx = build(make(live))
    assert "forecasts" not in ctx and any("forecasts omitted" in n for n in ctx["state_notes"])


def test_stale_forecast_is_kept_with_its_age_and_a_note(live):
    put(live / "forecast" / "latest.json", forecast_doc(made_ms=(NOW - 50 * 3600) * 1000))
    ctx = build(make(live))
    assert ctx["forecasts"]["age_h"] == 50.0 and any("old" in n for n in ctx["state_notes"])


# ---------------------------------------------------------------------------------------- news
def test_news_by_default_combines_bybit_and_rss_newest_first_within_24h(live):
    default_news(live)
    ctx = build(make(live))
    n = ctx["news"]
    assert n["window_h"] == 24 and n["count"] == 3
    assert [i["src"] for i in n["items"]] == ["bybit", "rss-cointelegraph", "bybit"]            # 1 h, 2 h, 5 h ago
    assert n["items"][0]["title"] == "Bybit Will List FOO (FOO/USDT) in the Innovation Zone"
    first = n["items"][0]
    assert first["type"] == "new_crypto" and first["t"] == X._utc((NOW - 3600) * 1000) and "tags" not in first   # the collector keeps no tags
    blob = json.dumps(ctx)
    assert "30 hours back" not in blob and "BODY NEVER SENT" not in blob
    assert "state_notes" not in ctx or not any("RSS" in x for x in ctx["state_notes"])


@pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
def test_rss_off_sends_bybit_announcements_only_and_says_so(live, monkeypatch, flag):
    default_news(live)
    monkeypatch.setenv("CTB_ALLOCATOR_RSS_HEADLINES", flag)
    ctx = build(make(live))
    assert ctx["news"]["count"] == 2 and {i["src"] for i in ctx["news"]["items"]} == {"bybit"}
    assert "RSS headline" not in json.dumps(ctx["news"]) and any("RSS headlines are switched off" in x for x in ctx["state_notes"])


@pytest.mark.parametrize("flag", ["1", "true", "ON"])
def test_rss_on_values(live, monkeypatch, flag):
    default_news(live)
    monkeypatch.setenv("CTB_ALLOCATOR_RSS_HEADLINES", flag)
    assert build(make(live))["news"]["count"] == 3


def test_news_is_capped_at_40_items_and_says_how_many_older_ones_are_left(tmp_path):
    its = [item(f"n{i}", BYBIT if i % 2 else "rss-decrypt", "new_crypto", NOW - 60 * (i + 1)) for i in range(55)]
    root = collector(tmp_path, its, [raw(f"n{i}", BYBIT if i % 2 else "rss-decrypt", f"Title {i}") for i in range(55)])
    block, _ = X.news_block(root, NOW_MS, rss=True)
    assert block["count"] == 40 and block["older_not_shown"] == 15
    assert block["items"][0]["title"] == "Title 0" and block["items"][-1]["title"] == "Title 39"


def test_news_tags_title_clean_up_and_clock_rules(tmp_path):
    its = [item("a", BYBIT, "new_crypto", NOW - 600, tags=["Spot", "USDT", 5, ""]),
           item("b", BYBIT, "new_crypto", None, rx_s=NOW - 1200),             # no pub time: the receive time
           item("c", BYBIT, "new_crypto", NOW + 3 * 3600, rx_s=NOW - 1800),   # pub in the future: the receive time
           item("d", BYBIT, "new_crypto", NOW - 900)]                         # no raw line: skipped and counted
    rs = [raw("a", BYBIT, "  Multi\n line\t\x00title  " + "x" * 400), raw("b", BYBIT, "B"), raw("c", BYBIT, "C"), None]
    root = collector(tmp_path, its, rs)
    block, notes = X.news_block(root, NOW_MS, rss=False)
    by = {i["title"][:1]: i for i in block["items"]}
    assert by["M"]["tags"] == ["Spot", "USDT"] and len(by["M"]["title"]) == X.TITLE_MAX and "\n" not in by["M"]["title"]
    assert by["B"]["t"] == X._utc((NOW - 1200) * 1000) and by["C"]["t"] == X._utc((NOW - 1800) * 1000)
    assert block["count"] == 3 and any("no stored title" in n for n in notes)


def test_news_item_is_joined_to_raw_by_key_and_sha(tmp_path):
    root = collector(tmp_path, [item("a", BYBIT, "new_crypto", NOW - 600, sha="s1")], [raw("a", BYBIT, "Revised later", sha="s2")])
    assert X.news_block(root, NOW_MS, rss=False)[0]["count"] == 0         # another text revision is not this item's title


def test_news_reads_yesterdays_file_across_midnight(tmp_path):
    t = DAY0 + 600                                                       # 00:10 UTC
    root = collector(tmp_path, [item("a", BYBIT, "new_crypto", t - 2 * 3600)], [raw("a", BYBIT, "Late night notice")])
    assert day_name((t - 2 * 3600) * 1000) != day_name(t * 1000)
    assert X.news_block(root, t * 1000, rss=False)[0]["items"][0]["title"] == "Late night notice"


def test_news_missing_files_are_omitted_with_a_state_note(live):
    ctx = build(make(live))
    assert "news" not in ctx and any("collector files are not available" in n for n in ctx["state_notes"])


def test_news_root_can_be_set(live, tmp_path, monkeypatch):
    root = collector(tmp_path / "elsewhere", [item("a", BYBIT, "new_crypto", NOW - 600)], [raw("a", BYBIT, "From the override")])
    monkeypatch.setenv("CTB_ALLOCATOR_NEWS_ROOT", str(root))
    assert build(make(live))["news"]["items"][0]["title"] == "From the override"


def test_news_garbage_lines_do_not_break_it(tmp_path):
    root = collector(tmp_path, [item("a", BYBIT, "new_crypto", NOW - 600)], [raw("a", BYBIT, "Fine")])
    with open(next((root / "items").iterdir()), "a") as f:
        f.write("{torn\n[1,2]\n")
    assert X.news_block(root, NOW_MS, rss=False)[0]["count"] == 1


# ---------------------------------------------------------------------------------------- payload bound and the log
def test_state_with_every_block_is_small_and_sizes_are_logged(live):
    put(live / "forecast" / "latest.json", forecast_doc())
    put(live / "regime" / "latest.json", REGIME)
    default_news(live)
    llm = FakeDecider(max_payload_bytes=1024 * 1024)
    rec = asyncio.run(make(live, llm).decide(trigger="cli"))
    assert rec["error"] is None and rec["qset"] == "alloc-v1.1"
    state, questions = llm.calls[0]
    assert {"regime", "forecasts", "news"} <= set(state) and "state_notes" not in state
    sizes = rec["state_bytes"]
    assert sizes["total"] == len(X.dumps(rec["inputs"]).encode())
    assert {"strategies", "market", "regime", "forecasts", "news", "total"} <= set(sizes)
    assert X.payload_bytes(state, questions) < 20_000
    last = json.loads((live / "allocator" / "decisions.jsonl").read_text().splitlines()[-1])
    assert last["state_bytes"] == sizes


def test_sizes_are_logged_when_the_model_fails_too(live):
    rec = asyncio.run(make(live, Boom()).decide(trigger="cli"))
    assert rec["error"] and rec["state_bytes"]["total"] > 0


def _big_news(tmp_path, monkeypatch, n=40, width=150):
    its = [item(f"n{i}", BYBIT, "new_crypto", NOW - 60 * (i + 1)) for i in range(n)]
    root = collector(tmp_path / "big", its, [raw(f"n{i}", BYBIT, f"Title {i} " + "y" * width) for i in range(n)])
    monkeypatch.setenv("CTB_ALLOCATOR_NEWS_ROOT", str(root))


def test_fit_state_cuts_news_first_then_whole_blocks(live, tmp_path, monkeypatch):
    put(live / "forecast" / "latest.json", forecast_doc())
    put(live / "regime" / "latest.json", REGIME)
    _big_news(tmp_path, monkeypatch)
    qs = build_questions(IDS)
    full = build(make(live))
    full_bytes = X.payload_bytes(full, qs)
    assert full["news"]["count"] == 40
    ctx = build(make(live))                          # a limit a little under the full request: news is halved until it fits
    notes = X.fit_state(ctx, [qs], full_bytes - 3000)
    assert 0 < ctx["news"]["count"] < 40 and X.payload_bytes(ctx, qs) <= full_bytes - 3000
    assert ctx["news"]["older_not_shown"] == 40 - ctx["news"]["count"] and "forecasts" in ctx and "regime" in ctx
    assert notes and any("news cut" in n for n in ctx["state_notes"])
    ctx = build(make(live))                          # a limit only the books and the market fit under: every optional block goes
    base = X.payload_bytes({k: ctx[k] for k in ("now_utc", "limits", "strategies", "market")}, qs)
    X.fit_state(ctx, [qs], base + 50)
    assert not ({"news", "forecasts", "regime"} & set(ctx)) and len([n for n in ctx["state_notes"] if "omitted" in n]) == 3
    ctx = build(make(live))                          # no limit known (a fake client): nothing is cut
    assert X.fit_state(ctx, [qs], None) == [] and ctx["news"]["count"] == 40


def test_the_allocator_cuts_the_state_before_asking(live, tmp_path, monkeypatch):
    _big_news(tmp_path, monkeypatch, n=40, width=140)
    qs = build_questions(IDS)
    base = X.payload_bytes(build(make(live)), qs)
    llm = FakeDecider(max_payload_bytes=base - 2500)
    rec = asyncio.run(make(live, llm).decide(trigger="cli"))
    state, questions = llm.calls[0]
    assert rec["error"] is None and X.payload_bytes(state, questions) <= base - 2500
    assert rec["inputs"]["news"]["count"] == state["news"]["count"] < 40         # the log shows what was sent
    assert rec["state_bytes"]["total"] == len(X.dumps(state).encode())


# ---------------------------------------------------------------------------------------- the paths
def test_the_computed_path_logs_state_bytes_and_still_sets_weights(live):
    put(live / "forecast" / "latest.json", forecast_doc())
    put(live / "regime" / "latest.json", REGIME)
    llm = FakeDecider()
    rec = asyncio.run(make(live, llm).decide(trigger="cli"))
    assert rec["weights_mode"] == "computed" and rec["error"] is None
    assert rec["state_bytes"]["forecasts"] > 0 and "forecasts" in llm.calls[0][0]


def test_the_decider_only_path_has_regime_and_news_but_a_note_for_the_missing_forecast(live):
    put(live / "regime" / "latest.json", REGIME)
    default_news(live)
    llm = FakeDecider()
    rec = asyncio.run(make(live, llm).decide(trigger="cli"))
    state = llm.calls[0][0]
    assert rec["error"] is None and rec.get("weights_mode") is None
    assert "regime" in state and "news" in state and "forecasts" not in state and any("forecasts omitted" in n for n in state["state_notes"])


def test_the_chat_fallback_state_is_unchanged(live):
    put(live / "forecast" / "latest.json", forecast_doc())
    put(live / "regime" / "latest.json", REGIME)
    default_news(live)
    assert set(build(make(live, mode="chat"))) == {"now_utc", "limits", "strategies", "market"}


def test_score_reads_both_question_sets(live):
    h = 24 * 3600_000
    t = 1_000_000_000_000
    lines = [{"t_ms": t + k * 3 * h, "qset": q, "error": None, "dry_run": False, "weights": {"carry": 1.0}, "answers": {"p": {"carry": 0.7}, "best": None},
              "inputs": {"strategies": {"carry": {"equity_usdt": 1000.0}}}} for k, q in enumerate(["alloc-v1.0", "alloc-v1.1", "alloc-v0.9"])]
    jl(live / "allocator" / "decisions.jsonl", lines)
    put(live / "allocator" / "state.json", {"peaks": {}, "history": {"carry": [[t + h, 1010.0], [t + 4 * h, 1010.0], [t + 7 * h, 1010.0]]}})
    assert A.score_decisions(live, now_ms=t + 20 * h)["strategies"]["carry"]["n"] == 2
    assert A.score_decisions(live, now_ms=t + 20 * h, qset="alloc-v1.1")["strategies"]["carry"]["n"] == 1
