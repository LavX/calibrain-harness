"""LLM strategy allocator: the language model decides which live strategies run, and how much.

Each live strategy (carry, trend, xsmom, ...) has a virtual book and reads <root>/<id>/control.json
every loop. This module is the only writer of the allocation part of those files ("enabled",
"weight", "set_by", "reason"), unless the owner locked a strategy ("owner_lock": true).

Files under the live root (CTB_LIVE_ROOT, default <repo>/data/live):
  strategies.json            written by the runner: [{"id", "label", ...}]
  <id>/status.json           written by the runner (equity, positions, targets, trades)
  <id>/trades.jsonl          written by the runner, one fill per line
  <id>/control.json          read by the runner; this module writes enabled/weight/set_by/reason
  allocator/control.json     written by the dashboard: {"enabled", "interval_min", "decide_now"}
  allocator/status.json      written here (see Allocator.status)
  allocator/decisions.jsonl  one line per decision (also risk-floor actions and failed attempts)
  allocator/state.json       equity samples and peaks, so P&L windows and drawdown survive restarts

Hard rules, enforced in code whatever the model says:
  - a weight moves at most max_step (0.5) per decision;
  - a strategy more than 15% below its book's peak gets weight 0 (checked every tick, even when the
    model is unreachable);
  - owner_lock is never overridden;
  - invalid model output is retried once, then the current allocation is kept and the error logged.

No orders are placed here. The only network calls are the model and Bybit public market data.

CLI:  python -m ctb.allocator once [--dry-run]   |   run   |   reset-peak [id ...]
Env:  CTB_LIVE_ROOT, CTB_ALLOCATOR_MODEL (default: the configured chat model, CTB_LLM_MODEL),
      CTB_ALLOCATOR_MAX_RPH (model requests per hour, default 12), CTB_ALLOCATOR_TREND_SYMBOLS.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

import aiohttp

from .config import ROOT
from .kairos.allocator_prompt import REASONING_MAX_CHARS, build_messages, build_schema, retry_messages
from .kairos.chat import _reasoning, parse_reply
from .kairos.client import KairosError, RateBudget, SystemOneClient
from .models import MODELS, ModelSpec, _OR_ID, OR_PREFIX

log = logging.getLogger("ctb.allocator")

MAINNET_REST = "https://api.bybit.com"
DEFAULT_TREND_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
                         "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
DAY_MS = 86_400_000
SAMPLE_EVERY_MS = 10 * 60_000       # equity samples kept for the P&L windows
HISTORY_MS = 8 * DAY_MS
ALLOCATOR_ID = "allocator"


def live_root() -> Path:
    return Path(os.environ.get("CTB_LIVE_ROOT") or ROOT / "data" / "live")


# ---------------------------------------------------------------------------------- file helpers
def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write-then-rename in the same directory, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"), default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def tail_jsonl(path: Path, n: int) -> list[dict]:
    out: deque[dict] = deque(maxlen=n)
    try:
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    out.append(row)
    except OSError:
        pass
    return list(out)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fnum(x: Any) -> float | None:
    """Like _num, but also reads Bybit's numeric strings."""
    try:
        return _num(float(x)) if not isinstance(x, bool) and x is not None else None
    except (TypeError, ValueError):
        return None


def _r(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


# ---------------------------------------------------------------------------------- the model
def allocator_spec(name: str | None = None) -> ModelSpec:
    """The chat model the allocator asks. Default: the harness's configured chat model ("llm",
    CTB_LLM_MODEL). "or:<openrouter-id>" picks any OpenRouter chat model. System One and decision
    models answer typed questions, not allocations, so they are refused."""
    name = (name or os.environ.get("CTB_ALLOCATOR_MODEL", "")).strip() or "llm"
    if name in MODELS:
        spec = MODELS[name]
    elif name.startswith(OR_PREFIX) and _OR_ID.match(name[len(OR_PREFIX):]):
        mid = name[len(OR_PREFIX):]
        spec = replace(MODELS["llm"], provider=name, label=mid.split("/", 1)[1].split(":")[0],
                       vendor=f"OpenRouter · {mid.split('/', 1)[0]}", model_env=None, default_model=mid)
    else:
        raise ValueError(f"unknown allocator model {name!r}; use llm or or:<openrouter-chat-model-id>")
    if spec.kind != "chat":
        raise ValueError(f"{name!r} is a decision model and cannot write an allocation; use a chat model (llm or or:<id>)")
    return spec


class AllocatorClient(SystemOneClient):
    """Chat completion with a strict JSON schema, on the same transport, key handling and retry
    policy as the rest of the harness (SystemOneClient._post)."""

    async def ask_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        key = self._key()
        if not key:
            raise KairosError("not_configured", f"{self.spec.key_env} is not set")
        reasoning = _reasoning()
        try:
            temperature = float(os.environ.get("CTB_LLM_TEMPERATURE", "0"))
            max_tokens = int(os.environ.get("CTB_LLM_MAX_TOKENS", 1500 if reasoning == {"enabled": False} else 8000))
        except ValueError as exc:
            raise KairosError("invalid_request", f"bad CTB_LLM_* setting: {exc}") from exc
        body = {
            "model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "reasoning": reasoning,
            "response_format": {"type": "json_schema", "json_schema": {"name": "allocation", "strict": True, "schema": schema}},
            "provider": {"require_parameters": True},
            "usage": {"include": True},  # OpenRouter reports the request cost in usage.cost
        }
        payload = json.dumps(body, separators=(",", ":"), allow_nan=False)
        data, _, _ = await self._post(payload, key)
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            err = data["error"]
            raise KairosError(f"upstream_{err.get('code', 'error')}", str(err.get("message", err))[:500])
        return parse_reply(data), (data.get("usage") if isinstance(data, dict) else None) or {}


# ---------------------------------------------------------------------------------- validation
class DecisionInvalid(ValueError):
    pass


@dataclass
class Decision:
    weights: dict[str, float]
    enabled: dict[str, bool]
    reasoning: str
    confidence: float


def validate_decision(raw: Any, ids: list[str]) -> Decision:
    """Schema, ranges and known ids. Too-long reasoning is cut (models miscount characters); anything
    else out of contract is rejected so the caller can retry."""
    if not isinstance(raw, dict):
        raise DecisionInvalid("answer is not a JSON object")
    out: dict[str, dict] = {}
    for key, typ in (("weights", float), ("enabled", bool)):
        m = raw.get(key)
        if not isinstance(m, dict):
            raise DecisionInvalid(f"{key} is missing or not an object")
        unknown = sorted(set(m) - set(ids))
        if unknown:
            raise DecisionInvalid(f"{key} has unknown ids {unknown}; known ids are {ids}")
        missing = [i for i in ids if i not in m]
        if missing:
            raise DecisionInvalid(f"{key} is missing ids {missing}")
        clean: dict[str, Any] = {}
        for i in ids:
            v = m[i]
            if typ is bool:
                if not isinstance(v, bool):
                    raise DecisionInvalid(f"enabled.{i} must be true or false, got {v!r}")
                clean[i] = v
            else:
                n = _num(v)
                if n is None or not 0.0 <= n <= 1.0:
                    raise DecisionInvalid(f"weights.{i} must be a number from 0 to 1, got {v!r}")
                clean[i] = n
        out[key] = clean
    reasoning = raw.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise DecisionInvalid("reasoning must be a non-empty string")
    conf = _num(raw.get("confidence"))
    if conf is None or not 0.0 <= conf <= 1.0:
        raise DecisionInvalid(f"confidence must be a number from 0 to 1, got {raw.get('confidence')!r}")
    return Decision(out["weights"], out["enabled"], reasoning.strip()[:REASONING_MAX_CHARS], conf)


# ---------------------------------------------------------------------------------- allocation rules
@dataclass
class Current:
    enabled: bool
    weight: float
    owner_lock: bool
    drawdown_pct: float | None


@dataclass
class Plan:
    enabled: dict[str, bool]
    weights: dict[str, float]
    floored: list[str]
    step_limited: list[str]
    skipped_owner_lock: list[str]


def resolve(current: dict[str, Current], proposal: Decision | None, *, max_step: float = 0.5, dd_floor_pct: float = 15.0) -> Plan:
    """The proposal after the hard rules. proposal None means "keep as is" (model unavailable): only
    the drawdown floor can still change anything."""
    enabled: dict[str, bool] = {}
    weights: dict[str, float] = {}
    floored: list[str] = []
    step_limited: list[str] = []
    locked: list[str] = []
    for i, c in current.items():
        want_w = c.weight if proposal is None else proposal.weights[i]
        want_e = c.enabled if proposal is None else proposal.enabled[i]
        w = min(c.weight + max_step, max(c.weight - max_step, want_w))
        if abs(w - want_w) > 1e-9:
            step_limited.append(i)
        e = want_e
        if c.drawdown_pct is not None and c.drawdown_pct > dd_floor_pct:
            w, e = 0.0, False
            floored.append(i)
        if c.owner_lock:
            locked.append(i)
            w, e = c.weight, c.enabled
        enabled[i], weights[i] = e, round(w, 4)
    return Plan(enabled, weights, [i for i in floored if i not in locked],
                [i for i in step_limited if i not in floored and i not in locked], locked)


# ---------------------------------------------------------------------------------- market summary
def _closes(rows: list[list[str]]) -> list[float]:
    """Bybit kline rows arrive newest first: [ts, open, high, low, close, ...]."""
    return [float(r[4]) for r in sorted(rows, key=lambda r: int(r[0]))]


def _ret(closes: list[float], back: int) -> float | None:
    if len(closes) <= back or closes[-1 - back] <= 0:
        return None
    return closes[-1] / closes[-1 - back] - 1


def _vol_ann(closes: list[float], periods: int, per_year: float) -> float | None:
    c = closes[-(periods + 1):]
    if len(c) < periods + 1 or min(c) <= 0:
        return None
    rets = [math.log(b / a) for a, b in zip(c, c[1:])]
    return statistics.pstdev(rets) * math.sqrt(per_year)


def _ema(closes: list[float], span: int) -> float | None:
    if len(closes) < span:
        return None
    k = 2 / (span + 1)
    e = sum(closes[:span]) / span
    for p in closes[span:]:
        e = p * k + e * (1 - k)
    return e


def summarize_market(daily: dict[str, list[float]], hourly: dict[str, list[float]], tickers: dict[str, dict],
                     funding_hist: dict[str, list[float]], trend_symbols: list[str]) -> dict[str, Any]:
    """Pure: compact market summary from closes and tickers. Returns are in %, vol is annualised %,
    funding is per 8h in basis points, with an annualised figure assuming 8h periods."""
    def pct(x: float | None) -> float | None:
        return None if x is None else round(100 * x, 2)

    major: dict[str, Any] = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        h, d, t = hourly.get(sym, []), daily.get(sym, []), tickers.get(sym, {})
        v7, v30 = _vol_ann(h, 168, 24 * 365), _vol_ann(d, 30, 365)
        rate, mark, index = _fnum(t.get("fundingRate")), _fnum(t.get("markPrice")), _fnum(t.get("indexPrice"))
        hist = funding_hist.get(sym) or []
        major[sym[:-4]] = {
            "price": _r(_fnum(t.get("lastPrice")) or (h[-1] if h else None), 2),
            "ret_1d_pct": pct(_ret(h, 24)), "ret_7d_pct": pct(_ret(h, 168)),
            "vol_7d_ann_pct": pct(v7), "vol_30d_ann_pct": pct(v30),
            "funding_bp_8h": None if rate is None else round(rate * 1e4, 3),
            "funding_ann_pct": None if rate is None else round(rate * 3 * 365 * 100, 2),
            "funding_7d_avg_ann_pct": None if not hist else round(statistics.fmean(hist) * 3 * 365 * 100, 2),
            "basis_pct": None if not (mark and index) else round(100 * (mark / index - 1), 3),
        }
    above, rets7, fund = [], {}, []
    for sym in trend_symbols:
        d = daily.get(sym, [])
        e50 = _ema(d, 50)
        if e50 is not None:
            above.append(d[-1] > e50)
        r7 = _ret(d, 7)
        if r7 is not None:
            rets7[sym] = r7
        f = _fnum(tickers.get(sym, {}).get("fundingRate"))
        if f is not None:
            fund.append(f)
    ranked = sorted(rets7.values())
    breadth: dict[str, Any] = {
        "universe": len(trend_symbols),
        "share_above_ema50": None if not above else round(sum(above) / len(above), 2),
        "dispersion_7d_std_pct": pct(statistics.pstdev(ranked)) if len(ranked) > 1 else None,
        "top_minus_bottom_7d_pct": pct(ranked[-1] - ranked[0]) if len(ranked) > 1 else None,
        "median_ret_7d_pct": pct(statistics.median(ranked)) if ranked else None,
        "median_funding_bp_8h": None if not fund else round(statistics.median(fund) * 1e4, 3),
    }
    return {"major": major, "trend_universe": breadth}


class BybitMarket:
    """Market summary from Bybit mainnet public REST (no keys). Partial failures are reported in
    "errors", and the summary is only refused when BTC data is missing altogether."""

    def __init__(self, session: aiohttp.ClientSession, trend_symbols: list[str] | None = None):
        self.session = session
        raw = os.environ.get("CTB_ALLOCATOR_TREND_SYMBOLS", "")
        self.trend_symbols = trend_symbols or [s.strip().upper() for s in raw.split(",") if s.strip()] or list(DEFAULT_TREND_SYMBOLS)

    async def _get(self, path: str, params: dict) -> dict:
        async with self.session.get(f"{MAINNET_REST}{path}", params=params, timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.json(content_type=None)
        if int(data.get("retCode", -1)) != 0:
            raise RuntimeError(f"{path} {data.get('retCode')} {data.get('retMsg')}")
        return data["result"]

    async def __call__(self) -> dict[str, Any]:
        errors: list[str] = []
        sem = asyncio.Semaphore(4)

        async def fetch(path: str, params: dict) -> dict | None:
            async with sem:
                try:
                    return await self._get(path, params)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{path} {params.get('symbol', '')}: {str(exc)[:120]}")
                    return None

        syms = list(dict.fromkeys(["BTCUSDT", "ETHUSDT", *self.trend_symbols]))
        jobs = {("d", s): fetch("/v5/market/kline", {"category": "linear", "symbol": s, "interval": "D", "limit": 120}) for s in syms}
        for s in ("BTCUSDT", "ETHUSDT"):
            jobs[("h", s)] = fetch("/v5/market/kline", {"category": "linear", "symbol": s, "interval": "60", "limit": 200})
            jobs[("f", s)] = fetch("/v5/market/funding/history", {"category": "linear", "symbol": s, "limit": 21})
        jobs[("t", "*")] = fetch("/v5/market/tickers", {"category": "linear"})
        done = dict(zip(jobs, await asyncio.gather(*jobs.values())))
        daily = {s: _closes(r["list"]) for (k, s), r in done.items() if k == "d" and r}
        hourly = {s: _closes(r["list"]) for (k, s), r in done.items() if k == "h" and r}
        funding = {s: [float(x["fundingRate"]) for x in r["list"]] for (k, s), r in done.items() if k == "f" and r}
        tick = done.get(("t", "*"))
        tickers = {t["symbol"]: t for t in (tick or {}).get("list", [])}
        if "BTCUSDT" not in daily or "BTCUSDT" not in hourly:
            raise RuntimeError("no BTC market data: " + "; ".join(errors[:3]))
        out = summarize_market(daily, hourly, tickers, funding, [s for s in self.trend_symbols if s in daily])
        if errors:
            out["errors"] = errors[:6]
        return out


# ---------------------------------------------------------------------------------- the allocator
class AllocatorError(Exception):
    pass


class Allocator:
    def __init__(self, root: Path | str, llm: Any, market: Callable[[], Awaitable[dict]], *, model: str,
                 budget: RateBudget | None = None, clock: Callable[[], float] = time.time,
                 default_interval_min: int = 60, max_step: float = 0.5, dd_floor_pct: float = 15.0,
                 flatten_on_floor: bool = True, tick_s: float = 30.0):
        self.root = Path(root)
        self.llm = llm
        self.market = market
        self.model = model
        self.budget = budget or RateBudget(12)
        self.clock = clock
        self.default_interval_min = default_interval_min
        self.max_step = max_step
        self.dd_floor_pct = dd_floor_pct
        self.flatten_on_floor = flatten_on_floor
        self.tick_s = tick_s
        self._stop = asyncio.Event()
        self._busy = asyncio.Lock()
        self.last_error: str | None = None
        self.state_cache: dict[str, Any] | None = None
        self._dirty = False
        self._seed_budget()

    # ---- paths and small readers
    @property
    def dir(self) -> Path:
        return self.root / ALLOCATOR_ID

    def now_ms(self) -> int:
        return int(self.clock() * 1000)

    def control(self) -> dict[str, Any]:
        c = read_json(self.dir / "control.json")
        c = c if isinstance(c, dict) else {}
        iv = _num(c.get("interval_min"))
        return {"enabled": c.get("enabled", True) is not False,
                "interval_min": int(iv) if iv and iv >= 1 else self.default_interval_min,
                "decide_now": bool(c.get("decide_now"))}

    def strategies(self) -> list[dict[str, Any]]:
        listed = read_json(self.root / "strategies.json")
        rows = [r for r in listed if isinstance(r, dict) and isinstance(r.get("id"), str)] if isinstance(listed, list) else []
        if not rows and self.root.is_dir():
            rows = [{"id": p.name, "label": p.name} for p in sorted(self.root.iterdir())
                    if p.is_dir() and p.name != ALLOCATOR_ID and (p / "status.json").exists()]
        return [r for r in rows if r["id"] != ALLOCATOR_ID]

    def _seed_budget(self) -> None:
        """The hourly request budget has to survive restarts: count the model attempts already logged."""
        cutoff = self.clock() - 3600
        for row in tail_jsonl(self.dir / "decisions.jsonl", 200):
            t = _num(row.get("t_ms"))
            if t and t / 1000 > cutoff:
                for _ in range(int(row.get("attempts") or 0)):
                    self.budget.calls.append(t / 1000)

    # ---- equity samples and peaks
    def _state(self) -> dict[str, Any]:
        if self.state_cache is None:
            s = read_json(self.dir / "state.json")
            self.state_cache = s if isinstance(s, dict) else {}
            self.state_cache.setdefault("peaks", {})
            self.state_cache.setdefault("history", {})
        return self.state_cache

    def observe(self, sid: str, equity: float | None, start_equity: float | None, now_ms: int) -> bool:
        """Track the peak and keep sparse equity samples. Returns True when the state changed."""
        if equity is None:
            return False
        st = self._state()
        changed = False
        peak = st["peaks"].get(sid)
        new_peak = max(equity, start_equity or 0.0, peak or 0.0)
        if peak is None or new_peak > peak:
            st["peaks"][sid] = new_peak
            changed = True
        hist = st["history"].setdefault(sid, [])
        if not hist or now_ms - hist[-1][0] >= SAMPLE_EVERY_MS:
            hist.append([now_ms, equity])
            st["history"][sid] = [p for p in hist if now_ms - p[0] <= HISTORY_MS]
            changed = True
        return changed

    def _pnl_window(self, sid: str, equity: float | None, window_ms: int, now_ms: int) -> float | None:
        if equity is None:
            return None
        older = [p for p in self._state()["history"].get(sid, []) if p[0] <= now_ms - window_ms]
        if not older or older[-1][1] <= 0:
            return None
        return _r(100 * (equity / older[-1][1] - 1), 2)

    def drawdown_pct(self, sid: str, status: dict[str, Any]) -> float | None:
        eq = _num(status.get("equity_usdt"))
        if eq is None:
            return None
        peak = max(self._state()["peaks"].get(sid) or 0.0, _num(status.get("start_equity_usdt")) or 0.0,
                   _num(status.get("peak_equity_usdt")) or 0.0, eq)
        return 100 * (peak - eq) / peak if peak > 0 else None

    # ---- context
    def strategy_context(self, row: dict[str, Any], now_ms: int) -> tuple[dict[str, Any], Current]:
        sid = row["id"]
        status = read_json(self.root / sid / "status.json")
        status = status if isinstance(status, dict) else {}
        ctl = read_json(self.root / sid / "control.json")
        ctl = ctl if isinstance(ctl, dict) else {}
        eq = _num(status.get("equity_usdt"))
        if self.observe(sid, eq, _num(status.get("start_equity_usdt")), now_ms):
            self._dirty = True
        dd = self.drawdown_pct(sid, status)
        w = _num(ctl.get("weight"))
        weight = 0.0 if w is None else min(1.0, max(0.0, w))  # nothing decided yet means off, as in the runner
        enabled = ctl.get("enabled", status.get("enabled", True)) is not False
        lock = ctl.get("owner_lock") is True
        trades = tail_jsonl(self.root / sid / "trades.jsonl", 5000)
        t_ms = [x for x in (_num(t.get("t_ms")) for t in trades) if x is not None]
        positions = [p for p in (status.get("positions") or []) if isinstance(p, dict)]
        targets = [t for t in (status.get("targets") or []) if isinstance(t, dict)]
        upd = _num(status.get("updated_ms"))
        hist = self._state()["history"].get(sid) or []
        ctx = {
            "label": row.get("label") or status.get("label") or sid,
            "state": status.get("state") or "unknown",
            "status_age_s": None if upd is None else max(0, round((now_ms - upd) / 1000)),
            "enabled": enabled, "weight": round(weight, 3), "owner_lock": lock, "set_by": ctl.get("set_by"),
            "equity_usdt": _r(eq, 2), "start_equity_usdt": _r(_num(status.get("start_equity_usdt")), 2),
            "pnl_1d_pct": self._pnl_window(sid, eq, DAY_MS, now_ms),
            "pnl_7d_pct": self._pnl_window(sid, eq, 7 * DAY_MS, now_ms),
            "pnl_since_start_pct": _r(_num(status.get("pnl_pct")), 2),
            "history_hours": round((now_ms - hist[0][0]) / 3_600_000, 1) if hist else 0,
            "drawdown_pct": _r(dd, 2),
            "fees_usdt": _r(_num(status.get("fees_usdt")), 2), "funding_usdt": _r(_num(status.get("funding_usdt")), 2),
            "positions": {"count": len(positions),
                          "gross_notional_usdt": _r(sum(abs((_num(p.get("qty")) or 0) * (_num(p.get("mark_price")) or _num(p.get("entry_price")) or 0)) for p in positions), 0),
                          "upnl_usdt": _r(sum(_num(p.get("upnl_usdt")) or 0 for p in positions), 2),
                          "top": [{"symbol": p.get("symbol"), "side": p.get("side"), "market": p.get("market")} for p in positions[:6]]},
            "targets": {"count": len(targets), "symbols": [t.get("symbol") for t in targets[:8]]},
            "trades_24h": sum(1 for t in t_ms if t >= now_ms - DAY_MS),
            "trades_7d": sum(1 for t in t_ms if t >= now_ms - 7 * DAY_MS),
            "last_error": status.get("last_error"),
        }
        return ctx, Current(enabled=enabled, weight=weight, owner_lock=lock, drawdown_pct=dd)

    async def build_context(self, now_ms: int) -> tuple[dict[str, Any], dict[str, Current], list[dict[str, Any]]]:
        rows = self.strategies()
        if not rows:
            raise AllocatorError(f"no strategies found under {self.root}")
        strategies, current = {}, {}
        for r in rows:
            strategies[r["id"]], current[r["id"]] = self.strategy_context(r, now_ms)
        try:
            market = await self.market()
        except Exception as exc:  # noqa: BLE001
            raise AllocatorError(f"market data unavailable: {str(exc)[:200]}") from exc
        ctx = {"now_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime(now_ms / 1000)),
               "limits": {"max_weight_step": self.max_step, "drawdown_floor_pct": self.dd_floor_pct},
               "strategies": strategies, "market": market}
        return ctx, current, rows

    # ---- the model
    async def ask(self, ctx: dict[str, Any], ids: list[str], labels: dict[str, str]) -> tuple[Decision, float, int, Any]:
        """One validated decision: a second attempt on invalid output, then give up. Returns
        (decision, cost_usd, attempts, last_raw_reply)."""
        messages = build_messages(ctx, ids, labels)
        schema = build_schema(ids)
        cost, problem = 0.0, ""
        for attempt in (1, 2):
            if not self.budget.try_take():
                raise AllocatorError(f"hourly model request budget exhausted ({self.budget.per_hour}/h)")
            raw: Any = None
            try:
                raw, usage = await self.llm.ask_json(messages, schema)
                cost += _num(usage.get("cost")) or 0.0
                return validate_decision(raw, ids), cost, attempt, raw
            except DecisionInvalid as exc:
                problem = str(exc)
            except KairosError as exc:
                if exc.kind != "malformed_response":
                    raise AllocatorError(str(exc)[:300]) from exc
                problem = exc.message
            log.warning("allocator: invalid model output (attempt %d): %s", attempt, problem)
            messages = retry_messages(messages, raw, problem)
        raise AllocatorError(f"invalid model output after one retry: {problem}")

    # ---- applying
    def _write_control(self, sid: str, enabled: bool, weight: float, reason: str, flatten_on_zero: bool, now_ms: int) -> bool:
        """Read-modify-write one strategy's control.json. Returns False when the owner locked it
        (checked again here, right before the write)."""
        path = self.root / sid / "control.json"
        ctl = read_json(path)
        ctl = ctl if isinstance(ctl, dict) else {}
        if ctl.get("owner_lock") is True:
            return False
        ctl.update({"enabled": enabled, "weight": weight, "set_by": "llm", "reason": reason[:REASONING_MAX_CHARS],
                    "updated_ms": now_ms})
        if flatten_on_zero:
            ctl["flatten_on_zero"] = True
        write_json_atomic(path, ctl)
        return True

    def _apply(self, plan: Plan, current: dict[str, Current], reasons: dict[str, str], now_ms: int) -> dict[str, bool]:
        applied: dict[str, bool] = {}
        for sid, c in current.items():
            if sid in plan.skipped_owner_lock:
                applied[sid] = False
                continue
            unchanged = plan.enabled[sid] == c.enabled and abs(plan.weights[sid] - c.weight) < 1e-9
            if unchanged:
                applied[sid] = False  # nothing to tell the runner; keep control.json (and its reason) as is
                continue
            applied[sid] = self._write_control(sid, plan.enabled[sid], plan.weights[sid], reasons[sid],
                                               self.flatten_on_floor and sid in plan.floored, now_ms)
        return applied

    def _save_state(self) -> None:
        self._dirty = False
        write_json_atomic(self.dir / "state.json", self._state())

    # ---- one decision
    async def decide(self, *, dry_run: bool = False, trigger: str = "schedule") -> dict[str, Any]:
        """Build the context, ask the model, apply the rules and write the controls. With dry_run
        nothing is written: the record comes back (context, answer, plan) and that is all."""
        async with self._busy:
            now_ms = self.now_ms()
            record: dict[str, Any] = {"t_ms": now_ms, "model": self.model, "trigger": trigger}
            current: dict[str, Current] = {}
            error: str | None = None
            decision: Decision | None = None
            try:
                ctx, current, rows = await self.build_context(now_ms)
                record["inputs"] = ctx
                ids = [r["id"] for r in rows]
                decision, cost, attempts, _ = await self.ask(ctx, ids, {r["id"]: str(r.get("label") or r["id"]) for r in rows})
                record.update(cost_usd=round(cost, 6), attempts=attempts)
            except AllocatorError as exc:
                error = str(exc)
                if not current:  # failed before the strategies were read: nothing to apply, not even the floor
                    current = self._read_current(now_ms)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                current = current or self._read_current(now_ms)
            plan = resolve(current, decision, max_step=self.max_step, dd_floor_pct=self.dd_floor_pct) if current else None
            reasons = {}
            if plan:
                for sid in current:
                    if sid in plan.floored:
                        dd = current[sid].drawdown_pct
                        reasons[sid] = f"risk floor: book {dd:.1f}% below its peak, weight forced to 0"
                    else:
                        reasons[sid] = decision.reasoning if decision else "kept (allocator error)"
            applied: dict[str, bool] = {}
            if plan and not dry_run:
                applied = self._apply(plan, current, reasons, now_ms)
            record.update({
                "weights": plan.weights if plan else {}, "enabled": plan.enabled if plan else {},
                "reasoning": decision.reasoning if decision else "", "confidence": decision.confidence if decision else None,
                "requested": {"weights": decision.weights, "enabled": decision.enabled} if decision else None,
                "applied": applied, "skipped_owner_lock": plan.skipped_owner_lock if plan else [],
                "risk_floor": plan.floored if plan else [], "step_limited": plan.step_limited if plan else [],
                "error": error, "dry_run": dry_run,
            })
            if not dry_run:
                append_jsonl(self.dir / "decisions.jsonl", record)
                self.last_error = error
                self._save_state()
                self.write_status(now_ms, last_decision_ms=now_ms)
            return record

    def _read_current(self, now_ms: int) -> dict[str, Current]:
        out: dict[str, Current] = {}
        for r in self.strategies():
            out[r["id"]] = self.strategy_context(r, now_ms)[1]
        return out

    async def risk_check(self) -> dict[str, Any] | None:
        """The drawdown floor on its own, no model: runs every tick so a bad book is cut within
        seconds of the runner reporting it, not at the next hourly decision."""
        async with self._busy:
            now_ms = self.now_ms()
            current = self._read_current(now_ms)
            plan = resolve(current, None, max_step=self.max_step, dd_floor_pct=self.dd_floor_pct)
            todo = [s for s in plan.floored if current[s].weight > 0 or current[s].enabled]
            if not todo:
                if self._dirty:
                    self._save_state()
                return None
            reasons = {s: f"risk floor: book {current[s].drawdown_pct:.1f}% below its peak, weight forced to 0" for s in todo}
            partial = Plan({s: plan.enabled[s] for s in todo}, {s: plan.weights[s] for s in todo}, todo, [], [])
            applied = self._apply(partial, {s: current[s] for s in todo}, reasons, now_ms)
            record = {"t_ms": now_ms, "model": "risk-floor", "trigger": "risk", "weights": plan.weights, "enabled": plan.enabled,
                      "reasoning": "; ".join(reasons.values()), "inputs": {"drawdown_pct": {s: _r(current[s].drawdown_pct, 2) for s in todo}},
                      "applied": applied, "skipped_owner_lock": plan.skipped_owner_lock, "risk_floor": todo, "attempts": 0}
            append_jsonl(self.dir / "decisions.jsonl", record)
            self._save_state()
            return record

    # ---- status
    def cost_usd_today(self, now_ms: int) -> float:
        day0 = now_ms - now_ms % DAY_MS
        return round(sum(_num(r.get("cost_usd")) or 0.0 for r in tail_jsonl(self.dir / "decisions.jsonl", 500)
                         if (_num(r.get("t_ms")) or 0) >= day0), 6)

    def status(self) -> dict[str, Any]:
        """What the dashboard shows: the last written status.json, with the cost refreshed."""
        s = read_json(self.dir / "status.json")
        return s if isinstance(s, dict) else {"state": "paused", "model": self.model, "last_decision_ms": None,
                                              "next_decision_ms": None, "last_error": None, "cost_usd_today": 0.0}

    def decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest first."""
        return list(reversed(tail_jsonl(self.dir / "decisions.jsonl", max(1, min(limit, 500)))))

    def write_status(self, now_ms: int, *, last_decision_ms: int | None = None, paused: bool = False) -> None:
        ctl = self.control()
        prev = read_json(self.dir / "status.json")
        prev = prev if isinstance(prev, dict) else {}
        last = last_decision_ms or prev.get("last_decision_ms")
        nxt = None if paused or not last else int(last) + ctl["interval_min"] * 60_000
        last_dec = (self.decisions(1) or [{}])[0]
        write_json_atomic(self.dir / "status.json", {
            "state": "paused" if paused else ("error" if self.last_error else "running"),
            "model": self.model, "last_decision_ms": last, "next_decision_ms": nxt,
            "last_error": self.last_error, "cost_usd_today": self.cost_usd_today(now_ms),
            "interval_min": ctl["interval_min"], "updated_ms": now_ms,
            "last_reasoning": last_dec.get("reasoning") or None, "last_confidence": last_dec.get("confidence"),
        })

    # ---- loop
    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """The loop the dashboard server starts as a task. Wakes every tick_s: the drawdown floor
        always runs; a model decision runs when the interval has passed or decide_now is set; nothing
        changes while allocator/control.json says enabled false."""
        log.info("allocator running: model %s, root %s", self.model, self.root)
        while not self._stop.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                log.exception("allocator tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), self.tick_s)
            except asyncio.TimeoutError:
                pass

    async def _tick(self) -> None:
        now_ms = self.now_ms()
        ctl = self.control()
        if not ctl["enabled"]:
            self.write_status(now_ms, paused=True)
            return
        await self.risk_check()
        st = read_json(self.dir / "status.json") or {}
        last = _num(st.get("last_decision_ms"))
        due = last is None or now_ms >= last + ctl["interval_min"] * 60_000
        if ctl["decide_now"]:
            raw = read_json(self.dir / "control.json")
            raw = raw if isinstance(raw, dict) else {}
            raw["decide_now"] = False
            write_json_atomic(self.dir / "control.json", raw)
            await self.decide(trigger="decide_now")
        elif due:
            await self.decide(trigger="schedule")
        else:
            self.write_status(now_ms)

    def reset_peak(self, ids: list[str] | None = None) -> list[str]:
        """Forget a book's peak (and samples) so the drawdown floor measures from now. Owner action,
        for a strategy that was cut by the floor and is being restarted on purpose."""
        st = self._state()
        done = []
        for r in self.strategies():
            if ids and r["id"] not in ids:
                continue
            if st["peaks"].pop(r["id"], None) is not None:
                done.append(r["id"])
            st["history"].pop(r["id"], None)
        self._save_state()
        return done


# ---------------------------------------------------------------------------------- wiring and CLI
def create_allocator(session: aiohttp.ClientSession, root: Path | str | None = None, spec: ModelSpec | None = None) -> Allocator:
    """What the dashboard server calls: an Allocator on the harness's aiohttp session."""
    spec = spec or allocator_spec()
    try:
        rph = int(os.environ.get("CTB_ALLOCATOR_MAX_RPH", "12"))
    except ValueError:
        rph = 12
    llm = AllocatorClient(spec, session, timeout_s=float(os.environ.get("CTB_KAIROS_TIMEOUT_S", "120")))
    return Allocator(root or live_root(), llm, BybitMarket(session), model=f"{spec.model}", budget=RateBudget(rph))


def _jprint(obj: Any) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


async def _main(args: argparse.Namespace) -> int:
    from .config import load_dotenv
    load_dotenv()
    async with aiohttp.ClientSession() as session:
        alloc = create_allocator(session, args.root)
        if args.cmd == "reset-peak":
            print("peaks reset:", ", ".join(alloc.reset_peak(args.ids or None)) or "none")
            return 0
        if args.cmd == "once":
            if not alloc.llm.configured:
                print(f"{alloc.llm.spec.key_env} is not set", file=sys.stderr)
                return 2
            rec = await alloc.decide(dry_run=args.dry_run, trigger="cli")
            if args.dry_run:
                print("== CONTEXT given to the model ==")
                _jprint(rec.get("inputs"))
                print("== MODEL ANSWER ==")
                _jprint(rec.get("requested") and {**rec["requested"], "reasoning": rec["reasoning"], "confidence": rec["confidence"]})
                print("== AFTER THE RULES (not applied) ==")
                _jprint({k: rec[k] for k in ("weights", "enabled", "risk_floor", "step_limited", "skipped_owner_lock", "cost_usd", "error") if k in rec})
            else:
                _jprint({k: v for k, v in rec.items() if k != "inputs"})
            return 1 if rec.get("error") else 0
        if not alloc.llm.configured:
            print(f"{alloc.llm.spec.key_env} is not set", file=sys.stderr)
            return 2
        await alloc.run()
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m ctb.allocator", description="LLM strategy allocator for the live strategies")
    p.add_argument("--root", help="live data root (default: CTB_LIVE_ROOT or data/live)")
    sub = p.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("once", help="one decision now")
    one.add_argument("--dry-run", action="store_true", help="print the context and the model's answer, write nothing")
    sub.add_parser("run", help="decide every interval until stopped")
    rp = sub.add_parser("reset-peak", help="restart a book's drawdown measurement from now")
    rp.add_argument("ids", nargs="*")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
