"""LLM strategy allocator: the decision model decides which live strategies run, and how much.

Default: the harness's decision model (CTB_MODEL_PROVIDER when it is a decision spec: kairos, jev,
ordecide, or:<id>; else ordecide) answers the alloc-v1.1 question set (ctb/kairos/allocator_questions.py):
per strategy "is its net book return over the next 24 h positive" plus "which strategy, or none, earns
the most". Code maps the probabilities to a "mapping" (weight = clip((p - 0.5) / 0.15, 0, 1), the
categorical winner lifted to 0.5 when its own p > 0.55) that is logged; the weights themselves come from the
computed layer below, never from the mapping. Fallback: a chat model (CTB_ALLOCATOR_MODEL=llm or or:<id>) that
answers one JSON allocation directly.

Computed weights (ctb/allocator_math.py): when <root>/forecast/latest.json exists (written daily by the
prediction layer) the decider is only one forecaster. It is still asked every hour, p is the mean of its
last 24 answers, and the weights are SET once a day at 23:50 UTC and on DECIDE NOW: core plus tilt (the
static inverse-volatility mix, tilted by at most +-T by each strategy's forecast Sharpe, with the tilt room kappa 0 until
the shadow tilt has earned it, so live weights start as the core), then the vol cap,
buffer, step, decider veto, exploration floor, the drawdown floor and owner_lock. A forecast that is missing, unreadable
or older than 36 h holds the last weights ("forecast missing: holding") and reports the error in status.json and the
feed: the decider alone never sets weights. Risk reductions (the drawdown floor) still apply while held.

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
  allocator/state.json       equity samples and peaks, so P&L windows and drawdown survive restarts, the
                             decider's last answers (p_hist) and when weights were last set
  forecast/latest.json       written by the prediction layer (read only here)
  regime/latest.json         the same (read only; shown in the one-line reasons and given to the decider)
  ../research/h7/collector   the H7 collector's item and raw day files (read only): the last 24 h of Bybit announcements
                             and RSS headlines go into the decider's state (ctb/allocator_context.py; RSS is on by
                             default, owner decision 2026-11-19, set CTB_ALLOCATOR_RSS_HEADLINES=0 for Bybit only;
                             the path can be set with CTB_ALLOCATOR_NEWS_ROOT)

Hard rules, enforced in code whatever the model says:
  - a weight moves at most max_step (0.5) per decision;
  - a strategy more than 15% below its book's peak gets weight 0 (checked every tick, even when the
    model is unreachable);
  - owner_lock is never overridden;
  - invalid model output is retried once, then the current allocation is kept and the error logged.
(The step limit, the floor and owner_lock apply after the probability mapping as well.)

Venues (CTB_LIVE_ROOTS, ctb/venues.py): the decision is made once per STRATEGY (core weight, shadow tilt, decider p,
veto, regime, forecasts are venue-independent; the model sees the allocator venue's books, bybit-demo) and applied to
every venue the allocator manages (CTB_ALLOCATOR_VENUES, default all configured venues) whose strategies.json lists
that strategy; a strategy a venue's runner does not list is never written there. Each venue is its own account, so the
risk rules run per venue: the portfolio vol cap over that venue's strategies, the drawdown floor per (venue, strategy)
book, the daily step-up from a day anchor per (venue, strategy), the exploration floor per (venue, strategy), and every
write's owner_lock, flatten and pause checks. State keys are "<venue>/<strategy>" (older files are migrated on load:
the bybit-only keys become "bybit-demo/<id>", values unchanged). The decision record's top-level fields stay the
allocator venue's; "venues" holds, per venue, what was written or skipped and why.

No orders are placed here. The only network calls are the model and Bybit public market data.

CLI:  python -m ctb.allocator once [--dry-run]   |   run   |   score   |   reset-peak [id ...]
Env:  CTB_LIVE_ROOT, CTB_ALLOCATOR_MODEL: kairos | jev | ordecide | ordecide:<id> (decision models) or
      llm | or:<id> (chat fallback); default CTB_MODEL_PROVIDER when that is a decision spec, else ordecide.
      CTB_ALLOCATOR_MAX_RPH (model requests per hour, default 12), CTB_ALLOCATOR_TREND_SYMBOLS,
      CTB_ALLOCATOR_P_ZERO / _P_FULL / _WINNER_MIN_P / _WINNER_WEIGHT (the mapping, see ctb/config.py).
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

from . import allocator_context as X
from . import allocator_math as M
from .config import ROOT, Settings
from . import strategies as S
from .control_lock import locked
from .kairos.allocator_prompt import REASONING_MAX_CHARS, build_messages, build_schema, retry_messages
from .kairos.allocator_questions import (BEST_QID, NONE_ID, QUESTION_SET_VERSION, SCORED_QSETS, MappingParams, batches,
                                         build_questions, map_answers, pos_qid)
from .kairos.chat import _reasoning, parse_reply
from .kairos.client import KairosError, RateBudget, SystemOneClient, make_client
from .kairos.parse import validate
from .models import MODELS, ModelSpec, _OR_ID, OR_PREFIX, get as get_model

log = logging.getLogger("ctb.allocator")

MAINNET_REST = "https://api.bybit.com"
DEFAULT_TREND_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
                         "BNBUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
DAY_MS = 86_400_000
SAMPLE_EVERY_MS = 10 * 60_000       # equity samples kept for the P&L windows
HISTORY_MS = 8 * DAY_MS
ALLOCATOR_ID = "allocator"
PRIMARY_VENUE = "bybit-demo"  # the venue whose root holds the allocator's own files and whose books the model sees
STATE_KEYS_VERSION = 2          # state.json keys peaks/history/day_anchor per "<venue>/<strategy>"


@dataclass(frozen=True)
class AllocVenue:
    """One venue the allocator applies its per-strategy decision to. managed False: computed and logged, never written."""
    id: str
    root: Path
    label: str
    managed: bool = True


def live_root() -> Path:
    return Path(os.environ.get("CTB_LIVE_ROOT") or ROOT / "data" / "live")


# ---------------------------------------------------------------------------------- file helpers
def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _join_errors(*parts: str | None) -> str | None:
    return "; ".join(p for p in parts if p) or None


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
    """The model the allocator asks.

    Decision models (default): kairos, jev, ordecide, or ordecide:<openrouter-id>. With no name the
    harness's CTB_MODEL_PROVIDER is used when it is a decision spec, else ordecide.
    Chat fallback: llm (CTB_LLM_MODEL) or or:<openrouter-chat-model-id>."""
    name = (name or os.environ.get("CTB_ALLOCATOR_MODEL", "")).strip()
    if not name:
        configured = os.environ.get("CTB_MODEL_PROVIDER", "kairos").strip().lower()
        try:
            spec = get_model(configured)
        except ValueError:
            spec = None
        return spec if spec is not None and spec.kind == "systemone" else MODELS["ordecide"]
    if name.startswith("ordecide:") and _OR_ID.match(name[len("ordecide:"):]):
        return get_model(OR_PREFIX + name[len("ordecide:"):])
    if name in MODELS:
        return MODELS[name]
    if name.startswith(OR_PREFIX) and _OR_ID.match(name[len(OR_PREFIX):]):
        mid = name[len(OR_PREFIX):]
        return replace(MODELS["llm"], provider=name, label=mid.split("/", 1)[1].split(":")[0],
                       vendor=f"OpenRouter · {mid.split('/', 1)[0]}", model_env=None, default_model=mid)
    raise ValueError(f"unknown allocator model {name!r}; use kairos, jev, ordecide, ordecide:<id>, llm or or:<chat-model-id>")


def model_display(spec: ModelSpec) -> str:
    """"provider / model" as the dashboard shows it, e.g. "ordecide / google/gemini-decision-1"."""
    provider = ("ordecide" if spec.kind == "systemone" else "chat") if spec.provider.startswith(OR_PREFIX) else spec.provider
    return f"{provider} / {spec.model}"


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
    confidence: float | None
    derive_enabled: tuple[str, ...] = ()  # ids whose enabled flag follows their final weight (decision model mapping)
    # ids the decision model left unanswered (decider-only mapping): their weights are the decision venue's own allocation,
    # kept; every other venue keeps its own allocation for them too
    held: tuple[str, ...] = ()


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
    gross: float | None = None   # gross exposure per unit of capital at weight 1 (None: unknown, 1 is used)


@dataclass
class Plan:
    enabled: dict[str, bool]
    weights: dict[str, float]
    floored: list[str]
    step_limited: list[str]
    skipped_owner_lock: list[str]


def apply_explore_floor(plan: Plan, current: dict[str, Current], floor: float) -> list[str]:
    """Decider-only mode on demo: every strategy keeps at least `floor` so it keeps producing fills.
    Never applied to a drawdown-floored or owner-locked strategy. Returns the ids that were lifted."""
    lifted: list[str] = []
    if floor <= 0:
        return lifted
    for sid, c in current.items():
        if sid in plan.floored or c.owner_lock or sid not in plan.weights:
            continue
        if plan.weights[sid] < floor:
            plan.weights[sid] = floor
            plan.enabled[sid] = True
            lifted.append(sid)
    return lifted


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
        if proposal is not None and i in proposal.derive_enabled:
            e = w > 0  # enabled = weight > 0, after the step limit: a book stepping down to 0.5 stays on
        if c.drawdown_pct is not None and c.drawdown_pct > dd_floor_pct:
            w, e = 0.0, False
            floored.append(i)
        if c.owner_lock:
            locked.append(i)
            w, e = c.weight, c.enabled
        enabled[i], weights[i] = e, round(w, 4)
    return Plan(enabled, weights, [i for i in floored if i not in locked],
                [i for i in step_limited if i not in floored and i not in locked], locked)


def gross_per_unit(positions: list[dict[str, Any]], equity: float | None, weight: float, min_weight: float = 0.10) -> float | None:
    """G_s: the book's gross exposure per unit of capital at weight 1. status.json positions over equity,
    divided by the weight, because the runner scales the book's targets by it. Below min_weight the
    scaling is too noisy to undo, and with no positions there is nothing to measure: None (caller uses 1)."""
    if equity is None or equity <= 0 or weight < min_weight or not positions:
        return None
    gross = sum(abs((_num(p.get("qty")) or 0.0) * (_num(p.get("mark_price")) or _num(p.get("entry_price")) or 0.0)) for p in positions)
    return gross / equity / weight if gross > 0 else None


# ---------------------------------------------------------------------------------- computed layer: settings and files
@dataclass(frozen=True)
class Calc:
    """The constants of the computed weights, from ctb/config.py Settings (tests pass their own)."""
    gamma: float = 2.0
    horizon_days: float = 7.0
    cost_bp: float = 10.0
    carry_cost_bp: float = 16.0
    sharpe_cap: float = 1.5
    decider_k: float = 0.8
    p_window: int = 24
    stale_h: float = 36.0
    weights_utc: tuple[int, int] = (23, 50)
    rules: M.RuleParams = M.RuleParams()
    tilt: M.TiltParams = M.TiltParams()
    start_weights: Any = None   # {"default": {...}, "<id>": {...}}

    @staticmethod
    def from_settings(s: Settings) -> "Calc":
        from .config import _venue_name
        try:
            hh, mm = (int(x) for x in s.allocator_weights_utc.split(":"))
            if not (0 <= hh < 24 and 0 <= mm < 60):
                raise ValueError
        except ValueError:
            hh, mm = 23, 50
        try:
            floor = s.allocator_explore_floor if _venue_name() in s.allocator_explore_venues else 0.0
        except Exception:  # noqa: BLE001 - an unreadable venue setting must not stop the allocator
            floor = 0.0
        return Calc(gamma=s.allocator_gamma, horizon_days=s.allocator_horizon_days, cost_bp=s.allocator_cost_bp,
                    carry_cost_bp=s.allocator_carry_cost_bp, sharpe_cap=s.allocator_sharpe_cap, decider_k=s.allocator_decider_k,
                    p_window=max(1, s.allocator_p_window), stale_h=s.allocator_stale_h, weights_utc=(hh, mm),
                    rules=M.RuleParams(vol_cap=s.allocator_vol_cap, buffer=s.allocator_buffer, step_up=s.allocator_step_up,
                                       veto_p=s.allocator_veto_p, veto_factor=s.allocator_veto_factor, explore_floor=floor),
                    tilt=M.TiltParams(sr_full=s.allocator_tilt_sr_full, t_max=s.allocator_tilt_max),
                    start_weights=s.allocator_start_weights)

    def cost(self, sid: str) -> float:
        """c_s as a return fraction per side."""
        return (self.carry_cost_bp if sid == "carry" else self.cost_bp) / 1e4

    def start(self, sid: str) -> dict[str, float]:
        sw = self.start_weights or {}
        return dict(sw.get(sid) or sw.get("default") or {})


@dataclass
class Forecast:
    state: str                     # "missing" | "invalid" | "stale" | "fresh"
    data: dict[str, Any] | None = None
    date: str | None = None
    made_ms: int | None = None
    age_h: float | None = None
    note: str = ""

    @property
    def error(self) -> str | None:
        """What the status shows as an error: a forecast that is missing or cannot be used (the weights are held)."""
        return self.note if self.state in ("missing", "invalid", "stale") else None

    def info(self) -> dict[str, Any]:
        return {"state": self.state, "date": self.date, "made_ms": self.made_ms,
                "age_h": None if self.age_h is None else round(self.age_h, 2), "note": self.note}


def read_forecast(root: Path, now_ms: int, stale_h: float) -> Forecast:
    path = root / "forecast" / "latest.json"
    if not path.exists():
        return Forecast("missing", note="forecast missing: holding the last weights (forecast/latest.json not found; risk cuts still apply)")
    raw = read_json(path)
    made = _num(raw.get("made_ms")) if isinstance(raw, dict) else None
    if not isinstance(raw, dict) or not isinstance(raw.get("strategies"), dict) or made is None:
        return Forecast("invalid", note="forecast invalid: holding the last weights (forecast/latest.json unreadable or incomplete; risk cuts still apply)")
    age = max(0.0, (now_ms - made) / 3_600_000)
    date = raw.get("date") if isinstance(raw.get("date"), str) else None
    if age > stale_h:
        return Forecast("stale", raw, date, int(made), age, f"forecast stale: holding the last weights (it is {age:.1f} h old, limit {stale_h:g} h; risk cuts still apply)")
    return Forecast("fresh", raw, date, int(made), age)


def latest_weights_slot_ms(now_ms: int, hh: int, mm: int) -> int:
    """The most recent hh:mm UTC at or before now."""
    day0 = now_ms - now_ms % DAY_MS
    slot = day0 + (hh * 60 + mm) * 60_000
    return slot if slot <= now_ms else slot - DAY_MS


def regime_text(root: Path) -> str | None:
    r = read_json(root / "regime" / "latest.json")
    if not isinstance(r, dict) or not isinstance(r.get("state"), str):
        return None
    p7 = r.get("p7") if isinstance(r.get("p7"), dict) else {}
    pr = _num(p7.get(r["state"]))
    return f"regime {r['state']}" + ("" if pr is None else f" {pr:.2f}")


def compute_weights(ids: list[str], forecast: dict[str, Any], current: dict[str, Current], p: dict[str, float | None],
                    calc: Calc, regime: str | None = None, dd_floor_pct: float = 15.0) -> dict[str, Any]:
    """The computed layer for one day, no I/O: forecasts to target weights (core plus tilt, then the rules) with the full
    breakdown. The tilt room kappa comes from the forecast file (scores.json via forecast/latest.json), 0 when absent: the live target
    is the core until the shadow tilt has earned room. The tilted target at full T is returned as the shadow.
    Strategies without a usable forecast (no sigma, or no forecaster with a value) are held at their weight."""
    fstrat = forecast.get("strategies") or {}
    fw = forecast.get("weights_k") if isinstance(forecast.get("weights_k"), dict) else {}
    cov_in = forecast.get("cov_d") if isinstance(forecast.get("cov_d"), dict) else {}
    cov_ids = [x for x in (cov_in.get("ids") or []) if isinstance(x, str)]
    matrix = cov_in.get("matrix") or []
    mu_log: dict[str, dict[str, float | None]] = {}
    sigma_log: dict[str, float | None] = {}
    clipped: dict[str, list[str]] = {}
    opt: list[str] = []
    held: dict[str, str] = {}
    for sid in ids:
        f = fstrat.get(sid) if isinstance(fstrat.get(sid), dict) else None
        sig = _num(f.get("sigma_d")) if f else None
        sigma_log[sid] = sig
        if not f or sig is None or sig <= 0:
            held[sid] = "no forecast"
            continue
        mus: dict[str, float | None] = {}
        clipped[sid] = []
        for k, v in (f.get("mu") or {}).items():
            n = _num(v)
            if n is None:
                continue
            mus[k], was = M.clip_mu(n, sig, calc.sharpe_cap)
            if was:
                clipped[sid].append(k)
        if p.get(sid) is not None:
            mus["decider"], was = M.clip_mu(M.decider_mu(p[sid], sig, calc.decider_k), sig, calc.sharpe_cap)
            if was:
                clipped[sid].append("decider")
        raw_w = fw.get(sid)
        weights = {k: n for k, v in raw_w.items() if (n := _num(v)) is not None} if isinstance(raw_w, dict) and raw_w else calc.start(sid)
        if "decider" not in weights:
            weights["decider"] = calc.start(sid).get("decider", 0.25)
        comb = M.combine(mus, weights)
        if comb is None:
            held[sid] = "no forecaster has a value"
            continue
        mus["combined"] = comb
        mu_log[sid] = mus
        opt.append(sid)
    for sid in ids:
        mu_log.setdefault(sid, {})
    w_prev = {sid: current[sid].weight for sid in ids}
    kappa = _num(forecast.get("kappa"))
    tilt = replace(calc.tilt, kappa=0.0 if kappa is None else min(1.0, max(0.0, kappa)))
    shadow = replace(calc.tilt, kappa=1.0)
    if not opt:
        return {"opt": [], "held": held, "mu": mu_log, "sigma_d": sigma_log, "clipped": clipped, "w_prev": w_prev, "w_target": {},
                "core": {}, "tau": {}, "sr": {}, "kappa": tilt.kappa, "t_eff": tilt.t_eff, "shadow_tau": {}, "shadow_w_target": {},
                "w_final": dict(w_prev), "steps": [], "reasons": {sid: f"{sid}: held at {w_prev[sid]:.2f} ({held[sid]})" for sid in ids}}

    def cov_ij(a: str, b: str) -> float:
        if a in cov_ids and b in cov_ids:
            try:
                v = _num(matrix[cov_ids.index(a)][cov_ids.index(b)])
            except (IndexError, TypeError):
                v = None
            if v is not None:
                return v
        return sigma_log[a] ** 2 if a == b else 0.0  # no covariance given: independent

    cov = [[cov_ij(a, b) for b in opt] for a in opt]
    core = dict(zip(opt, M.core_weights([sigma_log[sid] for sid in opt], cov, calc.rules.vol_cap, len(opt))))
    sr = {sid: M.forecast_sharpe(mu_log[sid]["combined"], sigma_log[sid]) for sid in opt}
    tau = {sid: M.tilt_tau(sr[sid], tilt) for sid in opt}
    w_target = dict(zip(opt, M.tilted_target([core[sid] for sid in opt], [tau[sid] for sid in opt])))
    shadow_tau = {sid: M.tilt_tau(sr[sid], shadow) for sid in opt}
    shadow_w = dict(zip(opt, M.tilted_target([core[sid] for sid in opt], [shadow_tau[sid] for sid in opt])))
    final, steps = M.apply_rules(opt, w_target, {sid: w_prev[sid] for sid in opt}, cov, {sid: p.get(sid) for sid in opt}, calc.rules,
                                 floored=[sid for sid in opt if current[sid].drawdown_pct is not None and current[sid].drawdown_pct > dd_floor_pct], locked=[sid for sid in opt if current[sid].owner_lock])
    reasons = {sid: f"{sid}: held at {w_prev[sid]:.2f} ({why})" for sid, why in held.items()}
    for sid in opt:
        binding = [st["step"] for st in steps if sid in st["changed"]]
        reasons[sid] = M.reason_line(sid, core[sid], tau[sid], sr[sid], w_target[sid], final[sid], binding, regime, tilt.kappa, shadow_tau[sid])
    return {"opt": opt, "held": held, "mu": mu_log, "sigma_d": sigma_log, "clipped": clipped, "w_prev": w_prev, "core": core, "tau": tau,
            "sr": sr, "kappa": tilt.kappa, "t_eff": tilt.t_eff, "w_target": w_target, "shadow_tau": shadow_tau, "shadow_w_target": shadow_w,
            "w_final": {**{sid: w_prev[sid] for sid in ids}, **final}, "steps": steps, "reasons": reasons}


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
                 flatten_on_floor: bool = True, tick_s: float = 30.0, mode: str | None = None,
                 params: MappingParams | None = None, max_questions: int | None = None, calc: Calc | None = None,
                 venues: list[AllocVenue] | None = None):
        # mode "decider": llm.ask(state, questions) answers alloc-v1.0; mode "chat": llm.ask_json(messages, schema) answers a JSON
        # allocation. Unset: chat when the client has ask_json, else decider (create_allocator always sets it).
        self.root = Path(root)
        # the first venue is the allocator venue (its root is self.root); without a list, that venue alone, as before
        self.venues: list[AllocVenue] = venues or [AllocVenue(PRIMARY_VENUE, self.root, "BYBIT·DEMO")]
        if Path(self.venues[0].root) != self.root:
            raise ValueError("the first venue must be the allocator's own root")
        self.primary = self.venues[0]
        self._kill_skips: list[str] = []  # the last _apply: strategies not switched on because a kill is pending
        self._paused_skips: list[str] = []  # the last _apply: not written because the allocator was paused meanwhile
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
        mode = mode or ("chat" if hasattr(llm, "ask_json") else "decider")
        if mode not in ('decider', 'chat'):
            raise ValueError(f"mode must be decider or chat, got {mode!r}")
        self.mode = mode
        self.params = params or MappingParams()
        self.max_questions = max_questions
        self.calc = calc or Calc.from_settings(Settings())
        self._stop = asyncio.Event()
        self._busy = asyncio.Lock()
        self.last_error: str | None = None
        # "<venue>/<strategy>": the drawdown floor's last failed write ({venue, venue_label, strategy, error, t_ms}). Rebuilt
        # by every risk check: an entry stays while the write keeps failing and goes once it succeeds or is no longer needed.
        self.floor_errors: dict[str, dict[str, Any]] = {}
        # the same for the last decision's writes (and a venue the decision could not handle at all); rebuilt by every
        # decision that is not a dry run, so an entry stays until a later decision writes that book
        self.decision_errors: dict[str, dict[str, Any]] = {}
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

    def vk(self, vid: str, sid: str) -> str:
        """The state key of one (venue, strategy) book."""
        return f"{vid}/{sid}"

    def strategies(self, venue: AllocVenue | None = None) -> list[dict[str, Any]]:
        """The strategies a venue's runner lists (strategies.json). Only the allocator venue falls back to its directories
        (older roots): another venue's strategy is written only when its runner lists it."""
        venue = venue or self.primary
        if venue is not self.primary:
            # the dashboard's registry rules (ids, units, duplicates): an unusable file writes nothing on that venue, and an
            # id can never reach outside the venue's root
            reg = S.load_registry(venue.root)
            return [] if reg.error else [{"id": e.id, "label": e.label} for e in reg.entries]
        listed = read_json(venue.root / "strategies.json")
        rows = [r for r in listed if isinstance(r, dict) and isinstance(r.get("id"), str) and S.ID_RE.fullmatch(r["id"])] if isinstance(listed, list) else []
        if not rows and self.root.is_dir():
            rows = [{"id": p.name, "label": p.name} for p in sorted(self.root.iterdir())
                    if p.is_dir() and p.name != ALLOCATOR_ID and S.ID_RE.fullmatch(p.name) and (p / "status.json").exists()]
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
            self.state_cache.setdefault("p_hist", {})   # the decider's last answers per strategy: [[t_ms, p], ...] (venue-independent)
            self._migrate_keys(self.state_cache)
        return self.state_cache

    def _migrate_keys(self, st: dict[str, Any]) -> None:
        """Older state.json files key peaks, history and the day anchor by strategy id alone (the allocator venue's books).
        They become "<allocator venue>/<id>", values unchanged, so the allocator venue's weights do not move."""
        if st.get("keys_version") == STATE_KEYS_VERSION:
            return
        pv = self.primary.id
        for k in ("peaks", "history"):
            st[k] = {(key if "/" in key else self.vk(pv, key)): v for key, v in (st.get(k) or {}).items()}
        da = st.get("day_anchor")
        if isinstance(da, dict) and isinstance(da.get("w"), dict):
            da["w"] = {(key if "/" in key else self.vk(pv, key)): v for key, v in da["w"].items()}
        st["keys_version"], st["primary_venue"] = STATE_KEYS_VERSION, pv
        self._dirty = True

    # ---- computed layer: forecast, decider history, schedule
    def load_forecast(self, now_ms: int) -> Forecast:
        return read_forecast(self.root, now_ms, self.calc.stale_h)

    def _push_p(self, p: dict[str, float], now_ms: int) -> None:
        hist = self._state()["p_hist"]
        for sid, v in p.items():
            hist[sid] = (hist.get(sid) or []) + [[now_ms, v]]
            hist[sid] = hist[sid][-self.calc.p_window:]
        self._dirty = True

    def p_means(self, ids: list[str], extra: dict[str, float] | None = None) -> dict[str, float | None]:
        """p per strategy: the mean of the decider's last p_window answers (`extra`: one more, not stored, for dry runs)."""
        hist = self._state()["p_hist"]
        out: dict[str, float | None] = {}
        for sid in ids:
            vals = [v for _, v in (hist.get(sid) or [])]
            if extra and sid in extra:
                vals.append(extra[sid])
            out[sid] = M.mean_p(vals[-self.calc.p_window:])
        return out

    def weights_due(self, now_ms: int) -> bool:
        """True when the daily weight-setting slot (23:50 UTC) has passed since weights were last set, or never were."""
        last = _num(self._state().get("last_weights_ms"))
        return last is None or last < latest_weights_slot_ms(now_ms, *self.calc.weights_utc)

    def observe(self, key: str, equity: float | None, start_equity: float | None, now_ms: int) -> bool:
        """Track the peak and keep sparse equity samples of one (venue, strategy) book. Returns True when the state changed."""
        if equity is None:
            return False
        st = self._state()
        changed = False
        peak = st["peaks"].get(key)
        new_peak = max(equity, start_equity or 0.0, peak or 0.0)
        if peak is None or new_peak > peak:
            st["peaks"][key] = new_peak
            changed = True
        hist = st["history"].setdefault(key, [])
        if not hist or now_ms - hist[-1][0] >= SAMPLE_EVERY_MS:
            hist.append([now_ms, equity])
            st["history"][key] = [p for p in hist if now_ms - p[0] <= HISTORY_MS]
            changed = True
        return changed

    def _pnl_window(self, key: str, equity: float | None, window_ms: int, now_ms: int) -> float | None:
        if equity is None:
            return None
        older = [p for p in self._state()["history"].get(key, []) if p[0] <= now_ms - window_ms]
        if not older or older[-1][1] <= 0:
            return None
        return _r(100 * (equity / older[-1][1] - 1), 2)

    def drawdown_pct(self, key: str, status: dict[str, Any]) -> float | None:
        """One (venue, strategy) book's drawdown from its peak, from that venue's status equity."""
        eq = _num(status.get("equity_usdt"))
        if eq is None:
            return None
        peak = max(self._state()["peaks"].get(key) or 0.0, _num(status.get("start_equity_usdt")) or 0.0,
                   _num(status.get("peak_equity_usdt")) or 0.0, eq)
        return 100 * (peak - eq) / peak if peak > 0 else None

    # ---- context
    def strategy_context(self, row: dict[str, Any], now_ms: int, venue: AllocVenue | None = None) -> tuple[dict[str, Any], Current]:
        venue = venue or self.primary
        sid = row["id"]
        key = self.vk(venue.id, sid)
        status = read_json(venue.root / sid / "status.json")
        status = status if isinstance(status, dict) else {}
        ctl = read_json(venue.root / sid / "control.json")
        ctl = ctl if isinstance(ctl, dict) else {}
        eq = _num(status.get("equity_usdt"))
        if self.observe(key, eq, _num(status.get("start_equity_usdt")), now_ms):
            self._dirty = True
        dd = self.drawdown_pct(key, status)
        w = _num(ctl.get("weight"))
        weight = 0.0 if w is None else min(1.0, max(0.0, w))  # nothing decided yet means off, as in the runner
        enabled = ctl.get("enabled", status.get("enabled", True)) is not False
        lock = ctl.get("owner_lock") is True
        trades = tail_jsonl(venue.root / sid / "trades.jsonl", 5000)
        t_ms = [x for x in (_num(t.get("t_ms")) for t in trades) if x is not None]
        positions = [p for p in (status.get("positions") or []) if isinstance(p, dict)]
        targets = [t for t in (status.get("targets") or []) if isinstance(t, dict)]
        upd = _num(status.get("updated_ms"))
        hist = self._state()["history"].get(key) or []
        ctx = {
            "label": row.get("label") or status.get("label") or sid,
            "state": status.get("state") or "unknown",
            "status_age_s": None if upd is None else max(0, round((now_ms - upd) / 1000)),
            "enabled": enabled, "weight": round(weight, 3), "owner_lock": lock, "set_by": ctl.get("set_by"),
            "equity_usdt": _r(eq, 2), "start_equity_usdt": _r(_num(status.get("start_equity_usdt")), 2),
            "pnl_1d_pct": self._pnl_window(key, eq, DAY_MS, now_ms),
            "pnl_7d_pct": self._pnl_window(key, eq, 7 * DAY_MS, now_ms),
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
        return ctx, Current(enabled=enabled, weight=weight, owner_lock=lock, drawdown_pct=dd,
                            gross=gross_per_unit(positions, eq, weight))

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
        if self.mode == "decider":   # alloc-v1.1 blocks; the chat fallback keeps its own prompt and context
            self.add_decider_blocks(ctx, current, now_ms)
        return ctx, current, rows

    def add_decider_blocks(self, ctx: dict[str, Any], current: dict[str, Current], now_ms: int) -> None:
        """alloc-v1.1: regime, forecasts and news next to the books and the market. A block whose file is missing is left out
        and a line in state_notes says so; nothing here raises."""
        notes: list[str] = []
        regime, note = X.regime_block(self.root)
        if regime:
            ctx["regime"] = regime
        elif note:
            notes.append(note)
        fblock, note = self.forecast_block(current, now_ms)
        if fblock:
            ctx["forecasts"] = fblock
        if note:
            notes.append(note)
        try:
            news, news_notes = X.news_block(X.news_root(self.root), now_ms, rss=X.rss_headlines_enabled())
        except Exception as exc:  # noqa: BLE001
            news, news_notes = None, [f"news omitted: {type(exc).__name__}"]
        if news is not None:
            ctx["news"] = news
        notes += news_notes
        if notes:
            ctx["state_notes"] = notes

    def forecast_block(self, current: dict[str, Current], now_ms: int) -> tuple[dict[str, Any] | None, str | None]:
        """The prediction layer's computed numbers per strategy, WITHOUT the decider: its answers are not passed in (p is empty),
        so the combined mu is the weighted mean of base, regime, ridge (carry also structural) only, clipped to the Sharpe cap.
        mu and sigma in %/yr, forecast Sharpe, core weight (the inverse-volatility size before the rules)."""
        fc = self.load_forecast(now_ms)
        if fc.state in ("missing", "invalid") or not fc.data:
            return None, f"forecasts omitted: {fc.note}"
        ids = list(current)
        data = dict(fc.data)
        data["strategies"] = {sid: ({**f, "mu": {k: v for k, v in f["mu"].items() if k != "decider"}} if isinstance(f, dict) and isinstance(f.get("mu"), dict) else f)
                              for sid, f in (fc.data.get("strategies") or {}).items()}   # belt and braces: never feed back a decider row
        try:
            out = compute_weights(ids, data, current, {}, self.calc, None, self.dd_floor_pct)
        except Exception as exc:  # noqa: BLE001
            return None, f"forecasts omitted: could not be computed ({type(exc).__name__})"
        rows: dict[str, Any] = {}
        for sid in ids:
            mu = out["mu"].get(sid, {}).get("combined")
            sig = out["sigma_d"].get(sid)
            if mu is None or sig is None or sig <= 0:
                continue
            rows[sid] = {"mu_pct_yr": round(M.ann_pct(mu), 2), "sigma_pct_yr": round(M.sigma_ann_pct(sig), 2),
                         "forecast_sharpe": round(out["sr"][sid], 3), "core_weight": round(out["core"][sid], 4),
                         "forecasters": sorted(k for k in out["mu"][sid] if k not in ("combined", "decider"))}
        if not rows:
            return None, "forecasts omitted: no strategy has a usable forecast"
        block = {"date": fc.date, "age_h": None if fc.age_h is None else round(fc.age_h, 1), "strategies": rows}
        missing = [sid for sid in ids if sid not in rows]
        note = None
        if fc.state == "stale":
            note = f"forecasts are old: {fc.note}"
        elif missing:
            note = f"forecasts: no usable forecast for {', '.join(missing)}"
        return block, note

    # ---- the model
    async def ask(self, ctx: dict[str, Any], ids: list[str], labels: dict[str, str],
                  current: dict[str, Current]) -> tuple[Decision, float, int, dict[str, Any]]:
        """One validated decision. Returns (decision, cost_usd, attempts, extra fields for the log)."""
        if self.mode == "chat":
            return await self._ask_chat(ctx, ids, labels)
        return await self._ask_decider(ctx, ids, labels, current)

    async def _ask_decider(self, ctx: dict[str, Any], ids: list[str], labels: dict[str, str],
                           current: dict[str, Current]) -> tuple[Decision, float, int, dict[str, Any]]:
        """alloc-v1.0 to the decision model, in as many requests as its question limit needs; each
        request gets a second attempt when answers are missing or off-contract. A strategy whose
        question never validates keeps its current allocation."""
        qs = build_questions(ids, labels)
        parsed: dict[str, Any] = {}
        issues_all: list[str] = []
        request_ids: list[str] = []
        usage_all: list[dict[str, Any]] = []
        cost, attempts = 0.0, 0
        limit = self.max_questions or getattr(getattr(self.llm, "spec", None), "max_questions", None)
        parts = batches(qs, limit)
        X.fit_state(ctx, parts, getattr(getattr(self.llm, "spec", None), "max_payload_bytes", None))   # in place: the log shows what was sent
        for batch in parts:
            for attempt in (1, 2):
                if not self.budget.try_take():
                    raise AllocatorError(f"hourly model request budget exhausted ({self.budget.per_hour}/h)")
                attempts += 1
                try:
                    res = await self.llm.ask(ctx, batch)
                except KairosError as exc:
                    raise AllocatorError(str(exc)[:300]) from exc
                ok, issues = validate(res.answers, batch)
                parsed.update(ok)
                usage_all.append(res.usage or {})
                cost += _num((res.usage or {}).get("cost")) or 0.0
                rid = (res.metadata or {}).get("request_id") or (res.metadata or {}).get("id")
                if rid:
                    request_ids.append(str(rid))
                if all(q in ok for q in batch):
                    break
                issues_all = issues
                log.warning("allocator: decision model answer incomplete (attempt %d): %s", attempt, "; ".join(issues)[:300])
        p = {sid: parsed[pos_qid(sid)]["p"] for sid in ids if pos_qid(sid) in parsed}
        best = parsed[BEST_QID]["probs"] if BEST_QID in parsed else None
        if not p and best is None:
            raise AllocatorError("decision model gave no valid answers: " + "; ".join(issues_all)[:300])
        hold = {sid: (c.enabled, c.weight) for sid, c in current.items()}
        weights, enabled, mapping, summary = map_answers(ids, p, best, hold, self.params)
        conf = [max(v, 1 - v) for v in p.values()]
        extra = {"qset": QUESTION_SET_VERSION, "provider": getattr(getattr(self.llm, "spec", None), "provider", None),
                 "state_bytes": X.block_sizes(ctx), "request_ids": request_ids, "usage": usage_all, "mapping": mapping,
                 "answers": {"p": {sid: round(v, 4) for sid, v in p.items()},
                             "best": None if best is None else {k: round(v, 4) for k, v in best.items()}},
                 "unanswered": [q for q in qs if q not in parsed], "issues": issues_all[:6] if len(parsed) < len(qs) else []}
        decision = Decision(weights, enabled, summary[:REASONING_MAX_CHARS], round(sum(conf) / len(conf), 3) if conf else None,
                            derive_enabled=tuple(sid for sid in ids if not mapping[sid]["held"]),
                            held=tuple(sid for sid in ids if mapping[sid]["held"]))
        return decision, cost, attempts, extra

    async def _ask_chat(self, ctx: dict[str, Any], ids: list[str], labels: dict[str, str]) -> tuple[Decision, float, int, dict[str, Any]]:
        """Chat fallback: one JSON allocation; a second attempt on invalid output, then give up."""
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
                return validate_decision(raw, ids), cost, attempt, {}
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
    def _write_control(self, sid: str, enabled: bool, weight: float, reason: str, flatten_on_zero: bool, now_ms: int,
                       root: Path | None = None) -> bool:
        """Read-modify-write one strategy's control.json on one venue's root. Returns False when the owner locked it
        (checked again here, right before the write), when a kill is pending and this would switch it on, or when the
        allocator was paused meanwhile and this does not reduce risk. The pause is the allocator's own control: it applies
        to every venue."""
        path = (root or self.root) / sid / "control.json"
        with locked(path):  # one read-modify-write at a time per file, shared with the dashboard (ctb.control_lock)
            ctl = read_json(path)
            ctl = ctl if isinstance(ctl, dict) else {}
            if ctl.get("owner_lock") is True:
                return False
            if enabled and ctl.get("flatten") is True:  # a pending kill or flatten ends when the runner, flat, clears it, or the owner cancels it
                log.info("allocator: %s not switched on: a kill or flatten is pending (control.json flatten true)", sid)
                self._kill_skips.append(sid)
                return False
            # A decision can take minutes (the model call); the owner may have paused the allocator meanwhile (AUTO-TRADE OFF, or
            # a switch to MANUAL or PER-SYMBOL BOT). Read its control again here, at the write: while paused only a write that
            # reduces risk goes through (switching off, or a lower weight on a strategy that is on), never an enable or a raise.
            if not self.control()["enabled"]:
                cur_on = ctl.get("enabled") is True
                cur_w = _num(ctl.get("weight"))
                reducing = (not enabled) or (cur_on and cur_w is not None and weight <= cur_w + 1e-12)
                if not reducing:
                    log.info("allocator: %s not written: the allocator was paused while this decision ran", sid)
                    self._paused_skips.append(sid)
                    return False
            ctl.update({"enabled": enabled, "weight": weight, "set_by": "llm", "reason": reason[:REASONING_MAX_CHARS],
                        "updated_ms": now_ms})
            if flatten_on_zero:
                ctl["flatten_on_zero"] = True
            write_json_atomic(path, ctl)
        return True

    def _apply(self, plan: Plan, current: dict[str, Current], reasons: dict[str, str], now_ms: int, *,
               refresh_reasons: bool = False, venue: AllocVenue | None = None,
               errors: dict[str, str] | None = None) -> dict[str, bool]:
        """Write the plan into each strategy's control.json on one venue. With errors given, a failed write (an unwritable
        directory, a full disk) is logged and recorded there as "<Type>: message" and the other strategies are still written;
        without it the failure propagates as before."""
        venue = venue or self.primary
        applied: dict[str, bool] = {}
        self._kill_skips = []  # filled by _write_control: strategies not switched on because a kill is pending
        self._paused_skips = []  # and those not written because the allocator was paused while the decision ran
        for sid, c in current.items():
            if sid in plan.skipped_owner_lock:
                applied[sid] = False
                continue
            unchanged = plan.enabled[sid] == c.enabled and abs(plan.weights[sid] - c.weight) < 1e-9
            if unchanged and refresh_reasons:
                ctl = read_json(venue.root / sid / "control.json")
                unchanged = isinstance(ctl, dict) and ctl.get("reason") == reasons[sid][:REASONING_MAX_CHARS]  # new text for the card
            if unchanged:
                applied[sid] = False  # nothing to tell the runner; keep control.json (and its reason) as is
                continue
            try:
                applied[sid] = self._write_control(sid, plan.enabled[sid], plan.weights[sid], reasons[sid],
                                                   self.flatten_on_floor and sid in plan.floored, now_ms, root=venue.root)
            except Exception as exc:  # noqa: BLE001 - only when the caller collects errors (the drawdown floor)
                if errors is None:
                    raise
                errors[sid] = f"{type(exc).__name__}: {exc}"[:300]
                applied[sid] = False
        return applied

    def _day_anchor(self, current: dict[str, "Current"], now_ms: int, *, persist: bool = True,
                    venue: AllocVenue | None = None) -> dict[str, "Current"]:
        """The step limit and the buffer count from the weights at the first weights run of the UTC day, so
        repeated DECIDE NOW presses cannot ratchet a weight up by more than one step per day. Keyed per
        (venue, strategy): each venue's book steps from its own anchor."""
        venue = venue or self.primary
        day = time.strftime("%Y-%m-%d", time.gmtime(now_ms / 1000))
        st = self._state()
        anchor = st.get("day_anchor") if isinstance(st.get("day_anchor"), dict) else {}
        if anchor.get("day") != day:
            anchor = {"day": day, "w": {}}
            if persist:
                st["day_anchor"] = anchor
        w = anchor.setdefault("w", {})
        for sid, c in current.items():
            if self.vk(venue.id, sid) not in w:
                w[self.vk(venue.id, sid)] = c.weight
        if persist:
            self._save_state()
        return {sid: replace(c, weight=float(w[self.vk(venue.id, sid)])) for sid, c in current.items()}

    def _save_state(self) -> None:
        self._dirty = False
        write_json_atomic(self.dir / "state.json", self._state())

    # ---- one decision
    async def decide(self, *, dry_run: bool = False, trigger: str = "schedule") -> dict[str, Any]:
        """One decision. With a decision model the computed layer runs; a missing, invalid or stale forecast holds the last
        weights there (the decider is still asked, its answers still go into the p history) and never falls back to the
        decider-only mapping. The chat fallback model has its own path. With dry_run nothing is written: the record comes
        back and that is all."""
        fc = self.load_forecast(self.now_ms()) if self.mode == "decider" else None
        if fc is None:
            return await self._decide_legacy(dry_run=dry_run, trigger=trigger)
        return await self._decide_computed(fc, dry_run=dry_run, trigger=trigger)

    async def _decide_legacy(self, *, dry_run: bool = False, trigger: str = "schedule") -> dict[str, Any]:
        """Build the context, ask the model, apply the rules and write the controls: the chat fallback model's allocation.
        (A decision model never comes here: its weights are computed, or held without a usable forecast.)"""
        async with self._busy:
            now_ms = self.now_ms()
            record: dict[str, Any] = {"t_ms": now_ms, "model": self.model, "trigger": trigger}
            current: dict[str, Current] = {}
            error: str | None = None
            decision: Decision | None = None
            try:
                ctx, current, rows = await self.build_context(now_ms)
                record["inputs"] = ctx
                record["state_bytes"] = X.block_sizes(ctx)
                ids = [r["id"] for r in rows]
                decision, cost, attempts, extra = await self.ask(ctx, ids, {r["id"]: str(r.get("label") or r["id"]) for r in rows}, current)
                record.update(cost_usd=round(cost, 6), attempts=attempts, **extra)
            except AllocatorError as exc:
                error = str(exc)
                if not current:  # failed before the strategies were read: nothing to apply, not even the floor
                    current = self._read_current(now_ms)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                current = current or self._read_current(now_ms)
            plan = resolve(current, decision, max_step=self.max_step, dd_floor_pct=self.dd_floor_pct) if current else None
            explored = apply_explore_floor(plan, current, self._explore_floor()) if plan and decision else []
            reasons = {}
            if plan:
                for sid in current:
                    if sid in plan.floored:
                        dd = current[sid].drawdown_pct
                        reasons[sid] = f"risk floor: book {dd:.1f}% below its peak, weight forced to 0"
                    else:
                        reasons[sid] = decision.reasoning if decision else "kept (allocator error)"
                        if sid in explored:
                            reasons[sid] += f" | demo minimum {plan.weights[sid]:.2f}"
            applied: dict[str, bool] = {}
            failed: dict[str, str] = {}
            if plan and not dry_run and self.primary.managed:
                applied = self._apply(plan, current, reasons, now_ms, errors=failed)
            venues = {self.primary.id: self._venue_record(self.primary, plan, current, reasons, applied, explored,
                                                          {"write_failed": failed} if failed else None)}
            record.update({
                "weights": plan.weights if plan else {}, "enabled": plan.enabled if plan else {},
                "reasoning": decision.reasoning if decision else "", "confidence": decision.confidence if decision else None,
                "requested": {"weights": decision.weights, "enabled": decision.enabled} if decision else None,
                "mode": self.mode,
                "applied": applied, "skipped_owner_lock": plan.skipped_owner_lock if plan else [],
                "skipped_kill_pending": [x for x in self._kill_skips if x in applied],
                "skipped_paused": [x for x in self._paused_skips if x in applied],
                "risk_floor": plan.floored if plan else [], "step_limited": plan.step_limited if plan else [],
                "explore_floor": explored,
                "error": error, "dry_run": dry_run,
            })
            for v in self.venues[1:]:  # the same per-strategy decision, under each venue's own rules
                venues[v.id] = self._other_venue(v, lambda v=v: self._legacy_venue(v, decision, now_ms, dry_run))
            record["venues"] = venues
            not_applied = self._write_failures(venues, now_ms, dry_run)
            record["error"] = _join_errors(record["error"], not_applied)
            if not dry_run:
                append_jsonl(self.dir / "decisions.jsonl", record)
                self.last_error = record["error"]
                self._save_state()
                self.write_status(now_ms, last_decision_ms=now_ms)
            return record

    async def _decide_computed(self, fc: Forecast, *, dry_run: bool = False, trigger: str = "schedule") -> dict[str, Any]:
        """The decider is asked (hourly: that is all, its answers go into the p history); the weights are set on a
        weights run: DECIDE NOW, the CLI, or the daily 23:50 UTC slot. A stale or unreadable forecast holds the weights."""
        async with self._busy:
            now_ms = self.now_ms()
            record: dict[str, Any] = {"t_ms": now_ms, "model": self.model, "trigger": trigger, "forecast": fc.info()}
            current: dict[str, Current] = {}
            error: str | None = None
            extra: dict[str, Any] = {}
            try:
                ctx, current, rows = await self.build_context(now_ms)
                record["inputs"] = ctx
                record["state_bytes"] = X.block_sizes(ctx)
                _, cost, attempts, extra = await self._ask_decider(ctx, [r["id"] for r in rows],
                                                                    {r["id"]: str(r.get("label") or r["id"]) for r in rows}, current)
                record.update(cost_usd=round(cost, 6), attempts=attempts)
            except AllocatorError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if not current:
                current = self._read_current(now_ms)
            ids = list(current)
            extra.pop("mapping", None)  # the decider-only mapping never sets weights
            record.update(extra)
            p_new = (extra.get("answers") or {}).get("p") or {}
            if p_new and not dry_run:
                self._push_p(p_new, now_ms)
            p_now = self.p_means(ids, p_new if dry_run else None)
            weights_run = trigger != "schedule" or self.weights_due(now_ms)
            calc_out: dict[str, Any] | None = None
            decision: Decision | None = None
            if not weights_run:
                kind, why = "refresh", f"decider answers refreshed; weights are set daily at {self.calc.weights_utc[0]:02d}:{self.calc.weights_utc[1]:02d} UTC"
            elif fc.state != "fresh":
                kind, why = "held", fc.note
            elif not current:
                kind, why = "held", "no strategies"
            else:
                kind = "computed"
                anchored = self._day_anchor(current, now_ms, persist=not dry_run)
                calc_out = compute_weights(ids, fc.data or {}, anchored, p_now, self.calc, regime_text(self.root), self.dd_floor_pct)
                why = " | ".join(calc_out["reasons"][sid] for sid in ids)
                decision = Decision(dict(calc_out["w_final"]), {sid: current[sid].enabled for sid in ids}, why, None,
                                    derive_enabled=tuple(calc_out["opt"]))
            plan = resolve(current, decision, max_step=1.0, dd_floor_pct=self.dd_floor_pct) if current else None
            reasons: dict[str, str] = {}
            if plan:
                for sid in current:
                    if sid in plan.floored:
                        reasons[sid] = f"risk floor: book {current[sid].drawdown_pct:.1f}% below its peak, weight forced to 0"
                    else:
                        reasons[sid] = calc_out["reasons"][sid] if calc_out else why
            applied: dict[str, bool] = {}
            failed: dict[str, str] = {}
            if plan and not dry_run and self.primary.managed:
                applied = self._apply(plan, current, reasons, now_ms, refresh_reasons=kind == "computed", errors=failed)
            venues = {self.primary.id: self._venue_record(self.primary, plan, current, reasons, applied, None,
                                                          {"write_failed": failed} if failed else None)}
            if kind == "computed" and plan and not dry_run:
                self._state()["last_weights_ms"] = now_ms
                self._dirty = True
            record.update({
                "weights": plan.weights if plan else {}, "enabled": plan.enabled if plan else {},
                "reasoning": why, "confidence": None, "requested": None, "mode": self.mode, "weights_mode": kind,
                "applied": applied, "skipped_owner_lock": plan.skipped_owner_lock if plan else [],
                "skipped_kill_pending": [x for x in self._kill_skips if x in applied],
                "skipped_paused": [x for x in self._paused_skips if x in applied],
                "risk_floor": plan.floored if plan else [],
                "step_limited": next((st["changed"] for st in (calc_out or {}).get("steps", []) if st["step"] == "step"), []),
                "p_decider": p_now,
                "error": error or fc.error, "dry_run": dry_run,
            })
            if calc_out:
                record.update({
                    "mu": calc_out["mu"], "mu_ann_pct": {sid: {k: round(M.ann_pct(v), 2) for k, v in m.items() if v is not None} for sid, m in calc_out["mu"].items()},
                    "sigma_d": calc_out["sigma_d"], "w_prev": calc_out["w_prev"],
                    "core": {k: round(v, 6) for k, v in calc_out["core"].items()}, "tau": {k: round(v, 6) for k, v in calc_out["tau"].items()},
                    "sr": {k: round(v, 4) for k, v in calc_out["sr"].items()}, "kappa": calc_out["kappa"], "t_eff": calc_out["t_eff"],
                    "shadow_tau": {k: round(v, 6) for k, v in calc_out["shadow_tau"].items()},
                    "shadow_w_target": {k: round(v, 6) for k, v in calc_out["shadow_w_target"].items()},
                    "w_target": {k: round(v, 6) for k, v in calc_out["w_target"].items()},
                    "w_final": plan.weights if plan else {}, "steps": calc_out["steps"], "reasons": reasons, "clipped": calc_out["clipped"],
                    "no_forecast": calc_out["held"],
                })
            for v in self.venues[1:]:  # the same forecasts and decider p, each venue's own books, anchor and rules
                venues[v.id] = self._other_venue(v, lambda v=v: self._computed_venue(v, fc, kind, why, p_new, now_ms, dry_run, decided_ids=ids))
            record["venues"] = venues
            # only a weights run rewrites every book: an hourly refresh or a held run leaves earlier failures standing
            not_applied = self._write_failures(venues, now_ms, dry_run, replace=kind == "computed")
            record["error"] = _join_errors(record["error"], not_applied)
            if not dry_run:
                append_jsonl(self.dir / "decisions.jsonl", record)
                self.last_error = _join_errors(error, not_applied)  # the forecast's error is said by write_status, as before
                self._save_state()
                self.write_status(now_ms, last_decision_ms=now_ms)
            return record

    def _other_venue(self, v: AllocVenue, run: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """One other venue's part of a decision. Anything that goes wrong there (beyond a single write, which _apply keeps
        per strategy) is that venue's error: the decision, its record and the other venues go on."""
        try:
            return run()
        except Exception as exc:  # noqa: BLE001
            log.exception("allocator: decision on %s failed", v.id)
            return {"label": v.label, "managed": v.managed, "strategies": [], "note": "nothing applied (error)",
                    "error": f"{type(exc).__name__}: {exc}"[:300]}

    def _write_failures(self, venues: dict[str, dict[str, Any]], now_ms: int, dry_run: bool, replace: bool = True) -> str | None:
        """What a decision could not apply: every write that failed and every venue that failed, as one line for the
        decision's error (None when all went through). Unless a dry run, also kept in decision_errors: status.json and the
        venue alerts show them until a later decision that sets the weights (replace) writes those books."""
        found: dict[str, dict[str, Any]] = {}
        for vid, vr in venues.items():
            label = vr.get("label") or vid
            for sid, err in (vr.get("write_failed") or {}).items():
                found[self.vk(vid, sid)] = {"venue": vid, "venue_label": label, "strategy": sid, "error": err, "t_ms": now_ms}
            if vr.get("error"):
                found[self.vk(vid, "*")] = {"venue": vid, "venue_label": label, "strategy": None, "error": vr["error"], "t_ms": now_ms}
        if not dry_run:
            for key, e in found.items():
                if self.decision_errors.get(key, {}).get("error") != e["error"]:
                    log.error("allocator: decision on %s %s not applied: %s", e["venue"], e["strategy"] or "(venue)", e["error"])
            self.decision_errors = found if replace else {**self.decision_errors, **found}
        if not found:
            return None
        txt = "; ".join(f"{e['venue_label']} {e['strategy'] or '(venue)'}: {e['error']}" for e in found.values())
        return f"control.json not written, kept as it was ({txt})"

    def _venue_record(self, v: AllocVenue, plan: Plan | None, current: dict[str, Current], reasons: dict[str, str],
                      applied: dict[str, bool], explored: list[str] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """What the feed shows for one venue: per strategy the weight, whether it was written, and why not (read right after
        that venue's _apply: the skip lists are that call's)."""
        return {"label": v.label, "managed": v.managed, "strategies": list(current),
                "weights": plan.weights if plan else {}, "enabled": plan.enabled if plan else {}, "applied": applied,
                "skipped_owner_lock": plan.skipped_owner_lock if plan else [],
                "skipped_kill_pending": [x for x in self._kill_skips if x in applied],
                "skipped_paused": [x for x in self._paused_skips if x in applied],
                "risk_floor": plan.floored if plan else [], "explore_floor": explored or [], "reasons": reasons, **(extra or {})}

    def _floor_reason(self, v: AllocVenue, c: Current) -> str:
        where = "" if v is self.primary else f"{v.label} "  # the allocator venue keeps its old wording
        return f"risk floor: {where}book {c.drawdown_pct:.1f}% below its peak, weight forced to 0"

    def _legacy_venue(self, v: AllocVenue, decision: Decision | None, now_ms: int, dry_run: bool) -> dict[str, Any]:
        """The decider-only (or chat) decision on another venue: its strategies that the decision covers take the decided
        weight under that venue's step limit, drawdown floor, exploration floor and owner locks; the others are held at this
        venue's own allocation: a strategy the decision venue does not list, and one the decision model left unanswered
        (its decided weight is only the decision venue's current allocation, kept)."""
        current = self._read_current(now_ms, v)
        if not current:
            return {"label": v.label, "managed": v.managed, "strategies": [], "note": "no usable strategies.json (missing or invalid): nothing applied"}
        decided = [s for s in current if decision is not None and s in decision.weights and s not in decision.held]
        sub = None if decision is None else Decision(
            {s: decision.weights[s] if s in decided else current[s].weight for s in current},
            {s: decision.enabled[s] if s in decided else current[s].enabled for s in current},
            decision.reasoning, decision.confidence, derive_enabled=tuple(s for s in decision.derive_enabled if s in decided))
        plan = resolve(current, sub, max_step=self.max_step, dd_floor_pct=self.dd_floor_pct)
        explored = apply_explore_floor(plan, {s: current[s] for s in decided}, self._explore_floor()) if decision else []
        reasons: dict[str, str] = {}
        for sid, c in current.items():
            if sid in plan.floored:
                reasons[sid] = self._floor_reason(v, c)
            elif decision is None:
                reasons[sid] = "kept (allocator error)"
            elif sid not in decided and sid in decision.held:
                reasons[sid] = "no decision: the decision model gave no valid answer for it, held"
            elif sid not in decided:
                reasons[sid] = f"no decision: strategy not on the decision venue ({self.primary.label}), held"
            else:
                reasons[sid] = decision.reasoning + (f" | demo minimum {plan.weights[sid]:.2f}" if sid in explored else "")
        writable = {s: c for s, c in current.items() if s in decided or s in plan.floored or decision is None}
        failed: dict[str, str] = {}
        applied = self._apply(plan, writable, reasons, now_ms, venue=v, errors=failed) if not dry_run and v.managed else {}
        applied.update({s: False for s in current if s not in writable})
        return self._venue_record(v, plan, current, reasons, applied, explored, {"write_failed": failed} if failed else None)

    def _computed_venue(self, v: AllocVenue, fc: Forecast, kind: str, why: str, p_new: dict[str, float], now_ms: int,
                        dry_run: bool, decided_ids: list[str] | None = None) -> dict[str, Any]:
        """The computed weights on another venue: the same forecasts and decider p per strategy, that venue's own books,
        day anchor, vol cap over its strategies, drawdown floor, exploration floor and owner locks. Only strategies the
        decision venue lists (decided_ids) are computed; a strategy only this venue lists is held (no decision covers it),
        whatever the forecast file holds for it."""
        current = self._read_current(now_ms, v)
        if not current:
            return {"label": v.label, "managed": v.managed, "strategies": [], "note": "no usable strategies.json (missing or invalid): nothing applied"}
        ids = list(current)
        covered = [s for s in ids if decided_ids is None or s in decided_ids]
        calc_out: dict[str, Any] | None = None
        decision: Decision | None = None
        if kind == "computed" and covered:
            anchored = self._day_anchor({s: current[s] for s in covered}, now_ms, persist=not dry_run, venue=v)
            p_now = self.p_means(covered, p_new if dry_run else None)
            calc_out = compute_weights(covered, fc.data or {}, anchored, p_now, self.calc, regime_text(self.root), self.dd_floor_pct)
            decision = Decision({**{s: current[s].weight for s in ids}, **calc_out["w_final"]}, {sid: current[sid].enabled for sid in ids},
                                " | ".join(calc_out["reasons"][sid] for sid in covered), None, derive_enabled=tuple(calc_out["opt"]))
        plan = resolve(current, decision, max_step=1.0, dd_floor_pct=self.dd_floor_pct)
        held = f"no decision: strategy not on the decision venue ({self.primary.label}), held"
        reasons = {sid: (self._floor_reason(v, c) if sid in plan.floored
                         else held if sid not in covered else (calc_out["reasons"][sid] if calc_out else why))
                   for sid, c in current.items()}
        # a held strategy (no decision covers it) is not written at all; only the drawdown floor may still cut it
        writable = {s: c for s, c in current.items() if s in covered or s in plan.floored}
        failed: dict[str, str] = {}
        applied = (self._apply(plan, writable, reasons, now_ms, refresh_reasons=kind == "computed", venue=v, errors=failed)
                   if not dry_run and v.managed else {})
        applied.update({s: False for s in current if s not in writable})
        extra = {"core": {k: round(x, 6) for k, x in calc_out["core"].items()},
                 "w_target": {k: round(x, 6) for k, x in calc_out["w_target"].items()}, "steps": calc_out["steps"]} if calc_out else {}
        if failed:
            extra["write_failed"] = failed
        return self._venue_record(v, plan, current, reasons, applied, None, extra)

    def _explore_floor(self) -> float:
        """The demo minimum weight (CTB_ALLOCATOR_EXPLORE_FLOOR) on the venues it applies to, else 0."""
        from .config import _venue_name, load_settings
        try:
            s = load_settings()
            raw = os.environ.get("CTB_ALLOCATOR_EXPLORE_FLOOR")  # read now: Settings defaults are fixed at import time
            floor = float(raw) if raw not in (None, "") else s.allocator_explore_floor
            return floor if _venue_name() in s.allocator_explore_venues else 0.0
        except Exception:  # noqa: BLE001 - an unreadable setting must not stop the allocator
            return 0.0

    def _read_current(self, now_ms: int, venue: AllocVenue | None = None) -> dict[str, Current]:
        out: dict[str, Current] = {}
        for r in self.strategies(venue):
            out[r["id"]] = self.strategy_context(r, now_ms, venue)[1]
        return out

    async def risk_check(self) -> dict[str, Any] | None:
        """The drawdown floor on its own, no model: runs every tick so a bad book is cut within
        seconds of the runner reporting it, not at the next hourly decision. Each (venue, strategy) stands alone: a write
        that fails (or a venue that cannot be read) is logged, kept in floor_errors (status.json and the venue alerts show
        it), named in the feed when it is new, and the floor goes on with the other books; the next tick tries again."""
        async with self._busy:
            now_ms = self.now_ms()
            last: dict[str, Any] | None = None
            prev, errs = self.floor_errors, {}
            for v in self.venues:  # each venue's books against their own peaks
                if not v.managed:
                    continue
                failed: dict[str, str] = {}
                try:
                    current = self._read_current(now_ms, v)
                    plan = resolve(current, None, max_step=self.max_step, dd_floor_pct=self.dd_floor_pct)
                    todo = [s for s in plan.floored if current[s].weight > 0 or current[s].enabled]
                    if not todo:
                        continue
                    reasons = {s: self._floor_reason(v, current[s]) for s in todo}
                    partial = Plan({s: plan.enabled[s] for s in todo}, {s: plan.weights[s] for s in todo}, todo, [], [])
                    applied = self._apply(partial, {s: current[s] for s in todo}, reasons, now_ms, venue=v, errors=failed)
                except Exception as exc:  # noqa: BLE001 - this venue's books could not even be read: the others still run
                    todo, applied, plan, reasons = [], {}, None, {}
                    failed[""] = f"{type(exc).__name__}: {exc}"[:300]
                new = False
                for sid, err in failed.items():
                    key = self.vk(v.id, sid or "*")
                    errs[key] = {"venue": v.id, "venue_label": v.label, "strategy": sid or None, "error": err, "t_ms": now_ms}
                    if prev.get(key, {}).get("error") != err:
                        new = True
                        log.error("allocator: drawdown floor on %s %s failed: %s", v.id, sid or "(reading the venue)", err)
                if failed and not new and not any(applied.values()):
                    continue  # the same failure as last tick and nothing written: the alert and status.json already say it
                txt = "; ".join(f"{sid or 'reading the venue'}: {err}" for sid, err in failed.items())
                last = {"t_ms": now_ms, "model": "risk-floor", "trigger": "risk", "venue": v.id, "venue_label": v.label,
                        "weights": plan.weights if plan else {}, "enabled": plan.enabled if plan else {},
                        "reasoning": "; ".join(reasons.values()), "inputs": {"drawdown_pct": {s: _r(current[s].drawdown_pct, 2) for s in todo}},
                        "applied": applied, "skipped_owner_lock": plan.skipped_owner_lock if plan else [],
                        "skipped_kill_pending": [x for x in self._kill_skips if x in applied],
                        "skipped_paused": [x for x in self._paused_skips if x in applied], "risk_floor": todo, "attempts": 0,
                        "write_failed": {k: e for k, e in failed.items() if k},
                        "error": f"drawdown floor not applied, tried again every tick ({txt})" if failed else None}
                append_jsonl(self.dir / "decisions.jsonl", last)
            self.floor_errors = errs
            if last is not None or self._dirty:
                self._save_state()
            return last

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
        fc = self.load_forecast(now_ms) if self.mode == "decider" else None
        err = self.last_error or (fc.error if fc else None)  # an unusable forecast is an error; a missing one is a mode
        floor = [f"{e['venue_label']} {e['strategy'] or '(venue)'}: {e['error']}" for e in self.floor_errors.values()]
        if floor:  # the drawdown floor could not cut a book: said even while paused
            err = "; ".join(([err] if err else []) + [f"drawdown floor could not write control.json ({'; '.join(floor)})"])
        weights_mode = {None: "chat", "fresh": "computed"}.get(fc.state if fc else None, "held")
        nxt_w = None if fc is None or paused else latest_weights_slot_ms(now_ms, *self.calc.weights_utc) + DAY_MS
        write_json_atomic(self.dir / "status.json", {
            "state": "paused" if paused else ("error" if err else "running"),
            "model": self.model, "last_decision_ms": last, "next_decision_ms": nxt,
            "last_error": err, "cost_usd_today": self.cost_usd_today(now_ms),
            "interval_min": ctl["interval_min"], "updated_ms": now_ms,
            "last_reasoning": last_dec.get("reasoning") or None, "last_confidence": last_dec.get("confidence"),
            "weights_mode": weights_mode, "forecast": fc.info() if fc else None,
            "last_weights_ms": _num(self._state().get("last_weights_ms")), "next_weights_ms": nxt_w,
            "risk_floor_errors": list(self.floor_errors.values()),
            "decision_write_errors": list(self.decision_errors.values()),
        })

    # ---- loop
    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """The loop the dashboard server starts as a task. Wakes every tick_s: the drawdown floor
        always runs, on every managed venue, also while paused; a model decision runs when the interval has passed or
        decide_now is set, and none runs while allocator/control.json says enabled false."""
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
        # The drawdown floor runs on every managed venue every tick, paused or not: it only cuts (the paused rule in
        # _write_control lets a risk-reducing write through). A pause (AUTO-TRADE OFF, MANUAL, PER-SYMBOL BOT, a bybit-demo
        # KILL) stops decisions, never the floor on books that are still open.
        await self.risk_check()
        if not ctl["enabled"]:
            self.write_status(now_ms, paused=True)
            return
        st = read_json(self.dir / "status.json") or {}
        last = _num(st.get("last_decision_ms"))
        due = last is None or now_ms >= last + ctl["interval_min"] * 60_000
        if ctl["decide_now"]:
            with locked(self.dir / "control.json"):  # the dashboard writes this file too (pause, interval)
                raw = read_json(self.dir / "control.json")
                raw = raw if isinstance(raw, dict) else {}
                raw["decide_now"] = False
                write_json_atomic(self.dir / "control.json", raw)
            await self.decide(trigger="decide_now")
        elif self.mode == "decider" and self.load_forecast(now_ms).state == "fresh" and self.weights_due(now_ms):
            await self.decide(trigger="daily")  # the 23:50 UTC slot (or the first run): ask the decider, then set the weights
        elif due:
            await self.decide(trigger="schedule")
        else:
            self.write_status(now_ms)

    def reset_peak(self, ids: list[str] | None = None, venue: AllocVenue | None = None) -> list[str]:
        """Forget a (venue, strategy) book's peak (and samples) so the drawdown floor measures from now. Owner action,
        for a strategy that was cut by the floor and is being restarted on purpose. Default: the allocator venue."""
        venue = venue or self.primary
        st = self._state()
        done = []
        for r in self.strategies(venue):
            if ids and r["id"] not in ids:
                continue
            if st["peaks"].pop(self.vk(venue.id, r["id"]), None) is not None:
                done.append(r["id"])
            st["history"].pop(self.vk(venue.id, r["id"]), None)
        self._save_state()
        return done


# ---------------------------------------------------------------------------------- wiring and CLI
def create_allocator(session: aiohttp.ClientSession, root: Path | str | None = None, spec: ModelSpec | None = None,
                     venues: list[AllocVenue] | None = None) -> Allocator:
    """What the dashboard server calls: an Allocator on the harness's aiohttp session. The model is
    allocator_spec(): a decision model by default, a chat model when CTB_ALLOCATOR_MODEL says so. venues: the venues it
    applies its decision to, the allocator venue (root) first (ctb.venues.VenueHub.allocator_venues)."""
    spec = spec or allocator_spec()
    try:
        rph = int(os.environ.get("CTB_ALLOCATOR_MAX_RPH", "12"))
    except ValueError:
        rph = 12
    timeout = float(os.environ.get("CTB_KAIROS_TIMEOUT_S", "120"))
    s = Settings()
    params = MappingParams(s.allocator_p_zero, s.allocator_p_full, s.allocator_winner_min_p, s.allocator_winner_weight)
    if spec.kind == "chat":
        llm: Any = AllocatorClient(spec, session, timeout_s=timeout)
        mode = "chat"
    else:
        llm, mode = make_client(spec, session, timeout_s=timeout), "decider"
    return Allocator(root or live_root(), llm, BybitMarket(session), model=model_display(spec), budget=RateBudget(rph),
                     mode=mode, params=params, venues=venues)


# ---------------------------------------------------------------------------------- scoring
def _equity_at(hist: list, t_ms: int, tol_ms: int) -> float | None:
    best = min(hist, key=lambda x: abs(x[0] - t_ms), default=None)
    return float(best[1]) if best and abs(best[0] - t_ms) <= tol_ms else None


def score_decisions(root: Path | str, *, horizon_h: int = 24, now_ms: int | None = None, tol_min: int = 45,
                    min_weight: float = 0.1, qset: str | None = None) -> dict[str, Any]:
    """Score logged alloc-v1.0 and alloc-v1.1 answers (same questions; pass qset to keep one of them) against the realised net book return, from the equity at the
    decision (logged inputs) and the equity samples in allocator/state.json (kept 8 days: run this at
    least weekly). A book's return is divided by its weight, so it is the return on the capital the
    strategy deploys; a book held below min_weight cannot be labelled and is counted as flat."""
    alloc = Path(root) / ALLOCATOR_ID
    now_ms = now_ms or int(time.time() * 1000)
    state = read_json(alloc / "state.json") or {}
    hist = state.get("history", {})
    pv = state.get("primary_venue") or PRIMARY_VENUE  # the decisions are the allocator venue's: its keys, or legacy plain ids
    h_ms, tol = horizon_h * 3_600_000, tol_min * 60_000
    per: dict[str, dict[str, Any]] = {}
    best_stats = {"n": 0, "correct": 0, "p_realised": 0.0}
    pending, unlabelled = 0, {"flat": 0, "no_equity": 0}
    for d in tail_jsonl(alloc / "decisions.jsonl", 5000):
        ans = d.get("answers")
        if d.get("qset") not in (SCORED_QSETS if qset is None else (qset,)) or d.get("error") or d.get("dry_run") or not isinstance(ans, dict):
            continue
        t = int(d["t_ms"])
        if t + h_ms > now_ms:
            pending += 1
            continue
        strategies = ((d.get("inputs") or {}).get("strategies") or {})
        returns: dict[str, float] = {}
        for sid, p in (ans.get("p") or {}).items():
            e0 = _num((strategies.get(sid) or {}).get("equity_usdt"))
            e1 = _equity_at(hist.get(f"{pv}/{sid}") or hist.get(sid, []), t + h_ms, tol)
            w = _num((d.get("weights") or {}).get(sid)) or 0.0
            if w < min_weight:
                unlabelled["flat"] += 1
            elif e0 is None or e0 <= 0 or e1 is None:
                unlabelled["no_equity"] += 1
            else:
                r = (e1 / e0 - 1) / w
                returns[sid] = r
                st = per.setdefault(sid, {"n": 0, "pos": 0, "p_sum": 0.0, "brier": 0.0, "hits": 0})
                y = 1 if r > 0 else 0
                st["n"] += 1; st["pos"] += y; st["p_sum"] += p; st["brier"] += (p - y) ** 2; st["hits"] += int((p > 0.5) == bool(y))
        best = ans.get("best")
        if best and len(returns) == len(ans.get("p") or {}) and returns:
            top = max(returns, key=lambda k: returns[k])
            realised = top if returns[top] > 0 else NONE_ID
            best_stats["n"] += 1
            best_stats["correct"] += int(max(best, key=lambda k: best[k]) == realised)
            best_stats["p_realised"] += best.get(realised, 0.0)
    out_per = {}
    for sid, st in per.items():
        n, base = st["n"], st["pos"] / st["n"]
        out_per[sid] = {"n": n, "base_rate": round(base, 3), "mean_p": round(st["p_sum"] / n, 3), "brier": round(st["brier"] / n, 4),
                        "brier_always_0.5": 0.25, "brier_base_rate": round(base * (1 - base), 4), "hit_rate": round(st["hits"] / n, 3)}
    bn = best_stats["n"]
    return {"qset": qset or "+".join(SCORED_QSETS), "horizon_h": horizon_h, "pending": pending, "unlabelled": unlabelled, "strategies": out_per,
            "best": {"n": bn, "accuracy": round(best_stats["correct"] / bn, 3) if bn else None,
                     "mean_p_of_realised": round(best_stats["p_realised"] / bn, 3) if bn else None}}


FORECASTER_ORDER = ("base", "regime", "ridge", "structural", "decider", "combined")
RULE_COLUMNS = (("vol_cap", "vol"), ("buffer", "buf"), ("step", "step"), ("veto", "veto"), ("explore_floor", "floor"), ("floor_lock", "dd/lock"))


def format_table(rec: dict[str, Any]) -> str:
    """The computed weights as plain text tables (python -m ctb.allocator once --dry-run). Needs a record with
    the breakdown (mu, sigma_d, w_prev, core, tau, w_target, steps, w_final, p_decider, reasons); other records get one line."""
    fc = rec.get("forecast") or {}
    head = f"weights_mode {rec.get('weights_mode')}, trigger {rec.get('trigger')}, forecast {fc.get('state')} {fc.get('date') or ''} age {fc.get('age_h')} h"
    if "mu" not in rec:
        return head + "\n" + str(rec.get("reasoning") or "")
    ids = list(rec["w_final"])
    ks = [k for k in FORECASTER_ORDER if any(k in (rec["mu"].get(i) or {}) for i in ids)]
    out = [head, "", "mu, % per year (each forecast clipped to Sharpe +-1.5), p = mean of the decider's last answers:"]
    rows = [["id", *ks, "p", "sigma %/yr"]]
    for i in ids:
        mu = rec["mu"].get(i) or {}
        sd = (rec.get("sigma_d") or {}).get(i)
        p = (rec.get("p_decider") or {}).get(i)
        rows.append([i, *[("-" if mu.get(k) is None else f"{M.ann_pct(mu[k]):+.1f}") for k in ks], "-" if p is None else f"{p:.2f}",
                     "-" if sd is None else f"{M.sigma_ann_pct(sd):.1f}"])
    out += _grid(rows)
    out += ["", f"weights: previous, core, forecast Sharpe, live tilt (kappa {rec.get('kappa', 0.0)}), shadow tilt, target, after each rule in order, final:"]
    steps = {st["step"]: st for st in rec.get("steps") or []}
    rows = [["id", "prev", "core", "SR", "tilt", "shadow", "target", *[lab for _, lab in RULE_COLUMNS], "final"]]
    for i in ids:
        ws = (rec.get("w_target") or {}).get(i)
        co, sr, ta, sh = ((rec.get(k) or {}).get(i) for k in ("core", "sr", "tau", "shadow_tau"))
        rows.append([i, f"{rec['w_prev'][i]:.2f}", "-" if co is None else f"{co:.2f}", "-" if sr is None else f"{sr:+.2f}",
                     "-" if ta is None else f"{ta * 100:+.0f}%", "-" if sh is None else f"{sh * 100:+.0f}%", "-" if ws is None else f"{ws:.2f}",
                     *[("-" if key not in steps or i not in steps[key]["after"] else f"{steps[key]['after'][i]:.2f}") for key, _ in RULE_COLUMNS],
                     f"{rec['w_final'][i]:.2f}"])
    out += _grid(rows)
    out += ["", "reasons:", *[f"  {rec['reasons'][i]}" for i in ids if i in (rec.get("reasons") or {})]]
    return "\n".join(out)


def _grid(rows: list[list[str]]) -> list[str]:
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    return ["  " + "  ".join(cell.rjust(w) if c else cell.ljust(w) for c, (cell, w) in enumerate(zip(r, widths))) for r in rows]


def _jprint(obj: Any) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


async def _main(args: argparse.Namespace) -> int:
    from .config import load_dotenv
    load_dotenv()
    from .venues import allocator_venues_of, configs_from_env
    # the same venues the dashboard gives the allocator (CTB_LIVE_ROOTS, CTB_ALLOCATOR_VENUES); --root overrides bybit-demo's
    venues = allocator_venues_of(configs_from_env(), args.root)
    async with aiohttp.ClientSession() as session:
        alloc = create_allocator(session, venues[0].root, venues=venues)
        if args.cmd == "reset-peak":
            venue = next((v for v in alloc.venues if v.id == args.venue), None)
            if venue is None:
                print(f"unknown or unmanaged venue {args.venue!r}", file=sys.stderr)
                return 2
            print("peaks reset:", ", ".join(alloc.reset_peak(args.ids or None, venue=venue)) or "none")
            return 0
        if args.cmd == "score":
            _jprint(score_decisions(alloc.root, horizon_h=args.horizon_h, qset=args.qset))
            return 0
        if args.cmd == "once":
            if not alloc.llm.configured:
                print(f"{alloc.llm.spec.key_env} is not set", file=sys.stderr)
                return 2
            rec = await alloc.decide(dry_run=args.dry_run, trigger="cli")
            if args.dry_run and rec.get("weights_mode") in ("computed", "held", "refresh"):
                if rec.get("inputs"):
                    print(f"== STATE given to the model ({rec.get('qset')}, {(rec.get('state_bytes') or {}).get('total')} bytes) ==")
                    _jprint(rec["inputs"])
                    print("state bytes per block:", json.dumps(rec.get("state_bytes")))
                print(format_table(rec))
                print(f"decider: {rec.get('cost_usd', 0)} usd, error: {rec.get('error')}")
            elif args.dry_run:
                print(f"== CONTEXT given to the model ({rec.get('qset')}, {(rec.get('state_bytes') or {}).get('total')} bytes) ==")
                _jprint(rec.get("inputs"))
                print("state bytes per block:", json.dumps(rec.get("state_bytes")))
                print(f"== MODEL ANSWER ({rec['model']}) ==")
                _jprint(rec.get("answers") or (rec.get("requested") and {**rec["requested"], "confidence": rec["confidence"]}))
                if rec.get("mapping"):
                    print("== MAPPING (probability to target weight) ==")
                    _jprint(rec["mapping"])
                print("== AFTER THE RULES (not applied) ==")
                _jprint({k: rec[k] for k in ("reasoning", "weights", "enabled", "risk_floor", "step_limited", "skipped_owner_lock", "cost_usd", "error") if k in rec})
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
    sc = sub.add_parser("score", help="score logged alloc-v1.0 and v1.1 answers against realised book returns")
    sc.add_argument("--horizon-h", type=int, default=24)
    sc.add_argument("--qset", choices=list(SCORED_QSETS), help="score one question set only (default: both, same questions)")
    rp = sub.add_parser("reset-peak", help="restart a book's drawdown measurement from now")
    rp.add_argument("ids", nargs="*")
    rp.add_argument("--venue", default=PRIMARY_VENUE, help="the venue whose books to reset (default bybit-demo)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
