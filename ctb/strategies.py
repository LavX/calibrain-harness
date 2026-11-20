"""Strategies tab backend: watch and control the live strategies that trade on the Bybit DEMO account.

The strategies are not run by this bot. A separate live runner (a systemd user unit) trades them on the demo
account and writes plain files; this module

  * reads them READ-ONLY (never a credential, never a Bybit call): <root>/strategies.json, and per strategy
    <root>/<id>/status.json and trades.jsonl,
  * writes only <root>/<id>/control.json (enable, pause, flatten, reset latch), atomically,
  * asks systemd to start or stop the runner unit(s) named in strategies.json (`systemctl --user`, argument
    lists, no shell, only names found in strategies.json or in CTB_LIVE_UNITS).

Root: CTB_LIVE_ROOT (shared with ctb.allocator), default data/live; on aurora4 the runner's root,
/home/lavx/Documents/calibrain-study/data/live.

Auto-trade guard (both directions): the bot's own LLM auto-trade must never trade on the demo account the live
runner uses, because the positions would merge. It only matters on the demo venue. The bot's account label is
CTB_DASHBOARD_ACCOUNT; a runner strategy conflicts when its `account` equals it, or when the label is not set
(unknown means "assume the same").

Modes (the header's auto-trade; exactly one at a time):
  "llm"     LLM decides: the allocator (ctb/allocator.py) writes each strategy's control.json (enabled, weight, reason)
  "manual"  the owner switches strategies per card; the allocator is paused
  "bot"     the legacy per-symbol LLM bot (trading_enabled); no strategy may trade the same demo account
Only the chosen mode may trade a conflicting demo account. Switching to "bot" pauses the allocator and every strategy and
stops the runner unit (refused while any strategy holds a position: flatten it first); switching away from "bot" turns the
bot's auto-trade off (refused while it has open trades).
Control file ownership: a card switch flipped by the owner sets owner_lock=true, set_by="owner": the allocator must not
change that strategy until the owner unlocks it ("unlock to LLM").
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import ROOT
from .control_lock import locked

DEFAULT_ROOT = ROOT / "data" / "live"  # the same default as ctb.allocator; on aurora4 CTB_LIVE_ROOT points at the runner's root
UNIT_RE = re.compile(r"^[A-Za-z0-9_.:-]+(@[A-Za-z0-9_.:-]+)?\.service$")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
ACTIONS = ("enable", "pause", "unlock", "flatten", "reset", "cancel_kill")
# A pending kill (control.json flatten: true) ends only when the runner, flat, clears the flag itself (research/live/runner.py
# _flatten_clear), or when the owner explicitly cancels it. Nothing else may clear it or switch the strategy on over it.
KILL_PENDING_MSG = ("a kill is pending for {names}: the runner has not reported it flat yet. Cancel the kill first (its positions "
                    "then stay open), or wait until the runner closes them and clears the flag.")
MODES = ("llm", "manual", "bot")
ALLOCATOR = "allocator"  # data/live/allocator/ holds the LLM allocator's files; not a strategy id
# The one map from strategy id to the short name every screen shows (F1 matrix, F7 cards, alerts). Unknown ids fall back
# to the label up to its first bracket, then the id.
SHORT_NAMES = {"carry": "Carry", "dated_carry": "Dated carry", "trend": "Trend", "xsmom": "Momentum", "basis": "Basis",
               "longvol": "Long-only", "events": "Events"}
FEED_N = 15
RUNNER_ACTIONS = ("start", "stop")
CACHE_S = 15.0
CMD_TIMEOUT_S = 30.0
STALE_S = 180.0  # the runner rewrites status.json at least every 60 s
ACTIVE_STATES = ("active", "activating", "reloading", "deactivating")
TRADES_TAIL = 12
TRADES_READ_BYTES = 256 * 1024
EQUITY_POINTS = 720


class StrategyError(Exception):
    """A refusal with an HTTP status and a message fit to show the owner. `detail` carries what was done anyway (a kill
    that wrote its flags but could not confirm the runner reads them), returned next to the message."""

    def __init__(self, status: int, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail or {}


def live_root() -> Path:
    return Path(os.environ.get("CTB_LIVE_ROOT") or DEFAULT_ROOT)


def short_name(sid: str, label: str = "") -> str:
    if sid in SHORT_NAMES:
        return SHORT_NAMES[sid]
    head = re.split(r"[ (]", label.strip(), maxsplit=1)[0] if label.strip() else ""
    return head or sid


# ------------------------------------------------------------------------------------------------ registry
@dataclass(frozen=True)
class Entry:
    id: str
    label: str
    unit: str | None
    account: str


@dataclass(frozen=True)
class Registry:
    entries: tuple[Entry, ...] = ()
    units: tuple[str, ...] = ()  # the allowlist: units named in strategies.json plus CTB_LIVE_UNITS
    error: str | None = None  # strategies.json exists but cannot be used: the guard then fails closed

    def get(self, sid: str) -> Entry | None:
        return next((e for e in self.entries if e.id == sid), None)


def parse_registry(doc: Any, extra_units: tuple[str, ...] = ()) -> Registry:
    if not isinstance(doc, list):
        raise ValueError("expected a list of strategies")
    seen: set[str] = set()
    entries: list[Entry] = []
    for d in doc:
        if not isinstance(d, dict):
            raise ValueError("each strategy must be an object")
        sid = d.get("id")
        if not isinstance(sid, str) or not ID_RE.fullmatch(sid):
            raise ValueError(f"strategy id {sid!r}: lowercase letters, digits, - and _ only (max 32)")
        if sid == ALLOCATOR:
            raise ValueError(f"strategy id {sid!r} is reserved for the LLM allocator")
        if sid in seen:
            raise ValueError(f"duplicate strategy id {sid!r}")
        seen.add(sid)
        unit = d.get("unit")
        if unit is not None and (not isinstance(unit, str) or not UNIT_RE.fullmatch(unit)):
            raise ValueError(f"{sid}: unit {unit!r} is not a plain .service name")
        entries.append(Entry(sid, str(d.get("label") or sid), unit, str(d.get("account") or "")))
    units = tuple(dict.fromkeys([e.unit for e in entries if e.unit] + list(extra_units)))
    return Registry(tuple(entries), units)


def extra_units_from_env() -> tuple[str, ...]:
    out = []
    for u in (os.environ.get("CTB_LIVE_UNITS") or "").split(","):
        u = u.strip()
        if u and UNIT_RE.fullmatch(u):
            out.append(u)
    return tuple(out)


def load_registry(root: Path, extra_units: tuple[str, ...] = ()) -> Registry:
    """No strategies.json means the runner has not started yet (no strategies, guard on only for CTB_LIVE_UNITS)."""
    try:
        text = (root / "strategies.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return Registry(units=extra_units)
    except OSError as exc:
        return Registry(units=extra_units, error=f"cannot read strategies.json: {exc}")
    try:
        return parse_registry(json.loads(text), extra_units)
    except ValueError as exc:
        return Registry(units=extra_units, error=f"strategies.json: {exc}")


# ------------------------------------------------------------------------------------------------ files
def read_json(path: Path) -> dict | None:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def tail_jsonl(path: Path, n: int = TRADES_TAIL) -> list[dict]:
    """The last n complete lines of a JSON-lines file, newest first. Reads only the end of the file."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - TRADES_READ_BYTES))
            data = fh.read()
    except OSError:
        return []
    lines = data[:data.rfind(b"\n") + 1].splitlines()
    out: list[dict] = []
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
            if len(out) >= n:
                break
    return out


def write_control(path: Path, doc: dict) -> None:
    """Atomic write: the runner never reads half a file. The temp name is unique to this process and thread: the runner
    writes control.json through a fixed control.json.tmp, and two writers sharing one temp file can install a mixed file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def clean_status(doc: dict | None) -> dict | None:
    """Only the contract's fields, with the right types; anything else the runner wrote is dropped."""
    if doc is None:
        return None
    pos = []
    for p in doc.get("positions") if isinstance(doc.get("positions"), list) else []:
        if isinstance(p, dict):
            pos.append({"symbol": str(p.get("symbol", "")), "side": str(p.get("side", "")), "market": str(p.get("market", "")),
                        **{k: _num(p.get(k)) for k in ("qty", "entry_price", "mark_price", "upnl_usdt")}})
    tgt = [{"symbol": str(t.get("symbol", "")), "market": str(t.get("market", "")), "target_qty": _num(t.get("target_qty"))}
           for t in (doc.get("targets") if isinstance(doc.get("targets"), list) else []) if isinstance(t, dict)]
    return {"enabled": bool(doc.get("enabled")), "state": str(doc.get("state") or "error"),
            # what the runner applied (control.json is only the request): weight, who set it, why, and the owner lock
            "weight": _num(doc.get("weight")), "set_by": doc.get("set_by") if doc.get("set_by") in ("llm", "owner") else None,
            "reason": str(doc["reason"])[:500] if doc.get("reason") else None, "owner_lock": bool(doc.get("owner_lock")),
            "updated_ms": _num(doc.get("updated_ms")), "last_rebalance_ms": _num(doc.get("last_rebalance_ms")),
            "next_rebalance_ms": _num(doc.get("next_rebalance_ms")),
            "last_error": str(doc["last_error"])[:500] if doc.get("last_error") else None,
            **{k: _num(doc.get(k)) for k in ("equity_usdt", "start_equity_usdt", "pnl_usdt", "pnl_pct", "fees_usdt", "funding_usdt")},
            "positions": pos, "targets": tgt,
            "trades_today": int(doc["trades_today"]) if isinstance(doc.get("trades_today"), int) else None,
            "trades_total": int(doc["trades_total"]) if isinstance(doc.get("trades_total"), int) else None}


def is_idle(status: dict) -> bool:
    """Switched on and running, but holding nothing and wanting nothing: no position, no non-zero target."""
    return not status["positions"] and not any(t.get("target_qty") for t in status["targets"])


def pill(status: dict | None, now_ms: float, runner_active: bool) -> str:
    if status is None:
        return "no_data"
    if status["updated_ms"] is None or now_ms - status["updated_ms"] > STALE_S * 1000:
        return "stale" if runner_active else "runner_stopped"
    if status["state"] in ("latched", "error"):
        return status["state"]
    if status["enabled"] and status["state"] == "running":
        return "idle" if is_idle(status) else "running"
    return "paused"


# ------------------------------------------------------------------------------------------------ systemd
Runner = Callable[..., Awaitable[tuple[int, str, str]]]
SHOW_PROPS = "LoadState,ActiveState,SubState,UnitFileState,Result,ActiveEnterTimestamp"


def _env() -> dict[str, str]:
    """A small environment built from scratch: nothing from the dashboard's own environment (its keys) reaches a child."""
    base = {k: os.environ[k] for k in ("PATH", "HOME", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "LANG") if k in os.environ}
    base.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    return base


async def run_cmd(argv: list[str], *, timeout: float = CMD_TIMEOUT_S) -> tuple[int, str, str]:
    """Run an argument list (never a shell string). Returns (exit code, stdout, stderr), output capped."""
    try:
        proc = await asyncio.create_subprocess_exec(*argv, env=_env(), stdin=asyncio.subprocess.DEVNULL,
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    except OSError as exc:
        return 127, "", f"cannot run {argv[0]}: {exc}"
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"{argv[0]} timed out after {timeout:.0f}s"
    return proc.returncode or 0, out.decode(errors="replace")[-2000:], err.decode(errors="replace")[-2000:]


async def unit_state(run: Runner, unit: str) -> dict[str, Any]:
    """`systemctl --user show` of one unit. Timestamps are asked for as unix seconds ("@1795170550", systemd 248+), so the
    page can show them in UTC; an older systemd that rejects the flag is asked again without it (host-local text)."""
    argv = ["systemctl", "--user", "show", "--timestamp=unix", f"--property={SHOW_PROPS}", "--", unit]
    rc, out, err = await run(argv, timeout=5)
    if rc != 0 and "=" not in out:
        rc, out, err = await run([a for a in argv if a != "--timestamp=unix"], timeout=5)
    props = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    if rc != 0 and not props:
        return {"unit": unit, "load": "unknown", "active": "unknown", "error": (err or "systemctl failed").strip()[:200]}
    since = props.get("ActiveEnterTimestamp", "")
    return {"unit": unit, "load": props.get("LoadState", ""), "active": props.get("ActiveState", ""), "sub": props.get("SubState", ""),
            "enabled": props.get("UnitFileState", ""), "result": props.get("Result", ""), "since": since,
            "since_s": int(since[1:]) if since.startswith("@") and since[1:].isdigit() else None}


def is_active(st: dict[str, Any]) -> bool:
    return st.get("active") in ACTIVE_STATES


# ------------------------------------------------------------------------------------------------ service
DECISION_KEYS = ("t_ms", "model", "trigger", "weights", "enabled", "reasoning", "applied", "skipped_owner_lock", "skipped_kill_pending",
                 "skipped_paused", "risk_floor", "write_failed", "error", "venue", "venue_label", "venues")


def clean_allocator(status: dict | None, control: dict | None, decisions: list[dict]) -> dict[str, Any]:
    """The allocator panel's data: its status.json, the dashboard's control.json and the newest decisions (inputs dropped)."""
    stt = None
    if status is not None:
        stt = {"state": str(status.get("state") or "error"), "model": status.get("model"),
               # last/next_weights_ms: when the daily weights were last written and are next due (the hourly decision only
               # refreshes the decider's p); updated_ms: the loop's last tick
               **{k: _num(status.get(k)) for k in ("last_decision_ms", "next_decision_ms", "cost_usd_today", "last_weights_ms",
                                                   "next_weights_ms", "updated_ms")},
               "weights_mode": status.get("weights_mode") if isinstance(status.get("weights_mode"), str) else None,
               "last_error": str(status["last_error"])[:500] if status.get("last_error") else None,
               # the drawdown floor's failed writes, per (venue, strategy): the venue alerts name them
               "risk_floor_errors": [{"venue": str(e.get("venue") or ""), "venue_label": str(e.get("venue_label") or e.get("venue") or ""),
                                      "strategy": str(e["strategy"]) if e.get("strategy") else None, "error": str(e.get("error") or "")[:300],
                                      "t_ms": _num(e.get("t_ms"))}
                                     for e in (status.get("risk_floor_errors") or []) if isinstance(e, dict)][:50]}
    ctl = {"enabled": bool(control and control.get("enabled")), "interval_min": int(control["interval_min"]) if control and isinstance(control.get("interval_min"), int) else None,
           "decide_now_pending": bool(control and control.get("decide_now"))}
    return {"status": stt, "control": ctl, "decisions": [{k: d.get(k) for k in DECISION_KEYS} for d in decisions]}


def control_view(ctl: dict, st: dict | None) -> dict[str, Any]:
    return {"enabled": bool(ctl["enabled"]) if "enabled" in ctl else (st["enabled"] if st else False),
            "weight": _num(ctl.get("weight")), "set_by": ctl.get("set_by") if ctl.get("set_by") in ("llm", "owner") else None,
            "owner_lock": bool(ctl.get("owner_lock")), "reason": str(ctl["reason"])[:500] if ctl.get("reason") else None,
            "flatten_pending": bool(ctl.get("flatten")), "reset_pending": bool(ctl.get("reset_latch")), "updated_ms": _num(ctl.get("updated_ms"))}


class StrategyService:
    def __init__(self, root: Path | None = None, run: Runner = run_cmd, clock: Callable[[], float] = time.time,
                 extra_units: tuple[str, ...] | None = None, dashboard_account: str | None = None,
                 allocator_available: bool = False, allocator: Any = None, shares_bot_account: bool = True,
                 auto_set_path: Path | None = None, has_allocator: bool = True) -> None:
        self.root = root if root is not None else live_root()
        # False for a venue on another exchange (OKX, Deribit, Bitget): the per-symbol bot never trades there, so the
        # one-mode-at-a-time guard does not apply to its strategies.
        self.shares_bot_account = shares_bot_account
        # False for an owner-controlled venue: no LLM allocator writes its controls, so "unlock to the LLM" is refused there.
        self.has_allocator = has_allocator
        self.auto_set_path = auto_set_path  # the MANUAL header set, kept by the harness (None: every strategy)
        self.run, self.clock = run, clock
        self.extra_units = extra_units if extra_units is not None else extra_units_from_env()
        self.dashboard_account = dashboard_account if dashboard_account is not None else (os.environ.get("CTB_DASHBOARD_ACCOUNT") or None)
        self.allocator = allocator  # the running ctb.allocator.Allocator (set by the server at startup), or None
        self.allocator_available = allocator_available or allocator is not None
        self._cache: tuple[float, dict] | None = None
        self._lock = asyncio.Lock()
        self._equity: dict[str, deque] = {}
        self._mode: str | None = None

    def registry(self) -> Registry:
        return load_registry(self.root, self.extra_units)

    # ----------------------------------------------------------------- the guard
    def conflicting_units(self, reg: Registry, venue: str) -> list[str]:
        """Runner units that trade the account the bot's own auto-trade would use. Only the demo venue can collide."""
        if venue != "demo" or not self.shares_bot_account:
            return []
        mine = self.dashboard_account
        accounts: dict[str, set[str]] = {}
        for e in reg.entries:
            if e.unit:
                accounts.setdefault(e.unit, set()).add(e.account)
        return [u for u in reg.units if mine is None or u not in accounts or mine in accounts[u]]

    # ----------------------------------------------------------------- files
    def _control(self, e: Entry) -> dict:
        return read_json(self.root / e.id / "control.json") or {}

    def _control_by_id(self, sid: str) -> dict:
        return read_json(self.root / sid / "control.json") or {}

    def _enabled(self, e: Entry) -> bool:
        """Is the strategy switched on: control.json if there is one, else what the runner reports."""
        ctl = self._control(e)
        if "enabled" in ctl:
            return bool(ctl["enabled"])
        stt = read_json(self.root / e.id / "status.json")
        return bool(stt and stt.get("enabled"))

    def _allocator_on(self) -> bool:
        return (read_json(self.root / ALLOCATOR / "control.json") or {}).get("enabled") is True

    def attach_allocator(self, allocator: Any) -> None:
        """Called by the server at startup, before the allocator loop starts. The allocator treats a missing
        allocator/control.json as enabled, so write it first: the loop only writes strategy controls in LLM DECIDES."""
        self.allocator, self.allocator_available = allocator, allocator is not None
        reg = self.registry()
        self._write_allocator(enabled=self.mode(reg) == "llm")

    def _write(self, e: Entry, *, refuse_if_kill: bool = False, require_kill: bool = False, derive: Callable[[dict], dict] | None = None,
               **changes: Any) -> None:
        """Merge changes into the strategy's control.json (unknown keys the runner may use are kept), atomically, under the
        file's lock (ctb.control_lock, shared with the allocator). refuse_if_kill re-checks the kill flag inside the lock, so
        an enable that awaited (systemctl) in between cannot switch on over a kill written meanwhile; derive computes changes
        from the current file inside the lock."""
        path = self.root / e.id / "control.json"
        with locked(path):
            cur = read_json(path) or {}
            if refuse_if_kill and cur.get("flatten") is True:  # never cleared here: only the runner (flat) or the owner's cancel ends it
                raise StrategyError(409, KILL_PENDING_MSG.format(names=e.label), detail={"kill_pending": [e.id]})
            if require_kill and cur.get("flatten") is not True:  # checked here, at the write: the runner may have just cleared it
                raise StrategyError(409, f"no kill is pending for {e.label}")
            ctl = {**cur, "enabled": self._enabled(e), "flatten": bool(cur.get("flatten")), "reset_latch": bool(cur.get("reset_latch")),
                   **(derive(cur) if derive else {}), **changes, "updated_ms": int(self.clock() * 1000)}
            try:
                write_control(path, ctl)
            except OSError as exc:
                raise StrategyError(500, f"cannot write control.json: {exc.strerror or exc}") from exc

    def _write_owner(self, e: Entry, enabled: bool, *, cancel_kill: bool = False) -> None:
        """The owner flips a switch: the allocator may not change this strategy until it is unlocked. A pending kill is left
        alone unless the owner explicitly cancels it (cancel_kill), which clears flatten so the runner does not close the book;
        without cancel_kill an "on" is refused if a kill is pending when the write happens."""
        def weight(cur: dict) -> dict:
            w = _num(cur.get("weight"))
            return {"weight": 1.0 if enabled and not w else (w if w is not None else 1.0)}

        self._write(e, refuse_if_kill=enabled and not cancel_kill, derive=weight, enabled=enabled, owner_lock=True, set_by="owner",
                    reason="owner", **({"flatten": False} if cancel_kill else {}))

    def _kill_pending(self, e: Entry) -> bool:
        return self._control(e).get("flatten") is True

    def _refuse_if_kill_pending(self, entries: list[Entry], cancel_kill: bool) -> None:
        """A status.json that shows no position does not prove the book flat (it can predate a fill or a working order), so a
        pending kill is never waved through here: the runner clears it when flat, or the owner cancels it."""
        pending = [e for e in entries if self._kill_pending(e)]
        if pending and not cancel_kill:
            raise StrategyError(409, KILL_PENDING_MSG.format(names=", ".join(e.label for e in pending)),
                                detail={"kill_pending": [e.id for e in pending]})

    def _write_allocator(self, **changes: Any) -> None:
        d = self.root / ALLOCATOR
        try:
            d.mkdir(parents=True, exist_ok=True)
            with locked(d / "control.json"):
                cur = read_json(d / "control.json") or {}
                write_control(d / "control.json", {"enabled": bool(cur.get("enabled")), "interval_min": cur.get("interval_min", 60),
                                                   "decide_now": bool(cur.get("decide_now")), **cur, **changes,
                                                   "updated_ms": int(self.clock() * 1000)})
        except OSError as exc:
            raise StrategyError(500, f"cannot write the allocator's control.json: {exc.strerror or exc}") from exc

    async def _pause_all(self, reg: Registry) -> None:
        for e in reg.entries:
            if (self.root / e.id).is_dir():
                self._write(e, enabled=False)

    # ----------------------------------------------------------------- mode
    def mode(self, reg: Registry) -> str:
        """The mode. Kept in memory; after a restart: allocator switched on is "llm", else any strategy on or any strategy
        known is "manual", else "bot" (no strategies means the dashboard behaves as before)."""
        if self._mode is None:
            if self._allocator_on():
                self._mode = "llm"
            else:
                self._mode = "manual" if reg.entries else "bot"
        return self._mode

    def _holding(self, reg: Registry, venue: str) -> list[str]:
        units = set(self.conflicting_units(reg, venue))
        out = []
        for e in reg.entries:
            stt = read_json(self.root / e.id / "status.json")
            if e.unit in units and stt and stt.get("positions"):
                out.append(e.label)
        return out

    async def trading_block(self, venue: str) -> str | None:
        """Why the legacy per-symbol bot's auto-trade may not be enabled now, or None."""
        if venue != "demo":
            return None
        reg = self.registry()
        if reg.error:
            return f"the live runner's {reg.error}; fix it before trading on the demo account"
        conflicting = self.conflicting_units(reg, venue)
        for u in conflicting:
            if is_active(await unit_state(self.run, u)):
                return (f"the live strategy runner ({u}) is running on the demo account. Switch the mode to PER-SYMBOL BOT (it pauses "
                        "the strategies and stops the runner) or stop the runner first: the positions would merge.")
        if conflicting and self.mode(reg) != "bot":
            return f"the auto-trade mode is {self.mode(reg).upper()}. Switch the mode to PER-SYMBOL BOT first."
        return None

    async def set_mode(self, mode: str, *, venue: str, bot_open: bool) -> dict[str, Any]:
        """Switch the mode. Returns {"mode", "changed", "llm_off"}: the caller turns the per-symbol bot's auto-trade off when llm_off."""
        if mode not in MODES:
            raise StrategyError(400, f"mode must be one of {', '.join(MODES)}")
        reg = self.registry()
        cur = self.mode(reg)
        if mode == cur:
            return {"mode": mode, "changed": False, "llm_off": False}
        conflict = bool(self.conflicting_units(reg, venue))
        if mode == "bot":
            if conflict:
                held = self._holding(reg, venue)
                if held:
                    raise StrategyError(409, f"{', '.join(held)} hold positions on the demo account. FLATTEN them first, then switch to the per-symbol bot.")
                self._write_allocator(enabled=False)
                await self._pause_all(reg)
                for u in self.conflicting_units(reg, venue):
                    if is_active(await unit_state(self.run, u)):
                        rc, _, err = await self.run(["systemctl", "--user", "stop", "--", u])
                        if rc != 0:
                            raise StrategyError(502, f"stop {u} failed (exit {rc}): {(err or 'no output').strip()[-300:]}")
            self._mode = mode
            self._cache = None
            return {"mode": mode, "changed": True, "llm_off": False}
        if cur == "bot" and conflict and bot_open:
            raise StrategyError(409, "the per-symbol bot has open trades on the demo account. Close them (KILL, or wait for their exits) "
                                     "before the strategies take over the account.")
        if mode == "manual":
            self._write_allocator(enabled=False)  # the allocator is paused; strategies stay as they are
        self._mode = mode
        self._cache = None
        return {"mode": mode, "changed": True, "llm_off": cur == "bot"}

    async def _ensure_running(self, units: list[str]) -> None:
        for u in dict.fromkeys(units):
            st = await unit_state(self.run, u)
            if st.get("load") == "not-found":
                raise StrategyError(409, f"{u} is not installed on this machine (see docs/strategies-tab.md)")
            if not is_active(st):
                rc, _, err = await self.run(["systemctl", "--user", "start", "--", u])
                if rc != 0:
                    raise StrategyError(502, f"start {u} failed (exit {rc}): {(err or 'no output').strip()[-300:]}")

    def _may_trade(self, reg: Registry, e: Entry, *, llm_trading: bool, venue: str) -> None:
        """Raises unless strategy e may be switched on: the per-symbol bot is not the mode and its auto-trade is off."""
        if e.unit not in self.conflicting_units(reg, venue):
            return
        if llm_trading:
            raise StrategyError(409, "the per-symbol bot's AUTO-TRADE is ON for the demo account. Switch the mode to LLM DECIDES or MANUAL "
                                     "(it turns the bot's auto-trade off first) before switching a strategy on: the positions would merge.")
        if self.mode(reg) == "bot":
            raise StrategyError(409, "the auto-trade mode is PER-SYMBOL BOT. Switch the mode to LLM DECIDES or MANUAL first.")

    async def auto(self, enabled: bool, ids: list[str] | None, *, llm_trading: bool, venue: str,
                   cancel_kill: bool | list[str] = False) -> dict[str, Any]:
        """The header switch. LLM DECIDES: the allocator on (and the runner running), or the allocator off and every strategy paused.
        MANUAL: the selected strategies on (owner-locked, runner running), or every strategy off. In MANUAL a selected strategy with a
        pending kill is refused unless cancel_kill names it (a list of the ids the owner confirmed; True means every target); a
        kill on any other target is refused, also when it lands while the runner is being started (rechecked at the write). The
        allocator itself never switches on a strategy with a pending kill."""
        reg = self.registry()
        mode = self.mode(reg)
        if mode == "bot":
            raise StrategyError(409, "the mode is PER-SYMBOL BOT: its switch is /api/trading")
        if not enabled:
            if mode == "llm":
                self._write_allocator(enabled=False)
            await self._pause_all(reg)
            self._cache = None
            return {"ok": True, "enabled": []}
        if reg.error:
            raise StrategyError(409, f"strategies.json is unusable: {reg.error}")
        if ids is None and mode == "manual":
            ids = self.auto_set()
        targets = list(reg.entries) if mode == "llm" else [e for e in reg.entries if ids is None or e.id in ids]
        if not targets:
            raise StrategyError(409, "no strategy selected")
        for e in targets:
            self._may_trade(reg, e, llm_trading=llm_trading, venue=venue)
            if mode == "manual" and not (self.root / e.id).is_dir():
                raise StrategyError(409, f"{self.root / e.id} does not exist: the runner has not created {e.label} yet")
        cancel_ids = {e.id for e in targets} if cancel_kill is True else set(cancel_kill or [])
        if mode == "manual":
            self._refuse_if_kill_pending([e for e in targets if e.id not in cancel_ids], False)
        if mode == "llm" and not self.allocator_available:
            raise StrategyError(409, "the LLM allocator is not installed in this build (ctb.allocator); use MANUAL")
        await self._ensure_running([e.unit for e in targets if e.unit])
        if mode == "llm":
            self._write_allocator(enabled=True)
        else:
            for e in targets:
                self._write_owner(e, True, cancel_kill=e.id in cancel_ids)
        self._cache = None
        return {"ok": True, "enabled": [e.id for e in targets]}

    def auto_set(self) -> list[str] | None:
        """Strategy ids the header switch turns on in MANUAL mode, or None for every strategy (nothing saved yet)."""
        if self.auto_set_path is None:
            return None
        doc = read_json(self.auto_set_path)
        ids = doc.get("ids") if doc else None
        return [str(i) for i in ids if isinstance(i, str)] if isinstance(ids, list) else None

    def set_auto_set(self, ids: Any) -> dict[str, Any]:
        if self.auto_set_path is None:
            raise StrategyError(409, "this venue has no header set")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise StrategyError(400, "ids must be a list of strategy ids")
        known = {e.id for e in self.registry().entries}
        if unknown := sorted(set(ids) - known):
            raise StrategyError(400, f"unknown strategy ids: {', '.join(unknown)}")
        try:
            self.auto_set_path.parent.mkdir(parents=True, exist_ok=True)
            write_control(self.auto_set_path, {"ids": sorted(set(ids)), "updated_ms": int(self.clock() * 1000)})
        except OSError as exc:
            raise StrategyError(500, f"cannot save the header set: {exc.strerror or exc}") from exc
        self._cache = None
        return {"ok": True, "ids": sorted(set(ids))}

    # ----------------------------------------------------------------- reads
    def _read(self, e: Entry) -> dict[str, Any]:
        d = self.root / e.id
        return {"status": clean_status(read_json(d / "status.json")), "control": read_json(d / "control.json"),
                "trades": tail_jsonl(d / "trades.jsonl")}

    def _read_allocator(self) -> dict[str, Any]:
        a = self.allocator
        if a is not None:  # the running allocator's own readers (same files, same contract)
            try:
                return clean_allocator(a.status(), a.control(), a.decisions(FEED_N))
            except Exception:  # noqa: BLE001 - fall back to the files
                pass
        d = self.root / ALLOCATOR
        return clean_allocator(read_json(d / "status.json"), read_json(d / "control.json"), tail_jsonl(d / "decisions.jsonl", FEED_N))

    def _sample(self, sid: str, now: float, equity: float | None) -> list[list[float]]:
        ring = self._equity.setdefault(sid, deque(maxlen=EQUITY_POINTS))
        if equity is not None and (not ring or now - ring[-1][0] >= 60):
            ring.append((now, equity))
        return [[t, v] for t, v in ring]

    async def snapshot(self, force: bool = False) -> dict[str, Any]:
        async with self._lock:
            now = self.clock()
            if not force and self._cache and now - self._cache[0] < CACHE_S:
                # unit states and status files are cached; the allocator panel and the controls are small files, always fresh
                doc = dict(self._cache[1])
                alloc = await asyncio.to_thread(self._read_allocator)
                alloc["available"] = self.allocator_available
                doc["allocator"] = alloc
                doc["strategies"] = [{**v, "control": control_view(self._control_by_id(v["id"]), v["status"])} for v in doc["strategies"]]
                doc["auto_on"] = alloc["control"]["enabled"] if doc["mode"] == "llm" else any(v["control"]["enabled"] for v in doc["strategies"])
                doc["auto_set"] = self.auto_set()
                return doc
            reg = self.registry()
            units = {u: await unit_state(self.run, u) for u in reg.units}
            reads = await asyncio.gather(*(asyncio.to_thread(self._read, e) for e in reg.entries))
            alloc = await asyncio.to_thread(self._read_allocator)
            alloc["available"] = self.allocator_available
            views = []
            for e, r in zip(reg.entries, reads):
                st, ctl = r["status"], r["control"] or {}
                u = units.get(e.unit) if e.unit else None
                runner_active = bool(u and is_active(u))
                views.append({
                    "id": e.id, "name": short_name(e.id, e.label), "label": e.label, "unit": e.unit, "account": e.account, "status": st,
                    "pill": pill(st, now * 1000, runner_active),
                    "control": control_view(ctl, st),
                    "trades": r["trades"], "equity_history": self._sample(e.id, now, st["equity_usdt"] if st else None),
                    "runner_active": runner_active,
                })
            mode = self.mode(reg)
            doc = {"mode": mode, "error": reg.error, "root": str(self.root), "dashboard_account": self.dashboard_account, "generated_at": now,
                   "auto_on": alloc["control"]["enabled"] if mode == "llm" else any(v["control"]["enabled"] for v in views),
                   "allocator": alloc, "runners": list(units.values()), "strategies": views, "auto_set": self.auto_set()}
            self._cache = (now, doc)
            return doc

    # ----------------------------------------------------------------- actions
    def owns_account(self) -> bool:
        """True when the live strategies (not the per-symbol bot) own the demo account: any mode but "bot"."""
        return self.mode(self.registry()) != "bot"

    def kill_all(self, pause_allocator: bool = True) -> dict[str, Any]:
        """Dashboard kill switch: pause the allocator, then pause every strategy (owner-locked) and ask the runner to
        flatten it. The runner closes positions through its own books. Always records the owner's intent, whether or not
        the runner is up (ctb.venues.VenueHub.kill says whether it is confirmed). A failed write does not stop the others.
        Every strategy gets the flatten flag, also one whose status shows it flat: a status can predate a fill or a working
        order, and a live runner clears the flag itself on its first loop when the book really is flat.
        Returns {"written": [flagged ids], "failed": {id or "allocator": why}, "missing": [ids without a directory],
        "allocator_paused": bool}."""
        reg = self.registry()
        failed: dict[str, str] = {}
        if pause_allocator:
            try:
                self._write_allocator(enabled=False)
            except StrategyError as exc:
                failed[ALLOCATOR] = exc.message
        done: list[str] = []
        missing: list[str] = []
        for e in reg.entries:
            if not (self.root / e.id).is_dir():
                missing.append(e.id)  # no directory: nothing to write into; named so the answer can say so
                continue
            try:
                self._write_owner(e, False)
                self._write(e, flatten=True, reason="kill switch from the dashboard")
                done.append(e.id)
            except StrategyError as exc:
                failed[e.id] = exc.message
        self._cache = None
        return {"written": done, "failed": failed, "missing": missing, "allocator_paused": pause_allocator and ALLOCATOR not in failed}

    async def act(self, sid: str, action: str, *, llm_trading: bool, venue: str, cancel_kill: bool = False) -> dict[str, Any]:
        """Per-card actions: enable / pause (the Auto trade switch, owner-locked), unlock (back to the LLM), flatten, reset, and
        cancel_kill (clear a pending kill's flatten flag: the positions stay open). enable and unlock are refused while a kill is
        pending unless cancel_kill is set (the page asks first, saying the positions will NOT be closed)."""
        action = "pause" if action == "disable" else action  # "disable" reads better in the venue routes; same switch
        reg = self.registry()
        e = reg.get(sid)
        if e is None or action not in ACTIONS:
            raise StrategyError(404, "unknown strategy or action")
        if not (self.root / e.id).is_dir():
            raise StrategyError(409, f"{self.root / e.id} does not exist: the runner has not created this strategy yet")
        st = clean_status(read_json(self.root / e.id / "status.json"))
        if action == "enable":
            self._may_trade(reg, e, llm_trading=llm_trading, venue=venue)
            self._refuse_if_kill_pending([e], cancel_kill)
            if e.unit:
                await self._ensure_running([e.unit])
            self._write_owner(e, True, cancel_kill=cancel_kill)
        elif action == "pause":
            self._write_owner(e, False)
        elif action == "unlock":
            if not self.has_allocator:
                raise StrategyError(409, "this venue is controlled by the owner (CTB_ALLOCATOR_VENUES leaves it out): there is no LLM allocator to hand the strategy to")
            self._refuse_if_kill_pending([e], cancel_kill)
            self._write(e, refuse_if_kill=not cancel_kill, owner_lock=False, set_by="llm", reason="unlocked: the LLM decides at its next decision",
                        **({"flatten": False} if cancel_kill else {}))
        elif action == "cancel_kill":  # cleared, and the strategy paused and owner-locked, as the page's confirm says
            self._write(e, require_kill=True, flatten=False, enabled=False, owner_lock=True, set_by="owner",
                        reason="kill cancelled by the owner: positions stay open, paused")
        else:
            if not (e.unit and is_active(await unit_state(self.run, e.unit))):
                raise StrategyError(409, f"the runner ({e.unit or 'unit unknown'}) is not running, so nobody would act on a {action}. Start it first.")
            if action == "flatten":
                self._write(e, flatten=True)
            else:
                if not st or st["state"] != "latched":
                    raise StrategyError(409, f"{e.label} is not latched: nothing to reset")
                self._write(e, reset_latch=True)
        self._cache = None
        return {"ok": True, "action": action}

    def hand_to_allocator(self, cancel_kill: bool = False) -> dict[str, Any]:
        """Owner action: hand every strategy of this venue to the LLM allocator (owner_lock cleared, set_by llm). The
        allocator sets their weights at its next weights run. A strategy with a pending kill is refused unless cancel_kill,
        as for UNLOCK TO LLM; the others are handed all the same, and the answer says which were refused."""
        if not self.has_allocator:
            raise StrategyError(409, "this venue is not controlled by the LLM allocator (CTB_ALLOCATOR_VENUES leaves it out)")
        reg = self.registry()
        if reg.error:
            raise StrategyError(409, f"strategies.json is unusable: {reg.error}")
        handed: list[str] = []
        already: list[str] = []
        refused: dict[str, str] = {}
        for e in reg.entries:
            if not (self.root / e.id).is_dir():
                refused[e.id] = "no directory: the runner has not created it yet"
                continue
            ctl = self._control(e)
            if ctl.get("owner_lock") is not True and ctl.get("set_by") == "llm":
                already.append(e.id)
                continue
            try:
                self._write(e, refuse_if_kill=not cancel_kill, owner_lock=False, set_by="llm",
                            reason="handed to the LLM allocator by the owner: it decides at its next weights run",
                            **({"flatten": False} if cancel_kill else {}))
                handed.append(e.id)
            except StrategyError as exc:
                refused[e.id] = exc.message
        self._cache = None
        out = {"ok": not refused, "handed": handed, "already": already, "refused": refused}
        if refused:
            raise StrategyError(409, f"handed to the allocator: {', '.join(handed) or 'none'}; refused: "
                                     + "; ".join(f"{k} ({v})" for k, v in refused.items()), detail=out)
        return out

    async def allocator_act(self, action: str, interval_min: Any = None) -> dict[str, Any]:
        """Allocator panel: decide now, or set the interval (minutes)."""
        reg = self.registry()
        if self.mode(reg) != "llm":
            raise StrategyError(409, "the allocator only runs in the LLM DECIDES mode")
        if not self.allocator_available:
            raise StrategyError(409, "the LLM allocator is not installed in this build (ctb.allocator)")
        if action == "decide_now":
            if not self._allocator_on():
                raise StrategyError(409, "switch AUTO-TRADE on first: the allocator is paused")
            self._write_allocator(decide_now=True)
        elif action == "interval":
            if not isinstance(interval_min, int) or isinstance(interval_min, bool) or not 1 <= interval_min <= 1440:
                raise StrategyError(400, "interval_min must be a whole number of minutes, 1 to 1440")
            self._write_allocator(interval_min=interval_min)
        else:
            raise StrategyError(404, "unknown allocator action")
        self._cache = None
        return {"ok": True, "action": action}

    async def runner_act(self, unit: str, action: str, *, llm_trading: bool, venue: str) -> dict[str, Any]:
        reg = self.registry()
        if action not in RUNNER_ACTIONS or unit not in reg.units or not UNIT_RE.fullmatch(unit):
            raise StrategyError(404, "unknown runner unit or action")
        st = await unit_state(self.run, unit)
        if st.get("load") == "not-found":
            raise StrategyError(409, f"{unit} is not installed on this machine (see docs/strategies-tab.md)")
        if action == "start":
            if unit in self.conflicting_units(reg, venue):
                if llm_trading:
                    raise StrategyError(409, "the per-symbol bot's AUTO-TRADE is ON for the demo account. Turn it off (or switch the mode "
                                             "to LLM DECIDES or MANUAL) before starting the strategy runner: the positions would merge.")
                if self.mode(reg) == "bot":
                    raise StrategyError(409, "the auto-trade mode is PER-SYMBOL BOT. Switch the mode to LLM DECIDES or MANUAL first.")
            if reg.error:
                raise StrategyError(409, f"strategies.json is unusable: {reg.error}")
            if is_active(st):
                raise StrategyError(409, f"{unit} is already running")
        rc, _, err = await self.run(["systemctl", "--user", action, "--", unit])
        self._cache = None
        if rc != 0:
            raise StrategyError(502, f"{action} {unit} failed (exit {rc}): {(err or 'no output').strip()[-400:]}")
        return {"ok": True, "action": action, "unit": await unit_state(self.run, unit)}
