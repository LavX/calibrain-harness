"""Venues: the live strategy runners on several exchanges, one runner root each, seen and controlled from one dashboard.

Each venue is a runner root (strategies.json, <id>/status.json, control.json, trades.jsonl: the contract in
docs/strategies-tab.md) plus the systemd user unit that runs it. The venue list comes from configuration, never from a
directory glob (a quarantined copy such as data/live-okx-rejected-<date> must stay invisible):

    CTB_LIVE_ROOTS="bybit-demo=/path/live,okx-demo=/path/live-okx,deribit-test=/path/live-deribit,bitget-demo=/path/live-bitget"

Known venue ids get their label, unit, settlement currency and controller from KNOWN; another id gets a derived label,
no unit and USDT. bybit-demo must be listed: the LLM allocator, the predictions panel and the per-symbol bot's guard stay
bound to it. Unset: one venue, bybit-demo at CTB_LIVE_ROOT (or data/live), exactly as before.

Per venue the hub reports a presence:
    live     the unit is active and at least one status.json is fresher than 3 minutes
    stale    the unit is active but every status.json is older than 3 minutes (or there is none)
    stopped  the unit is not active
    absent   no root or no strategies.json yet: "not started"
    error    strategies.json exists but cannot be used (fail closed, with the message)

KILL (per venue) always records the owner's intent: it pauses the allocator (bybit-demo), and pauses (owner-locked) and
flags every strategy of the venue to flatten. It then answers ok only when every written strategy's runner unit is active
and its status.json is fresh; otherwise 409 "written, NOT confirmed" names the strategies nobody may be acting for yet
(a hung or crash-looping runner reads the flags when it comes back). Only a venue that has not started (no root or no
strategies.json) or whose strategies.json is unusable is refused with nothing written.

P&L today: the harness keeps its own 00:00 UTC baseline per venue (the first sample of each UTC day, from the status
files and, on bybit-demo, the account's USDT equity) in a small JSON file under its data directory, so the number
survives a restart. A background task samples every minute; when the harness was down at midnight the baseline is the
first sample after it came back, and the page says since when.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import strategies as S
from .config import ROOT

PRIMARY = "bybit-demo"  # the allocator venue
VID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
# id: (label, unit, settlement currency, controlled by)
KNOWN: dict[str, tuple[str, str | None, str, str]] = {
    "bybit-demo": ("BYBIT·DEMO", "calibrain-live.service", "USDT", "allocator"),
    "okx-demo": ("OKX·DEMO", "calibrain-live-okx.service", "USDT", "owner"),
    "deribit-test": ("DERIBIT·TEST", "calibrain-live-deribit.service", "USDC", "owner"),
    "bitget-demo": ("BITGET·DEMO", "calibrain-live-bitget.service", "USDT", "owner"),
}
IDLE_ALERT_S = 24 * 3600  # an enabled strategy that held nothing and traded nothing for this long is flagged
MIDNIGHT_SLACK_MS = 5 * 60 * 1000  # a baseline sampled this soon after 00:00 UTC counts as "since 00:00 UTC"
PRESENCE = ("live", "stale", "stopped", "absent", "error")


@dataclass(frozen=True)
class VenueConfig:
    id: str
    label: str
    root: Path
    unit: str | None
    ccy: str
    controlled_by: str  # "allocator" (the LLM allocator writes the controls) or "owner"


def venue_config(vid: str, root: Path, *, unit: str | None = None) -> VenueConfig:
    label, known_unit, ccy, by = KNOWN.get(vid) or (vid.upper().replace("-", "·"), None, "USDT", "owner")
    return VenueConfig(vid, label, root, unit if unit is not None else known_unit, ccy, by)


def parse_live_roots(text: str | None, default_root: Path) -> list[VenueConfig]:
    """CTB_LIVE_ROOTS -> venues, in the order given. Unset or empty: bybit-demo at the old single root, its unit taken
    from strategies.json as before. Raises ValueError on a malformed entry (the dashboard then refuses to start)."""
    if not (text or "").strip():
        return [venue_config(PRIMARY, default_root, unit="")]  # unit "" -> no fixed unit: strategies.json names it
    out: list[VenueConfig] = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        vid, sep, root = part.partition("=")
        vid, root = vid.strip().lower(), root.strip()
        if not sep or not root or not VID_RE.fullmatch(vid):
            raise ValueError(f"CTB_LIVE_ROOTS entry {part!r}: use <venue id>=<runner root>, ids in lowercase letters, digits and -")
        if any(v.id == vid for v in out):
            raise ValueError(f"CTB_LIVE_ROOTS lists {vid} twice")
        out.append(venue_config(vid, Path(root).expanduser()))
    if not any(v.id == PRIMARY for v in out):
        raise ValueError(f"CTB_LIVE_ROOTS must list {PRIMARY}: the LLM allocator and the per-symbol bot's guard are bound to it")
    return out


def state_dir() -> Path:
    """The harness's own data directory (next to its database), where the venue baselines and the header set live."""
    return Path(os.environ.get("CTB_DB_PATH") or ROOT / "data" / "ctb.db").parent / "venues"


# ------------------------------------------------------------------------------------------------ pure helpers
def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def utc_day(now_s: float) -> str:
    return datetime.fromtimestamp(now_s, timezone.utc).strftime("%Y-%m-%d")


def presence(cfg: VenueConfig, *, has_registry: bool, reg_error: str | None, units: list[dict], statuses: list[dict | None],
             now_ms: float) -> tuple[str, str | None]:
    """(state, why). `units` are the unit states of the runner unit(s) this venue's strategies name (or its configured unit);
    `statuses` the cleaned status.json of each strategy."""
    if not cfg.root.is_dir() or not has_registry:
        return "absent", f"not started: no strategies.json under {cfg.root}"
    if reg_error:
        return "error", reg_error
    if units and not any(S.is_active(u) for u in units):
        names = ", ".join(u.get("unit", "?") for u in units)
        state = units[0].get("active") or "unknown"
        if any(u.get("load") == "not-found" for u in units):
            state = "not installed"
        return "stopped", f"{names} is {state}"
    stamps = [s["updated_ms"] for s in statuses if s and s.get("updated_ms") is not None]
    newest = max(stamps) if stamps else None
    if newest is None:
        return "stale", "no strategy has written a status.json yet"
    if now_ms - newest > S.STALE_S * 1000:
        return "stale", f"every status.json is older than {int(S.STALE_S // 60)} min (newest {int((now_ms - newest) / 60000)} min ago)"
    return "live", None


def strategy_liveness(cfg: VenueConfig, view: dict, runners: list[dict], now_ms: float) -> tuple[bool, str]:
    """Is someone acting on this strategy's control.json right now: its own unit (or the venue's) active and its
    status.json fresh. Units may be per strategy (calibrain-live@<id>.service), so this is answered per strategy."""
    unit = view.get("unit") or cfg.unit or None
    if unit:
        u = next((r for r in runners if r.get("unit") == unit), None)
        if u is None or not S.is_active(u):
            state = "not installed" if u and u.get("load") == "not-found" else (u or {}).get("active") or "unknown"
            return False, f"unit {unit} is {state}"
    st = view.get("status")
    if not st or st.get("updated_ms") is None:
        return False, "no status.json yet"
    age_ms = now_ms - st["updated_ms"]
    if age_ms > S.STALE_S * 1000:
        return False, f"status.json is {int(age_ms // 60000)} min old"
    return True, "runner up, status fresh"


def _signed(side: str, qty: float) -> float:
    return -abs(qty) if side.lower() in ("short", "sell") else abs(qty)


def net_positions(strategies: list[dict]) -> list[dict]:
    """The books' legs netted per (symbol, market), as the exchange holds them: one row per coin and market, the signed
    size summed, uPnL summed, and which strategies hold a leg. Entry is the size-weighted entry only when every leg is on
    the net side (otherwise there is no single entry price)."""
    rows: dict[tuple[str, str], dict] = {}
    for s in strategies:
        st = s.get("status") or {}
        for p in st.get("positions") or []:
            qty = _num(p.get("qty"))
            if qty is None:
                continue
            key = (p.get("symbol", ""), p.get("market", ""))
            r = rows.setdefault(key, {"symbol": key[0], "market": key[1], "qty": 0.0, "upnl": 0.0, "upnl_known": True,
                                      "mark": None, "legs": [], "held_by": []})
            r["qty"] += _signed(p.get("side", ""), qty)
            if _num(p.get("upnl_usdt")) is None:
                r["upnl_known"] = False
            else:
                r["upnl"] += p["upnl_usdt"]
            r["mark"] = _num(p.get("mark_price")) if _num(p.get("mark_price")) is not None else r["mark"]
            r["legs"].append((_signed(p.get("side", ""), qty), _num(p.get("entry_price"))))
            if s["name"] not in r["held_by"]:
                r["held_by"].append(s["name"])
    out = []
    for r in rows.values():
        q = r["qty"]
        side = "long" if q > 0 else "short" if q < 0 else "flat"
        same = [(lq, e) for lq, e in r["legs"] if e is not None and (lq > 0) == (q > 0)]
        entry = (sum(abs(lq) * e for lq, e in same) / sum(abs(lq) for lq, _ in same)
                 if q and len(same) == len(r["legs"]) and same else None)
        out.append({"symbol": r["symbol"], "market": r["market"], "side": side, "size": abs(q), "entry": entry, "mark": r["mark"],
                    "upnl": r["upnl"] if r["upnl_known"] else None, "held_by": r["held_by"], "legs": len(r["legs"])})
    out.sort(key=lambda x: -abs(x["upnl"] or 0))
    return out


def exchange_positions(account: dict, strategies: list[dict]) -> list[dict]:
    """bybit-demo with the exchange account: its netted positions, each with the strategies that hold a leg in it."""
    held: dict[str, list[str]] = {}
    for s in strategies:
        for p in (s.get("status") or {}).get("positions") or []:
            names = held.setdefault(p.get("symbol", ""), [])
            if s["name"] not in names:
                names.append(s["name"])
    out = [{"symbol": p.get("symbol", ""), "market": "linear", "side": p.get("side", ""), "size": _num(p.get("size")),
            "entry": _num(p.get("entry")), "mark": _num(p.get("mark")), "upnl": _num(p.get("upnl")),
            "held_by": held.get(p.get("symbol", ""), []), "legs": None} for p in account.get("positions") or []]
    out.sort(key=lambda x: -abs(x["upnl"] or 0))
    return out


def strategy_row(v: dict, base: dict | None, now_ms: float) -> dict:
    """One strategy as the F1 matrix and alerts need it: the runner's applied values first, the request as context."""
    st, ctl = v.get("status"), v.get("control") or {}
    eq = st.get("equity_usdt") if st else None
    b = (base or {}).get("strategies", {}).get(v["id"]) if base else None
    pnl_today = (eq - b["equity"]) if (eq is not None and b and _num(b.get("equity")) is not None) else None
    trades = v.get("trades") or []
    last_trade = max((t.get("t_ms") for t in trades if _num(t.get("t_ms")) is not None), default=None)
    return {
        "id": v["id"], "name": v.get("name") or S.short_name(v["id"], v.get("label", "")), "label": v.get("label", v["id"]),
        "pill": v.get("pill"), "enabled": bool(st and st["enabled"]),
        "weight": st.get("weight") if st else None, "weight_requested": ctl.get("weight"),
        "set_by": st.get("set_by") if st and st.get("set_by") else ctl.get("set_by"),
        "owner_lock": bool(st.get("owner_lock")) if st else bool(ctl.get("owner_lock")),
        "reason": (st.get("reason") if st else None) or ctl.get("reason"),
        "flatten_pending": bool(ctl.get("flatten_pending")),
        "positions_n": len(st["positions"]) if st else None,
        "targets_n": sum(1 for t in st["targets"] if t.get("target_qty")) if st else None,
        "equity": eq, "start_equity": st.get("start_equity_usdt") if st else None, "pnl": st.get("pnl_usdt") if st else None,
        "pnl_today": pnl_today, "pnl_today_since_ms": b.get("t_ms") if b and pnl_today is not None else None,
        "unrealised": sum(p["upnl_usdt"] for p in st["positions"] if p.get("upnl_usdt") is not None) if st else None,
        "last_error": st.get("last_error") if st else None,
        "updated_ms": st.get("updated_ms") if st else None, "last_rebalance_ms": st.get("last_rebalance_ms") if st else None,
        "last_trade_ms": last_trade, "trades_today": st.get("trades_today") if st else None,
    }


def alerts(v: dict, now_ms: float) -> list[dict]:
    """What needs the owner's attention on one venue. crit: red (unit down, stale, error, latched); warn: amber."""
    out: list[dict] = []

    def add(level: str, kind: str, text: str, sid: str | None = None, name: str | None = None) -> None:
        out.append({"venue": v["id"], "venue_label": v["label"], "strategy": sid, "name": name, "kind": kind, "level": level, "text": text})

    pres = v["presence"]
    if pres == "stopped":
        add("crit", "unit_down", f"runner {v['presence_text']}: nothing trades or flattens on this venue")
    elif pres == "stale":
        add("crit", "stale", f"runner unit is active but {v['presence_text']}")
    elif pres == "error":
        add("crit", "error", f"strategies.json: {v['presence_text']}")
    if pres in ("absent", "error"):
        return out
    for s in v["strategies"]:
        sid, name, p = s["id"], s["name"], s["pill"]
        err = s.get("last_error") or ""
        if s["flatten_pending"]:
            add("warn", "kill_pending", "flatten requested, waiting for the runner", sid, name)
        if p == "latched":
            add("crit", "latched", err or "latched: check the account, then reset the latch (F7)", sid, name)
        elif p == "error":
            kind = "not_tradeable" if re.search(r"rejected|given up|not tradeable|restricted", err, re.I) else "error"
            add("crit" if kind == "error" else "warn", kind, err or "the runner reports an error", sid, name)
        elif p == "stale" and pres == "live":
            add("crit", "stale", "status.json older than 3 min", sid, name)
        elif p == "no_data" and pres == "live":
            add("warn", "no_data", "no status.json yet", sid, name)
        elif p == "idle":
            last = s.get("last_trade_ms")
            if last is None or now_ms - last > IDLE_ALERT_S * 1000:
                since = "no trade yet" if last is None else f"last trade {int((now_ms - last) / 3_600_000)} h ago"
                add("warn", "idle", f"enabled but holds nothing and wants nothing ({since})", sid, name)
        elif err and p in ("running", "paused"):
            if re.search(r"rejected|given up|not tradeable|restricted", err, re.I):
                add("warn", "not_tradeable", err, sid, name)
    return out


# ------------------------------------------------------------------------------------------------ baselines
class Baselines:
    """The persisted 00:00 UTC sample per venue: {"day", "t_ms", "strategies": {id: {"equity", "t_ms"}},
    "account": {"usdt_equity", "t_ms"} | null}. Written once per UTC day, topped up when a strategy or the account
    first appears later that day. Survives restarts (a file per venue under the harness's data directory)."""

    def __init__(self, directory: Path) -> None:
        self.dir = directory

    def path(self, vid: str) -> Path:
        return self.dir / f"baseline-{vid}.json"

    def get(self, vid: str, now_s: float) -> dict | None:
        doc = S.read_json(self.path(vid))
        return doc if doc and doc.get("day") == utc_day(now_s) else None

    def sample(self, vid: str, now_s: float, equities: dict[str, float], account_equity: float | None) -> dict:
        day, now_ms = utc_day(now_s), int(now_s * 1000)
        doc = self.get(vid, now_s)
        changed = doc is None
        if doc is None:
            doc = {"day": day, "t_ms": now_ms, "strategies": {}, "account": None}
        for sid, eq in equities.items():
            if eq is not None and sid not in doc["strategies"]:
                doc["strategies"][sid] = {"equity": eq, "t_ms": now_ms}
                changed = True
        if account_equity is not None and not doc.get("account"):
            doc["account"] = {"usdt_equity": account_equity, "t_ms": now_ms}
            changed = True
        if changed:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                S.write_control(self.path(vid), doc)
            except OSError:
                pass  # a read-only data dir costs the "since 00:00" figure, never the page
        return doc


def since_label(t_ms: float | None, now_s: float) -> str | None:
    if t_ms is None:
        return None
    midnight_ms = int(now_s // 86400) * 86400 * 1000
    if t_ms - midnight_ms <= MIDNIGHT_SLACK_MS:
        return "since 00:00 UTC"
    return "since " + datetime.fromtimestamp(t_ms / 1000, timezone.utc).strftime("%H:%M") + " UTC (harness started after midnight)"


# ------------------------------------------------------------------------------------------------ the hub
class VenueHub:
    def __init__(self, venues: list[VenueConfig], services: dict[str, S.StrategyService], *, state: Path | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        if not venues or venues[0].id not in services:
            raise ValueError("a hub needs at least one venue with a service")
        self.venues = venues
        self.services = services
        self.baselines = Baselines(state if state is not None else state_dir())
        self.clock = clock
        self.primary_id = PRIMARY if PRIMARY in services else venues[0].id

    @classmethod
    def from_env(cls) -> "VenueHub":
        venues = parse_live_roots(os.environ.get("CTB_LIVE_ROOTS"), S.live_root())
        sdir = state_dir()
        services: dict[str, S.StrategyService] = {}
        for v in venues:
            auto = sdir / f"auto-set-{v.id}.json"
            if v.id == PRIMARY:
                extra = S.extra_units_from_env() + ((v.unit,) if v.unit else ())
                services[v.id] = S.StrategyService(v.root, extra_units=tuple(dict.fromkeys(extra)), auto_set_path=auto)
            else:
                services[v.id] = S.StrategyService(v.root, extra_units=(v.unit,) if v.unit else (), dashboard_account="",
                                                   shares_bot_account=False, auto_set_path=None, has_allocator=False)
        return cls(venues, services, state=sdir)

    @classmethod
    def single(cls, svc: S.StrategyService, *, state: Path | None = None) -> "VenueHub":
        """One venue (bybit-demo) around an existing service: old configurations and tests."""
        return cls([venue_config(PRIMARY, svc.root, unit="")], {PRIMARY: svc}, state=state, clock=svc.clock)

    @property
    def primary(self) -> S.StrategyService:
        return self.services[self.primary_id]

    def config(self, vid: str) -> VenueConfig:
        cfg = next((v for v in self.venues if v.id == vid), None)
        if cfg is None:
            raise S.StrategyError(404, f"unknown venue {vid!r}")
        return cfg

    def service(self, vid: str) -> S.StrategyService:
        self.config(vid)
        return self.services[vid]

    # ----------------------------------------------------------------- reads
    async def venue_doc(self, vid: str, *, account: dict | None = None, force: bool = False) -> dict[str, Any]:
        """The F7 document of one venue (the /api/strategies shape) plus venue fields and today's P&L per strategy."""
        cfg, svc = self.config(vid), self.service(vid)
        doc = dict(await svc.snapshot(force=force))
        now = self.clock()
        base = self._sample(cfg, doc, account, now)
        rows = {r["id"]: r for r in (strategy_row(v, base, now * 1000) for v in doc["strategies"])}
        doc["strategies"] = [{**v, "pnl_today": rows[v["id"]]["pnl_today"], "pnl_today_since_ms": rows[v["id"]]["pnl_today_since_ms"]}
                             for v in doc["strategies"]]
        doc["venue_id"], doc["venue_label"], doc["ccy"], doc["controlled_by"] = cfg.id, cfg.label, cfg.ccy, cfg.controlled_by
        doc["allocator_venue"] = cfg.id == self.primary_id
        return doc

    def _account_for(self, cfg: VenueConfig, doc: dict, account: dict | None) -> dict | None:
        """The exchange account only on bybit-demo, and only when it is the account the strategies trade (the per-symbol
        bot's executor reads the CTB_DASHBOARD_ACCOUNT demo account; unset counts as the same, like the guard)."""
        if cfg.id != self.primary_id or not account or not isinstance(account.get("wallet"), dict):
            return None
        mine = self.primary.dashboard_account
        accounts = {s.get("account") for s in doc["strategies"]}
        return account if mine is None or not accounts or mine in accounts else None

    def _sample(self, cfg: VenueConfig, doc: dict, account: dict | None, now: float) -> dict | None:
        equities = {v["id"]: v["status"]["equity_usdt"] for v in doc["strategies"] if v.get("status")}
        acct = self._account_for(cfg, doc, account)
        usdt = _num((acct or {}).get("wallet", {}).get("usdt_equity")) if acct else None
        if not equities and usdt is None:
            return self.baselines.get(cfg.id, now)
        return self.baselines.sample(cfg.id, now, equities, usdt)

    async def summary(self, account: dict | None = None) -> dict[str, Any]:
        """GET /api/venues: per venue presence, books, account, today's P&L, strategies and netted positions, plus the alerts."""
        out, all_alerts = [], []
        now = self.clock()
        now_ms = now * 1000
        for cfg in self.venues:
            v = await self._venue_summary(cfg, account, now)
            all_alerts += alerts(v, now_ms)
            out.append(v)
        prim = await self.primary.snapshot()
        return {"generated_at": now, "primary": self.primary_id, "venues": out, "alerts": all_alerts,
                "mode": prim["mode"], "auto_on": prim["auto_on"], "allocator": prim["allocator"]}

    async def _venue_summary(self, cfg: VenueConfig, account: dict | None, now: float, force: bool = False) -> dict[str, Any]:
        svc = self.services[cfg.id]
        now_ms = now * 1000
        has_registry = (cfg.root / "strategies.json").is_file()
        doc = await svc.snapshot(force=force)
        base = self._sample(cfg, doc, account, now) if has_registry else None
        rows = [strategy_row(v, base, now_ms) for v in doc["strategies"]]
        entry_units = list(dict.fromkeys(s["unit"] for s in doc["strategies"] if s.get("unit")))
        wanted = entry_units or ([cfg.unit] if cfg.unit else [])
        units = [u for u in doc["runners"] if u.get("unit") in wanted]
        pres, why = presence(cfg, has_registry=has_registry, reg_error=doc.get("error"), units=units,
                             statuses=[s.get("status") for s in doc["strategies"]], now_ms=now_ms)
        known = [r for r in rows if r["equity"] is not None]
        today = [r for r in rows if r["pnl_today"] is not None]
        unreal = [r["unrealised"] for r in rows if r["unrealised"] is not None]
        books = {"equity": sum(r["equity"] for r in known) if known else None,
                 "start": sum(r["start_equity"] for r in rows if r["start_equity"] is not None) if rows else None,
                 "pnl": sum(r["pnl"] for r in rows if r["pnl"] is not None) if rows else None,
                 "unrealised": sum(unreal) if unreal else None,
                 "legs": sum(r["positions_n"] or 0 for r in rows)}
        since_ms = min((r["pnl_today_since_ms"] for r in today if r["pnl_today_since_ms"] is not None), default=None)
        acct = self._account_for(cfg, doc, account)
        account_block = None
        if acct:
            w = acct["wallet"]
            ab = (base or {}).get("account") or {}
            usdt = _num(w.get("usdt_equity"))
            acc_unreal = _num(w.get("unrealized_pnl"))
            account_block = {
                "source": "exchange", "ccy": "USDT", "equity": usdt, "collateral_usd": _num(w.get("equity")),
                "available": _num(w.get("available")), "unrealised": acc_unreal,
                "pnl_today": (usdt - ab["usdt_equity"]) if usdt is not None and _num(ab.get("usdt_equity")) is not None else None,
                "pnl_today_since": since_label(ab.get("t_ms"), now) if ab else None,
                "unattributed": (acc_unreal - books["unrealised"]) if acc_unreal is not None and books["unrealised"] is not None else None,
                "positions_n": len(acct.get("positions") or []), "last_sync": acct.get("last_sync"),
            }
        elif has_registry and (extra := S.read_json(cfg.root / "account.json")):
            eq, start = _num(extra.get("equity")), _num(extra.get("day_start_equity"))
            account_block = {"source": "account.json", "ccy": str(extra.get("ccy") or cfg.ccy), "equity": eq, "collateral_usd": None,
                             "available": _num(extra.get("available")), "unrealised": _num(extra.get("unrealised")),
                             "pnl_today": (eq - start) if eq is not None and start is not None else None,
                             "pnl_today_since": "since the runner's day start" if start is not None else None,
                             "unattributed": None, "positions_n": None, "last_sync": _num(extra.get("updated_ms"))}
        positions = exchange_positions(acct, doc["strategies"]) if acct else net_positions(doc["strategies"])
        counts = {k: sum(1 for r in rows if r["pill"] == k) for k in ("running", "idle", "paused")}
        counts["total"] = len(rows)
        counts["abnormal"] = counts["total"] - counts["running"] - counts["idle"] - counts["paused"]
        return {
            "id": cfg.id, "label": cfg.label, "ccy": cfg.ccy, "controlled_by": cfg.controlled_by, "root": str(cfg.root),
            "unit": wanted[0] if wanted else None, "units": units, "presence": pres, "presence_text": why,
            "updated_ms": max((r["updated_ms"] for r in rows if r["updated_ms"] is not None), default=None),
            "counts": counts, "books": books,
            "pnl_today": sum(r["pnl_today"] for r in today) if today else None,
            "pnl_today_partial": bool(today) and len(today) < len(known),
            "pnl_today_since": since_label(since_ms, now),
            "account": account_block, "strategies": rows, "positions": positions,
            # available: KILL writes the flags (everything but absent/error); ok: and the runner would act on them now
            "kill": {"ok": pres == "live", "available": pres not in ("absent", "error"),
                     "why": None if pres == "live" else kill_refusal(cfg, pres, why)},
        }

    async def sample_all(self, account: dict | None = None) -> None:
        """The background sampler: keeps the 00:00 UTC baselines and the F7 sparklines filled without a browser."""
        await self.summary(account)

    # ----------------------------------------------------------------- kill
    async def kill(self, vid: str) -> dict[str, Any]:
        """Pause and flag to flatten every strategy of one venue (and the allocator on bybit-demo), then say whether the
        runner confirms. Raises 409 with nothing written for an absent or unusable root; 409 "written, NOT confirmed" when a
        written strategy's runner is down or stale; 500 when a write failed (detail lists what was written)."""
        cfg = self.config(vid)
        svc = self.services[vid]
        if not cfg.root.is_dir() or not (cfg.root / "strategies.json").is_file():
            raise S.StrategyError(409, kill_refusal(cfg, "absent", None))
        doc = await svc.snapshot(force=True)
        if doc.get("error"):
            raise S.StrategyError(409, kill_refusal(cfg, "error", doc["error"]))
        now_ms = self.clock() * 1000
        live = {v["id"]: strategy_liveness(cfg, v, doc["runners"], now_ms) for v in doc["strategies"]}
        res = svc.kill_all(pause_allocator=vid == self.primary_id)
        written, failed = res["written"], res["failed"]
        confirmed = [i for i in written if live.get(i, (False, ""))[0]]
        unconfirmed = {i: live.get(i, (False, "unknown"))[1] for i in written if i not in confirmed}
        out = {"ok": not failed and not unconfirmed, "venue": vid, "label": cfg.label, "written": written, "flattening": written,
               "confirmed": confirmed, "unconfirmed": unconfirmed, "failed": failed,
               "allocator_paused": res["allocator_paused"] if vid == self.primary_id else None}
        if failed:
            raise S.StrategyError(500, kill_message(cfg, out), detail=out)
        if unconfirmed:
            raise S.StrategyError(409, kill_message(cfg, out), detail=out)
        return out

    async def kill_all(self) -> dict[str, Any]:
        """Every venue: each one that has started gets its flags written and says whether that is confirmed; a venue that
        has not started is skipped."""
        res = []
        for cfg in self.venues:
            try:
                res.append(await self.kill(cfg.id))
            except S.StrategyError as exc:
                absent = not cfg.root.is_dir() or not (cfg.root / "strategies.json").is_file()
                res.append({**exc.detail, "ok": False, "venue": cfg.id, "label": cfg.label, "skipped": absent, "error": exc.message})
        return {"ok": all(r["ok"] or r.get("skipped") for r in res), "venues": res}


def kill_message(cfg: VenueConfig, out: dict) -> str:
    written = ", ".join(out["written"]) or "none"
    alloc = ", allocator paused" if out.get("allocator_paused") else ""
    if out["failed"]:
        fails = "; ".join(f"{k}: {v}" for k, v in out["failed"].items())
        return (f"{cfg.label} KILL partially written: could not write {fails}. Written (paused, owner-locked, flatten "
                f"requested{alloc}): {written}. Fix the file permissions and kill again.")
    why = "; ".join(f"{k} ({v})" for k, v in out["unconfirmed"].items())
    return (f"{cfg.label} KILL written, NOT confirmed: {why}. The flags are on disk for {written} (paused, owner-locked, "
            f"flatten requested{alloc}); nothing closes until the runner reads them. Confirmed: {', '.join(out['confirmed']) or 'none'}.")


def kill_refusal(cfg: VenueConfig, pres: str, why: str | None) -> str:
    """Absent and error: why KILL writes nothing. Stopped and stale: why it would not be confirmed (it still writes)."""
    if pres == "absent":
        return f"{cfg.label} has not started: nothing to kill"
    if pres == "error":
        return f"{cfg.label} KILL refused: strategies.json is unusable ({why}). Nothing was written."
    if pres == "stopped":
        return (f"{why}: KILL writes the flags (paused, owner-locked, flatten requested) but no runner reads them until it "
                f"starts. Start it (F7) or close the positions on the exchange.")
    if pres == "stale":
        return (f"{why}: the runner may be hung. KILL writes the flags but cannot confirm they are read. Check its journal "
                f"(journalctl --user -u {cfg.unit or 'the runner unit'}).")
    return f"{cfg.label}: presence {pres}"
