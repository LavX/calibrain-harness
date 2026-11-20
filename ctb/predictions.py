"""Predictions panel backend (Strategies screen, F7): the prediction layer's numbers, read-only.

GET /api/predictions answers one JSON document built from files under the live root (CTB_LIVE_ROOT, the same root the
Strategies tab reads). Nothing here writes a file, calls an exchange or reads a credential.

  regime/latest.json          the day's state, the 8 features and the 7-day state probabilities (research/live/regime.py)
  forecast/latest.json        sigma and mu per forecaster per strategy, kappa (research/live/forecast.py run)
  forecast/scores.json        L and W per forecaster, the decider's Brier score, tilt_skill and kappa (forecast.py score)
  allocator/decisions.jsonl   the newest weights line carries core, sr, tau, shadow_tau, mu_ann_pct, p_decider, reasons
  allocator/policies.jsonl    the paired policies (core, live, tilt, core_veto) of every weights run: each day's own weights
                              for the P&L and history comparisons (never today's weight applied backwards)
  returns/<id>.csv            the strategy's modelled daily net return at weight 1 (research/live/backtest.py)
  <id>/status.json, control.json   live P&L and the live weight

Every file may be missing or half-written (the writers are other jobs): that section is null and the page says
"not computed yet". A section never fails the whole answer.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

from . import allocator_math as M
from . import decision_log as DL
from .strategies import live_root, load_registry, read_json

GO_LIVE = os.environ.get("CTB_GO_LIVE_DATE") or "2026-11-19"  # the first day the live book trades; P&L comparisons start here
CACHE_S = 30.0
TILT_DAYS = 90                  # the shadow tilt must lead the core over this many days before it gets live control
STALE_FORECAST_H = 36.0         # the allocator holds weights past this age (CTB_ALLOCATOR_STALE_H)
DECISIONS_READ_BYTES = 8 * 1024 * 1024   # decision lines carry the full model inputs and can be large
FORECASTERS = ("base", "regime", "ridge", "structural", "decider")
STATES = ("up_calm", "up_wild", "down_calm", "down_wild")
ORDER = ("carry", "trend", "xsmom", "basis", "longvol")

STATE_HELP = {
    "up_calm": "BTC in an uptrend, volatility below its one-year median",
    "up_wild": "BTC in an uptrend, volatility above its one-year median",
    "down_calm": "BTC in a downtrend, volatility below its one-year median",
    "down_wild": "BTC in a downtrend, volatility above its one-year median",
}
# key, one-line meaning, unit ("pct" = the stored value is a fraction shown as %/yr)
FEATURES = (
    ("btc_trend", "BTC trend strength (the trend rules' forecast divided by 10); above 0 is an uptrend", ""),
    ("breadth", "share of the 40 largest perps trading above their 50-day average", ""),
    ("vol_rank", "where BTC's 30-day volatility sits in the last year, 0 calmest to 1 wildest", ""),
    ("vol_ratio", "ln of BTC's 7-day over 30-day volatility; above 0 means volatility is rising", ""),
    ("dispersion", "how far apart the coins' 20-day returns are, in volatility units", ""),
    ("avg_corr", "average pairwise correlation of the 20 largest perps over 30 days", ""),
    ("funding_level", "median perp funding rate, %/yr; positive means longs pay shorts", "pct"),
    ("funding_spread", "interquartile range of funding across perps, %/yr; wide means funding is dispersed", "pct"),
)


FEATURES_KEYS = tuple(k for k, _, _ in FEATURES)


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _order(sid: str) -> tuple[int, str]:
    return (ORDER.index(sid) if sid in ORDER else len(ORDER), sid)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


# ------------------------------------------------------------------------------------------------ file readers
def read_last_with(path: Path, key: str, max_bytes: int = DECISIONS_READ_BYTES) -> dict | None:
    """The newest line of a JSON-lines file with a non-empty `key`, reading only the end of the file."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return None
    for line in reversed(data.splitlines()):
        if f'"{key}"'.encode() not in line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get(key):  # a non-empty value: a "held" line has an empty core
            return rec
    return None


_ARCHIVE_HITS: dict[tuple[str, str], tuple[int, int, dict | None]] = {}  # (archive, key) -> (mtime_ns, size, newest match)


def _newest_in(path: Path, key: str) -> dict | None:
    """The newest line of a whole (possibly gzipped) archive with a non-empty `key`, remembered per file version: an
    archive no longer changes once written, so it is read once."""
    try:
        st = path.stat()
    except OSError:
        return None
    hit = _ARCHIVE_HITS.get((str(path), key))
    if hit and hit[:2] == (st.st_mtime_ns, st.st_size):
        return hit[2]
    found = None
    for rec in DL._lines(path):  # a damaged archive yields what is readable
        if rec.get(key):
            found = rec
    _ARCHIVE_HITS[(str(path), key)] = (st.st_mtime_ns, st.st_size, found)
    return found


def newest_with(alloc: Path, key: str) -> dict | None:
    """The newest decision line with a non-empty `key`: the end of the current decisions.jsonl, else the file being split
    and the archives, newest first, compressed ones too (days of held runs can push the last weights line far back)."""
    line = read_last_with(alloc / DL.LOG, key)
    if line is not None:
        return line
    for path in DL._newest_first(alloc)[1:]:
        line = _newest_in(path, key)
        if line is not None:
            return line
    return None


def read_returns(path: Path) -> dict[str, float]:
    """{ISO date: ret_net} of one returns/<id>.csv; unreadable rows are skipped, a missing file is empty."""
    out: dict[str, float] = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    v = float(row["ret_net"])
                    d = str(row["date"])[:10]
                    date.fromisoformat(d)
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    out[d] = v
    except OSError:
        pass
    return out


# ------------------------------------------------------------------------------------------------ sections
def regime_section(doc: dict | None) -> dict | None:
    if not doc or not isinstance(doc.get("state"), str):
        return None
    feats = doc.get("features") if isinstance(doc.get("features"), dict) else {}
    p7 = doc.get("p7") if isinstance(doc.get("p7"), dict) else {}
    return {
        "date": doc.get("date") if isinstance(doc.get("date"), str) else None,
        "state": doc["state"], "state_help": STATE_HELP.get(doc["state"], ""),
        "features": [{"key": k, "meaning": meaning, "unit": unit, "value": _r(_num(feats.get(k)), 6)} for k, meaning, unit in FEATURES],
        "p7": [{"state": s, "p": _r(_num(p7.get(s)), 4), "help": STATE_HELP[s]} for s in STATES],
    }


def forecast_section(doc: dict | None, now_s: float) -> dict | None:
    if not doc or not isinstance(doc.get("strategies"), dict):
        return None
    made = _num(doc.get("made_ms"))
    age_h = None if made is None else max(0.0, (now_s * 1000 - made) / 3_600_000)
    return {"date": doc.get("date") if isinstance(doc.get("date"), str) else None, "made_ms": made, "age_h": _r(age_h, 2),
            "stale": age_h is not None and age_h > STALE_FORECAST_H, "kappa": _num(doc.get("kappa"))}


def strategy_rows(ids: list[str], labels: dict[str, str], fc: dict | None, line: dict | None, root: Path, scores: dict | None) -> list[dict]:
    """One row per strategy: core, shadow tilt, forecast Sharpe, mu per forecaster, sigma, decider p, live weight and the reason.
    The allocator line is the source of the breakdown the allocator actually used (it adds the decider and the Sharpe clip);
    without one, the forecast file alone gives sigma and the stored forecasters."""
    fstrat = (fc or {}).get("strategies") if isinstance((fc or {}).get("strategies"), dict) else {}
    line = line or {}
    mu_ann = line.get("mu_ann_pct") if isinstance(line.get("mu_ann_pct"), dict) else {}
    sigma_l = line.get("sigma_d") if isinstance(line.get("sigma_d"), dict) else {}
    rows = []
    for sid in ids:
        ctl = read_json(root / sid / "control.json") or {}
        f = fstrat.get(sid) if isinstance(fstrat.get(sid), dict) else {}
        sigma = _num(sigma_l.get(sid)) or _num(f.get("sigma_d"))
        mus: dict[str, float | None] = {}
        raw = mu_ann.get(sid) if isinstance(mu_ann.get(sid), dict) else None
        if raw:
            for k in (*FORECASTERS, "combined"):
                mus[k] = _r(_num(raw.get(k)), 2)
        else:  # no allocator line yet: the stored forecasters, annualised, no decider, no combined
            for k in FORECASTERS:
                v = _num((f.get("mu") or {}).get(k)) if isinstance(f.get("mu"), dict) else None
                mus[k] = None if v is None else _r(M.ann_pct(v), 2)
            comb = _num(((fc or {}).get("mu_combined") or {}).get(sid)) if isinstance((fc or {}).get("mu_combined"), dict) else None
            mus["combined"] = None if comb is None else _r(M.ann_pct(comb), 2)
        reasons = line.get("reasons") if isinstance(line.get("reasons"), dict) else {}
        p = line.get("p_decider") if isinstance(line.get("p_decider"), dict) else {}
        rows.append({
            "id": sid, "label": labels.get(sid, sid),
            "core": _r(_num((line.get("core") or {}).get(sid)) if isinstance(line.get("core"), dict) else None, 4),
            "shadow_tau": _r(_num((line.get("shadow_tau") or {}).get(sid)) if isinstance(line.get("shadow_tau"), dict) else None, 4),
            "shadow_w": _r(_num((line.get("shadow_w_target") or {}).get(sid)) if isinstance(line.get("shadow_w_target"), dict) else None, 4),
            "tau": _r(_num((line.get("tau") or {}).get(sid)) if isinstance(line.get("tau"), dict) else None, 4),
            "sr": _r(_num((line.get("sr") or {}).get(sid)) if isinstance(line.get("sr"), dict) else None, 3),
            "mu_ann_pct": mus,
            "sigma_ann_pct": _r(None if sigma is None else M.sigma_ann_pct(sigma), 2),
            "p_decider": _r(_num(p.get(sid)), 3),
            "live_weight": _r(_num(ctl.get("weight")), 3),
            "reason": str(reasons[sid])[:500] if isinstance(reasons.get(sid), str) else None,
            "clipped": [str(k) for k in ((line.get("clipped") or {}).get(sid) or [])] if isinstance(line.get("clipped"), dict) else [],
        })
    return rows


def scoreboard_section(scores: dict | None) -> dict | None:
    if not scores or not isinstance(scores, dict):
        return None
    loss = scores.get("loss") if isinstance(scores.get("loss"), dict) else {}
    wk = scores.get("weights_k") if isinstance(scores.get("weights_k"), dict) else {}
    brier = scores.get("brier_decider") if isinstance(scores.get("brier_decider"), dict) else {}
    base = scores.get("brier_base_rate") if isinstance(scores.get("brier_base_rate"), dict) else {}
    nb = scores.get("n_brier") if isinstance(scores.get("n_brier"), dict) else {}
    ns = scores.get("n_scored") if isinstance(scores.get("n_scored"), dict) else {}
    nosk = scores.get("decider_no_skill") if isinstance(scores.get("decider_no_skill"), list) else []
    ids = sorted({*loss, *wk, *brier, *ns}, key=_order)
    per = []
    for sid in ids:
        L = loss.get(sid) if isinstance(loss.get(sid), dict) else {}
        W = wk.get(sid) if isinstance(wk.get(sid), dict) else {}
        per.append({"id": sid, "loss": {k: _r(_num(L.get(k)), 6) for k in FORECASTERS if k in L or k in W},
                    "weight": {k: _r(_num(W.get(k)), 4) for k in FORECASTERS if k in L or k in W},
                    "brier": _r(_num(brier.get(sid)), 4), "brier_base": _r(_num(base.get(sid)), 4),
                    "n_brier": int(nb[sid]) if isinstance(nb.get(sid), int) else None,
                    "n_scored": int(ns[sid]) if isinstance(ns.get(sid), int) else None,
                    "decider_no_skill": sid in nosk})
    ts = scores.get("tilt_skill") if isinstance(scores.get("tilt_skill"), dict) else {}
    n_days = int(ts["n_days"]) if isinstance(ts.get("n_days"), int) else 0
    kappa = _num(scores.get("kappa"))
    kappa = _num(ts.get("kappa")) if kappa is None else kappa
    return {"date": scores.get("date") if isinstance(scores.get("date"), str) else None,
            "per_strategy": per,
            "tilt": {"n_days": n_days, "days_needed": TILT_DAYS, "kappa": kappa,
                     "skill_90_ann_pct": _r(_num(ts.get("skill_90_ann_pct")), 3), "skill_180_ann_pct": _r(_num(ts.get("skill_180_ann_pct")), 3),
                     "live": bool(kappa and kappa > 0),
                     "line": tilt_line(kappa, n_days)}}


def tilt_line(kappa: float | None, n_days: int) -> str:
    if kappa and kappa > 0:
        return f"tilt live control: on at room {kappa:.2f} (the shadow tilt has beaten the core over 90 days; {n_days} days scored)"
    return f"tilt live control: off until the shadow tilt beats the core over {TILT_DAYS} days ({min(n_days, 999)}/{TILT_DAYS} days so far)"


def compound_pct(rets: list[float]) -> float:
    g = 1.0
    for r in rets:
        g *= 1.0 + r
    return (g - 1.0) * 100.0


def policy_days(root: Path, venue: str | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """{day: {policy: {sid: weight}}}: the weights each policy held on each UTC day, from allocator/policies.jsonl. A day's
    weights are those of the last weights run during the day before it (the 23:50 slot, or a DECIDE NOW); a day with no
    weights run the day before has no entry: it is missing, never weight 0. `venue` is the allocator venue by default."""
    venue = venue or ((read_json(root / "allocator" / "state.json") or {}).get("primary_venue") or "bybit-demo")
    lines = []
    try:
        with open(root / "allocator" / "policies.jsonl") as f:
            for raw in f:
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("venue") == venue and isinstance(rec.get("policies"), dict) and _num(rec.get("t_ms")) is not None:
                    lines.append(rec)
    except OSError:
        return {}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for rec in sorted(lines, key=lambda r: r["t_ms"]):
        start = int(rec["t_ms"]) // 86_400_000 * 86_400_000
        governed = date.fromordinal(date(1970, 1, 1).toordinal() + start // 86_400_000 + 1).isoformat()
        out[governed] = {k: {sid: float(w) for sid, w in v.items() if _num(w) is not None}
                         for k, v in rec["policies"].items() if isinstance(v, dict)}
    return out


def pnl_section(ids: list[str], labels: dict[str, str], root: Path, go_live: str,
                by_day: dict[str, dict[str, dict[str, float]]] | None = None) -> list[dict]:
    """Live P&L (status.json, since the book started) next to the modelled shadow return of returns/<id>.csv from the go-live day.
    The shadow is at weight 1 and ends at yesterday's close; the live book runs at its weight up to now. The weighted shadow
    uses each day's recorded live weight (policy_days); a day without one is missing and left out, counted in missing_days."""
    by_day = policy_days(root) if by_day is None else by_day
    out = []
    for sid in ids:
        st = read_json(root / sid / "status.json") or {}
        ctl = read_json(root / sid / "control.json") or {}
        rets = read_returns(root / "returns" / f"{sid}.csv")
        days = sorted(d for d in rets if d >= go_live)
        shadow = compound_pct([rets[d] for d in days]) if days else None
        recorded = [(d, by_day[d]["live"][sid]) for d in days if sid in ((by_day.get(d) or {}).get("live") or {})]
        out.append({"id": sid, "label": labels.get(sid, sid), "live_pnl_pct": _r(_num(st.get("pnl_pct")), 3),
                    "live_pnl_usdt": _r(_num(st.get("pnl_usdt")), 2), "shadow_pct": _r(shadow, 3),
                    "shadow_weighted_pct": _r(compound_pct([rets[d] * w for d, w in recorded]), 3) if recorded else None,
                    "weighted_days": len(recorded), "missing_days": len(days) - len(recorded),
                    "weight": _r(_num(ctl.get("weight")), 3), "days": len(days), "last_day": days[-1] if days else None})
    return out


POLICY_KEYS = (("core", "core_pct"), ("tilt", "tilted_pct"), ("live", "live_pct"), ("core_veto", "core_veto_pct"))


def history_section(root: Path, ids: list[str], go_live: str, by_day: dict[str, dict[str, dict[str, float]]] | None = None) -> dict | None:
    """Cumulative return since go-live of the recorded policies (core, tilt at kappa 1, live, core with the veto), each day at
    the weights that policy actually held that day (policy_days) and the modelled returns of returns/<id>.csv: the venue's
    portfolio is the average of its books, sum(w r) / N. A day with no recorded weights, or a weighted book without a
    return, is missing: listed in missing_days and left out, never counted as a zero return. None until a day is recorded."""
    by_day = policy_days(root) if by_day is None else by_day
    rets = {sid: read_returns(root / "returns" / f"{sid}.csv") for sid in ids}
    all_days = sorted({d for r in rets.values() for d in r if d >= go_live} | {d for d in by_day if d >= go_live})
    growth = {k: 1.0 for k, _ in POLICY_KEYS}
    days: list[str] = []
    series: dict[str, list[float | None]] = {out: [] for _, out in POLICY_KEYS}
    missing: list[str] = []
    for d in all_days:
        pol = by_day.get(d)
        day_r: dict[str, float] = {}
        for k, _ in POLICY_KEYS:
            w = (pol or {}).get(k)
            if not w or any(x > 0 and d not in rets.get(sid, {}) for sid, x in w.items()):
                break
            day_r[k] = sum(x * rets[sid][d] for sid, x in w.items() if x > 0) / len(w)
        if len(day_r) < len(POLICY_KEYS):
            missing.append(d)
            continue
        days.append(d)
        for k, out in POLICY_KEYS:
            growth[k] *= 1.0 + day_r[k]
            series[out].append(_r((growth[k] - 1.0) * 100.0, 4))
    if not days:
        return None
    return {"since": go_live, "days": days, **series, "missing_days": missing}


# ------------------------------------------------------------------------------------------------ service
def build(root: Path, now_s: float, go_live: str = GO_LIVE) -> dict[str, Any]:
    reg = load_registry(root)
    labels = {e.id: e.label for e in reg.entries}
    regime = read_json(root / "regime" / "latest.json")
    fc = read_json(root / "forecast" / "latest.json")
    scores = read_json(root / "forecast" / "scores.json")
    line = newest_with(root / "allocator", "core")
    ids = list(labels)
    for src in ((fc or {}).get("strategies"), (line or {}).get("core"), (scores or {}).get("loss")):
        if isinstance(src, dict):
            ids += [k for k in src if isinstance(k, str) and k not in ids]
    ids = sorted(dict.fromkeys(ids), key=_order)
    missing = [name for name, doc in (("regime", regime), ("forecast", fc), ("scores", scores), ("allocator", line)) if not doc]
    by_day = policy_days(root)
    return {
        "generated_at": now_s, "go_live": go_live, "root": str(root),
        "regime": regime_section(regime), "forecast": forecast_section(fc, now_s),
        "allocator": None if not line else {"t_ms": _num(line.get("t_ms")), "weights_mode": line.get("weights_mode"),
                                           "kappa": _num(line.get("kappa")), "t_eff": _num(line.get("t_eff")),
                                           "model": line.get("model") if isinstance(line.get("model"), str) else None},
        "strategies": strategy_rows(ids, labels, fc, line, root, scores) if (fc or line) else [],
        "scoreboard": scoreboard_section(scores),
        "pnl": pnl_section(ids, labels, root, go_live, by_day) if ids else [],
        "history": history_section(root, ids, go_live, by_day) if ids else None,
        "missing": missing,
    }


class PredictionsService:
    """The /api/predictions answer, cached for 30 s. The reads run in a thread: a slow disk never blocks the event loop."""

    def __init__(self, root: Path | None = None, clock: Callable[[], float] = time.time) -> None:
        self._root, self.clock = root, clock
        self._cache: tuple[float, dict] | None = None
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else live_root()

    async def snapshot(self, force: bool = False) -> dict[str, Any]:
        async with self._lock:
            now = self.clock()
            if not force and self._cache and now - self._cache[0] < CACHE_S:
                return self._cache[1]
            try:
                doc = await asyncio.to_thread(build, self.root, now)
            except Exception as exc:  # noqa: BLE001 - the panel degrades, it never errors the page
                doc = {"generated_at": now, "go_live": GO_LIVE, "root": str(self.root), "regime": None, "forecast": None, "allocator": None,
                       "strategies": [], "scoreboard": None, "pnl": [], "history": None, "missing": ["regime", "forecast", "scores", "allocator"],
                       "error": f"{type(exc).__name__}: {exc}"[:300]}
            self._cache = (now, doc)
            return doc
