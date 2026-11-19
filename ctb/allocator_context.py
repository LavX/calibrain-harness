"""Extra state for the decider (question set alloc-v1.1): regime, news and the payload bound. No network.

alloc-v1.0 gave the decision model the strategy books and a market summary. alloc-v1.1 adds, as separate
blocks of the same state object:

  regime      <live root>/regime/latest.json: today's state, its eight features and the 7-day row (P7).
  forecasts   the prediction layer's computed forecasts per strategy (built in ctb/allocator.py, which owns
              the weights math): combined mu and sigma in %/yr, forecast Sharpe, core weight. The decider's own
              past answers are not in them.
  news        the last 24 h of exchange announcements and RSS headlines the H7 collector stored, newest first.
  state_notes one plain line for each block that is missing or was cut, so the model is told, not left to guess.

The H7 collector (calibrain-study, research/v3/h7/) writes, under its root (default
<live root>/../research/h7/collector, on aurora4 /home/lavx/Documents/calibrain-study/data/research/h7/collector,
override CTB_ALLOCATOR_NEWS_ROOT):

  items/h7-items-YYYY-MM-DD.jsonl   {"k":"item","key","s" (source),"cat" (type),"pub" (ms or null),"rx" (ms),"sha",...}
  raw/h7-raw-YYYY-MM-DD.jsonl       {"k":"raw","key","sha","title","text","url","cat"}  (the text is never read here)

Day files are named by the UTC day the line was written, so every file from the window's first day on is read.
The item line has no title: it is joined to its raw line by (key, sha). The collector does not store the
announcement's tags; "tags" is passed on when a line carries a list of them and left out otherwise.

RSS items (any source whose name does not start with "bybit") are sent together with the Bybit announcements in one news
block, newest first: CTB_ALLOCATOR_RSS_HEADLINES is on by default (owner decision 2026-11-19); set 0 to send Bybit
announcements only. Headlines are text from third parties: they are data for the model, never instructions, and the question text says so.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

NEWS_WINDOW_H = 24
NEWS_MAX_ITEMS = 40
TITLE_MAX = 160
TAG_MAX = 5
MAX_FILE_BYTES = 256 * 1024 * 1024     # a day file larger than this is skipped, with a note
PAYLOAD_MARGIN = 4096                  # room for the model id, the JSON frame and the questions' own growth
BYBIT_PREFIX = "bybit"
_TRUE = {"1", "true", "yes", "on"}
_CTRL = re.compile(r"[\x00-\x1f\x7f]+")
_DAY_RE = re.compile(r"-(\d{4}-\d{2}-\d{2})\.jsonl$")


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str, allow_nan=False)


def rss_headlines_enabled() -> bool:
    """CTB_ALLOCATOR_RSS_HEADLINES: on by default (owner decision 2026-11-19); set 0 to send Bybit announcements only."""
    return os.environ.get("CTB_ALLOCATOR_RSS_HEADLINES", "1").strip().lower() in _TRUE


def news_root(live: Path) -> Path:
    raw = os.environ.get("CTB_ALLOCATOR_NEWS_ROOT", "").strip()
    return Path(raw) if raw else live.parent / "research" / "h7" / "collector"


# ---------------------------------------------------------------------------------- regime
def regime_block(live: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(block, note). The block is None, with a note, when the file is missing or unreadable."""
    r = _read_json(live / "regime" / "latest.json")
    if not isinstance(r, dict) or not isinstance(r.get("state"), str):
        return None, "regime omitted: regime/latest.json is missing or unreadable"
    feats = r.get("features") if isinstance(r.get("features"), dict) else {}
    p7 = r.get("p7") if isinstance(r.get("p7"), dict) else {}
    block: dict[str, Any] = {
        "date": r.get("date") if isinstance(r.get("date"), str) else None,
        "state": r["state"],
        "features": {k: round(n, 4) for k, v in feats.items() if (n := _num(v)) is not None},
        "p7": {k: round(n, 4) for k, v in p7.items() if (n := _num(v)) is not None},
    }
    return block, None


# ---------------------------------------------------------------------------------- news
def _day_files(directory: Path, prefix: str, first_day: date) -> list[Path]:
    out: list[tuple[date, Path]] = []
    try:
        names = list(directory.iterdir())
    except OSError:
        return []
    for p in names:
        m = _DAY_RE.search(p.name)
        if not (p.name.startswith(prefix) and m):
            continue
        try:
            d = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if d >= first_day:
            out.append((d, p))
    return [p for _, p in sorted(out)]


def _jsonl(path: Path, notes: list[str]):
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            notes.append(f"news: {path.name} skipped (over {MAX_FILE_BYTES // 2**20} MiB)")
            return
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _clean(text: Any, n: int) -> str:
    s = _CTRL.sub(" ", str(text or "")).strip()
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _utc(ms: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


def news_block(root: Path, now_ms: int, *, rss: bool, window_h: int = NEWS_WINDOW_H,
               max_items: int = NEWS_MAX_ITEMS) -> tuple[dict[str, Any] | None, list[str]]:
    """(block, notes): the announcements of the last window_h hours, newest first, at most max_items.
    None with a note when the collector's files are not there at all."""
    notes: list[str] = []
    items_dir = root / "items"
    if not items_dir.is_dir():
        return None, ["news omitted: the H7 collector files are not available"]
    since_ms = now_ms - window_h * 3_600_000
    first_day = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).date()
    wanted: dict[tuple[str, str], dict[str, Any]] = {}
    skipped_rss = 0
    for path in _day_files(items_dir, "h7-items-", first_day):
        for row in _jsonl(path, notes):
            if row.get("k") != "item":
                continue
            key, sha, src = row.get("key"), row.get("sha"), row.get("s")
            if not (isinstance(key, str) and isinstance(sha, str) and isinstance(src, str)):
                continue
            pub, rx = _num(row.get("pub")), _num(row.get("rx"))
            t = pub if pub is not None else rx
            if t is None:
                continue
            if rx is not None and t > rx:      # a source clock ahead of ours: the receive time is the honest one
                t = rx
            if not since_ms <= t <= now_ms + 60_000:
                continue
            if not src.lower().startswith(BYBIT_PREFIX) and not rss:
                skipped_rss += 1
                continue
            wanted[(key, sha)] = {"t": t, "src": "bybit" if src.lower().startswith(BYBIT_PREFIX) else src[:40],
                                  "type": _clean(row.get("cat"), 40), "tags": row.get("tags")}
    titles: dict[tuple[str, str], str] = {}
    if wanted:
        for path in _day_files(root / "raw", "h7-raw-", first_day):
            for row in _jsonl(path, notes):
                k = (row.get("key"), row.get("sha"))
                if k in wanted and isinstance(row.get("title"), str):
                    titles[k] = row["title"]
                    if not wanted[k]["tags"] and isinstance(row.get("tags"), list):
                        wanted[k]["tags"] = row["tags"]
    rows: list[dict[str, Any]] = []
    untitled = 0
    for k, w in wanted.items():
        title = _clean(titles.get(k), TITLE_MAX)
        if not title:
            untitled += 1
            continue
        item: dict[str, Any] = {"t": _utc(w["t"]), "title": title, "type": w["type"], "src": w["src"]}
        tags = [_clean(x, 24) for x in w["tags"] if isinstance(x, str) and x.strip()][:TAG_MAX] if isinstance(w["tags"], list) else []
        if tags:
            item["tags"] = tags
        rows.append({**item, "_ms": w["t"]})
    rows.sort(key=lambda r: r["_ms"], reverse=True)
    more = max(0, len(rows) - max_items)
    out = [{k: v for k, v in r.items() if k != "_ms"} for r in rows[:max_items]]
    if untitled:
        notes.append(f"news: {untitled} item(s) skipped, no stored title")
    block: dict[str, Any] = {"window_h": window_h, "count": len(out), "items": out}
    if more:
        block["older_not_shown"] = more
    if skipped_rss:
        notes.append("news: RSS headlines are switched off (CTB_ALLOCATOR_RSS_HEADLINES=0), only Bybit announcements are shown")
    return block, notes


# ---------------------------------------------------------------------------------- sizes and the payload bound
def block_sizes(ctx: dict[str, Any]) -> dict[str, int]:
    """Encoded bytes of the state, in total and per top-level block, for decisions.jsonl."""
    sizes = {k: len(dumps(v).encode()) for k, v in ctx.items()}
    sizes["total"] = len(dumps(ctx).encode())
    return sizes


def payload_bytes(ctx: dict[str, Any], questions: dict[str, Any]) -> int:
    return len(dumps({"state": ctx, "questions": questions}).encode()) + PAYLOAD_MARGIN


def fit_state(ctx: dict[str, Any], batches: list[dict[str, Any]], limit: int | None) -> list[str]:
    """Make the state fit the decision model's request limit, in place. Returns the notes to add.
    Order of cuts: older news items (down to none), then the news block, the forecasts, the regime. The books and the
    market are never cut. The check uses the largest request of the set."""
    if not limit:
        return []
    notes: list[str] = []

    def too_big() -> bool:
        return max((payload_bytes(ctx, b) for b in batches), default=0) > limit

    news = ctx.get("news")
    if too_big() and isinstance(news, dict) and news.get("items"):
        total = news["count"] + news.get("older_not_shown", 0)
        while too_big() and news["items"]:
            news["items"] = news["items"][: len(news["items"]) // 2]
        news["count"] = len(news["items"])
        if total > news["count"]:
            news["older_not_shown"] = total - news["count"]
        notes.append(f"news cut to {news['count']} item(s) to fit the {limit // 1024} KiB request limit")
    for key in ("news", "forecasts", "regime"):
        if too_big() and key in ctx:
            del ctx[key]
            notes.append(f"{key} omitted: the request would exceed the {limit // 1024} KiB limit")
    if notes:
        ctx["state_notes"] = [*ctx.get("state_notes", []), *notes]
    return notes
