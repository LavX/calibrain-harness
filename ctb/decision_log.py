"""allocator/decisions.jsonl and its daily archives.

The allocator appends one line per decision to decisions.jsonl. When the first line of a new UTC day is appended, the
file is renamed to decisions-<day>.jsonl (<day> = the UTC day of its first line), the archives before it are gzipped
(decisions-<day>.jsonl.gz) and archives older than KEEP_DAYS are deleted. The newest archive stays uncompressed so
the feed and the predictions panel can read the hours just before midnight cheaply.

Readers: tail() for the newest lines (current file, then the newest archive), iter_all() for everything kept, oldest
first (the scorers). No other module writes these files.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

LOG = "decisions.jsonl"
KEEP_DAYS = 90
ARCHIVE_RE = re.compile(r"^decisions-(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$")


def utc_day(t_ms: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t_ms / 1000))


def archives(d: Path) -> list[tuple[str, Path]]:
    """(day, path) of every archive, oldest first; an uncompressed one sorts after a compressed one of the same day."""
    try:
        names = os.listdir(d)
    except OSError:
        return []
    out = []
    for name in names:
        m = ARCHIVE_RE.match(name)
        if m:
            out.append((m.group(1), not name.endswith(".gz"), d / name))
    return [(day, p) for day, _, p in sorted(out)]


def _first_day(path: Path) -> str | None:
    """The UTC day of the first line of the current file, None when it is empty or unreadable."""
    try:
        with open(path) as f:
            for line in f:
                try:
                    t = json.loads(line).get("t_ms")
                except (ValueError, AttributeError):
                    continue
                return utc_day(float(t)) if isinstance(t, (int, float)) else None
    except OSError:
        return None
    return None


def _gzip(src: Path) -> None:
    dst = src.with_name(src.name + ".gz")
    with open(src, "rb") as fin, gzip.open(dst, "ab") as fout:  # "ab": a second member if the day was split, gzip reads both
        shutil.copyfileobj(fin, fout)
    src.unlink()


def rotate(d: Path, now_ms: float, keep_days: int = KEEP_DAYS) -> None:
    """Before appending a line at now_ms: archive the current file if it holds an earlier UTC day."""
    cur = d / LOG
    first = _first_day(cur)
    today = utc_day(now_ms)
    if first is None or first >= today:
        return
    target = d / f"decisions-{first}.jsonl"
    if target.exists():  # a day split by a restart: keep both parts in order
        with open(cur, "rb") as fin, open(target, "ab") as fout:
            shutil.copyfileobj(fin, fout)
        cur.unlink()
    else:
        os.replace(cur, target)
    cutoff = (date.fromisoformat(today) - timedelta(days=keep_days)).isoformat()
    for day, p in archives(d):
        if day < cutoff:
            p.unlink(missing_ok=True)
        elif p.name.endswith(".jsonl") and p != target:
            _gzip(p)


def append(d: Path, obj: Any, now_ms: float) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rotate(d, now_ms)
    with open(d / LOG, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _lines(path: Path) -> Iterator[dict]:
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
    except (OSError, EOFError):
        return


def tail(d: Path, n: int) -> list[dict]:
    """The newest n lines, oldest first: the current file, topped up from the archives (newest first) after a rotation."""
    out: list[dict] = list(deque(_lines(d / LOG), maxlen=n))
    for _, p in reversed(archives(d)):
        if len(out) >= n:
            break
        out = list(deque(_lines(p), maxlen=n - len(out))) + out
    return out


def iter_all(d: Path) -> Iterator[dict]:
    """Every kept line, oldest first: the archives, then the current file."""
    for _, p in archives(d):
        yield from _lines(p)
    yield from _lines(d / LOG)
