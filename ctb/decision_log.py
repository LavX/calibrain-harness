"""allocator/decisions.jsonl and its daily archives.

The allocator appends one line per decision to decisions.jsonl. When the first line of a new UTC day is appended, the
file's lines move to decisions-<day>.jsonl, each line to the archive of its own UTC day (a log from before rotation,
spanning many days, is split the same way), the archives before the newest are gzipped (decisions-<day>.jsonl.gz) and
archives older than KEEP_DAYS are deleted. The newest archive stays uncompressed so
the feed and the predictions panel can read the hours just before midnight cheaply.

Archiving is never in the way of a decision: append() runs the rotation in its own try and writes the line to the
current file whatever happens (rotation_error() says why the last rotation failed; it is retried after RETRY_MS). A
.gz is written to a temp file, synced and renamed, never appended to. Readers skip what they cannot read of a damaged
archive and report it once.

Readers: tail() for the newest lines (current file, the file being split, then the archives), iter_all() for everything
kept, oldest first (the scorers). No other module writes these files.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import shutil
import time
from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Iterator

log = logging.getLogger(__name__)

LOG = "decisions.jsonl"
WORK = LOG + ".rotating"  # the file being split; its lines count as the newest before the current file
KEEP_DAYS = 90
RETRY_MS = 15 * 60_000  # a failed rotation is tried again after this long; lines keep going to the current file meanwhile
ARCHIVE_RE = re.compile(r"^decisions-(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?$")

# the rotation's file operations, so a test can make each one fail without touching the decision line's own write
_replace = os.replace
_fsync = os.fsync


def _unlink(p: Path) -> None:
    p.unlink(missing_ok=True)


_failures: dict[str, tuple[float, str]] = {}  # directory -> (when the last rotation failed, why)
_unreadable: set[str] = set()  # archives already reported as damaged


def rotation_error(d: Path) -> str | None:
    """Why the last rotation of this directory failed, None when it did not (or has since succeeded)."""
    f = _failures.get(str(d))
    return f[1] if f else None


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


def _first_day(path: Path, scan: int = 200) -> str | None:
    """The UTC day of the first dated line of the current file (the first `scan` lines are looked at), None if none."""
    try:
        with open(path, "rb") as f:
            for i, line in enumerate(f):
                if i >= scan:
                    break
                try:
                    t = json.loads(line).get("t_ms")
                except (ValueError, AttributeError):
                    continue
                if isinstance(t, (int, float)) and not isinstance(t, bool):
                    return utc_day(float(t))
    except OSError:
        return None
    return None


def _raw_lines(path: Path, strict: bool = False) -> Iterator[bytes]:
    """The lines of a log file as bytes, plain or gzipped. A damaged file (truncated, corrupt) yields what is readable,
    is reported once and is never an error for the caller; with strict (the split, which deletes its source after) a
    read error raises instead, so nothing unread is ever removed."""
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rb") as f:
            for raw in f:
                yield raw if raw.endswith(b"\n") else raw + b"\n"
    except FileNotFoundError:
        return
    except Exception as exc:  # noqa: BLE001 - zlib.error, BadGzipFile, EOFError, OSError: a damaged archive is skipped
        if strict:
            raise
        if str(path) not in _unreadable:
            _unreadable.add(str(path))
            log.warning("decision log: %s is damaged after its readable part, the rest is skipped (%s: %s)", path, type(exc).__name__, exc)


def _sync_close(f: BinaryIO) -> None:
    f.flush()
    _fsync(f.fileno())
    f.close()


def _gzip(src: Path) -> None:
    """Compress a plain archive to decisions-<day>.jsonl.gz atomically: a temp file (holding what is readable of an
    existing .gz of that day first) is written, synced and renamed over the .gz; only then is the plain file removed. A
    crash leaves the old .gz and the plain file as they were (at worst a later run holds some lines twice)."""
    dst = src.with_name(src.name + ".gz")
    tmp = src.with_name(f".{dst.name}.tmp")
    raw = open(tmp, "wb")
    try:
        with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
            if dst.exists():
                for line in _raw_lines(dst):
                    gz.write(line)
            with open(src, "rb") as fin:
                shutil.copyfileobj(fin, gz)
        _sync_close(raw)
    except BaseException:
        raw.close()
        _unlink(tmp)
        raise
    _replace(tmp, dst)
    _unlink(src)


def _split(d: Path, work: Path, today: str) -> None:
    """Move the lines of `work` into the plain archive of their UTC day (a line without t_ms goes with the line before
    it), streaming: a large log from before rotation is never held in memory. A line of today goes to the current file.
    Every file written is synced before `work` is removed."""
    handles: dict[Path, BinaryIO] = {}
    lead: list[bytes] = []
    day = None

    def write(day: str, raw: bytes) -> None:
        path = d / LOG if day >= today else d / f"decisions-{day}.jsonl"
        if path not in handles:
            handles[path] = open(path, "ab")
        handles[path].write(raw)

    try:
        for raw in _raw_lines(work, strict=True):
            try:
                t = json.loads(raw).get("t_ms")
            except (ValueError, AttributeError):
                t = None
            if isinstance(t, (int, float)) and not isinstance(t, bool):
                day = utc_day(t)
                for x in lead:  # leading lines without a time go with the first dated one
                    write(day, x)
                lead = []
            if day is None:
                lead.append(raw)
            else:
                write(day, raw)
        for x in lead:
            write(today, x)
        for f in handles.values():
            _sync_close(f)
    finally:
        for f in handles.values():
            if not f.closed:
                f.close()
    _unlink(work)


def rotate(d: Path, now_ms: float, keep_days: int = KEEP_DAYS) -> None:
    """When the current file holds an earlier UTC day: move its lines into the archive of their own day (all its days, if
    it is a log from before rotation), gzip every archive but the newest, then delete the archives older than keep_days.
    Splitting comes first, so no line is deleted for the day of an older neighbour. A split cut short by a crash is
    finished at the next rotation (its lines may then appear twice, never get lost). Raises on any failure: append()
    catches it."""
    cur, work = d / LOG, d / WORK
    today = utc_day(now_ms)
    for p in d.glob(".decisions-*.tmp"):  # a gzip cut short: the archive it was replacing is intact
        _unlink(p)
    if work.exists():
        _split(d, work, today)
    first = _first_day(cur)
    if first is None or first >= today:
        return
    _replace(cur, work)  # appends go to a fresh file from here on
    _split(d, work, today)
    days = archives(d)
    cutoff = (date.fromisoformat(today) - timedelta(days=keep_days)).isoformat()
    newest = max((day for day, _ in days), default=None)
    for day, p in days:
        if day < cutoff:
            _unlink(p)
        elif p.name.endswith(".jsonl") and day != newest:
            _gzip(p)


def append(d: Path, obj: Any, now_ms: float) -> None:
    """Append one line to the current file. The rotation before it runs in its own try: if it fails, the failure is
    logged, kept for rotation_error() (the allocator's status shows it), the line is appended to the current file all the
    same and the rotation is tried again after RETRY_MS."""
    d.mkdir(parents=True, exist_ok=True)
    key = str(d)
    failed = _failures.get(key)
    if failed is None or now_ms - failed[0] >= RETRY_MS or now_ms < failed[0]:
        try:
            rotate(d, now_ms)
            _failures.pop(key, None)
        except Exception as exc:  # noqa: BLE001 - archiving must never stop a decision line
            msg = f"{type(exc).__name__}: {exc}"[:300]
            if failed is None or failed[1] != msg:
                log.exception("decision log: rotation failed, lines keep going to %s", d / LOG)
            _failures[key] = (now_ms, msg)
    with open(d / LOG, "a") as f:
        f.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _lines(path: Path) -> Iterator[dict]:
    for raw in _raw_lines(path):
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def _newest_first(d: Path) -> list[Path]:
    """The current file, the file being split, then the archives, newest first."""
    return [d / LOG, d / WORK, *(p for _, p in reversed(archives(d)))]


def tail(d: Path, n: int) -> list[dict]:
    """The newest n lines, oldest first: the current file, topped up from the file being split and the archives."""
    out: list[dict] = []
    for p in _newest_first(d):
        if len(out) >= n:
            break
        out = list(deque(_lines(p), maxlen=n - len(out))) + out
    return out


def iter_all(d: Path) -> Iterator[dict]:
    """Every kept line, oldest first: the archives, the file being split, then the current file."""
    for p in reversed(_newest_first(d)):
        yield from _lines(p)
