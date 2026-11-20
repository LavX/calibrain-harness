"""allocator/decisions.jsonl and its archives.

Live path (append, used by every process that writes decisions: the dashboard's allocator and `python -m ctb.allocator
once`): under an fcntl lock on allocator/decisions.lock, at the first line of a new UTC day the current file is renamed
in one os.replace to decisions-<its day>.jsonl, then the line is appended. Nothing else: no splitting, no gzip, no read
of the file. If the rename fails it is logged once, the line goes to the current file all the same and the rename is
tried again at the next line. The day of the current file is kept in allocator/decisions.day ({"day", "ino"}).

The log from before this existed (decisions.jsonl without decisions.day) is renamed once, the same way, to
decisions-legacy-<first day>-<last day>.jsonl and is never touched again (not by retention either).

Compression and retention are not in the live path: `python -m ctb.decision_log compact`, run by a nightly timer outside
the allocator (docs/deploy/calibrain-decision-log-compact.*), gzips dated archives older than two days (temp file,
fsync, rename, then the plain file is removed) and deletes dated archives older than 90 days. It is safe to rerun and
never touches the current file or a legacy archive.

Readers: tail() for the newest lines, iter_all() for everything kept, oldest first. Both read plain and gzipped files,
skip the unreadable rest of a damaged archive (reported once) and never raise. With rename only, no line is ever
written twice.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import logging
import os
import re
import shutil
import sys
import time
from collections import deque
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

LOG = "decisions.jsonl"
DAY_FILE = "decisions.day"
LOCK_FILE = "decisions.lock"
KEEP_DAYS = 90
PLAIN_DAYS = 2
DATED_RE = re.compile(r"^decisions-(\d{4}-\d{2}-\d{2})(?:\.(\d+))?\.jsonl(\.gz)?$")
LEGACY_RE = re.compile(r"^decisions-legacy-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})(?:\.(\d+))?\.jsonl$")

_replace = os.replace  # the one rename of the live path; a test makes it fail
_failures: dict[str, tuple[float, str]] = {}  # directory -> (when the last rotation failed, why)
_unreadable: set[str] = set()  # archives already reported as damaged


def rotation_error(d: Path) -> str | None:
    """Why the last rotation of this directory failed, None when it did not (or has since succeeded)."""
    f = _failures.get(str(d))
    return f[1] if f else None


def utc_day(t_ms: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(t_ms / 1000))


# ------------------------------------------------------------------------------------------------ the live path
@contextmanager
def _locked(d: Path) -> Iterator[None]:
    """One appender at a time across processes (the dashboard's allocator and the CLI)."""
    with open(d / LOCK_FILE, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_day(d: Path) -> dict | None:
    try:
        doc = json.loads((d / DAY_FILE).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and isinstance(doc.get("day"), str) and isinstance(doc.get("ino"), int) else None


def _write_day(d: Path, day: str, ino: int) -> None:
    tmp = d / f".{DAY_FILE}.tmp"
    with open(tmp, "w") as f:
        json.dump({"day": day, "ino": ino}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, d / DAY_FILE)


def _free(path: Path) -> Path:
    """`path`, or the first decisions-<...>.<n>.jsonl next to it that does not exist yet (a name is never reused)."""
    if not path.exists() and not path.with_name(path.name + ".gz").exists():
        return path
    stem = path.name[: -len(".jsonl")]
    n = 1
    while (path.with_name(f"{stem}.{n}.jsonl")).exists() or path.with_name(f"{stem}.{n}.jsonl.gz").exists():
        n += 1
    return path.with_name(f"{stem}.{n}.jsonl")


def _edge_days(path: Path, default: str) -> tuple[str, str]:
    """The UTC days of the first and the last dated line of a legacy log: a few lines at its start and its end only."""
    def day_of(raw: bytes) -> str | None:
        try:
            t = json.loads(raw).get("t_ms")
        except (ValueError, AttributeError):
            return None
        return utc_day(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else None

    first = last = None
    try:
        with open(path, "rb") as f:
            for _, raw in zip(range(100), f):
                first = day_of(raw)
                if first:
                    break
            f.seek(max(0, os.fstat(f.fileno()).st_size - 256 * 1024))
            for raw in reversed(f.read().splitlines()):
                last = day_of(raw)
                if last:
                    break
    except OSError:
        pass
    return first or last or default, last or first or default


def _rotate(d: Path, now_ms: float) -> None:
    """The one rename, if one is due. Raises when it fails (append() catches it)."""
    cur = d / LOG
    try:
        ino = os.stat(cur).st_ino
    except FileNotFoundError:
        return  # nothing to rename: the next line starts the file
    rec = _read_day(d)
    if rec is None:  # a log from before rotation (or one whose day is unknown): renamed once, never split
        first, last = _edge_days(cur, utc_day(now_ms))
        _replace(cur, _free(d / f"decisions-legacy-{first}-{last}.jsonl"))
        return
    if rec["ino"] != ino:
        return  # started after a rename whose day record failed: append() records it as today's
    if rec["day"] < utc_day(now_ms):
        _replace(cur, _free(d / f"decisions-{rec['day']}.jsonl"))


def append(d: Path, obj: Any, now_ms: float) -> None:
    """Append one line. The rename before it is the only rotation step here; if it fails the line still goes to the
    current file and the rename is tried again with the next line (rotation_error() says why, the status shows it)."""
    d.mkdir(parents=True, exist_ok=True)
    key = str(d)
    with _locked(d):
        rotated_ok = True
        try:
            _rotate(d, now_ms)
        except Exception as exc:  # noqa: BLE001 - archiving must never stop a decision line
            rotated_ok = False
            msg = f"{type(exc).__name__}: {exc}"[:300]
            if rotation_error(d) != msg:
                log.exception("decision log: rename failed, lines keep going to %s", d / LOG)
            _failures[key] = (now_ms, msg)
        cur = d / LOG
        with open(cur, "a") as f:
            f.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if rotated_ok:
            try:
                ino = os.stat(cur).st_ino
                rec = _read_day(d)
                if rec is None or rec["ino"] != ino:  # a file just started: it holds this day's lines
                    _write_day(d, utc_day(now_ms), ino)
                _failures.pop(key, None)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"[:300]
                if rotation_error(d) != msg:
                    log.exception("decision log: day record not written")
                _failures[key] = (now_ms, msg)


# ------------------------------------------------------------------------------------------------ archives and readers
def archives(d: Path) -> list[Path]:
    """Every archive, oldest first: legacy logs (by their last day), then the dated archives (by day, then number). When a
    dated archive exists both plain and gzipped (compaction in between), the plain one is read, so no line counts twice."""
    try:
        names = set(os.listdir(d))
    except OSError:
        return []
    keyed = []
    for name in names:
        m = LEGACY_RE.match(name)
        if m:
            keyed.append(((m.group(2), 0, m.group(1), int(m.group(3) or 0)), d / name))
            continue
        m = DATED_RE.match(name)
        if m:
            if m.group(3) and name[: -len(".gz")] in names:
                continue
            keyed.append(((m.group(1), 1, "", int(m.group(2) or 0)), d / name))
    return [p for _, p in sorted(keyed)]


def _raw_lines(path: Path) -> Iterator[bytes]:
    """The lines of a log file as bytes, plain or gzipped. A damaged file yields what is readable, is reported once and is
    never an error for the caller."""
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rb") as f:
            for raw in f:
                yield raw if raw.endswith(b"\n") else raw + b"\n"
    except FileNotFoundError:
        return
    except Exception as exc:  # noqa: BLE001 - zlib.error, BadGzipFile, EOFError, OSError, decode errors
        if str(path) not in _unreadable:
            _unreadable.add(str(path))
            log.warning("decision log: %s is damaged after its readable part, the rest is skipped (%s: %s)", path, type(exc).__name__, exc)


def _lines(path: Path) -> Iterator[dict]:
    for raw in _raw_lines(path):
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict):
            yield row


def _newest_first(d: Path) -> list[Path]:
    """The current file, then the archives, newest first."""
    return [d / LOG, *reversed(archives(d))]


def tail(d: Path, n: int) -> list[dict]:
    """The newest n lines, oldest first: the current file, topped up from the archives."""
    out: list[dict] = []
    for p in _newest_first(d):
        if len(out) >= n:
            break
        out = list(deque(_lines(p), maxlen=n - len(out))) + out
    return out


def iter_all(d: Path) -> Iterator[dict]:
    """Every kept line, oldest first: the archives, then the current file."""
    for p in reversed(_newest_first(d)):
        yield from _lines(p)


# ------------------------------------------------------------------------------------------------ compaction (nightly)
def _gzip(src: Path) -> None:
    """src -> src.gz: a temp file, synced and renamed over any earlier .gz; the plain file is removed only after. A crash
    leaves the plain file (the one readers use) as it was."""
    dst = src.with_name(src.name + ".gz")
    tmp = src.with_name(f".{dst.name}.tmp")
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as gz, open(src, "rb") as fin:
                shutil.copyfileobj(fin, gz)
            raw.flush()
            os.fsync(raw.fileno())
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    src.unlink()


def compact(d: Path, now_ms: float, keep_days: int = KEEP_DAYS, plain_days: int = PLAIN_DAYS) -> dict[str, list[str]]:
    """Gzip the dated archives older than plain_days and delete the dated archives older than keep_days. Never touches the
    current file, its day record or a legacy archive. Idempotent: a rerun finds nothing left to do."""
    today = date.fromisoformat(utc_day(now_ms))
    drop_before = (today - timedelta(days=keep_days)).isoformat()
    zip_before = (today - timedelta(days=plain_days)).isoformat()
    done: dict[str, list[str]] = {"gzipped": [], "deleted": []}
    for p in d.glob(".decisions-*.tmp"):  # a gzip cut short
        p.unlink(missing_ok=True)
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return done
    for name in names:
        m = DATED_RE.match(name)
        if not m:
            continue
        p = d / name
        if m.group(1) < drop_before:
            p.unlink(missing_ok=True)
            done["deleted"].append(name)
        elif not m.group(3) and m.group(1) < zip_before:
            _gzip(p)
            done["gzipped"].append(name)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ctb.decision_log", description="Maintain allocator/decisions.jsonl archives.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compact", help="gzip dated archives older than --plain-days, delete those older than --keep-days")
    c.add_argument("--dir", type=Path, default=None, help="the allocator directory (default: <CTB_LIVE_ROOT>/allocator)")
    c.add_argument("--keep-days", type=int, default=KEEP_DAYS)
    c.add_argument("--plain-days", type=int, default=PLAIN_DAYS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = args.dir
    if d is None:
        from .strategies import live_root
        d = live_root() / "allocator"
    out = compact(d, time.time() * 1000, args.keep_days, args.plain_days)
    print(json.dumps({"dir": str(d), **out}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
