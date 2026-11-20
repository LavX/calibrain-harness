"""One lock per control.json for every read-modify-write the harness does on it: the dashboard (ctb.strategies) and the
LLM allocator (ctb.allocator) run in the same process, and without it a write built from an older read could drop keys
the other just wrote (a kill's owner_lock and flatten, or the owner's pause). A threading lock, so it holds whether the
write runs on the event loop or in a worker thread; never held across an await.

The runner is another process and writes control.json too (it clears flatten when it is flat); that cross-process race
is the runner's to close (it would need a file lock both sides honour)."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


@contextmanager
def locked(path: Path | str) -> Iterator[None]:
    key = str(Path(path).resolve())
    with _GUARD:
        lock = _LOCKS.setdefault(key, threading.Lock())
    with lock:
        yield
