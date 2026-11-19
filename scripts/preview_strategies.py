"""Local preview of the Strategies tab on fixture files (no runner, no systemd, no orders).

    .venv/bin/python scripts/preview_strategies.py [--port 8801]

Copies tests/fixtures/live to a temp dir with fresh timestamps, points CTB_LIVE_ROOT at it and starts the dashboard
with a throw-away database and no keys (observe-only, orders are impossible). The LLM allocator runs with a FAKE model
(no network call to any model, a canned allocation) and a 2 s tick, so DECIDE NOW answers within seconds. Start/stop of
the runner unit answers "not installed" because no such unit exists on this machine; the file-based controls work.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED = 1_800_000_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8801")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="ctb-preview-"))
    live = tmp / "live"
    shutil.copytree(ROOT / "tests" / "fixtures" / "live", live)
    shift = int(time.time() * 1000) - FIXED
    for f in live.rglob("*.json"):
        doc = json.loads(f.read_text())
        if isinstance(doc, dict):
            for k, v in doc.items():
                if k.endswith("_ms") and isinstance(v, (int, float)):
                    doc[k] = int(v) + shift
            f.write_text(json.dumps(doc, indent=1) + "\n")
    for f in live.rglob("*.jsonl"):
        rows = []
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            for k in ("t_ms",):
                if k in r:
                    r[k] += shift
            rows.append(json.dumps(r, separators=(",", ":")))
        f.write_text("\n".join(rows) + ("\n" if rows else ""))
    env = {**os.environ, "CTB_LIVE_ROOT": str(live), "CTB_ENV_FILE": os.devnull, "CTB_PORT": args.port,
           "CTB_DB_PATH": str(tmp / "ctb.db"), "CTB_TRADING_ENABLED": "0", "CTB_DASHBOARD_ACCOUNT": "demo-main"}
    for k in ("BYBIT_API_KEY", "BYBIT_API_SECRET", "BYBIT_DEMO_API_KEY", "BYBIT_DEMO_API_SECRET"):
        env.pop(k, None)
    print(f"preview data in {live}; open http://127.0.0.1:{args.port} and press F7")
    os.chdir(ROOT)
    os.environ.update(env)
    sys.path.insert(0, str(ROOT))
    from ctb import allocator as alloc_mod
    from ctb import server

    class FakeModel:
        """Stands in for the allocator's chat model: always the same valid answer, free of charge."""

        async def ask_json(self, messages, schema):
            props = schema["properties"]["weights"]["properties"]
            return ({"weights": {i: (0.8 if i == "carry" else 0.0) for i in props}, "enabled": {i: i == "carry" for i in props},
                     "reasoning": "fake model: carry only, everything else flat", "confidence": 0.5}, {"cost": 0.0})

    async def no_market():
        return {}

    def fake_create(session, root=None, spec=None):
        return alloc_mod.Allocator(root or live, FakeModel(), no_market, model="fake-model", tick_s=2.0)

    alloc_mod.create_allocator = fake_create
    server.main()


if __name__ == "__main__":
    main()
