import os

import pytest

# Keep the developer's local .env (live strategy settings, keys) out of the tests. Must run before
# ctb.config is imported, since its field defaults read the environment at import time.
os.environ["CTB_ENV_FILE"] = os.devnull


@pytest.fixture(autouse=True)
def _venue_testnet(monkeypatch):
    """Tests assume the testnet venue whatever the shell environment says; venue tests override it."""
    monkeypatch.setenv("CTB_VENUE", "testnet")


@pytest.fixture(autouse=True)
def _state_in_tmp(monkeypatch, tmp_path):
    """The harness's own data dir (venue baselines, the header set) follows CTB_DB_PATH: keep it out of the repo's data/."""
    monkeypatch.setenv("CTB_DB_PATH", str(tmp_path / "state" / "ctb.db"))
    monkeypatch.delenv("CTB_LIVE_ROOTS", raising=False)
