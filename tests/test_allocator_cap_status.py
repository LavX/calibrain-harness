"""An infeasible final cap stays visible until a weights run checks it again."""

import asyncio
import math

from ctb import allocator as A
from ctb.kairos.client import RateBudget
from test_allocator_computed import Clock, FakeDecider, T0, answers, market, put


def test_infeasible_cap_error_survives_refresh_and_restart_until_weights_run(tmp_path):
    root = tmp_path / "live"
    sigma = 0.20 / math.sqrt(365)
    put(root / "strategies.json", [{"id": "trend", "label": "Trend"}])
    put(root / "trend" / "status.json", {"state": "running", "equity_usdt": 1000.0,
                                          "start_equity_usdt": 1000.0, "pnl_pct": 0.0})
    put(root / "trend" / "control.json", {"enabled": True, "weight": 0.6, "owner_lock": True})
    put(root / "forecast" / "latest.json", {
        "date": "2026-11-20", "made_ms": int((T0 - 3600) * 1000), "kappa": 0.0,
        "strategies": {"trend": {"sigma_d": sigma, "mu": {"base": 0.0}, "n_days": 900}},
        "cov_d": {"ids": ["trend"], "matrix": [[sigma ** 2]]},
        "weights_k": {"trend": {"base": 1.0, "decider": 0.25}},
    })
    clock = Clock(T0)
    reply = answers({"trend": 0.5})
    reply["best_net_return"] = {"type": "choice", "choice": "trend", "probabilities": {"trend": 0.5, "none": 0.5},
                                "confidence": 0.5}

    def make():
        return A.Allocator(root, FakeDecider(reply), market, model="ordecide / test",
                           clock=clock, budget=RateBudget(50))

    a = make()
    rec = asyncio.run(a.decide(trigger="decide_now"))
    assert rec["final_cap"]["infeasible"] and "cap infeasible" in rec["error"]
    cap_error = a.status()["last_error"]
    assert "cap infeasible" in cap_error and a.status()["state"] == "error"

    clock.t += 3600
    rec = asyncio.run(a.decide(trigger="schedule"))
    assert rec["weights_mode"] == "refresh"
    assert a.status()["last_error"] == cap_error and a.status()["state"] == "error"

    a = make()
    a.write_status(a.now_ms())
    assert a.status()["last_error"] == cap_error and a.status()["state"] == "error"

    # The owner fixes the book, but only a weights run checks the cap again.
    put(root / "trend" / "control.json", {"enabled": True, "weight": 0.4, "owner_lock": False})
    clock.t += 3600
    rec = asyncio.run(a.decide(trigger="schedule"))
    assert rec["weights_mode"] == "refresh"
    assert a.status()["last_error"] == cap_error and a.status()["state"] == "error"

    rec = asyncio.run(a.decide(trigger="decide_now", dry_run=True))
    assert not rec["final_cap"]["infeasible"]
    a.write_status(a.now_ms())
    assert a.status()["last_error"] == cap_error

    rec = asyncio.run(a.decide(trigger="decide_now"))
    assert rec["weights_mode"] == "computed" and not rec["final_cap"]["infeasible"]
    assert rec["error"] is None
    assert a.status()["last_error"] is None and a.status()["state"] == "running"
