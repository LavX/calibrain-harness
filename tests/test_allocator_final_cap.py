"""The vol cap on final holdings: after the buffer, step, veto, exploration floor, drawdown floor and owner locks, each
venue's portfolio is checked again with every book in it (locked, held, without a forecast) and only unlocked books are
cut. The committee examples: the cap sets 0.50, the buffer restores 0.55 or 0.59, the book ends at 11% or 11.8%."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from ctb import allocator as A
from ctb import allocator_math as M
from ctb.kairos.client import RateBudget
from test_allocator_computed import Clock, FakeDecider, answers, market

DAY = 86_400
DAY0 = 1_800_000_000 - 1_800_000_000 % DAY
T0 = DAY0 + 8 * 3600
S20 = 0.20 / math.sqrt(365)  # a sleeve with 20%/yr volatility, daily
S40 = 0.40 / math.sqrt(365)


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def ctl(root, sid):
    return json.loads((root / sid / "control.json").read_text())


def forecast(made_s, sigmas):
    ids = list(sigmas)
    return {"date": "2026-11-20", "made_ms": int(made_s * 1000), "kappa": 0.0,
            "strategies": {i: {"sigma_d": s, "mu": {"base": 0.0}, "n_days": 900} for i, s in sigmas.items()},
            "cov_d": {"ids": ids, "matrix": [[sigmas[a] ** 2 if a == b else 0.0 for b in ids] for a in ids]},
            "weights_k": {i: {"base": 1.0, "decider": 0.25} for i in ids}}


def setup(tmp_path, books, sigmas, t=T0):
    """books: {id: (weight, owner_lock)}."""
    root = tmp_path / "live"
    put(root / "strategies.json", [{"id": i, "label": i.title()} for i in books])
    for i, (w, lock) in books.items():
        put(root / i / "status.json", {"state": "running", "equity_usdt": 1000.0, "start_equity_usdt": 1000.0, "pnl_pct": 0.0})
        put(root / i / "control.json", {"enabled": True, "weight": w, "owner_lock": lock})
    put(root / "forecast" / "latest.json", forecast(t - 3600, sigmas))
    return root


def make(root, clock):
    return A.Allocator(root, FakeDecider(answers({i: 0.5 for i in ("trend", "longvol")})), market, model="ordecide / test",
                       clock=clock, budget=RateBudget(50))


def decide(a, **kw):
    return asyncio.run(a.decide(**kw))


# ---------------------------------------------------------------- the math
def test_final_vol_cap_cuts_only_unlocked_books_and_never_raises():
    out, info = M.final_vol_cap(["t"], {"t": 0.55}, [[S20 ** 2]], [], 0.10, 1)
    assert out["t"] == pytest.approx(0.50) and info["vol_before"] == pytest.approx(0.11) and info["cut"] == ["t"]
    out, info = M.final_vol_cap(["t"], {"t": 0.40}, [[S20 ** 2]], [], 0.10, 1)
    assert out == {"t": 0.40} and info["cut"] == [] and not info["infeasible"]  # under the cap: nothing moves, nothing rises
    cov = [[S20 ** 2, 0.0], [0.0, S40 ** 2]]
    out, info = M.final_vol_cap(["t", "l"], {"t": 0.5, "l": 0.45}, cov, ["l"], 0.10, 2)
    assert out["l"] == 0.45 and out["t"] == pytest.approx(0.4359, abs=1e-4) and info["vol_after"] == pytest.approx(0.10)
    out, info = M.final_vol_cap(["t", "l"], {"t": 0.5, "l": 0.6}, cov, ["l"], 0.10, 2)  # the locked book alone is at 12%
    assert out == {"t": 0.0, "l": 0.6} and info["infeasible"] and info["vol_after"] == pytest.approx(0.12)


# ---------------------------------------------------------------- the committee examples, end to end
@pytest.mark.parametrize("prev, vol_before", [(0.55, 0.11), (0.59, 0.118)])
def test_the_buffer_can_no_longer_restore_a_weight_over_the_cap(tmp_path, prev, vol_before):
    """One 20%-vol sleeve: the core and the first cap give 0.50; the buffer keeps the previous weight (the change is under
    0.10), which used to be written: 11% or 11.8% volatility. The final cap brings it back to 0.50."""
    root = setup(tmp_path, {"trend": (prev, False)}, {"trend": S20})
    rec = decide(make(root, Clock(T0)), trigger="decide_now")
    assert rec["w_target"]["trend"] == pytest.approx(0.5, abs=1e-6)
    assert next(s for s in rec["steps"] if s["step"] == "buffer")["after"]["trend"] == prev  # the buffer restored it
    assert ctl(root, "trend")["weight"] == 0.5 and ctl(root, "trend")["enabled"] is True
    cap = rec["final_cap"]
    assert cap["vol_before"] == pytest.approx(vol_before) and cap["vol_after"] == pytest.approx(0.10) and cap["cut"] == ["trend"]
    assert "after the final vol cap" in ctl(root, "trend")["reason"]


def test_a_sleeve_without_a_forecast_counts_at_its_last_known_volatility(tmp_path):
    clock = Clock(T0)
    root = setup(tmp_path, {"trend": (0.5, False), "longvol": (0.5, False)}, {"trend": S20, "longvol": S40})
    a = make(root, clock)
    decide(a, trigger="decide_now")  # a forecast with both: their volatilities are remembered
    for i in ("trend", "longvol"):
        put(root / i / "control.json", {"enabled": True, "weight": 0.5, "owner_lock": False})
    clock.t = T0 + DAY
    put(root / "forecast" / "latest.json", forecast(clock.t - 3600, {"trend": S20}))  # longvol has no forecast today
    rec = decide(a, trigger="decide_now")
    assert "longvol" in rec["no_forecast"]  # held at its weight by compute_weights
    # 0.5 and 0.5 at 20% and 40%, two books: 11.2% > 10%; both unlocked, both cut by the same factor
    assert ctl(root, "longvol")["weight"] == pytest.approx(0.4472, abs=1e-4) and ctl(root, "trend")["weight"] == pytest.approx(0.4472, abs=1e-4)
    assert rec["final_cap"]["notes"] == {"longvol": "last known volatility"}


def test_locked_books_are_never_moved_and_an_infeasible_cap_is_said(tmp_path):
    root = setup(tmp_path, {"trend": (0.5, False), "longvol": (0.6, True)}, {"trend": S20, "longvol": S40})
    a = make(root, Clock(T0))
    rec = decide(a, trigger="decide_now")
    assert ctl(root, "longvol")["weight"] == 0.6 and ctl(root, "longvol")["owner_lock"] is True  # 12% on its own, locked
    assert ctl(root, "trend")["weight"] == 0.0 and ctl(root, "trend")["enabled"] is False  # cut as far as it goes
    assert rec["final_cap"]["infeasible"] and "cap infeasible" in rec["error"]
    st = a.status()
    assert "cap infeasible" in st["last_error"] and st["state"] == "error"


def test_a_held_run_still_cuts_to_the_cap_from_the_last_known_volatility(tmp_path):
    """The forecast goes missing (weights held); a book above the cap is still cut, from the remembered volatility."""
    clock = Clock(T0)
    root = setup(tmp_path, {"trend": (0.5, False)}, {"trend": S20})
    a = make(root, clock)
    decide(a, trigger="decide_now")
    put(root / "trend" / "control.json", {"enabled": True, "weight": 0.7, "owner_lock": False})
    (root / "forecast" / "latest.json").unlink()
    clock.t = T0 + 3600
    rec = decide(a, trigger="decide_now")
    assert rec["weights_mode"] == "held" and ctl(root, "trend")["weight"] == 0.5
    assert rec["final_cap"]["notes"] == {"trend": "last known volatility"} and "forecast missing: holding" in rec["error"]
