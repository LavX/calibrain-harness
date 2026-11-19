"""ctb/allocator_math.py: pure formulas. Known cases for the optimiser, and each rule on its own in order."""

from __future__ import annotations

import math

import pytest

from ctb import allocator_math as M

G = 2.0


def test_one_asset_unconstrained_is_mu_over_gamma_sigma_squared():
    sigma = 0.01
    w = M.solve_weights([0.0001], [[sigma ** 2]], [0.0], [0.0], gamma=G)
    assert w[0] == pytest.approx(0.0001 / (G * sigma ** 2), abs=1e-9)  # 0.5
    # the same with a different starting point and gamma
    w = M.solve_weights([0.0001], [[sigma ** 2]], [0.9], [0.0], gamma=4.0)
    assert w[0] == pytest.approx(0.25, abs=1e-9)


def test_box_constraints():
    cov = [[1e-4]]
    assert M.solve_weights([0.01], cov, [0.2], [0.0], gamma=G) == [1.0]   # would be 50: clipped to 1
    assert M.solve_weights([-0.01], cov, [0.2], [0.0], gamma=G) == [0.0]  # negative forecast: flat, never short


def test_two_assets_match_the_closed_form():
    cov = [[1e-4, 0.5e-4], [0.5e-4, 4e-4]]
    mu = [0.00008, 0.0002]
    det = cov[0][0] * cov[1][1] - cov[0][1] ** 2
    inv = [[cov[1][1] / det, -cov[0][1] / det], [-cov[1][0] / det, cov[0][0] / det]]
    want = [sum(inv[i][j] * mu[j] for j in range(2)) / G for i in range(2)]
    assert 0 < min(want) and max(want) < 1  # interior, so the box is not what decides
    got = M.solve_weights(mu, cov, [0.0, 0.0], [0.0, 0.0], gamma=G)
    assert got == pytest.approx(want, abs=1e-8)
    assert M.solve_weights(mu, cov, [0.0, 0.0], [0.0, 0.0], gamma=G) == got  # deterministic


def test_trading_cost_gives_a_no_trade_band_around_the_previous_weight():
    sigma2, mu, prev = 1e-4, 0.0001, 0.3
    drift = mu - G * sigma2 * prev   # marginal gain of moving up from prev
    assert drift == pytest.approx(0.00004)
    stay = M.solve_weights([mu], [[sigma2]], [prev], [0.0001], gamma=G)          # cost above the gain: no trade
    assert stay[0] == pytest.approx(prev, abs=1e-9)
    move = M.solve_weights([mu], [[sigma2]], [prev], [0.00001], gamma=G)         # cost below it: trade the difference
    assert move[0] == pytest.approx(prev + (drift - 0.00001) / (G * sigma2), abs=1e-9)
    down = M.solve_weights([-0.0001], [[sigma2]], [prev], [0.00001], gamma=G)    # and symmetrically down, to the box
    assert down[0] == 0.0


def test_forecast_helpers():
    sigma = 0.02
    cap = 1.5 * sigma / math.sqrt(365)
    assert M.clip_mu(1.0, sigma) == (pytest.approx(cap), True) and M.clip_mu(-1.0, sigma) == (pytest.approx(-cap), True)
    assert M.clip_mu(cap / 2, sigma) == (cap / 2, False)
    assert M.decider_mu(0.5, sigma) == 0 and M.decider_mu(0.75, sigma) == pytest.approx(0.5 * 0.8 * sigma) and M.decider_mu(0.25, sigma) < 0
    assert M.mean_p([0.4, 0.6, 0.8]) == pytest.approx(0.6) and M.mean_p([]) is None
    # weights renormalise over the forecasters that have a value; zero or missing weights are ignored
    assert M.combine({"a": 1.0, "b": 3.0, "c": None}, {"a": 0.3, "b": 0.1, "c": 0.6}) == pytest.approx((0.3 + 0.3) / 0.4)
    assert M.combine({"a": 1.0}, {"b": 1.0}) is None


def test_portfolio_vol_divides_by_the_number_of_books():
    cov = [[4e-4, 0.0], [0.0, 4e-4]]
    assert M.portfolio_vol_ann([1, 1], cov) == pytest.approx(math.sqrt(365 * 8e-4) / 2)


IDS = ["a", "b"]
COV = [[4e-4, 0.0], [0.0, 4e-4]]      # 38%/yr each, uncorrelated: the vol cap bites
CALM = [[2.5e-5, 0.0], [0.0, 2.5e-5]]  # 9.6%/yr each: it does not
NO_FLOOR = M.RuleParams(explore_floor=0.0)


def rules(star, prev, p=None, params=NO_FLOOR, cov=CALM, **kw):
    return M.apply_rules(IDS, dict(zip(IDS, star)), dict(zip(IDS, prev)), cov, dict(zip(IDS, p or (None, None))), params, **kw)


def after(steps, name):
    return next(s for s in steps if s["step"] == name)["after"]


def test_rules_run_in_the_documented_order():
    final, steps = rules([1.0, 1.0], [0.0, 0.0], cov=COV)
    assert [s["step"] for s in steps] == ["vol_cap", "buffer", "step", "veto", "explore_floor", "floor_lock"]
    # vol cap: 2 books at weight 1 are 27%/yr, scaled to 10%
    factor = 0.10 / (math.sqrt(365 * 8e-4) / 2)
    assert after(steps, "vol_cap")["a"] == pytest.approx(factor, abs=1e-5) and steps[0]["factor"] == pytest.approx(factor, abs=1e-5)
    assert M.portfolio_vol_ann([after(steps, "vol_cap")[i] for i in IDS], COV) == pytest.approx(0.10, abs=1e-4)
    assert after(steps, "buffer")["a"] == pytest.approx(factor, abs=1e-5)   # 0.37 from 0 is a real change
    assert after(steps, "step")["a"] == 0.25 and final == {"a": 0.25, "b": 0.25}  # then the 0.25 a day step


def test_vol_cap_does_not_act_below_the_cap():
    _, steps = rules([0.3, 0.1], [0.3, 0.1], cov=COV)
    assert steps[0]["factor"] == 1.0 and steps[0]["changed"] == []


def test_buffer_ignores_small_changes_and_lets_big_ones_through():
    final, steps = rules([0.55, 0.65], [0.50, 0.50])
    assert final == {"a": 0.50, "b": 0.65} and steps[1]["changed"] == ["a"]


def test_step_limits_increases_only():
    final, _ = rules([0.9, 0.1], [0.2, 0.9])
    assert final == {"a": pytest.approx(0.45), "b": 0.1}  # up 0.25 at most, down at once


def test_decider_veto_caps_at_half_of_the_previous_weight():
    final, steps = rules([0.8, 0.8], [0.8, 0.8], p=(0.30, 0.45))
    assert final == {"a": 0.4, "b": 0.8} and steps[3]["changed"] == ["a"]
    assert rules([0.8, 0.8], [0.8, 0.8], p=(None, None))[0] == {"a": 0.8, "b": 0.8}  # no answers: no veto


def test_exploration_floor_comes_after_buffer_step_and_veto():
    p = M.RuleParams()  # floor 0.10
    final, steps = rules([0.0, 0.0], [0.05, 0.9], params=p)
    # a: the 0.05 change is inside the buffer, so it stays 0.05; the floor lifts it to 0.10. b goes to 0 first, then the floor.
    assert after(steps, "buffer")["a"] == 0.05 and final == {"a": 0.10, "b": 0.10}
    final, _ = rules([0.2, 0.2], [0.12, 0.12], p=(0.2, None), params=p)  # veto halves 0.12 to 0.06, the floor wins back 0.10
    assert final["a"] == 0.10
    assert rules([0.0, 0.0], [0.0, 0.0], params=M.RuleParams(explore_floor=0.0))[0] == {"a": 0.0, "b": 0.0}


def test_drawdown_floor_and_owner_lock_override_everything():
    p = M.RuleParams()
    final, steps = rules([1.0, 1.0], [0.4, 0.4], params=p, floored=["a"])
    assert final["a"] == 0.0 and steps[-1]["changed"] == ["a"]  # beats the exploration floor
    final, _ = rules([0.0, 0.0], [0.4, 0.4], params=p, locked=["b"])
    assert final["b"] == 0.4  # the lock keeps the owner's weight, whatever the optimum says
    final, _ = rules([0.0, 0.0], [0.4, 0.4], params=p, floored=["a", "b"], locked=["b"])
    assert final == {"a": 0.0, "b": 0.4}  # lock beats the floor, as in resolve()


def test_reason_line():
    line = M.reason_line("trend", 0.061 / 365, 0.18 / math.sqrt(365), 0.41, 0.35, ["step"], "regime up_calm 0.62")
    assert line == "trend: mu +6.1%/yr (regime up_calm 0.62), sigma 18%, w* 0.41 -> 0.35 after step cap"
    assert M.reason_line("carry", -0.01 / 365, 0.1 / math.sqrt(365), 0.0, 0.0, []) == "carry: mu -1.0%/yr, sigma 10%, w* 0.00 -> 0.00"
    assert M.reason_line("x", 0.0, 0.01, 0.5, 0.1, ["vol_cap", "veto"]).endswith("after vol cap, decider veto")
