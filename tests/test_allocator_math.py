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
    line = M.reason_line("trend", 0.42, 0.18, 0.36, 0.50, 0.45, ["step"], "regime up_calm 0.62")
    assert line == "trend: core 0.42, tilt +18% (SR +0.36), w 0.50 -> 0.45 after step cap (regime up_calm 0.62)"
    assert M.reason_line("trend", 0.42, 0.18, 0.36, 0.50, 0.50, []) == "trend: core 0.42, tilt +18% (SR +0.36), w 0.50"
    assert M.reason_line("carry", 0.3, -0.5, -1.2, 0.15, 0.10, ["explore_floor"], kappa=0.5) == \
        "carry: core 0.30, tilt -50% (SR -1.20), w 0.15 -> 0.10 after exploration floor [tilt room 0.50]"
    assert M.reason_line("x", 0.5, 0.0, 0.0, 0.5, 0.1, ["vol_cap", "veto"]).endswith("after vol cap, decider veto")


# ---------------------------------------------------------------------------------- core plus tilt
def cov_of(sig, corr=0.0):
    n = len(sig)
    return [[sig[i] * sig[j] * (1.0 if i == j else corr) for j in range(n)] for i in range(n)]


def test_forecast_sharpe_is_the_annual_return_over_the_annual_volatility():
    sigma_d = 0.01
    assert M.forecast_sharpe(0.10 / 365, sigma_d) == pytest.approx(0.10 / (sigma_d * math.sqrt(365)))
    assert M.forecast_sharpe(0.001, 0.0) == 0.0 and M.forecast_sharpe(-0.001, sigma_d) < 0


def test_core_is_the_inverse_vol_mix_at_the_cap():
    sig = [0.01, 0.02, 0.04]
    cov = cov_of(sig)
    w = M.core_weights(sig, cov, 0.10)
    assert M.portfolio_vol_ann(w, cov) == pytest.approx(0.10, rel=1e-6)
    assert w[1] / w[2] == pytest.approx(2.0) and w[0] / w[1] == pytest.approx(2.0) and max(w) <= 1.0
    # a quiet book is capped at 1 and the cap is not reached: all at 1
    assert M.core_weights([1e-4] * 3, cov_of([1e-4] * 3), 0.10) == [1.0, 1.0, 1.0]
    # one very quiet book is held at 1 while the others are scaled to the cap
    q = M.core_weights([0.001, 0.02, 0.04], cov_of([0.001, 0.02, 0.04]), 0.10)
    assert q[0] == 1.0 and q[1] > q[2]
    assert M.core_weights([], [], 0.10) == []


def test_tau_zero_gives_the_core_exactly():
    core = [0.42, 0.3, 1.0, 0.0]
    assert M.tilted_target(core, [0.0] * 4) == core
    p = M.TiltParams()
    assert M.tilt_tau(0.0, p) == 0.0 and M.tilted_target(core, [M.tilt_tau(0.0, p)] * 4) == core


def test_tilt_is_bounded_by_t_and_the_box():
    p = M.TiltParams()                                     # T 0.5, SR_full 1
    assert M.tilt_tau(0.36, p) == pytest.approx(0.36)
    assert M.tilt_tau(3.0, p) == 0.5 and M.tilt_tau(-3.0, p) == -0.5 and M.tilt_tau(-0.2, p) == pytest.approx(-0.2)
    assert M.tilt_tau(0.8, M.TiltParams(sr_full=2.0, t_max=1.0)) == pytest.approx(0.4)   # SR_full 2: SR 0.8 is 40% of full
    assert M.tilt_tau(9.0, M.TiltParams(t_max=0.25)) == 0.25 and M.tilt_tau(9.0, M.TiltParams(t_max=1.0)) == 1.0
    w = M.tilted_target([0.4, 0.9, 0.5, 0.0], [0.5, 0.5, -0.5, 0.5])
    assert w == pytest.approx([0.6, 1.0, 0.25, 0.0])        # 0.9 x 1.5 is clipped to the box
    assert all(0.0 <= x <= 1.0 for x in M.tilted_target([1.0, 0.0], [1.0, -1.0]))
    # a tilt never flips a weight's sign or exceeds core x (1 + T)
    for core in (0.1, 0.5, 0.8):
        for sr in (-5.0, -0.3, 0.0, 0.3, 5.0):
            t = M.tilted_target([core], [M.tilt_tau(sr, p)])[0]
            assert core * 0.5 - 1e-12 <= t <= min(1.0, core * 1.5) + 1e-12


def test_kappa_scales_the_tilt_room():
    assert M.TiltParams(kappa=1.0).t_eff == 0.5 and M.TiltParams(kappa=0.5).t_eff == 0.25 and M.TiltParams(kappa=0.0).t_eff == 0.0
    assert M.TiltParams(kappa=3.0).t_eff == 0.5 and M.TiltParams(kappa=-1.0).t_eff == 0.0   # clamped to [0, 1]
    assert M.tilt_tau(5.0, M.TiltParams(kappa=0.5)) == 0.25 and M.tilt_tau(5.0, M.TiltParams(kappa=0.0)) == 0.0


def test_kappa_steps_down_with_a_negative_tilt_skill():
    assert M.kappa_for_skill(None, None) == 1.0            # not enough days: full room
    assert M.kappa_for_skill(0.0001, 0.0001) == 1.0
    assert M.kappa_for_skill(0.0001, -0.0001) == 1.0       # the last 90 days recovered
    assert M.kappa_for_skill(-0.0001, None) == 0.5         # 90 days negative
    assert M.kappa_for_skill(-0.0001, 0.0002) == 0.5       # but 180 days are not
    assert M.kappa_for_skill(-0.0001, -0.0001) == 0.25     # both negative
    assert M.kappa_for_skill(0.0, 0.0) == 1.0              # a tie is not negative


def test_rule_order_is_unchanged_after_the_tilt():
    cov = cov_of([0.02, 0.02])
    ids = ["a", "b"]
    p = M.RuleParams(explore_floor=0.10)
    w_t = dict(zip(ids, M.tilted_target(M.core_weights([0.02, 0.02], cov, 0.10, 2), [0.5, -0.5])))
    final, steps = M.apply_rules(ids, w_t, {"a": 0.2, "b": 0.5}, cov, {"a": None, "b": 0.3}, p, floored=[], locked=[])
    assert [st["step"] for st in steps] == ["vol_cap", "buffer", "step", "veto", "explore_floor", "floor_lock"]
    assert steps[0]["before"] == {i: round(w_t[i], 6) for i in ids}   # the first rule sees the tilted target
    assert final["b"] <= 0.5 * 0.5 + 1e-12                          # the decider veto (p 0.3 < 0.4) caps b at half of yesterday's
