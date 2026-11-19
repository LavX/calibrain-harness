from ctb.allocator import Current, Plan, apply_explore_floor



def _plan(weights, floored=()):
    return Plan(**{f: [] for f in Plan.__dataclass_fields__} | {"weights": dict(weights), "enabled": {k: v > 0 for k, v in weights.items()}, "floored": list(floored)})


class C:
    def __init__(self, lock=False):
        self.owner_lock = lock


def test_floor_lifts_off_strategies_on_demo():
    plan = _plan({"carry": 0.5, "trend": 0.0, "xsmom": 0.0})
    lifted = apply_explore_floor(plan, {"carry": C(), "trend": C(), "xsmom": C()}, 0.10)
    assert lifted == ["trend", "xsmom"]
    assert plan.weights == {"carry": 0.5, "trend": 0.10, "xsmom": 0.10}
    assert plan.enabled == {"carry": True, "trend": True, "xsmom": True}


def test_floor_skips_drawdown_floored_and_owner_locked():
    plan = _plan({"carry": 0.0, "trend": 0.0}, floored=["carry"])
    lifted = apply_explore_floor(plan, {"carry": C(), "trend": C(lock=True)}, 0.10)
    assert lifted == []
    assert plan.weights == {"carry": 0.0, "trend": 0.0}


def test_zero_floor_does_nothing():
    plan = _plan({"trend": 0.0})
    assert apply_explore_floor(plan, {"trend": C()}, 0.0) == []
    assert plan.weights["trend"] == 0.0
