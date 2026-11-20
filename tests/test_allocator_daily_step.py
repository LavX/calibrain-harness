from ctb.allocator import Current


class _A:
    from ctb.allocator import Allocator as _Alloc
    from ctb.allocator import AllocVenue as _Venue
    _day_anchor = _Alloc._day_anchor
    vk = _Alloc.vk

    def __init__(self):
        self.st = {}
        self.primary = self._Venue("bybit-demo", None, "BYBIT·DEMO")  # the anchor is keyed per (venue, strategy)

    def _state(self):
        return self.st

    def _save_state(self):
        pass


def _cur(w):
    return {"trend": Current(enabled=True, weight=w, owner_lock=False, drawdown_pct=0.0)}


DAY1 = 1_795_000_000_000  # 2026-11-18 UTC
DAY2 = DAY1 + 86_400_000


def test_anchor_holds_the_first_weight_of_the_day():
    a = _A()
    assert a._day_anchor(_cur(0.10), DAY1)["trend"].weight == 0.10
    # a second run the same day after the weight rose to 0.35 still steps from 0.10
    assert a._day_anchor(_cur(0.35), DAY1 + 3_600_000)["trend"].weight == 0.10
    # next UTC day: the anchor moves to the current weight
    assert a._day_anchor(_cur(0.35), DAY2)["trend"].weight == 0.35


def test_dry_run_does_not_persist_the_anchor():
    a = _A()
    a._day_anchor(_cur(0.10), DAY1, persist=False)
    assert "day_anchor" not in a.st
