"""An unknown book uses risk estimates from all known strategies."""

import pytest

from ctb import allocator as A
from test_allocator_computed import make


@pytest.mark.parametrize("source", ["forecast", "cache"])
@pytest.mark.parametrize("ids", [["unknown"], ["low", "unknown"]])
def test_unknown_volatility_uses_the_largest_known_strategy_outside_the_venue(tmp_path, source, ids):
    a = make(tmp_path)
    fc = A.Forecast("fresh", {
        "strategies": {"low": {"sigma_d": 0.02}, "high": {"sigma_d": 0.04}},
    }, made_ms=1)
    if source == "cache":
        a._remember_risk(fc)
        fc = A.Forecast("missing")

    cov, notes = a._risk_cov(ids, fc)

    assert cov is not None
    assert cov[-1][-1] == pytest.approx(0.0016)
    assert notes["unknown"] == "volatility unknown, the largest known one assumed"
    assert set(notes) <= set(ids)
    if len(ids) == 2:
        assert cov[0] == pytest.approx([0.0004, 0.0])
