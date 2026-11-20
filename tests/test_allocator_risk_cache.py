"""Risk estimates stay usable when forecasts omit known strategies."""

import pytest

from ctb import allocator as A
from test_allocator_computed import make


@pytest.mark.parametrize("latest_pair", ["omitted", None])
def test_partial_forecasts_preserve_last_known_covariances(tmp_path, latest_pair):
    a = make(tmp_path)
    a._remember_risk(A.Forecast("fresh", {
        "strategies": {"trend": {"sigma_d": 0.02}, "longvol": {"sigma_d": 0.04}},
        "cov_d": {"ids": ["trend", "longvol"], "matrix": [[0.0004, 0.0007], [0.0007, 0.0016]]},
    }, made_ms=1))
    latest = A.Forecast("fresh", {
        "strategies": {"trend": {"sigma_d": 0.03}},
        "cov_d": {"ids": ["trend"], "matrix": [[0.0009]]} if latest_pair == "omitted" else {
            "ids": ["trend", "longvol"], "matrix": [[0.0009, latest_pair], [latest_pair, None]],
        },
    }, made_ms=2)
    a._remember_risk(latest)
    a._save_state()

    cov, notes = make(tmp_path)._risk_cov(["trend", "longvol"], latest)

    assert cov[0] == pytest.approx([0.0009, 0.0007])
    assert cov[1] == pytest.approx([0.0007, 0.0016])
    assert notes == {"longvol": "last known volatility"}


def test_incompatible_cached_hedge_cannot_cancel_venue_risk(tmp_path):
    a = make(tmp_path)
    a._remember_risk(A.Forecast("fresh", {
        "strategies": {"trend": {"sigma_d": 0.02}, "longvol": {"sigma_d": 0.02}},
        "cov_d": {"ids": ["trend", "longvol"], "matrix": [[0.0004, -0.00036], [-0.00036, 0.0004]]},
    }, made_ms=1))
    latest = A.Forecast("fresh", {
        "strategies": {"trend": {"sigma_d": 0.001}},
        "cov_d": {"ids": ["trend"], "matrix": [[0.000001]]},
    }, made_ms=2)
    a._remember_risk(latest)

    ids = ["trend", "longvol"]
    cov, _ = a._risk_cov(ids, latest)
    _, cap = A.M.final_vol_cap(ids, {"trend": 0.9, "longvol": 0.9}, cov, [], 0.10)

    assert cap["vol_before"] > 0.16  # Even a perfect hedge between these vols cannot get below 16%.
    assert cap["cut"] == ids
