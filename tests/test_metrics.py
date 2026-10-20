"""Small examples with errors and exclusions calculated by hand."""

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    busiest_hour_measures,
    equal_cell_mean,
    finalize_predictions,
    mae,
    mase,
    rmse,
    seasonal_scale,
)


def test_mae_rmse_and_mase_by_hand():
    actual, forecast = [10.0, 20.0, 30.0], [12.0, 19.0, 27.0]
    assert mae(actual, forecast) == 2.0  # Absolute errors: 2, 1, 3.
    assert rmse(actual, forecast) == pytest.approx(np.sqrt(14 / 3))
    assert mase(actual, forecast, 4.0) == 0.5


@pytest.mark.parametrize("actual,forecast", [
    ([1, 2], [[1], [2]]),
    ([[1], [2]], [1, 2]),
    ([1, 2], [1]),
    ([], []),
    ([1, np.nan], [1, 2]),
    ([1, 2], [1, np.nan]),
    ([1, np.inf], [1, 2]),
    ([1, 2], [1, -np.inf]),
])
def test_scores_reject_shape_or_nonfinite_targets(actual, forecast):
    for metric in (mae, rmse):
        with pytest.raises(ValueError):
            metric(actual, forecast)
    with pytest.raises(ValueError):
        mase(actual, forecast, 0)  # Invalid scale cannot conceal bad forecasts.


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, -np.inf])
def test_undefined_mase_scale_is_not_replaced_with_epsilon(scale):
    assert np.isnan(mase([10], [12], scale))
    assert mae([10], [12]) == 2
    assert rmse([10], [12]) == 2


def test_mase_rejects_vector_scale():
    with pytest.raises(ValueError, match="scalar"):
        mase([1, 2], [2, 3], [1, 2])


def test_mase_under_one_does_not_imply_beating_evaluation_baseline():
    # Training scale 10 gives MASE .5, yet the evaluated baseline is perfect.
    assert mase([10], [15], 10) == 0.5
    assert mase([10], [10], 10) == 0


def test_seasonal_scale_observed_pairs_only():
    traffic = [10, 10, 13, 1000, 14, 1000]
    observed = [1, 1, 1, 0, 1, 0]
    # Only observed pairs (2, 0) and (4, 2): differences 3 and 1.
    assert seasonal_scale(traffic, observed, end=6, lag=2) == 2
    traffic[3], traffic[5] = np.nan, -500
    assert seasonal_scale(traffic, observed, end=6, lag=2) == 2


def test_seasonal_scale_uses_only_fitting_prefix():
    traffic = np.array([1, 2, 4, 6, 100, 200], dtype=float)
    observed = np.ones(6, dtype=bool)
    assert seasonal_scale(traffic, observed, end=4, lag=2) == 3.5
    traffic[4:] = [np.inf, np.nan]
    assert seasonal_scale(traffic, observed, end=4, lag=2) == 3.5


def test_seasonal_scale_default_is_weekly():
    assert seasonal_scale(np.arange(170), np.ones(170), end=170) == 168


def test_seasonal_scale_without_pairs_and_constant_series():
    assert np.isnan(seasonal_scale([1, 2], [1, 1], end=2, lag=2))
    assert np.isnan(seasonal_scale([1, 2, 3], [1, 0, 0], end=3, lag=1))
    assert seasonal_scale([2, 2, 2], [1, 1, 1], end=3, lag=1) == 0


@pytest.mark.parametrize("end,lag", [(-1, 1), (4, 1), (2.0, 1), (True, 1), (3, 0), (3, 1.5)])
def test_seasonal_scale_rejects_invalid_boundaries(end, lag):
    with pytest.raises(ValueError):
        seasonal_scale([1, 2, 3], [1, 1, 1], end, lag)


def test_seasonal_scale_rejects_bad_masks_and_observed_nan():
    for observed in ([1, 1], [1, 2, 1], [[1], [1], [1]]):
        with pytest.raises(ValueError):
            seasonal_scale([1, 2, 3], observed, 3, lag=1)
    with pytest.raises(ValueError, match="observed fitting"):
        seasonal_scale([1, np.nan, 3], [1, 1, 1], 3, lag=1)


def test_equal_cell_mean_gives_each_cell_one_vote():
    # Cell A has one error of 2; cell B has three errors of 4.
    scores = {"A": mae([0], [2]), "B": mae([0, 0, 0], [4, 4, 4])}
    assert equal_cell_mean(scores, ["A", "B"]) == 3  # Pooled MAE would be 3.5.


@pytest.mark.parametrize("scores,expected", [
    ({"A": 1}, ["A", "B"]),
    ({"A": 1, "B": 2}, ["A"]),
    ({"A": np.nan}, ["A"]),
    ({"A": np.inf}, ["A"]),
    ({"A": [1]}, ["A"]),
    ({"A": 1}, ["A", "A"]),
    ({}, []),
])
def test_equal_cell_mean_rejects_incomplete_or_invalid_cohort(scores, expected):
    with pytest.raises(ValueError):
        equal_cell_mean(scores, expected)


def _block(start="2026-01-01", hours=48):
    times = pd.date_range(start, periods=hours, freq="h")
    actual = np.ones(hours)
    observed = np.ones(hours, dtype=bool)
    eligible = np.ones(hours, dtype=bool)
    forecast = np.ones(hours)
    return times, actual, observed, eligible, forecast


def test_busiest_hour_averages_ties_then_days_and_preserves_error_sign():
    times, actual, observed, eligible, forecast = _block()
    actual[[3, 4, 25]] = 10
    forecast[[3, 4, 25]] = [8, 14, 1]
    result = busiest_hour_measures(times, actual, observed, eligible, forecast)
    # Day 1: |errors| mean 3, signed mean 1; day 2: |error| 9, signed -9.
    assert result["absolute_error"] == 6
    assert result["signed_error"] == -4
    assert result["qualifying_days"] == 2
    assert result["tied_maximum_hours"] == 3
    assert result["status"] == "available"


def test_busiest_hour_requires_every_tied_maximum_to_be_eligible():
    times, actual, observed, eligible, forecast = _block(hours=24)
    actual[[3, 4]] = 10
    eligible[4] = False
    forecast[4] = np.nan
    result = busiest_hour_measures(times, actual, observed, eligible, forecast)
    assert result["complete_observed_days"] == 1
    assert result["excluded_ineligible_maximum_days"] == 1
    assert result["qualifying_days"] == 0
    assert result["absolute_error"] is None
    assert result["signed_error"] is None
    assert result["status"] == "unavailable"


def test_busiest_hour_rejects_partial_edge_days_not_arbitrary_24_hour_chunks():
    times, actual, observed, eligible, forecast = _block("2026-01-01 12:00", hours=48)
    actual[18], forecast[18] = 10, 7
    result = busiest_hour_measures(times, actual, observed, eligible, forecast)
    assert result["calendar_days_seen"] == 3
    assert result["excluded_incomplete_calendar_days"] == 2
    assert result["complete_observed_days"] == 1
    assert result["qualifying_days"] == 1
    assert result["absolute_error"] == 3
    assert result["signed_error"] == -3


def test_busiest_hour_missing_day_is_excluded_even_if_missing_hour_was_not_peak():
    times, actual, observed, eligible, forecast = _block(hours=24)
    actual[5], forecast[5] = 10, 6
    actual[1], forecast[1] = np.nan, np.nan
    observed[1], eligible[1] = False, False
    result = busiest_hour_measures(times, actual, observed, eligible, forecast)
    assert result["excluded_missing_observation_days"] == 1
    assert result["complete_observed_days"] == 0
    assert result["status"] == "unavailable"


def test_busiest_hour_allows_nonpeak_ineligibility_but_not_failed_eligible_forecast():
    times, actual, observed, eligible, forecast = _block(hours=24)
    actual[5], forecast[5] = 10, 6
    eligible[1], forecast[1] = False, np.nan
    result = busiest_hour_measures(times, actual, observed, eligible, forecast)
    assert result["absolute_error"] == 4
    forecast[2] = np.nan  # A failed forecast cannot be silently discarded.
    with pytest.raises(ValueError, match="every eligible"):
        busiest_hour_measures(times, actual, observed, eligible, forecast)


def test_busiest_hour_rejects_observed_nan_and_ineligible_observation_mask():
    times, actual, observed, eligible, forecast = _block(hours=24)
    actual[0] = np.nan
    with pytest.raises(ValueError, match="actual must"):
        busiest_hour_measures(times, actual, observed, eligible, forecast)
    observed[0] = False
    with pytest.raises(ValueError, match="genuinely observed"):
        busiest_hour_measures(times, actual, observed, eligible, forecast)


def test_busiest_hour_rejects_shape_mismatch():
    times, actual, observed, eligible, forecast = _block(hours=24)
    with pytest.raises(ValueError):
        busiest_hour_measures(times, actual, observed, eligible, forecast[:, None])
    with pytest.raises(ValueError):
        busiest_hour_measures(times, actual[:-1], observed, eligible, forecast)


@pytest.mark.parametrize("times", [
    pd.to_datetime(["2026-01-01 00:00", "2026-01-01 02:00"]),
    pd.to_datetime(["2026-01-01 01:00", "2026-01-01 00:00"]),
    pd.to_datetime(["2026-01-01 00:30", "2026-01-01 01:30"]),
    pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00"]),
    pd.to_datetime(["2026-01-01", None]),
    pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
    [0, 3600000000000],
])
def test_busiest_hour_rejects_invalid_calendars(times):
    with pytest.raises(ValueError):
        busiest_hour_measures(times, [1, 2], [1, 1], [1, 1], [1, 2])


def test_busiest_hour_empty_block_is_unavailable():
    result = busiest_hour_measures([], [], [], [], [])
    assert result["status"] == "unavailable"
    assert result["calendar_days_seen"] == 0
    assert result["qualifying_days"] == 0


def test_final_prediction_clipping_keeps_raw_and_uses_all_predictions_for_mean():
    raw = np.array([-4.0, 2.0, -2.0, 0.0])
    original = raw.copy()
    final, diagnostics = finalize_predictions(raw)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(final, [0, 2, 0, 0])
    assert not np.shares_memory(raw, final)
    assert diagnostics == {
        "prediction_count": 4,
        "clipped_count": 2,
        "clipped_fraction": 0.5,
        "mean_upward_adjustment": 1.5,
    }


@pytest.mark.parametrize("raw", [[-np.inf, 1], [np.inf], [np.nan], [[1], [2]], []])
def test_clipping_rejects_failed_forecasts_before_zero_bound(raw):
    with pytest.raises(ValueError):
        finalize_predictions(raw)
