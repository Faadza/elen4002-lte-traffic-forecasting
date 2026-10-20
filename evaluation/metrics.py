"""Shared error measures and final nonnegative prediction handling.

Weekly MASE uses observed lag-168 pairs inside the fitting prefix. A score
below one compares against that *training* scale, not the evaluated weekly
seasonal-naive forecast. Compare the two models' evaluation scores directly.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _vector(values, name, *, finite=True, allow_empty=False):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not allow_empty and values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if finite and not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _mask(values, name, size):
    values = np.asarray(values)
    if values.shape != (size,) or not np.isin(values, [False, True]).all():
        raise ValueError(f"{name} must be a matching one-dimensional 0/1 mask")
    return values.astype(bool)


def _prediction_pair(actual, forecast):
    actual = _vector(actual, "actual")
    forecast = _vector(forecast, "forecast")
    if actual.shape != forecast.shape:
        raise ValueError("actual and forecast must have identical shapes")
    return actual, forecast


def _finite_result(value):
    if not np.isfinite(value):
        raise ValueError("error calculation produced a nonfinite result")
    return float(value)


def mae(actual, forecast):
    """Mean absolute error on identical, nonempty target vectors."""
    actual, forecast = _prediction_pair(actual, forecast)
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.mean(np.abs(forecast - actual))
    return _finite_result(value)


def rmse(actual, forecast):
    """Root mean squared error; calculate separately for every LSTM seed."""
    actual, forecast = _prediction_pair(actual, forecast)
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.sqrt(np.mean((forecast - actual) ** 2))
    return _finite_result(value)


def seasonal_scale(traffic, observed, end, lag=168):
    """Observed lag-difference mean entirely in [0, end); NaN if no pairs.

    Missing or filled hours never enter the denominator. A constant fitting
    series gives a zero scale, making MASE unavailable without an epsilon.
    """
    traffic = _vector(traffic, "traffic", finite=False, allow_empty=True)
    observed = _mask(observed, "observed", len(traffic))
    if isinstance(end, (bool, np.bool_)) or not isinstance(end, (int, np.integer)):
        raise ValueError("end must be an integer prefix boundary")
    if not 0 <= end <= len(traffic):
        raise ValueError("end must lie inside the available calendar")
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError("lag must be a positive integer")
    values, genuine = traffic[:end], observed[:end]
    if not np.isfinite(values[genuine]).all():
        raise ValueError("observed fitting values must be finite")
    if end <= lag:
        return float("nan")
    paired = genuine[lag:] & genuine[:-lag]
    if not paired.any():
        return float("nan")
    return mae(values[:-lag][paired], values[lag:][paired])


def mase(actual, forecast, scale):
    """MAE divided by the fitting scale; NaN for nonpositive/nonfinite scale.

    Undefined MASE must be reported explicitly; do not hide failed forecasts
    or drop cells when aggregating. Original-unit MAE/RMSE remain meaningful.
    """
    error = mae(actual, forecast)  # Validate forecasts even when scale is undefined.
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim != 0:
        raise ValueError("scale must be a scalar")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return _finite_result(error / scale)


def equal_cell_mean(scores_by_cell, expected_cells):
    """Give every cell in a predeclared common scoring cohort equal weight.

    Establish the cohort from data eligibility before comparing methods. A
    missing or nonfinite model score is a failure, not a reason to drop a cell.
    """
    if isinstance(expected_cells, str):
        raise ValueError("expected_cells must be a sequence of unique cell IDs")
    expected = list(expected_cells)
    if not expected or any(not isinstance(cell, str) or not cell for cell in expected):
        raise ValueError("expected_cells must contain nonempty cell IDs")
    if len(set(expected)) != len(expected):
        raise ValueError("expected_cells must not contain duplicates")
    if not isinstance(scores_by_cell, Mapping) or set(scores_by_cell) != set(expected):
        raise ValueError("scores must contain exactly the expected cells")
    scores = []
    for cell in expected:
        score = np.asarray(scores_by_cell[cell], dtype=np.float64)
        if score.ndim != 0 or not np.isfinite(score):
            raise ValueError(f"score for {cell} must be a finite scalar")
        scores.append(float(score))
    return _finite_result(np.mean(scores))


def finalize_predictions(raw):
    """Check original-unit forecasts, then bound them at zero without mutation.

    Keep the caller's raw values when saving forecasts. The mean adjustment
    includes all predictions, including those that needed no change.
    """
    raw = _vector(raw, "raw")
    final = np.maximum(0.0, raw)
    clipped = raw < 0
    diagnostics = {
        "prediction_count": int(raw.size),
        "clipped_count": int(clipped.sum()),
        "clipped_fraction": float(clipped.mean()),
        "mean_upward_adjustment": _finite_result(np.mean(final - raw)),
    }
    return final, diagnostics


def _hourly_calendar(timestamps):
    values = np.asarray(timestamps)
    if values.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if any(isinstance(value, (int, float, complex, np.number)) for value in values):
        raise ValueError("timestamps must be dates, not numeric positions")
    times = pd.DatetimeIndex(pd.to_datetime(values))
    if times.tz is not None or times.hasnans:
        raise ValueError("timestamps must be valid timezone-naive recorded hours")
    if not (times == times.floor("h")).all():
        raise ValueError("timestamps must be aligned to whole hours")
    if len(times) > 1 and not (times[1:] - times[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("timestamps must be consecutive chronological hours")
    return times


def busiest_hour_measures(timestamps, actual, observed, eligible, forecast):
    """Score actual daily maxima, averaging ties within each day, then days.

    Pass full consecutive evaluation-block arrays, including ineligible hours.
    Only complete 24-observed-hour days qualify, and every tied daily maximum
    must be eligible. Forecast failure at any eligible hour raises an error.
    Outside eligibility, forecasts may be NaN; no substitute peak is chosen.
    """
    times = _hourly_calendar(timestamps)
    actual = _vector(actual, "actual", finite=False, allow_empty=True)
    forecast = _vector(forecast, "forecast", finite=False, allow_empty=True)
    size = len(times)
    if actual.shape != (size,) or forecast.shape != (size,):
        raise ValueError("actual and forecast must match the full hourly calendar")
    observed = _mask(observed, "observed", size)
    eligible = _mask(eligible, "eligible", size)
    if np.isinf(actual).any() or not np.isfinite(actual[observed]).all():
        raise ValueError("actual must be finite at observed hours; only missing hours may be NaN")
    if (eligible & ~observed).any():
        raise ValueError("eligible targets must be genuinely observed")
    if np.isinf(forecast).any() or not np.isfinite(forecast[eligible]).all():
        raise ValueError("forecast must be finite at every eligible hour")

    dates = times.normalize()
    result = {
        "status": "unavailable",
        "absolute_error": None,
        "signed_error": None,
        "calendar_days_seen": int(len(dates.unique())),
        "complete_observed_days": 0,
        "qualifying_days": 0,
        "tied_maximum_hours": 0,
        "excluded_incomplete_calendar_days": 0,
        "excluded_missing_observation_days": 0,
        "excluded_ineligible_maximum_days": 0,
    }
    absolute_errors, signed_errors = [], []
    for date in dates.unique():
        positions = np.flatnonzero(dates == date)
        if len(positions) != 24:
            result["excluded_incomplete_calendar_days"] += 1
            continue
        if not observed[positions].all():
            result["excluded_missing_observation_days"] += 1
            continue
        result["complete_observed_days"] += 1
        maximum = positions[actual[positions] == actual[positions].max()]
        if not eligible[maximum].all():
            result["excluded_ineligible_maximum_days"] += 1
            continue
        absolute_errors.append(mae(actual[maximum], forecast[maximum]))
        signed_errors.append(_finite_result(np.mean(forecast[maximum] - actual[maximum])))
        result["tied_maximum_hours"] += int(len(maximum))

    result["qualifying_days"] = len(absolute_errors)
    if absolute_errors:
        result.update(
            status="available",
            absolute_error=_finite_result(np.mean(absolute_errors)),
            signed_error=_finite_result(np.mean(signed_errors)),
        )
    return result
