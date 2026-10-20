# The three reference forecasters.
#
# Each takes the prepared traffic series (after the permitted filling) and
# an array of target indices, and returns the forecast for each target.
# For a target at hour s, only values up to s-1 are used:
#   persistence     y[s-1]
#   daily naive     y[s-24]
#   weekly naive    y[s-168]
# Target eligibility (see preprocessing/windows.py) guarantees the whole
# 168 hour window before s has values, so these lookups never hit a hole.

import numpy as np


def lag_forecast(traffic, targets, lag):
    """Look up one past value per target, rejecting unsafe indices or inputs."""
    values = np.asarray(traffic, dtype=np.float64)
    positions = np.asarray(targets)
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError("Lag must be a positive integer")
    if values.ndim != 1 or positions.ndim != 1:
        raise ValueError("Traffic and target indices must be one-dimensional")
    if positions.size == 0:
        return np.empty(0, dtype=np.float64)
    if positions.dtype.kind not in "iu" or positions.dtype.kind == "b":
        raise ValueError("Target indices must be integers")
    if (positions < lag).any() or (positions > len(values)).any():
        raise ValueError("A target lacks the requested historical lag")
    if (np.diff(positions.astype(np.int64)) <= 0).any():
        raise ValueError("Targets must be strictly increasing and unique")
    # A target at len(values) is a valid next-hour forecast without a label yet.
    forecast = values[positions - lag].copy()
    if not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError("Requested baseline inputs must be finite and nonnegative")
    return forecast


def persistence(traffic, targets):
    return lag_forecast(traffic, targets, 1)


def daily_naive(traffic, targets):
    return lag_forecast(traffic, targets, 24)


def weekly_naive(traffic, targets):
    return lag_forecast(traffic, targets, 168)
