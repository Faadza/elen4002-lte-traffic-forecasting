"""Build causal one-hour examples and fit one traffic scaler per cell/prefix.

Inputs must already be on the complete hourly calendar and causally filled by
the verified preparation code. No filling, model fitting or clipping occurs here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from preprocessing.splits import LOOKBACK
from preprocessing.windows import checked_arrays, eligible_targets


@dataclass(frozen=True)
class Examples:
    """One row per genuine target, with oldest-to-newest historical inputs."""

    targets: np.ndarray
    timestamps: pd.DatetimeIndex
    history: np.ndarray
    missing: np.ndarray
    calendar: np.ndarray
    actual: np.ndarray


def _index(value, length, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer calendar index")
    if not 0 <= value <= length:
        raise ValueError(f"{name} must lie between 0 and {length}")
    return int(value)


def _hourly_clock(timestamps, length):
    raw = np.asarray(timestamps)
    if raw.ndim != 1 or len(raw) != length or (length and raw.dtype.kind in "biufc"):
        raise ValueError("Timestamps must match the one-dimensional traffic calendar")
    try:
        clock = pd.DatetimeIndex(timestamps)
    except (TypeError, ValueError) as exc:
        raise ValueError("Timestamps must be datetime values") from exc
    if clock.tz is not None or clock.hasnans:
        raise ValueError("Use the recorded timezone-naive clock, without missing timestamps")
    if not clock.equals(clock.floor("h")):
        raise ValueError("Timestamps must fall on whole recorded hours")
    if len(clock) > 1 and not np.all(clock[1:] - clock[:-1] == pd.Timedelta(hours=1)):
        raise ValueError("Calendar must be increasing with exactly one hour between rows")
    return clock


def make_examples(traffic, observed, timestamps, start, end):
    """Select eligible targets in [start, end), allowing history before start.

    For target s, history is [s-168, s); traffic at s never becomes an input.
    Original missing flags remain 1 even where causal filling supplied a value.
    """
    values, mask = checked_arrays(traffic, observed)
    start = _index(start, len(values), "start")
    end = _index(end, len(values), "end")
    if start > end:
        raise ValueError("start must not exceed end")
    clock = _hourly_clock(timestamps, len(values))
    eligible = eligible_targets(values, mask)
    targets = np.flatnonzero(eligible[start:end]) + start
    positions = targets[:, None] + np.arange(-LOOKBACK, 0)
    target_clock = clock[targets]
    hour = target_clock.hour.to_numpy(dtype=np.float64)
    hour_of_week = 24 * target_clock.dayofweek.to_numpy(dtype=np.float64) + hour
    calendar = np.column_stack((
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * hour_of_week / 168),
        np.cos(2 * np.pi * hour_of_week / 168),
    ))
    return Examples(
        targets=targets,
        timestamps=target_clock,
        history=values[positions],
        missing=(~mask[positions]).astype(np.float64),
        calendar=calendar,
        actual=values[targets],
    )


@dataclass(frozen=True)
class TrafficScaler:
    """Frozen population mean/SD from genuine hours in one fitting prefix."""

    mean: float
    sd: float
    fit_end: int
    observed_count: int

    def __post_init__(self):
        if not np.isfinite(self.mean) or not np.isfinite(self.sd) or self.sd <= 0:
            raise ValueError("Scaler requires a finite mean and strictly positive finite SD")
        _index(self.fit_end, self.fit_end, "fit_end")
        _index(self.observed_count, self.fit_end, "observed_count")
        if self.observed_count < 2:
            raise ValueError("Scaler requires at least two genuine observations")

    def transform(self, values):
        """Standardise traffic only; flags and calendar features remain unchanged."""
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Cannot scale nonfinite values")
        with np.errstate(over="ignore", invalid="ignore"):
            result = (values - self.mean) / self.sd
        if not np.isfinite(result).all():
            raise ValueError("Scaling produced a nonfinite value")
        return result

    def inverse(self, values):
        """Return to dataset units; retain negative values for later final clipping."""
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Cannot invert nonfinite values")
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.mean + self.sd * values
        if not np.isfinite(result).all():
            raise ValueError("Inverse scaling produced a nonfinite value")
        return result


def fit_scaler(traffic, observed, end):
    """Use unique genuine hours in [0, end), never filled or evaluation values."""
    values = np.asarray(traffic, dtype=np.float64)
    mask = np.asarray(observed)
    if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
        raise ValueError("Traffic and observation mask must be matching one-dimensional arrays")
    end = _index(end, len(values), "end")
    values, mask = checked_arrays(values[:end], mask[:end])
    genuine = values[mask]
    if len(genuine) < 2:
        raise ValueError("Scaler requires at least two genuine observations in the fitting prefix")
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(np.mean(genuine))
        sd = float(np.std(genuine, ddof=0))
    return TrafficScaler(mean, sd, end, len(genuine))


def tabular_inputs(examples, scaler):
    """Return 340 float64 features: 168 traffic, 168 flags, four calendar values."""
    return np.column_stack((
        scaler.transform(examples.history), examples.missing, examples.calendar,
    ))


def lstm_inputs(examples, scaler):
    """Return float32 sequence (n, 168, 2) and separate calendar (n, 4).

    Channel 0 is scaled traffic; channel 1 records original missingness. Only
    this helper casts to the approved LSTM precision; it does not train a model.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        sequence = np.stack((scaler.transform(examples.history), examples.missing), axis=2).astype(np.float32)
        calendar = examples.calendar.astype(np.float32)
    if not np.isfinite(sequence).all() or not np.isfinite(calendar).all():
        raise ValueError("LSTM float32 conversion produced a nonfinite feature")
    return sequence, calendar


def scaled_targets(examples, scaler):
    """Return float64 targets using the same traffic mean and SD as the inputs."""
    return scaler.transform(examples.actual)
