"""Score and save a pilot's fixed targets using the approved error definitions.

The caller supplies the exact common target indices from the validated bundle.
This module checks their order, observed labels and interval membership; input
eligibility itself needs the prepared history and is checked by the runner.
"""

import hashlib
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import (busiest_hour_measures, finalize_predictions,
                                mae, mase, rmse, seasonal_scale)


def _vector(values, name, *, finite=True):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric array")
    array = array.astype(np.float64)
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _targets(values):
    array = np.asarray(values)
    if (array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu"
            or np.any(array < 0) or np.any(array > np.iinfo(np.int64).max)):
        raise ValueError("targets must be a nonempty vector of nonnegative integer indices")
    array = array.astype(np.int64)
    if np.any(array[1:] <= array[:-1]):
        raise ValueError("targets must be sorted and unique")
    return array


def _clock(timestamps, size, *, consecutive):
    raw = np.asarray(timestamps)
    if (raw.ndim != 1 or len(raw) != size or raw.dtype.kind in "biufc"
            or any(isinstance(value, (int, float, complex, np.number)) for value in raw)):
        raise ValueError("timestamps must be a matching vector of recorded hours")
    try:
        clock = pd.DatetimeIndex(timestamps)
    except (TypeError, ValueError) as exc:
        raise ValueError("timestamps must be valid recorded hours") from exc
    if clock.hasnans or clock.tz is not None or not clock.equals(clock.floor("h")):
        raise ValueError("timestamps must be valid timezone-naive whole hours")
    if len(clock) > 1:
        gaps = clock[1:] - clock[:-1]
        if not np.all(gaps == pd.Timedelta(hours=1) if consecutive else gaps > pd.Timedelta(0)):
            raise ValueError("timestamps must be chronological, with no duplicate hours")
    return clock


def _predictions(raw, final, count):
    raw, final = _vector(raw, "raw"), _vector(final, "final")
    if raw.shape != (count,) or final.shape != (count,):
        raise ValueError("raw and final predictions must match every supplied target")
    expected, clipping = finalize_predictions(raw)
    if not np.array_equal(final, expected):
        raise ValueError("final predictions must equal max(0, raw) exactly")
    return raw, final, clipping


def _check_arithmetic(actual, expected, name):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"{name} differs from independent arithmetic")


def score_predictions(values_observed, observed, timestamps, fit_end, start, end,
                      targets, raw, final):
    """Score one cell/model/seed, retaining unavailable scales explicitly.

    Observations and dates use full calendar indexing. Only [0,end) is read;
    weekly lag-168 pairs use [0,fit_end). No model or seed aggregation occurs.
    """
    fit_end, start, end = (_index(fit_end, "fit_end"), _index(start, "start"), _index(end, "end"))
    if not (0 <= fit_end <= start < end <= len(values_observed)):
        raise ValueError("Expected fitting prefix before a nonempty scoring interval")
    if not (len(values_observed) == len(observed) == len(timestamps)):
        raise ValueError("Values, observation mask and timestamps must match")
    # Slice before conversion or validation: later traffic cannot enter scores.
    values = _vector(values_observed[:end], "values_observed", finite=False)
    mask = np.asarray(observed[:end])
    if mask.shape != values.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("observed must be a matching one-dimensional 0/1 mask")
    mask = mask.astype(bool)
    if np.isinf(values).any() or not np.isfinite(values[mask]).all() or np.any(values[mask] < 0):
        raise ValueError("Genuine traffic observations must be finite and nonnegative")
    clock = _clock(timestamps[:end], end, consecutive=True)
    targets = _targets(targets)
    if targets[0] < start or targets[-1] >= end or not mask[targets].all():
        raise ValueError("targets must be genuine observations inside the scoring interval")
    raw, final, clipping = _predictions(raw, final, len(targets))
    actual = values[targets]

    errors = [float(prediction) - float(truth) for prediction, truth in zip(final, actual)]
    manual_mae = math.fsum(abs(error) for error in errors) / len(errors)
    manual_rmse = math.sqrt(math.fsum(error * error for error in errors) / len(errors))
    scores = {"MAE": mae(actual, final), "RMSE": rmse(actual, final), "MASE": None}
    _check_arithmetic(scores["MAE"], manual_mae, "MAE")
    _check_arithmetic(scores["RMSE"], manual_rmse, "RMSE")
    pairs = [abs(float(values[i]) - float(values[i - 168])) for i in range(168, fit_end)
             if mask[i] and mask[i - 168]]
    scale = seasonal_scale(values[:fit_end], mask[:fit_end], fit_end, lag=168)
    if pairs:
        _check_arithmetic(scale, math.fsum(pairs) / len(pairs), "Weekly scale")
    else:
        if not math.isnan(scale):
            raise AssertionError("Weekly scale must be unavailable without genuine pairs")
    if pairs and scale > 0:
        scores["MASE"] = mase(actual, final, scale)
        _check_arithmetic(scores["MASE"], manual_mae / scale, "MASE")
        scale_status = "available"
    else:
        scale_status = "zero_scale" if pairs else "no_observed_weekly_pairs"

    _check_arithmetic(clipping["mean_upward_adjustment"],
                      math.fsum(float(f) - float(r) for f, r in zip(final, raw)) / len(targets),
                      "Mean clipping adjustment")
    clipped_count = sum(float(value) < 0 for value in raw)
    if clipping["clipped_count"] != clipped_count:
        raise AssertionError("Clipped count differs from independent arithmetic")
    _check_arithmetic(clipping["clipped_fraction"], clipped_count / len(targets), "Clipped fraction")
    block_forecasts = np.full(end - start, np.nan)
    block_eligible = np.zeros(end - start, dtype=bool)
    block_forecasts[targets - start] = final
    block_eligible[targets - start] = True
    busy = busiest_hour_measures(clock[start:end], values[start:end], mask[start:end],
                                block_eligible, block_forecasts)
    observed_count = int(mask[start:end].sum())
    return {
        "scores": scores, "clipping": clipping, "busiest_hour": busy,
        "weekly_scale": float(scale) if pairs else None, "weekly_scale_status": scale_status,
        "mase_status": "available" if scores["MASE"] is not None else "unavailable",
        "counts": {"calendar_hours": end - start, "observed_targets": observed_count,
                   "scored_targets": len(targets), "fitting_observed_weekly_pairs": len(pairs)},
        "coverage": {"fraction_calendar_scored": len(targets) / (end - start),
                     "fraction_observed_scored": len(targets) / observed_count},
        "independent_arithmetic_checked": True,
    }


def save_predictions(path, cell, method, seed, targets, timestamps, actual, raw, final):
    """Save target-aligned rows once and verify exact CSV round-trip values.

    Here timestamps and actual contain one entry per target (not full-calendar
    arrays). A blank seed records a deterministic method, rather than seed zero.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Prediction evidence already exists: {path}")
    if any(not isinstance(value, str) or not value for value in (cell, method)):
        raise ValueError("cell and method must be nonempty strings")
    seed_text = "" if seed is None else str(_index(seed, "seed"))
    targets = _targets(targets)
    clock = _clock(timestamps, len(targets), consecutive=False)
    if len(targets) > 1 and not np.array_equal(
            np.diff(targets), np.diff(clock.asi8) // pd.Timedelta(hours=1).value):
        raise ValueError("Timestamp gaps must match target-index gaps")
    actual = _vector(actual, "actual")
    if actual.shape != (len(targets),) or np.any(actual < 0):
        raise ValueError("actual must be nonnegative and match every target")
    raw, final, _ = _predictions(raw, final, len(targets))
    table = pd.DataFrame({"cell": cell, "method": method, "seed": seed_text,
        "target_index": targets, "timestamp": clock,
        "actual": actual, "prediction_raw": raw, "prediction": final})
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects evidence if another process writes first.
    with path.open("x", newline="") as handle:
        table.to_csv(handle, index=False, float_format="%.17g", date_format="%Y-%m-%d %H:%M:%S")
    loaded = pd.read_csv(path, keep_default_na=False, float_precision="round_trip",
                         dtype={"cell": str, "method": str, "seed": str,
                                "target_index": np.int64, "actual": np.float64,
                                "prediction_raw": np.float64, "prediction": np.float64},
                         parse_dates=["timestamp"])
    for name in ("cell", "method", "seed", "target_index", "timestamp", "actual", "prediction_raw", "prediction"):
        if not np.array_equal(loaded[name].to_numpy(), table[name].to_numpy()):
            raise AssertionError(f"Saved prediction column failed round-trip check: {name}")
    return {"rows": len(table), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "round_trip_checked": True}
