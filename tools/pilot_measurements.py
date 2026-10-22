"""Small timing and memory helpers for the bounded development pilot.

The caller supplies an already loaded model and data, applies CPU controls, and
runs fit/prediction in separate fresh processes. These helpers do not score errors.
"""

import math
from numbers import Integral, Real
import platform
import resource
import time

import numpy as np


def elapsed_call(call):
    """Time only the supplied operation using elapsed wall-clock seconds."""
    start = time.perf_counter_ns()
    result = call()
    stop = time.perf_counter_ns()
    if stop < start:
        raise RuntimeError("The performance counter moved backwards")
    return result, (stop - start) / 1_000_000_000


def clock_information():
    """Record timer properties; nanosecond units do not imply that resolution."""
    info = time.get_clock_info("perf_counter")
    if not info.monotonic:
        raise RuntimeError("Elapsed timings require a monotonic performance counter")
    return {"function": "time.perf_counter_ns", "implementation": info.implementation,
            "resolution_seconds": info.resolution, "monotonic": info.monotonic,
            "adjustable": info.adjustable, "unit": "seconds",
            "includes_sleep_and_scheduler_delays": True}


def _indices(values, name, allow_empty=False):
    indices = np.asarray(values)
    if allow_empty and indices.ndim == 1 and len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or (not allow_empty and len(indices) == 0)
            or np.any(indices > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} must be a one-dimensional integer target list")
    indices = indices.astype(np.int64, copy=True)
    if np.any(indices < 0) or (len(indices) > 1 and np.any(indices[1:] <= indices[:-1])):
        raise ValueError(f"{name} must be nonnegative, sorted and unique")
    return indices


def select_origins(eligible_targets, maximum=100):
    """Choose evenly spaced positions from common eligibility, before predictions.

    A missing common workload stays empty; the caller must disclose it rather
    than silently changing the cell panel. Rounded positions include both ends.
    """
    if (isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, Integral)
            or not 1 <= maximum <= 100):
        raise ValueError("maximum must be an integer from 1 through 100")
    targets = _indices(eligible_targets, "Eligible targets", allow_empty=True)
    if len(targets) <= maximum:
        return targets
    positions = np.rint(np.linspace(0, len(targets) - 1, maximum)).astype(np.int64)
    return targets[positions]


def timing_summary(seconds):
    """Keep every sample elsewhere; describe median/range and linear empirical p95."""
    values = np.asarray(seconds, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Timing samples must be a finite, nonnegative, nonempty vector")
    return {"count": len(values), "median_seconds": float(np.median(values)),
            "minimum_seconds": float(np.min(values)), "maximum_seconds": float(np.max(values)),
            "p95_seconds": float(np.quantile(values, .95, method="linear")),
            "p95_method": "linear empirical quantile; descriptive, not a confidence interval"}


def _predictions(result, count):
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("Prediction callback must return (raw, final) arrays")
    checked = []
    for label, values in zip(("raw", "final"), result):
        array = np.asarray(values)
        if array.dtype.kind not in "iuf" or array.shape != (count,) or not np.isfinite(array).all():
            raise ValueError(f"{label} predictions must be finite real values matching targets")
        checked.append(np.array(array, dtype=np.float64, copy=True))
    if np.any(checked[1] < 0) or not np.array_equal(checked[1], np.maximum(0., checked[0])):
        raise ValueError("Final predictions must equal the raw predictions lower-bounded at zero")
    return tuple(checked)


def _exact_repeat(actual, expected):
    if any(not np.array_equal(left, right) for left, right in zip(actual, expected)):
        raise AssertionError("Repeated identical prediction work changed its outputs")


def measure_predictions(predict, targets, origins, *, rtol, atol):
    """Measure one warmup plus three passes, with checks outside each timer.

    predict(indices) includes feature construction, scaling, every model stage,
    inverse transformation and final clipping. Loading, exporting and metric
    scoring belong outside the callback. Tolerances compare single-origin with
    batch calculations in original dataset units; repeat checks remain exact.
    """
    targets = _indices(targets, "Targets")
    origins = _indices(origins, "Origins")
    if len(origins) > 100 or not np.isin(origins, targets).all():
        raise ValueError("Origins must be at most 100 members of the common target list")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    clock = clock_information()
    reference = _predictions(predict(targets), len(targets))
    batch_seconds = []
    for _ in range(3):
        result, seconds = elapsed_call(lambda: predict(targets))
        _exact_repeat(_predictions(result, len(targets)), reference)
        batch_seconds.append(seconds)

    # Construct the one-target workloads before their timers.
    workloads = [np.array([origin], dtype=np.int64) for origin in origins]
    positions = np.searchsorted(targets, origins)
    single_reference = []
    for workload, position in zip(workloads, positions):
        result = _predictions(predict(workload), 1)
        for observed, batch in zip(result, reference):
            np.testing.assert_allclose(observed, batch[position:position + 1], rtol=rtol, atol=atol)
        single_reference.append(result)
    single_seconds = []
    for _ in range(3):
        row = []
        for workload, expected in zip(workloads, single_reference):
            result, seconds = elapsed_call(lambda: predict(workload))
            _exact_repeat(_predictions(result, 1), expected)
            row.append(seconds)
        single_seconds.append(row)
    return {"raw": reference[0], "final": reference[1], "timings": {
        "clock": clock, "warmup_passes": 1, "timed_passes": 3,
        "target_count": len(targets), "origins": origins.tolist(), "origin_count": len(origins),
        "batch_seconds": batch_seconds, "batch_summary": timing_summary(batch_seconds),
        "single_origin_seconds": single_seconds,
        "single_origin_summary": timing_summary(np.asarray(single_seconds).ravel()),
        "single_origin_sample_order": "pass then ascending origin",
        "single_vs_batch_tolerance": {"rtol": float(rtol), "atol": float(atol),
                                      "units": "original dataset traffic units"},
        "identical_workload_repeats_equal": True,
        "single_vs_batch_checked": True,
        "timer_excludes_output_checks_and_metric_scoring": True,
    }}


def peak_rss_bytes(native_value, system):
    """Convert documented ru_maxrss units; never guess another platform's units.

    Linux getrusage(2): KiB. Apple XNU bsd/man/man2/getrusage.2: bytes.
    """
    if system not in ("Linux", "Darwin"):
        raise ValueError("Peak RSS units are verified only for Linux and Darwin")
    if (isinstance(native_value, (bool, np.bool_)) or not isinstance(native_value, Real)
            or not math.isfinite(native_value) or native_value < 0 or int(native_value) != native_value):
        raise ValueError("Native peak RSS must be a finite nonnegative integer")
    return int(native_value) * (1024 if system == "Linux" else 1)


def memory_snapshot():
    """Read total process peak RSS immediately after work, before export/auditing.

    Peak RSS includes earlier imports/loading and, for prediction, warmups,
    repeated calls and their output checks. It is not incremental model memory.
    Separate fresh processes are required for fit and prediction measurements.
    """
    system = platform.system()
    native = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"process_peak_rss_bytes": peak_rss_bytes(native, system),
            "native_peak_value": int(native), "native_peak_unit": "KiB" if system == "Linux" else "bytes",
            "system": system, "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            "scope": "total current worker lifetime, including imports/data/loading and measured work",
            "incremental_model_memory": False,
            "requires_separate_fresh_fit_and_prediction_workers": True}
