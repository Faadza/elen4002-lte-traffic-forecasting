"""Common, data-only eligibility: observed target and complete preceding history."""
import numpy as np
from preprocessing.splits import LOOKBACK


def checked_arrays(traffic, observed):
    values = np.asarray(traffic, dtype=float)
    mask = np.asarray(observed)
    if values.ndim != 1 or mask.ndim != 1 or values.shape != mask.shape:
        raise ValueError("Traffic and observation mask must be matching one-dimensional arrays")
    if not np.isin(mask, [0, 1]).all() or np.isinf(values).any() or np.any(values < 0):
        raise ValueError("Invalid traffic values or observation mask")
    mask = mask.astype(bool)
    if not np.isfinite(values[mask]).all():
        raise ValueError("Observed positions must have finite traffic")
    return values, mask


def eligible_targets(traffic, observed, lookback=LOOKBACK):
    values, observed = checked_arrays(traffic, observed)
    if isinstance(lookback, bool) or not isinstance(lookback, (int, np.integer)) or lookback < 1:
        raise ValueError("lookback must be a positive integer")
    ok = np.zeros(len(values), dtype=bool)
    # Prefix sums let us count valid inputs in [s-lookback, s), excluding target s.
    counts = np.r_[0, np.cumsum(np.isfinite(values))]
    for s in range(lookback, len(values)):
        ok[s] = observed[s] and counts[s] - counts[s-lookback] == lookback
    return ok
