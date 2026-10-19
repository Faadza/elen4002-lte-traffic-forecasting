"""Validate the original CSV and build a complete hourly input table.

Run the full audit with: python -m preprocessing.validate_data --help
The original file is never changed. Missing targets remain missing.
"""
import numpy as np
import pandas as pd
from preprocessing.splits import N_HOURS, CALENDAR_START, MAX_FILL_HOURS


def load_raw(path):
    """Return sorted unique observations; reject malformed or conflicting rows."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = ["Date", "Hour", "CellName", "Traffic"]
    if list(df.columns) != required or df.empty:
        raise ValueError("Expected a nonempty CSV with Date,Hour,CellName,Traffic")
    if not df["Date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Dates must have YYYY-MM-DD format")
    dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")
    hours = pd.to_numeric(df["Hour"], errors="raise")
    values = pd.to_numeric(df["Traffic"], errors="raise")
    if not (np.isfinite(hours) & hours.between(0, 23) & (hours == np.floor(hours))).all():
        raise ValueError("Hour must be an integer from 0 to 23")
    if not df["CellName"].str.fullmatch(r"Cell_\d+").all():
        raise ValueError("Missing or malformed cell ID")
    if not (np.isfinite(values) & (values >= 0)).all():
        raise ValueError("Observed traffic must be finite and nonnegative")
    timestamps = dates + pd.to_timedelta(hours, unit="h")
    calendar = pd.date_range(CALENDAR_START, periods=N_HOURS, freq="h")
    if not timestamps.isin(calendar).all():
        raise ValueError("A timestamp falls outside the declared hourly calendar")
    # Exact means identical parsed CSV fields, before numeric conversion.
    keep = ~df.duplicated()
    out = pd.DataFrame({"cell": df.CellName, "ts": timestamps, "traffic": values})[keep]
    if out.duplicated(["cell", "ts"]).any():
        raise ValueError("Repeated cell-hour keys remain after exact-row deduplication")
    out = out.sort_values(["cell", "ts"]).reset_index(drop=True)
    out.attrs["integrity"] = {"raw_rows": len(df), "exact_duplicates": int((~keep).sum()),
        "observed_rows": len(out), "cells": int(out.cell.nunique()),
        "invalid_rows": 0, "conflicting_keys": 0}
    return out


def causal_fill(values, max_fill=MAX_FILL_HOURS):
    """Fill the first max_fill missing hours from the last genuine observation."""
    filled = np.asarray(values, dtype=float).copy()
    if filled.ndim != 1 or np.isinf(filled).any() or np.any(filled < 0):
        raise ValueError("Expected one-dimensional nonnegative values or NaN")
    if isinstance(max_fill, bool) or not isinstance(max_fill, (int, np.integer)) or max_fill < 0:
        raise ValueError("max_fill must be a nonnegative integer")
    # Preserve this mask: a filled input must never become an observed target.
    observed = np.isfinite(filled)
    last_value, run = np.nan, 0
    for i in range(len(filled)):
        if observed[i]:
            last_value, run = filled[i], 0
        else:
            run += 1
            if run <= max_fill and np.isfinite(last_value):
                filled[i] = last_value
    return filled, observed.astype(int)


def prepare(raw, max_fill=MAX_FILL_HOURS):
    """Preserve original observations separately from the filled input column."""
    calendar = pd.date_range(CALENDAR_START, periods=N_HOURS, freq="h")
    if raw.empty or raw.duplicated(["cell", "ts"]).any() or not raw.ts.isin(calendar).all():
        raise ValueError("Expected unique observations on the declared calendar")
    if raw.cell.isna().any() or not (np.isfinite(raw.traffic) & (raw.traffic >= 0)).all():
        raise ValueError("Invalid original observations")
    frames = []
    for cell, group in raw.groupby("cell", sort=True):
        # Insert absent hours before forming lags, so a lag always means elapsed time.
        original = group.set_index("ts").traffic.reindex(calendar).to_numpy()
        filled, observed = causal_fill(original, max_fill)
        frames.append(pd.DataFrame({"cell": cell, "ts": calendar,
            "traffic_observed": original, "traffic": filled, "observed": observed}))
    return pd.concat(frames, ignore_index=True)
