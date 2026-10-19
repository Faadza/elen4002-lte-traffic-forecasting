"""Development-only computability checks and deterministic early tuning selection."""
import numpy as np
import pandas as pd
from preprocessing.windows import checked_arrays, eligible_targets
from preprocessing.splits import TRAIN_END, FOLDS


def weekly_scale(traffic, observed, end):
    values, observed = checked_arrays(traffic, observed)
    if not isinstance(end, (int, np.integer)) or not 0 <= end <= len(values):
        raise ValueError("Invalid prefix end")
    if end <= 168:
        return np.nan
    # The weekly error scale uses genuine observations at both ends of each pair.
    pairs = observed[168:end] & observed[:end-168]
    diffs = np.abs(values[168:end] - values[:end-168])
    return float(diffs[pairs].mean()) if pairs.any() else np.nan


def prefix_stats(traffic, observed, end):
    values, observed = checked_arrays(traffic, observed)
    if not isinstance(end, (int, np.integer)) or not 0 < end <= len(values):
        raise ValueError("Invalid prefix end")
    sample = values[:end][observed[:end]]
    mean = float(sample.mean()) if len(sample) else np.nan
    sd = float(sample.std(ddof=0)) if len(sample) else np.nan
    scale = weekly_scale(values, observed, end)
    windows = int(eligible_targets(values[:end], observed[:end]).sum())
    pairs = int((observed[168:end] & observed[:end-168]).sum()) if end > 168 else 0
    reasons = []
    if windows == 0:
        reasons.append("no usable fitting window")
    if not np.isfinite(sd) or sd <= 0:
        reasons.append("observed population SD not positive")
    if not np.isfinite(scale) or scale <= 0:
        reasons.append("weekly scale not positive")
    return {"prefix_end": end, "observed_count": len(sample),
        "observed_fraction": len(sample)/end, "mean": mean, "sd": sd,
        "cv": sd/mean if mean > 0 else np.nan, "weekly_pairs": pairs,
        "weekly_scale": scale, "usable_windows": windows,
        "exclusion_reason": "; ".join(reasons)}


def cell_statistics(prepared, end):
    rows = []
    for cell, group in prepared.groupby("cell", sort=True):
        group = group.sort_values("ts")
        rows.append({"cell": cell, **prefix_stats(group.traffic.to_numpy(),
                    group.observed.to_numpy(), end)})
    return pd.DataFrame(rows)


def choose_panel(early_stats):
    """Use only early-prefix quality and unrounded CV; never later main eligibility."""
    pool = early_stats[early_stats.exclusion_reason == ""].copy()
    if not np.isfinite(pool.cv).all():
        raise ValueError("Eligible tuning cells must have finite CV")
    if len(pool) < 9:
        panel = pool.sort_values("cell").copy()
        panel["quantile"] = np.nan
        return panel.reset_index(drop=True)
    chosen = []
    levels = np.arange(1, 10)/10
    targets = np.quantile(pool.cv, levels, method="linear")
    for level, target in zip(levels, targets):
        candidates = pool[~pool.cell.isin([row["cell"] for row in chosen])].copy()
        candidates["distance"] = (candidates.cv - target).abs()
        row = candidates.sort_values(["distance", "cell"]).iloc[0].to_dict()
        row.update(quantile=float(level), target_cv=float(target))
        chosen.append(row)
    return pd.DataFrame(chosen).reset_index(drop=True)


def select_cells(prepared):
    # Select tuning cells using only the prefix available before the first fold.
    main = cell_statistics(prepared, TRAIN_END)
    early = cell_statistics(prepared, FOLDS[0]["train_end"])
    return main, early, choose_panel(early)
