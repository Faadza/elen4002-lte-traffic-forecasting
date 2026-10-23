"""Recompute the declared weight grids from verified saved raw forecasts.

This module performs no filesystem access and no model fitting. The caller
checks source provenance before providing original-unit arrays, then saves the
returned records with those provenance hashes. Earlier residual/partial models
are not accepted inputs to this independent-forecast experiment.
"""
from collections.abc import Mapping

import numpy as np

from evaluation.metrics import finalize_predictions, mae, rmse
from ensemble_study.selection import (
    CAPACITIES, METRICS, OBJECTIVES, WEIGHTS, _panel, _capacity,
    build_selection, select_ensembles, selected_components,
)

FOLD_SCORE_INTERVALS = {0: (4116, 4788), 1: (4788, 5460), 2: (5460, 6132)}


def _vector(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite nonempty one-dimensional vector")
    return values


def score_raw_predictions(actual, raw_by_seed, weekly_scale, *, expected_seeds=None):
    """Clip each seed's raw forecasts, score separately, then average scores.

    This also supports an existing LSTM comparator without treating its three
    seed forecasts as a deployment ensemble. Weight replay itself has one
    deterministic forecast per cell/fold.
    """
    actual = _vector(actual, "actual")
    if np.any(actual < 0):
        raise ValueError("Actual traffic must be nonnegative")
    scale = np.asarray(weekly_scale, dtype=np.float64)
    if scale.ndim != 0 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Development selection requires an available positive fitting weekly scale")
    if not isinstance(raw_by_seed, Mapping) or not raw_by_seed:
        raise ValueError("Provide every declared seed in a nonempty mapping")
    if len({str(s) for s in raw_by_seed}) != len(raw_by_seed):
        raise ValueError("Seed IDs must remain unique after string conversion")
    if expected_seeds is not None:
        expected = [str(s) for s in expected_seeds]
        if len(set(expected)) != len(expected) or set(expected) != {str(s) for s in raw_by_seed}:
            raise ValueError("Predictions do not contain exactly the declared seeds")
    seeds = {}
    for seed, values in raw_by_seed.items():
        raw = _vector(values, "raw forecast")
        if raw.shape != actual.shape:
            raise ValueError("Every seed must predict the identical complete target cohort")
        final, clipping = finalize_predictions(raw)
        absolute = mae(actual, final)
        seeds[str(seed)] = {"scores": {"MAE": absolute, "RMSE": rmse(actual, final),
                                         "MASE": absolute / float(scale)}, "clipping": clipping}
    return {"scores": {metric: float(np.mean([s["scores"][metric] for s in seeds.values()]))
                       for metric in METRICS}, "seed_scores": seeds,
            "target_count": len(actual), "seed_count": len(seeds)}


def _cohorts(cohorts, expected_cells, expected_folds):
    cells, folds = _panel(expected_cells, expected_folds)
    if not isinstance(cohorts, Mapping):
        raise ValueError("Cohorts must be a mapping keyed by objective/capacity/cell/fold")
    expected = {(o, t, l, c, f) for o in OBJECTIVES for t, l in CAPACITIES
                for c in cells for f in folds}
    if set(cohorts) != expected:
        raise ValueError("Replay requires both complete six-capacity common development cohorts")
    checked, common = {}, {}
    for objective in OBJECTIVES:
        for trees, leaves in CAPACITIES:
            for cell in cells:
                for fold in folds:
                    key = (objective, trees, leaves, cell, fold)
                    entry = cohorts[key]
                    targets = np.asarray(entry["targets"])
                    start, end = FOLD_SCORE_INTERVALS[fold]
                    if (targets.ndim != 1 or targets.dtype.kind not in "iu" or len(targets) == 0
                            or targets[0] < start or targets[-1] >= end
                            or (len(targets) > 1 and np.any(targets[1:] <= targets[:-1]))):
                        raise ValueError("Targets must be unique ordered hours inside the declared development fold")
                    actual = _vector(entry["actual"], "actual")
                    ridge = _vector(entry["ridge_raw"], "ridge_raw")
                    direct = _vector(entry["direct_raw"], "direct_raw")
                    if actual.shape != targets.shape or actual.shape != ridge.shape or actual.shape != direct.shape:
                        raise ValueError("Targets and all raw forecasts must share one exact cohort")
                    # Score validates nonnegative actual and a positive scalar denominator.
                    score_raw_predictions(actual, {"deterministic": direct}, entry["weekly_scale"])
                    scale = float(entry["weekly_scale"])
                    if (cell, fold) not in common:
                        common[(cell, fold)] = (targets, actual, ridge, scale)
                    else:
                        previous = common[(cell, fold)]
                        if (any(not np.array_equal(a, b) for a, b in
                                zip((targets, actual, ridge), previous[:3])) or scale != previous[3]):
                            raise ValueError("All objectives/capacities must share targets, actuals, ridge raw and fitting scale")
                    checked[key] = {"targets": targets, "actual": actual, "ridge_raw": ridge,
                                    "direct_raw": direct, "weekly_scale": scale}
    return checked, cells, folds


def capacity_records(cohorts, expected_cells, expected_folds=(0, 1, 2)):
    """Recompute all standalone capacity scores from raw predictions."""
    checked, cells, folds = _cohorts(cohorts, expected_cells, expected_folds)
    return [{"objective": objective, "trees": trees, "leaves": leaves,
             "cell": cell, "fold_index": fold,
             **score_raw_predictions(entry["actual"], {"deterministic": entry["direct_raw"]},
                                     entry["weekly_scale"])}
            for (objective, trees, leaves, cell, fold), entry in checked.items()]


def replay_weights(cohorts, capacity_selection, expected_cells, expected_folds=(0, 1, 2)):
    """Return both eight-weight grids, using only each frozen tree component.

    Inputs are keyed by ``(objective,trees,leaves,cell,fold_index)`` and contain
    targets, actual, ridge_raw, direct_raw and weekly_scale, in original units.
    The caller must supply the whole capacity search; recomputing its scores
    confirms that the supplied capacity selection precedes weight selection.
    """
    checked, cells, folds = _cohorts(cohorts, expected_cells, expected_folds)
    records = [{"objective": objective, "trees": trees, "leaves": leaves,
                "cell": cell, "fold_index": fold,
                "scores": score_raw_predictions(entry["actual"], {"deterministic": entry["direct_raw"]},
                                                 entry["weekly_scale"])["scores"]}
               for (objective, trees, leaves, cell, fold), entry in checked.items()]
    recomputed = build_selection(records, cells, folds)
    if recomputed != capacity_selection:
        raise ValueError("Capacity selection does not exactly reproduce from these saved forecasts")
    components = selected_components(capacity_selection)
    result = []
    for objective in OBJECTIVES:
        trees, leaves = components[objective]["trees"], components[objective]["leaves"]
        for weight in WEIGHTS:
            for cell in cells:
                for fold in folds:
                    entry = checked[(objective, trees, leaves, cell, fold)]
                    if weight == 0:
                        raw = entry["direct_raw"].copy()
                    elif weight == 1:
                        raw = entry["ridge_raw"].copy()
                    else:
                        raw = weight * entry["ridge_raw"] + (1 - weight) * entry["direct_raw"]
                    result.append({"objective": objective, "trees": trees, "leaves": leaves,
                        "ridge_weight": weight, "cell": cell, "fold_index": fold,
                        **score_raw_predictions(entry["actual"], {"deterministic": raw}, entry["weekly_scale"])})
    return result


def replay_study(cohorts, expected_cells, expected_folds=(0, 1, 2)):
    """One zero-fit entry point: capacities, full grids and deployment selection."""
    records = capacity_records(cohorts, expected_cells, expected_folds)
    capacities = build_selection(records, expected_cells, expected_folds)
    weights = replay_weights(cohorts, capacities, expected_cells, expected_folds)
    return {"capacity_selection": capacities, "weight_records": weights,
            "selection": select_ensembles(capacities, weights, expected_cells, expected_folds),
            "new_model_fits": 0, "clipping_order": "combine_raw_then_final_zero_bound"}
