"""Finite development selection for independent ridge/LightGBM ensembles.

Every tree capacity is assessed before selecting one component per objective.
Only those frozen components enter the weight grids. MASE remains the selection
criterion; MAE and RMSE remain independently reported supporting outcomes.
"""
from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


OBJECTIVES = ("l2", "l1")
CAPACITIES = tuple((trees, leaves) for trees in (32, 128, 512) for leaves in (7, 31))
WEIGHTS = (0., .05, .1, .2, .3, .5, .75, 1.)
BASELINES = ("persistence", "daily_seasonal_naive", "weekly_seasonal_naive")
STANDALONES = (*BASELINES, "ridge", "lightgbm_l2", "lightgbm_l1", "lstm")
METRICS = ("MAE", "RMSE", "MASE")
CONDITION_KEYS = {"id", "kind", "objective", "trees", "leaves", "ridge_weight"}


def _panel(expected_cells, expected_folds):
    if isinstance(expected_cells, str):
        raise ValueError("expected_cells must be a sequence")
    cells, folds = list(expected_cells), list(expected_folds)
    if (not cells or len(set(cells)) != len(cells)
            or any(not isinstance(c, str) or not c for c in cells)):
        raise ValueError("Expected unique nonempty cell identifiers")
    if (not folds or len(set(folds)) != len(folds)
            or any(isinstance(f, (bool, np.bool_)) or not isinstance(f, Integral)
                   or int(f) not in (0, 1, 2) for f in folds)):
        raise ValueError("Expected unique development fold indices from 0, 1, 2")
    return cells, [int(f) for f in folds]


def _capacity(objective, trees, leaves):
    if objective not in OBJECTIVES:
        raise ValueError("Objective must be l1 or l2")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in (trees, leaves)):
        raise ValueError("Tree capacity must contain integers")
    if (int(trees), int(leaves)) not in CAPACITIES:
        raise ValueError("Tree capacity is outside the declared six configurations")
    return objective, int(trees), int(leaves)


def _weight(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or float(value) not in WEIGHTS):
        raise ValueError("Ridge weight must belong to the declared eight-weight grid")
    return float(value)


def make_condition(kind_or_baseline, *, objective=None, trees=None, leaves=None,
                   ridge_weight=None):
    """Return the canonical exact six-field deployment condition.

    Use a baseline's full ID, ``ridge``, ``lstm``, ``direct`` or ``blend``.
    Endpoints are represented only by standalone IDs; no duplicate ridge model
    is created for the two objective families.
    """
    empty = {"objective": None, "trees": None, "leaves": None, "ridge_weight": None}
    if kind_or_baseline in BASELINES:
        if any(v is not None for v in (objective, trees, leaves, ridge_weight)):
            raise ValueError("Baseline conditions have no learned settings")
        return {"id": kind_or_baseline, "kind": "baseline", **empty}
    if kind_or_baseline in ("ridge", "lstm"):
        expected_weight = 1. if kind_or_baseline == "ridge" else None
        if (any(v is not None for v in (objective, trees, leaves))
                or (ridge_weight is not None and
                    (isinstance(ridge_weight, (bool, np.bool_)) or ridge_weight != expected_weight))):
            raise ValueError("Standalone ridge/LSTM cannot carry tree settings")
        return {"id": kind_or_baseline, "kind": kind_or_baseline,
                **empty, "ridge_weight": expected_weight}
    objective, trees, leaves = _capacity(objective, trees, leaves)
    if kind_or_baseline == "direct":
        if ridge_weight is not None and _weight(ridge_weight) != 0:
            raise ValueError("Direct LightGBM requires ridge weight zero")
        weight, identifier = 0., f"lightgbm_{objective}"
    elif kind_or_baseline == "blend":
        weight = _weight(ridge_weight)
        if weight in (0., 1.):
            raise ValueError("Use canonical standalone conditions for endpoints")
        identifier = f"ensemble_{objective}_r{format(weight, '.12g').replace('.', 'p')}"
    else:
        raise ValueError("Unknown condition kind")
    return {"id": identifier, "kind": kind_or_baseline, "objective": objective,
            "trees": trees, "leaves": leaves, "ridge_weight": weight}


def condition_for_weight(objective, trees, leaves, ridge_weight):
    objective, trees, leaves = _capacity(objective, trees, leaves)
    weight = _weight(ridge_weight)
    if weight == 1:
        return make_condition("ridge")
    return make_condition("direct" if weight == 0 else "blend", objective=objective,
                          trees=trees, leaves=leaves, ridge_weight=weight)


def validate_condition(condition):
    """Reject extra fields, noncanonical IDs and duplicate endpoint encodings."""
    if not isinstance(condition, Mapping) or set(condition) != CONDITION_KEYS:
        raise ValueError("Condition must contain exactly the declared six fields")
    kind = condition["id"] if condition["kind"] == "baseline" else condition["kind"]
    canonical = make_condition(kind, **{key: condition[key] for key in
                                       ("objective", "trees", "leaves", "ridge_weight")})
    if dict(condition) != canonical:
        raise ValueError("Condition differs from its canonical settings or ID")
    return canonical


def _scores(values):
    if not isinstance(values, Mapping) or any(m not in values for m in METRICS):
        raise ValueError("Every record needs MAE, RMSE and MASE")
    result = {}
    for metric in METRICS:
        value = values[metric]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError("Selection scores must be finite, nonnegative scalar values")
        result[metric] = float(value)
    return result


def _aggregate(records, cells, folds):
    by_key = {}
    for record in records:
        fold = record["fold_index"]
        if isinstance(fold, (bool, np.bool_)) or not isinstance(fold, Integral):
            raise ValueError("Fold index must be an integer")
        key = (record["cell"], int(fold))
        if key in by_key:
            raise ValueError("Duplicate cell/fold record")
        by_key[key] = _scores(record["scores"])
    if set(by_key) != {(cell, fold) for cell in cells for fold in folds}:
        raise ValueError("Every candidate requires the exact complete common cell/fold cohort")
    def mean(values):
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.mean(values))
        if not np.isfinite(result):
            raise ValueError("Aggregation produced a nonfinite score")
        return result

    fold_results = []
    for fold in folds:
        scores = {m: mean([by_key[(cell, fold)][m] for cell in cells]) for m in METRICS}
        fold_results.append({"fold_index": fold, "scores": scores,
                             "cells": [{"cell": cell, "scores": by_key[(cell, fold)]} for cell in cells]})
    return {"scores": {m: mean([f["scores"][m] for f in fold_results]) for m in METRICS},
            "folds": fold_results, "cell_count": len(cells), "fold_count": len(folds),
            "record_count": len(by_key)}


def build_selection(records, expected_cells, expected_folds=(0, 1, 2)):
    """Select one standalone tree capacity per objective before weight search.

    Records carry objective, trees, leaves, cell, fold_index and scores. Both
    complete six-capacity searches are required. No successful-subset winner,
    rounding or supporting-metric substitution is permitted.
    """
    cells, folds = _panel(expected_cells, expected_folds)
    grouped = {(o, t, l): [] for o in OBJECTIVES for t, l in CAPACITIES}
    for record in records:
        key = _capacity(record["objective"], record["trees"], record["leaves"])
        grouped[key].append(record)
    objectives = {}
    for objective in OBJECTIVES:
        candidates = []
        for trees, leaves in CAPACITIES:
            candidates.append({"objective": objective, "trees": trees, "leaves": leaves,
                "condition": make_condition("direct", objective=objective, trees=trees, leaves=leaves),
                **_aggregate(grouped[(objective, trees, leaves)], cells, folds)})
        selected = min(candidates, key=lambda c: (c["scores"]["MASE"], c["trees"], c["leaves"]))
        objectives[objective] = {"selected": selected, "candidates": candidates}
    return {"status": "passed", "selection_metric": "equal_cell_then_fold_MASE",
            "capacity_tie_rule": "fewer_trees_then_fewer_leaves_exact_ties_only",
            "expected_cells": cells, "expected_folds": folds,
            "record_count": len(OBJECTIVES) * len(CAPACITIES) * len(cells) * len(folds),
            "objectives": objectives}


def selected_components(capacity_selection):
    if capacity_selection.get("status") != "passed" or set(capacity_selection["objectives"]) != set(OBJECTIVES):
        raise ValueError("Both complete objective selections are required")
    result = {}
    for objective in OBJECTIVES:
        winner = capacity_selection["objectives"][objective]["selected"]
        _, trees, leaves = _capacity(objective, winner["trees"], winner["leaves"])
        result[objective] = {"trees": trees, "leaves": leaves}
    return result


def _weight_tie_key(candidate):
    weight = candidate["ridge_weight"]
    return candidate["scores"]["MASE"], 0 if weight in (0., 1.) else 1, weight


def select_ensembles(capacity_selection, weight_records, expected_cells, expected_folds=(0, 1, 2)):
    """Keep both equal blends, both objective winners, and one overall primary.

    The primary may be a standalone endpoint; an endpoint is not described as
    successful ensembling. Equal 50/50 blends remain reportable even when they
    lose. Deduplication removes only identical canonical deployment conditions.
    """
    cells, folds = _panel(expected_cells, expected_folds)
    if (cells != capacity_selection["expected_cells"] or folds != capacity_selection["expected_folds"]):
        raise ValueError("Capacity and weight selection cohorts must match")
    components = selected_components(capacity_selection)
    grouped = {(objective, weight): [] for objective in OBJECTIVES for weight in WEIGHTS}
    for record in weight_records:
        objective, trees, leaves = _capacity(record["objective"], record["trees"], record["leaves"])
        if components[objective] != {"trees": trees, "leaves": leaves}:
            raise ValueError("Weight search must use the already selected component capacity")
        grouped[(objective, _weight(record["ridge_weight"]))].append(record)
    candidates, winners = {}, {}
    for objective in OBJECTIVES:
        capacity = components[objective]
        candidates[objective] = [{"objective": objective, **capacity, "ridge_weight": weight,
            "condition": condition_for_weight(objective, **capacity, ridge_weight=weight),
            **_aggregate(grouped[(objective, weight)], cells, folds)} for weight in WEIGHTS]
        winners[objective] = min(candidates[objective], key=_weight_tie_key)
        direct = candidates[objective][0]
        selected = capacity_selection["objectives"][objective]["selected"]
        if direct["scores"] != selected["scores"] or direct["folds"] != selected["folds"]:
            raise ValueError("Weight zero must reproduce the frozen standalone component exactly")
    ridge_l2, ridge_l1 = candidates["l2"][-1], candidates["l1"][-1]
    if ridge_l2["scores"] != ridge_l1["scores"] or ridge_l2["folds"] != ridge_l1["folds"]:
        raise ValueError("Weight one must be the same standalone ridge for both objectives")
    primary = min(winners.values(), key=lambda c: (*_weight_tie_key(c), OBJECTIVES.index(c["objective"])))
    conditions = [make_condition(m) for m in (*BASELINES, "ridge")]
    conditions += [make_condition("direct", objective=o, **components[o]) for o in OBJECTIVES]
    conditions += [make_condition("lstm")]
    conditions += [condition_for_weight(o, **components[o], ridge_weight=.5) for o in OBJECTIVES]
    conditions += [winners[o]["condition"] for o in OBJECTIVES]
    unique = {}
    for condition in conditions:
        validate_condition(condition)
        if condition["id"] in unique and unique[condition["id"]] != condition:
            raise ValueError("Canonical ID collision")
        unique.setdefault(condition["id"], condition)
    return {
        "status": "passed", "conditions": list(unique.values()),
        "primary_candidate": primary["condition"]["id"], "primary_condition": primary["condition"],
        "primary_is_interior_ensemble": primary["condition"]["kind"] == "blend",
        "selected_components": components,
        "selected_weights": {o: winners[o]["ridge_weight"] for o in OBJECTIVES},
        "weight_selections": winners, "weight_candidates": candidates,
        "selection_metric": "equal_cell_then_fold_MASE",
        "weight_tie_rule": "endpoint_first_then_smaller_ridge_weight_then_l2_before_l1",
        "equal_weight_conditions_always_retained": True,
        "expected_cells": cells, "expected_folds": folds,
        "heldout_scores_used": False, "new_model_fits": 0,
    }
