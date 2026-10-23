"""Check complete development results, then apply the declared selection rule.

This module only reads score records. It cannot fit models, see traffic or
choose the final primary comparator, which belongs to confirmation.
"""
from collections.abc import Mapping
from math import ceil, fsum, isclose, isfinite
from numbers import Integral, Real
from statistics import median, stdev

from forecasting.development_plan import FOLDS, TUNING_CELLS, score_candidates


METRICS = ("MAE", "RMSE", "MASE")
TIE_RULES = {
    "ridge": "larger alpha",
    "direct": "fewer trees then fewer leaves",
    "hybrid": "lambda=0 first when tied for best, else fewer trees then fewer leaves then smaller lambda",
    "lstm": "fewer hidden units then lower dropout",
}


def _finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def _mean_scores(rows):
    # The nesting is deliberate: equal seeds, then cells, then folds.
    return {metric: fsum(row[metric] / len(rows) for row in rows) for metric in METRICS}


def _checked_records(records, config, stage, choices):
    """Require every declared cell/fold/seed, without pruning failed records."""
    seeds = config["models"]["lstm"]["seeds_all_formal_periods"] if stage == "lstm" else [None]
    expected = {(candidate["candidate_id"], cell, fold, seed)
                for candidate in choices for cell in TUNING_CELLS
                for fold in range(len(FOLDS)) for seed in seeds}
    checked, scales = {}, {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Every result must be a score record")
        try:
            candidate, cell = record["candidate_id"], record["cell"]
            fold, seed = record["fold_index"], record["seed"]
            if (isinstance(fold, bool) or not isinstance(fold, Integral)
                    or (seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)))):
                raise ValueError("Fold index and seed must use declared integer identities")
            key = (candidate, cell, fold, seed)
            if key not in expected:
                raise ValueError("Unexpected candidate, cell, fold or seed in development results")
            if key in checked:
                raise ValueError("Duplicate development score record")
            if record["status"] != "passed":
                raise ValueError("Failed or incomplete development records cannot enter selection")
            scores = {metric: _finite_number(record["scores"][metric], metric) for metric in METRICS}
            scale = _finite_number(record["mase_scale"], "mase_scale", positive=True)
            if not isclose(scores["MASE"], scores["MAE"] / scale, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("MASE must agree with MAE and the fitting-prefix scale")
            if (cell, fold) in scales and scales[cell, fold] != scale:
                raise ValueError("All candidates and seeds must share each cell/fold MASE scale")
            scales[cell, fold] = scale
            item = {"scores": scores}
            if stage == "lstm":
                epochs = record["selected_epochs"]
                if (isinstance(epochs, bool) or not isinstance(epochs, Integral)
                        or not 1 <= epochs <= config["models"]["lstm"]["maximum_epochs"]):
                    raise ValueError("selected_epochs must be an integer inside the declared budget")
                item["selected_epochs"] = int(epochs)
            checked[key] = item
        except (KeyError, TypeError) as exc:
            raise ValueError("Development score record is malformed or lacks required fields") from exc
    if set(checked) != expected:
        raise ValueError(f"Incomplete development cohort: expected {len(expected)} records, received {len(checked)}")
    return checked, seeds


def aggregate_stage(records, config, stage):
    """Aggregate a complete stage at full precision; never pool seed forecasts.

Each record supplies candidate_id, cell, fold_index, seed, status='passed',
scores={MAE, RMSE, MASE}, and positive mase_scale. LSTM also supplies
selected_epochs. Prediction arithmetic and common target masks are separately
checked by the runner before these records are written.
"""
    choices = score_candidates(config, stage)
    checked, seeds = _checked_records(records, config, stage, choices)
    summaries = []
    for candidate in choices:
        name = candidate["candidate_id"]
        folds = []
        for fold in range(len(FOLDS)):
            cells = []
            for cell in TUNING_CELLS:
                scores = _mean_scores([checked[name, cell, fold, seed]["scores"] for seed in seeds])
                cells.append({"cell": cell, "scores": scores})
            folds.append({"fold_index": fold, "scores": _mean_scores([r["scores"] for r in cells]),
                          "cells": cells})
        summary = {**candidate, "scores": _mean_scores([fold["scores"] for fold in folds]),
                   "folds": folds, "cell_count": len(TUNING_CELLS), "fold_count": len(FOLDS),
                   "seed_count": len(seeds), "record_count": len(TUNING_CELLS)*len(FOLDS)*len(seeds)}
        if stage == "lstm":
            seed_scores = []
            for seed in seeds:
                per_fold = [_mean_scores([checked[name, cell, fold, seed]["scores"] for cell in TUNING_CELLS])
                            for fold in range(len(FOLDS))]
                seed_scores.append({"seed": seed, "scores": _mean_scores(per_fold)})
            epochs = [checked[name, cell, fold, seed]["selected_epochs"]
                      for cell in TUNING_CELLS for fold in range(len(FOLDS)) for seed in seeds]
            summary.update(seed_scores=seed_scores,
                           seed_score_sample_sd={metric: stdev(r["scores"][metric] for r in seed_scores)
                                                 for metric in METRICS},
                           selected_epoch_values=epochs,
                           median_selected_epochs=median(epochs))
        summaries.append(summary)
    return summaries


def _tie_key(candidate, stage):
    params = candidate["parameters"]
    if stage == "ridge":
        return (-params["alpha"],)
    if stage == "direct":
        return (params["trees"], params["leaves"])
    if stage == "hybrid":
        if params["weight"] == 0:
            return (0, 0, 0, 0)
        return (1, params["trees"], params["leaves"], params["weight"])
    return (params["hidden_units"], params["dropout"])


def select_stage(records, config, stage):
    """Select one global family configuration; baseline methods all remain."""
    summaries = aggregate_stage(records, config, stage)
    result = {"status": "passed", "stage": stage,
              "scope": "development configuration selection; primary comparator remains pending confirmation",
              "candidates": summaries, "selected": None, "tie_break": None,
              "primary_comparator": None}
    if stage == "baselines":
        return result
    minimum = min(candidate["scores"]["MASE"] for candidate in summaries)
    tied = [candidate for candidate in summaries if candidate["scores"]["MASE"] == minimum]
    selected = min(tied, key=lambda candidate: _tie_key(candidate, stage))
    result.update(selected=selected, tie_break={"metric": "MASE", "minimum_score": minimum,
                  "exact_tie": len(tied) > 1, "tied_candidate_ids": [r["candidate_id"] for r in tied],
                  "rule": TIE_RULES[stage], "rounded_before_selection": False})
    if stage == "hybrid":
        result["ridge_preferred"] = selected["parameters"]["weight"] == 0
        result["hybrid_success_claimed"] = False
    if stage == "lstm":
        result["fixed_epochs"] = ceil(selected["median_selected_epochs"])
        result["epoch_rule"] = "ceil(median(selected-architecture best epochs across all development cells/folds/seeds))"
        result["selected_seed"] = None
        result["prediction_ensemble"] = False
    return result
