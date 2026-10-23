"""Finite extension workload and complete-cohort, full-precision selection.

The original configuration stays immutable. Only the numerical fit adapter
receives an explicit copy with the two extra ridge penalties admitted.
"""
from copy import deepcopy
from itertools import product
from math import fsum

from forecasting.development_plan import (FOLDS, TUNING_CELLS, candidates,
                                         score_candidates, validate_plan)
from forecasting.development_selection import METRICS, _checked_records

NEW_ALPHAS = (10000, 100000)
PLAN_SHA256 = "2e8ac69fa8d596d34311b4e970ed4d87c2d2396e0e0124f32197a157018e63b2"
SUMMARY_SHA256 = "a3d6905021770195c8ff813c0bbbc4536ecc227371c03420f140e7fe086cb3b2"


def effective_config(config):
    validate_plan(config)
    result = deepcopy(config)
    result["models"]["ridge"]["alphas"] += list(NEW_ALPHAS)
    return result


def fit_candidates(config, stage):
    validate_plan(config)
    if stage == "ridge":
        return [{"candidate_id": f"ridge-a{alpha}", "method": "ridge",
                 "parameters": {"alpha": alpha}} for alpha in NEW_ALPHAS]
    if stage == "hybrid":
        return [{**c, "candidate_id": "extension-" + c["candidate_id"]}
                for c in candidates(config, "hybrid")]
    raise ValueError("Extension fits only ridge or conditional hybrid")


def build_jobs(config, stage):
    result = []
    for c, cell, fold in product(fit_candidates(config, stage), TUNING_CELLS, range(3)):
        start, end = FOLDS[fold]
        result.append({"id": f"{c['candidate_id']}--{cell}--f{fold+1}",
            "stage": stage, "method": c["method"], "cell": cell, "fold_index": fold,
            "fit_end": start, "score_start": start, "score_end": end,
            "candidate_id": c["candidate_id"], "parameters": deepcopy(c["parameters"]), "seed": None})
    return result


def scoring_candidates(config, stage):
    choices = fit_candidates(config, stage)
    if stage == "ridge":
        return choices
    return [{**c, "candidate_id": c["candidate_id"] + "-w" + format(w, "g").replace(".", "p"),
             "parameters": {**c["parameters"], "weight": w}}
            for c in choices for w in config["models"]["hybrid"]["weights"][1:]]


def aggregate(records, config, choices):
    """Reuse the checked record contract, then equal cells followed by folds."""
    checked, _ = _checked_records(records, config, "ridge", choices)
    def mean(rows):
        return {metric: fsum(row[metric] / len(rows) for row in rows) for metric in METRICS}
    summaries = []
    for candidate in choices:
        cid = candidate["candidate_id"]
        folds = []
        for fold in range(3):
            cells = [{"cell": cell, "scores": checked[cid, cell, fold, None]["scores"]}
                     for cell in TUNING_CELLS]
            folds.append({"fold_index": fold, "cells": cells,
                          "scores": mean([c["scores"] for c in cells])})
        summaries.append({**candidate, "folds": folds,
                          "scores": mean([f["scores"] for f in folds]),
                          "cell_count": 9, "fold_count": 3, "seed_count": 1, "record_count": 27})
    return summaries


def select_ridge(original_records, new_records, config):
    choices = candidates(config, "ridge") + fit_candidates(config, "ridge")
    summaries = aggregate(original_records + new_records, config, choices)
    selected = min(summaries, key=lambda c: (c["scores"]["MASE"], -c["parameters"]["alpha"]))
    minimum = selected["scores"]["MASE"]
    return {"status": "passed", "stage": "ridge", "candidates": summaries,
            "selected": selected, "rebuild_corrections": selected["parameters"]["alpha"] in NEW_ALPHAS,
            "tie_break": {"rule": "larger alpha on exact tie", "rounded_before_selection": False,
                          "tied_candidate_ids": [c["candidate_id"] for c in summaries if c["scores"]["MASE"] == minimum]},
            "primary_comparator": None}


def select_hybrid(original_records, original_alpha, ridge_selection, ridge_records,
                  new_records, config):
    """Old positive correction remains tied to its old ridge; zero uses best ridge."""
    positive = score_candidates(config, "hybrid")[1:]
    old_summaries = aggregate(original_records, config, positive)
    old = min(old_summaries, key=lambda c: (c["scores"]["MASE"], c["parameters"]["trees"],
                                          c["parameters"]["leaves"], c["parameters"]["weight"]))
    old = {**old, "origin": "initial", "ridge_alpha": original_alpha}
    best_ridge = ridge_selection["selected"]
    cid = best_ridge["candidate_id"]
    zero_rows = [{**r, "candidate_id": "hybrid-zero"} for r in ridge_records if r["candidate_id"] == cid]
    zero = aggregate(zero_rows, config, [{"candidate_id": "hybrid-zero", "method": "ridge_lightgbm_hybrid",
                                        "parameters": {"weight": 0}}])[0]
    zero.update(origin="expanded_ridge", ridge_alpha=best_ridge["parameters"]["alpha"])
    if ridge_selection["rebuild_corrections"]:
        added = [{**c, "origin": "extension", "ridge_alpha": best_ridge["parameters"]["alpha"]}
                 for c in aggregate(new_records, config, scoring_candidates(config, "hybrid"))]
    elif new_records:
        raise ValueError("An original ridge winner forbids additional correction fits")
    else:
        added = []
    # Comparisons must share the same scale across rounds, not only within one.
    aggregate(original_records + zero_rows + new_records, config,
              positive + [{"candidate_id": "hybrid-zero", "method": "ridge_lightgbm_hybrid",
                           "parameters": {"weight": 0}}]
              + (scoring_candidates(config, "hybrid") if ridge_selection["rebuild_corrections"] else []))
    eligible = [zero, old, *added]
    def key(c):
        p = c["parameters"]
        order = 0 if p["weight"] == 0 else 1 if c["origin"] == "initial" else 2
        return (c["scores"]["MASE"], order, p.get("trees", 0), p.get("leaves", 0), p["weight"])
    selected = min(eligible, key=key)
    return {"status": "passed", "stage": "hybrid", "selected": selected, "candidates": eligible,
            "original_positive_candidates": old_summaries,
            "correction_stage_skipped": not ridge_selection["rebuild_corrections"],
            "skip_reason": "Original ridge alpha remains selected; retain the original positive hybrid without refitting" if not ridge_selection["rebuild_corrections"] else None,
            "ridge_preferred": selected["parameters"]["weight"] == 0,
            "tie_break": {"rule": "zero first; original positive before new positive; then fewer trees, fewer leaves, smaller weight",
                          "rounded_before_selection": False},
            "primary_comparator": None, "hybrid_success_claimed": False}
