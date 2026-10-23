"""Frozen eight-candidate development revision, with a matched direct control."""
from copy import deepcopy
from itertools import product

from adaptation.plan import aggregate
from forecasting.development_plan import FOLDS, TUNING_CELLS, validate_plan

BETAS = (0, 0.25, 0.5, 1)
TREE_SETTINGS = ((128, 7), (32, 31))
RIDGE_ALPHA = 1000
STAGE = "l1"
COUNTS = {"direct_l1": 54, "partial_ridge_l1": 162, "new_ridge": 0}


def candidates(config):
    validate_plan(config)
    return [{"candidate_id": f"l1-b{format(beta, 'g').replace('.', 'p')}-t{trees}-l{leaves}",
             "method": "direct_lightgbm_l1" if beta == 0 else "partial_ridge_lightgbm_l1",
             "parameters": {"beta": beta, "trees": trees, "leaves": leaves}}
            for beta, (trees, leaves) in product(BETAS, TREE_SETTINGS)]


def build_jobs(config):
    return [{"id": f"{candidate['candidate_id']}--{cell}--f{fold+1}",
             "stage": STAGE, "method": candidate["method"], "cell": cell,
             "fold_index": fold, "fit_end": start, "score_start": start,
             "score_end": end, "candidate_id": candidate["candidate_id"],
             "parameters": deepcopy(candidate["parameters"]), "seed": None}
            for candidate, cell, (fold, (start, end)) in
            product(candidates(config), TUNING_CELLS, enumerate(FOLDS))]


def scientific_specification(config):
    validate_plan(config)
    return {"scope": "single bounded development-only partial-ridge L1 revision",
            "candidates": candidates(config), "stage_counts": COUNTS,
            "maximum_new_successful_tree_fits": 216, "ridge_alpha": RIDGE_ALPHA,
            "ridge_dependencies": "27 immutable original ridge-a1000 fitted bundles; reused only for beta>0",
            "equation": "z_hat = beta * ridge_z + tree_z; tree targets = z_actual - beta * ridge_z",
            "tree_objective": "regression_l1", "tree_parameters_other_than_objective": deepcopy(config["models"]["trees"]),
            "objective_exception": "Native regression_l1 replaces regression (squared_error) for every new tree, including beta=0 controls",
            "final_prediction": "inverse scaler once, then max(0, raw); no clipping intermediate outputs",
            "selection": "full precision equal-cell mean weekly MASE, then equal fold mean; all 27 cohorts required",
            "ties": "lower beta (zero first), then fewer trees, then fewer leaves",
            "reporting": "retain best beta=0 and best beta>0 separately, plus capacity-matched contrasts",
            "fairness": "same data, features, fitting prefixes, targets, scaling, objective, tree settings and deterministic seed; hybrid has three beta choices, direct has one",
            "tuning_limitation": "additional adaptive development search, not equal historical tuning budgets or unbiased evaluation; never tune on confirmation/final",
            "historical_references": ["original selected L2 direct", "original residual hybrid", "fixed 20% ridge plus 80% original L2 direct"],
            "holdout_forecasts_enabled": False, "formal_comparative_timing": False,
            "primary_comparator": None, "automatic_further_expansion": False}


def select(records, config):
    summaries = aggregate(records, config, candidates(config))
    key = lambda row: (row["scores"]["MASE"], row["parameters"]["beta"],
                       row["parameters"]["trees"], row["parameters"]["leaves"])
    direct = [row for row in summaries if row["parameters"]["beta"] == 0]
    positive = [row for row in summaries if row["parameters"]["beta"] > 0]
    best_direct, best_positive = min(direct, key=key), min(positive, key=key)
    pairs = []
    for row in positive:
        p = row["parameters"]
        control = next(d for d in direct if all(d["parameters"][k] == p[k] for k in ("trees", "leaves")))
        pairs.append({"candidate_id": row["candidate_id"], "control_candidate_id": control["candidate_id"],
                      "MASE_difference": row["scores"]["MASE"] - control["scores"]["MASE"]})
    return {"status": "passed", "candidates": summaries, "selected": min(summaries, key=key),
            "best_standalone_l1": best_direct, "best_positive_beta_l1": best_positive,
            "positive_beats_best_standalone_l1_on_development": best_positive["scores"]["MASE"] < best_direct["scores"]["MASE"],
            "capacity_matched_comparisons": pairs,
            "tie_rule": scientific_specification(config)["ties"], "rounded_before_selection": False,
            "primary_comparator": None, "holdout_forecasts_evaluated": False,
            "hybrid_success_claimed": False}
