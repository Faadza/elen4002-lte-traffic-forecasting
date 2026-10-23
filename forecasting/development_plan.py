"""Pure descriptions of the finite development workload; no data or fitting.

Fit candidates and scoring candidates differ only for the hybrid: one stored
correction supports four positive weights, and ridge supplies weight zero.
"""
from copy import deepcopy
from itertools import product
from numbers import Integral


STAGES = ("baselines", "ridge", "direct", "hybrid", "lstm")
BASELINES = ("persistence", "daily_seasonal_naive", "weekly_seasonal_naive")
TUNING_CELLS = (
    "Cell_003791", "Cell_000233", "Cell_001792", "Cell_001791", "Cell_007371",
    "Cell_001033", "Cell_039871", "Cell_000112", "Cell_001912",
)
FOLDS = ((4116, 4788), (4788, 5460), (5460, 6132))


def validate_plan(config):
    """Reject a changed search grid, chronology or panel before making jobs."""
    models = config["models"]
    grids = ((models["ridge"]["alphas"], [0.1, 1, 10, 100, 1000]),
             (models["trees"]["tree_counts"], [32, 128, 512]),
             (models["trees"]["leaf_caps"], [7, 31]),
             (models["hybrid"]["weights"], [0, 0.25, 0.5, 0.75, 1]),
             (models["lstm"]["hidden_units"], [16, 32]),
             (models["lstm"]["output_dropout"], [0.0, 0.2]),
             (models["lstm"]["seeds_all_formal_periods"], [2026, 2027, 2028]))
    if any(actual != expected or any(isinstance(x, bool) for x in actual)
           for actual, expected in grids):
        raise ValueError("Development planning requires the unchanged declared grids")
    integer_grids = (models["trees"]["tree_counts"], models["trees"]["leaf_caps"],
                     models["lstm"]["hidden_units"], models["lstm"]["seeds_all_formal_periods"])
    if any(not isinstance(value, Integral) for grid in integer_grids for value in grid):
        raise ValueError("Tree sizes, hidden units and seeds must remain integers")
    expected_folds = [{"fit": [0, start], "score": [start, end]}
                      for start, end in FOLDS]
    if (config["chronology"]["folds"] != expected_folds
            or config["chronology"]["development"] != [0, 6132]
            or config["tuning_panel"]["ids"] != list(TUNING_CELLS)):
        raise ValueError("Development planning requires the declared nine cells and three folds")
    if (config["selection"]["ridge_first"] is not True
            or config["selection"]["rounding_before_tie_detection"] is not False
            or config["metrics"]["primary"] != "equal_cell_mean_weekly_MASE"):
        raise ValueError("Development selection rules must remain unchanged")


def _number_label(value):
    return format(value, "g").replace(".", "p")


def candidates(config, stage):
    """Return fit candidates in a fixed order, without selecting any winner."""
    if stage not in STAGES:
        raise ValueError("Unknown development stage")
    validate_plan(config)
    models = config["models"]
    result = []
    if stage == "baselines":
        return [{"candidate_id": method, "method": method, "parameters": {}}
                for method in BASELINES]
    if stage == "ridge":
        for alpha in models["ridge"]["alphas"]:
            result.append({"candidate_id": f"ridge-a{_number_label(alpha)}",
                           "method": "ridge", "parameters": {"alpha": alpha}})
    elif stage in ("direct", "hybrid"):
        method = "direct_lightgbm" if stage == "direct" else "ridge_lightgbm_hybrid"
        for trees, leaves in product(models["trees"]["tree_counts"],
                                     models["trees"]["leaf_caps"]):
            result.append({"candidate_id": f"{stage}-t{trees}-l{leaves}",
                           "method": method,
                           "parameters": {"trees": trees, "leaves": leaves}})
    else:
        for hidden, dropout in product(models["lstm"]["hidden_units"],
                                       models["lstm"]["output_dropout"]):
            result.append({"candidate_id": f"lstm-h{hidden}-d{_number_label(dropout)}",
                           "method": "lstm",
                           "parameters": {"hidden_units": hidden, "dropout": dropout}})
    return result


def score_candidates(config, stage):
    """Describe every comparison, including ridge alone once for the hybrid."""
    fitted = candidates(config, stage)
    if stage != "hybrid":
        return fitted
    result = [{"candidate_id": "hybrid-zero", "method": "ridge_lightgbm_hybrid",
               "parameters": {"weight": 0}}]
    for candidate in fitted:
        for weight in config["models"]["hybrid"]["weights"][1:]:
            result.append({"candidate_id": f"{candidate['candidate_id']}-w{_number_label(weight)}",
                           "method": candidate["method"],
                           "parameters": {**candidate["parameters"], "weight": weight}})
    return result


def build_jobs(config, stage):
    """Build deterministic filenames and chronology for one complete stage.

Hybrid jobs fit only correction trees after the global ridge choice. The
runner supplies and verifies that selected ridge dependency separately.
"""
    choices = candidates(config, stage)
    seeds = config["models"]["lstm"]["seeds_all_formal_periods"] if stage == "lstm" else [None]
    result = []
    for candidate, cell, fold_index, seed in product(choices, TUNING_CELLS, range(3), seeds):
        start, end = FOLDS[fold_index]
        job_id = f"{candidate['candidate_id']}--{cell}--f{fold_index+1}"
        if seed is not None:
            job_id += f"--s{seed}"
        result.append({"id": job_id, "stage": stage, "method": candidate["method"],
                       "cell": cell, "fold_index": fold_index, "fit_end": start,
                       "score_start": start, "score_end": end,
                       "candidate_id": candidate["candidate_id"],
                       "parameters": deepcopy(candidate["parameters"]), "seed": seed})
    return result
