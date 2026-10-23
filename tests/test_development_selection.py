"""Hand arithmetic and failure cases for development-only score selection."""
import copy
from pathlib import Path

import pytest
import yaml

from forecasting.development_plan import score_candidates, TUNING_CELLS
from forecasting.development_selection import aggregate_stage, select_stage


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())


def records(config, stage, score_fn=lambda candidate, cell, fold, seed: 1.):
    """Invent score records with a common scale; no real traffic is read."""
    result = []
    seeds = [2026, 2027, 2028] if stage == "lstm" else [None]
    for candidate in score_candidates(config, stage):
        for cell_index, cell in enumerate(TUNING_CELLS):
            for fold in range(3):
                for seed in seeds:
                    value = score_fn(candidate, cell_index, fold, seed)
                    result.append({"candidate_id": candidate["candidate_id"], "cell": cell,
                        "fold_index": fold, "seed": seed, "status": "passed", "mase_scale": 2.,
                        "scores": {"MAE": value*2, "RMSE": value*3, "MASE": value},
                        "selected_epochs": 20})
    return result


def test_equal_seed_cell_fold_arithmetic_and_rmse_not_pooled(config):
    rows = records(config, "lstm", lambda candidate, cell, fold, seed: 1+cell+10*fold+100*(seed-2026))
    # Counts vary: aggregation must not weight cells by the number of forecasts.
    for row in rows:
        row["target_count"] = 1 if row["cell"] == TUNING_CELLS[0] else 672
    summary = aggregate_stage(rows, config, "lstm")[0]
    assert summary["scores"] == pytest.approx({"MASE": 115, "MAE": 230, "RMSE": 345})
    assert [f["scores"]["MASE"] for f in summary["folds"]] == pytest.approx([105, 115, 125])
    assert summary["folds"][0]["cells"][0]["scores"]["MASE"] == pytest.approx(101)
    assert [r["scores"]["MASE"] for r in summary["seed_scores"]] == pytest.approx([15, 115, 215])
    assert summary["seed_score_sample_sd"]["MASE"] == pytest.approx(100)
    assert summary["record_count"] == 81


@pytest.mark.parametrize("stage,parameters", [
    ("ridge", {"alpha": 1000}), ("direct", {"trees": 32, "leaves": 7}),
    ("hybrid", {"weight": 0}), ("lstm", {"hidden_units": 16, "dropout": 0.}),
])
def test_declared_exact_tie_breakers(config, stage, parameters):
    selected = select_stage(records(config, stage), config, stage)
    assert selected["selected"]["parameters"] == parameters
    assert selected["tie_break"]["exact_tie"] is True
    assert selected["tie_break"]["rounded_before_selection"] is False
    assert selected["primary_comparator"] is None
    if stage == "hybrid":
        assert selected["ridge_preferred"] is True
        assert selected["hybrid_success_claimed"] is False


def test_positive_hybrid_ties_use_capacity_then_weight(config):
    rows = records(config, "hybrid", lambda candidate, *args: 3. if candidate["parameters"]["weight"] == 0 else 1.)
    result = select_stage(rows, config, "hybrid")
    assert result["selected"]["parameters"] == {"trees": 32, "leaves": 7, "weight": .25}
    assert result["ridge_preferred"] is False
    assert result["hybrid_success_claimed"] is False


def test_full_precision_not_display_rounding_decides_winner(config):
    rows = records(config, "ridge", lambda candidate, *args: 1. if candidate["parameters"]["alpha"] == .1 else 1.00000001)
    result = select_stage(rows, config, "ridge")
    assert result["selected"]["parameters"]["alpha"] == .1
    assert result["tie_break"]["exact_tie"] is False


def test_primary_metric_is_mase_not_rmse_or_runtime(config):
    rows = records(config, "ridge", lambda candidate, *args: 1. if candidate["parameters"]["alpha"] == 10 else 2.)
    for row in rows:
        row["scores"]["RMSE"] = 100 if row["candidate_id"] == "ridge-a10" else 6
        row["fit_seconds"] = 999 if row["candidate_id"] == "ridge-a10" else .001
    assert select_stage(rows, config, "ridge")["selected"]["parameters"]["alpha"] == 10


def test_global_choice_not_per_cell_optima(config):
    def value(candidate, cell, fold, seed):
        alpha = candidate["parameters"]["alpha"]
        return (0 if cell == 0 else 10) if alpha == .1 else (2 if alpha == 1 else 3)
    result = select_stage(records(config, "ridge", value), config, "ridge")
    assert result["selected"]["parameters"] == {"alpha": 1}


def test_lstm_epochs_use_all_selected_architecture_records_not_best_seed(config):
    rows = records(config, "lstm", lambda candidate, *args: 1 if candidate["parameters"]["hidden_units"] == 32 else 2)
    chosen = [row for row in rows if row["candidate_id"] == "lstm-h32-d0"]
    for index, row in enumerate(chosen):
        row["selected_epochs"] = 11 if index < 40 else (23 if index == 40 else 90)
    result = select_stage(rows, config, "lstm")
    assert result["selected"]["candidate_id"] == "lstm-h32-d0"
    assert len(result["selected"]["selected_epoch_values"]) == 81
    assert result["fixed_epochs"] == 23
    assert result["selected_seed"] is None
    assert result["prediction_ensemble"] is False


def test_baselines_are_preserved_without_selecting_primary_comparator(config):
    result = select_stage(records(config, "baselines"), config, "baselines")
    assert result["selected"] is None
    assert len(result["candidates"]) == 3
    assert result["primary_comparator"] is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "candidate", "cell", "fold", "seed",
    "float_seed", "bool_fold", "nan", "negative", "bool_score", "zero_scale", "mismatched_scale", "bad_mase", "missing_field",
    "zero_epochs", "over_budget", "float_epochs"])
def test_incomplete_malformed_or_mismatched_cohort_cannot_win(config, mutation):
    rows = records(config, "lstm")
    row = rows[0]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(row))
    elif mutation == "failed":
        row["status"] = "failed"
    elif mutation in ("candidate", "cell", "fold", "seed", "float_seed", "bool_fold"):
        key, value = {"candidate": ("candidate_id", "unknown"), "cell": ("cell", "unknown"),
            "fold": ("fold_index", 3), "seed": ("seed", 2029), "float_seed": ("seed", 2026.),
            "bool_fold": ("fold_index", False)}[mutation]
        row[key] = value
    elif mutation in ("nan", "negative", "bool_score"):
        row["scores"]["RMSE"] = {"nan": float("nan"), "negative": -1, "bool_score": True}[mutation]
    elif mutation == "zero_scale":
        row["mase_scale"] = 0
    elif mutation == "mismatched_scale":
        row["mase_scale"] = 4
        row["scores"]["MASE"] = .5
    elif mutation == "bad_mase":
        row["scores"]["MASE"] = 99
    elif mutation == "missing_field":
        del row["seed"]
    else:
        row["selected_epochs"] = {"zero_epochs": 0, "over_budget": 101, "float_epochs": 20.}[mutation]
    with pytest.raises(ValueError):
        select_stage(rows, config, "lstm")


def test_permutation_does_not_change_selection_or_mutate_records(config):
    rows = records(config, "ridge")
    original = copy.deepcopy(rows)
    result = select_stage(rows, config, "ridge")
    assert result == select_stage(list(reversed(rows)), config, "ridge")
    assert rows == original
