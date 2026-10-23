"""Invented-data tests for the frozen evaluation arithmetic and failure rules."""

import copy
import math

import numpy as np
import pandas as pd
import pytest

from final_evaluation.analysis import (
    STANDALONES, aggregate_method, compare_methods, paired_calendar_intervals,
    score_cell, select_primary_comparator,
)


def cell(raw=None, actual=None, eligible=None, observed=None, scale=2.0, history=None):
    actual = np.ones(24) * 10 if actual is None else np.asarray(actual, dtype=float)
    n = len(actual)
    raw = {"0": actual + 2} if raw is None else raw
    return score_cell(
        pd.date_range("2026-01-01", periods=n, freq="h"), actual,
        np.ones(n, dtype=bool) if observed is None else observed,
        np.ones(n, dtype=bool) if eligible is None else eligible,
        raw, scale, fully_observed_history=history,
    )


def summary(error=2):
    return aggregate_method({"A": cell(raw={"0": np.ones(24) * (10 + error)})}, ["A"])


def panel(n=12):
    y = np.arange(n, dtype=float) + 10
    return {
        "A": {"actual": y, "eligible": np.ones(n, dtype=bool), "weekly_scale": 2,
              "hybrid_by_seed": {"0": y + np.arange(n) % 4},
              "comparator_by_seed": {"2026": y + 2, "2027": y - 2, "2028": y + 4}},
        "B": {"actual": y, "eligible": np.arange(n) % 3 != 0, "weekly_scale": 4,
              "hybrid_by_seed": {"0": y + 1},
              "comparator_by_seed": {"2026": y + 1, "2027": y + 3, "2028": y - 2}},
    }


def test_scores_average_seed_rmse_instead_of_ensemble_forecast():
    result = cell(actual=[10, 10], raw={"2026": [8, 10], "2027": [12, 10]})
    assert result["scores"] == {"MAE": 1, "RMSE": math.sqrt(2), "MASE": .5}
    assert result["seed_count"] == 2
    # Averaging forecasts would be perfect, an explicitly forbidden ensemble.
    assert result["scores"]["MAE"] > 0


def test_clipping_precedes_scores_and_sensitivity_keeps_original_cohort():
    result = cell(actual=[1, 10, 10], raw={"0": [-3, 12, 100]},
                  eligible=[1, 1, 0], history=[1, 0, 1])
    assert result["scores"]["MAE"] == 1.5
    assert result["clipping"]["clipped_fraction"] == .5
    assert result["clipping"]["mean_upward_adjustment"] == 1.5
    assert result["fully_observed_history"]["scores"]["MAE"] == 1
    assert result["fully_observed_history"]["target_count"] == 1
    assert result["coverage"]["fraction_observed_scored"] == 2 / 3


def test_busiest_hour_ties_then_days_then_seeds():
    y = np.ones(48)
    y[[3, 4, 25]] = 10
    p, q = y.copy(), y.copy()
    p[[3, 4, 25]], q[[3, 4, 25]] = [8, 14, 1], [10, 10, 10]
    result = cell(actual=y, raw={"1": p, "2": q})
    assert result["busiest_hour"]["absolute_error"] == 3
    assert result["busiest_hour"]["signed_error"] == -2
    assert result["busiest_hour"]["qualifying_days"] == 2


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_failed_forecast_rejected_even_with_zero_scale(bad):
    with pytest.raises(ValueError):
        cell(actual=[1, 2], raw={"0": [1, bad]}, scale=0)


def test_missing_observation_cannot_become_a_target():
    with pytest.raises(ValueError, match="genuine"):
        cell(actual=[1, 2], observed=[1, 0])


def test_missing_lstm_seed_is_not_silently_accepted():
    with pytest.raises(ValueError, match="frozen seed"):
        score_cell(pd.date_range("2026-01-01", periods=2, freq="h"),
                   [1, 2], [1, 1], [1, 1], {2026: [1, 2]}, 1,
                   expected_seeds=[2026, 2027, 2028])


def test_zero_scale_preserves_unit_metrics_and_explicit_common_exclusion():
    a, b = cell(scale=0), cell(scale=2)
    result = aggregate_method({"A": a, "B": b}, ["A", "B"])
    assert result["scores"] == {"MAE": 2, "RMSE": 2, "MASE": 1}
    assert result["contributors"]["MAE"] == ["A", "B"]
    assert result["contributors"]["MASE"] == ["B"]
    assert result["structural_exclusions"]["MASE"] == {"A": "unavailable_fitting_scale"}


def test_equal_cells_not_pooled_targets_and_no_missing_cell_or_seed():
    a = cell(actual=[10], raw={"0": [12]})
    b = cell(actual=[10, 10, 10], raw={"0": [14, 14, 14]})
    assert aggregate_method({"A": a, "B": b}, ["A", "B"])["scores"]["MAE"] == 3
    with pytest.raises(ValueError, match="complete declared panel"):
        aggregate_method({"A": a}, ["A", "B"])
    del b["seed_scores"]["0"]
    with pytest.raises(ValueError, match="seed list"):
        aggregate_method({"A": a, "B": b}, ["A", "B"])


def test_no_target_cell_is_preserved_and_not_confused_with_model_failure():
    result = cell(actual=[1, 2], eligible=[0, 0], raw={"0": [np.nan, np.nan]})
    assert result["scores"] == {"MAE": None, "RMSE": None, "MASE": None}
    total = aggregate_method({"A": result}, ["A"])
    assert total["scores"]["MASE"] is None
    assert total["cell_count"] == 1
    result["status"] = "model_failed"
    with pytest.raises(ValueError, match="model failure"):
        aggregate_method({"A": result}, ["A"])


def test_confirmation_primary_metric_full_precision_and_fixed_order():
    summaries = {m: summary() for m in STANDALONES}
    summaries["lstm"]["scores"]["MASE"] -= 1e-12
    chosen = select_primary_comparator(summaries)
    assert chosen["method"] == "lstm"
    summaries["lstm"] = summary()
    assert select_primary_comparator(summaries)["method"] == "persistence"
    # Better RMSE cannot quietly replace the declared MASE rule.
    summaries["lstm"]["scores"]["RMSE"] = 0
    assert select_primary_comparator(summaries)["method"] == "persistence"


def test_confirmation_exact_tie_latency_only_when_all_tied_values_available():
    summaries = {m: summary() for m in STANDALONES}
    latency = {m: 10 for m in STANDALONES}
    latency["lstm"] = 1
    assert select_primary_comparator(summaries, latency)["method"] == "lstm"
    latency["ridge"] = 0
    assert select_primary_comparator(summaries, latency)["method"] == "ridge"
    del latency["ridge"]
    assert select_primary_comparator(summaries, latency)["method"] == "persistence"


def test_confirmation_refuses_incomplete_candidates_or_different_cohorts():
    summaries = {m: summary() for m in STANDALONES}
    del summaries["ridge"]
    with pytest.raises(ValueError, match="all standalone"):
        select_primary_comparator(summaries)
    summaries["ridge"] = summary()
    summaries["ridge"]["contributors"]["MASE"] = []
    with pytest.raises(ValueError, match="common MASE"):
        select_primary_comparator(summaries)


def test_comparison_reports_signed_difference_and_fraction_improved():
    result = compare_methods(summary(1), summary(2))
    assert result["MASE"]["hybrid_minus_comparator"] == -.5
    assert result["MAE"]["fraction_cells_improved"] == 1


def naive_bootstrap(data, replicates, blocks, seed):
    """Independent loop oracle that explicitly recomputes every seed's score."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(data.values()))["actual"])
    output = {}
    for length in blocks:
        values = []
        for _ in range(replicates):
            starts = rng.integers(0, n - length + 1, size=math.ceil(n / length))
            indices = np.concatenate([np.arange(start, start + length) for start in starts])[:n]
            cells = []
            for entry in data.values():
                selected = indices[entry["eligible"][indices]]
                scores = []
                for key in ("hybrid_by_seed", "comparator_by_seed"):
                    seeds = [np.mean(np.abs(np.maximum(0, p[selected]) - entry["actual"][selected]))
                             / entry["weekly_scale"] for p in entry[key].values()]
                    scores.append(np.mean(seeds))
                cells.append(scores[0] - scores[1])
            values.append(np.mean(cells))
        output[str(length)] = np.quantile(values, [.025, .975], method="linear")
    return output


def test_block_intervals_match_independent_recompute_seed_cell_calendar_oracle():
    data = panel()
    original = copy.deepcopy(data)
    result = paired_calendar_intervals(data, ["A", "B"], replicates=30,
                                      block_lengths=(3, 6), seed=72)
    expected = naive_bootstrap(data, 30, (3, 6), 72)
    for length in (3, 6):
        np.testing.assert_allclose(result["blocks"][str(length)]["interval"], expected[str(length)])
    np.testing.assert_array_equal(data["A"]["hybrid_by_seed"]["0"], original["A"]["hybrid_by_seed"]["0"])
    assert result == paired_calendar_intervals(data, ["A", "B"], replicates=30,
                                               block_lengths=(3, 6), seed=72)


def test_undefined_replicate_keeps_fixed_panel_without_redraw_or_pruning():
    data = panel(n=12)
    data["B"]["eligible"][:] = False
    data["B"]["eligible"][0] = True
    result = paired_calendar_intervals(data, ["A", "B"], replicates=50, block_lengths=(3,))
    assert result["point_status"] == "available"
    block = result["blocks"]["3"]
    assert block["status"] == "unavailable"
    assert 0 < block["undefined_replicates"] < 50
    assert block["interval"] is None
    assert result["expected_cells"] == ["A", "B"]


def test_undefined_scale_makes_all_replicates_unavailable_and_no_epsilon():
    data = panel()
    data["B"]["weekly_scale"] = 0
    result = paired_calendar_intervals(data, ["A", "B"], replicates=5, block_lengths=(3,))
    assert result["point_difference"] is None
    assert result["blocks"]["3"]["undefined_replicates"] == 5


def test_uncertainty_requires_complete_panel_and_full_calendar():
    data = panel()
    with pytest.raises(ValueError, match="complete fixed"):
        paired_calendar_intervals(data, ["A", "B", "C"])
    data["B"]["actual"] = data["B"]["actual"][:-1]
    data["B"]["eligible"] = data["B"]["eligible"][:-1]
    with pytest.raises(ValueError, match="calendar length"):
        paired_calendar_intervals(data, ["A", "B"])


def test_uncertainty_rejects_seed_loss_in_one_cell():
    data = panel()
    del data["B"]["comparator_by_seed"]["2028"]
    with pytest.raises(ValueError, match="seed lists"):
        paired_calendar_intervals(data, ["A", "B"])


def test_protocol_defaults_are_2000_joint_replicates_week_and_fortnight():
    data = panel(n=336)
    result = paired_calendar_intervals(data, ["A", "B"])
    assert result["seed"] == 20260928
    assert result["blocks"]["168"]["replicates"] == 2000
    assert result["blocks"]["336"]["replicates"] == 2000
    assert result["blocks"]["336"]["interval"][0] == pytest.approx(result["point_difference"])
    assert result["blocks"]["336"]["interval"][1] == pytest.approx(result["point_difference"])


def test_block_larger_than_calendar_is_explicitly_unavailable():
    result = paired_calendar_intervals(panel(), ["A", "B"], replicates=5, block_lengths=(168,))
    assert result["blocks"]["168"]["reason"] == "block_longer_than_calendar"
