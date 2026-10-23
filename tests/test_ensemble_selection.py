"""Declared two-objective searches, endpoint rules and zero-fit replay."""
import copy

import numpy as np
import pytest

from ensemble_study.selection import (
    BASELINES, CAPACITIES, OBJECTIVES, STANDALONES, WEIGHTS, build_selection,
    condition_for_weight, make_condition, select_ensembles, validate_condition,
)
from ensemble_study.replay import (
    FOLD_SCORE_INTERVALS, capacity_records, replay_study, replay_weights, score_raw_predictions,
)

CELLS = ("A", "B")
FOLDS = (0, 1, 2)


def cohorts(actual=(10., 10.), ridge=(8., 8.), direct=(12., 12.)):
    result = {}
    for objective in OBJECTIVES:
        for index, (trees, leaves) in enumerate(CAPACITIES):
            for cell in CELLS:
                for fold in FOLDS:
                    start, _ = FOLD_SCORE_INTERVALS[fold]
                    result[(objective, trees, leaves, cell, fold)] = {
                        "targets": np.arange(start, start + len(actual)),
                        "actual": np.asarray(actual), "ridge_raw": np.asarray(ridge),
                        "direct_raw": np.asarray(direct) + index * .1,
                        "weekly_scale": 2.,
                    }
    return result


def records():
    return capacity_records(cohorts(), CELLS)


def test_full_two_objective_grid_selects_components_before_weights():
    selected = build_selection(records(), CELLS)
    assert selected["record_count"] == 2 * 6 * 2 * 3
    for objective in OBJECTIVES:
        winner = selected["objectives"][objective]["selected"]
        assert (winner["trees"], winner["leaves"]) == (32, 7)
        assert winner["scores"] == {"MAE": 2., "RMSE": 2., "MASE": 1.}
        assert len(winner["folds"]) == 3


def test_capacity_exact_ties_fewer_rounds_then_leaves_without_rounding():
    all_records = records()
    for record in all_records:
        record["scores"] = {"MAE": 2., "RMSE": 2., "MASE": 1.}
    selected = build_selection(all_records, CELLS)
    assert selected["objectives"]["l1"]["selected"]["trees"] == 32
    assert selected["objectives"]["l1"]["selected"]["leaves"] == 7
    for record in all_records:
        if record["objective"] == "l1" and (record["trees"], record["leaves"]) == (512, 31):
            record["scores"]["MASE"] -= 1e-12
    selected = build_selection(all_records, CELLS)
    assert selected["objectives"]["l1"]["selected"]["trees"] == 512
    assert selected["objectives"]["l1"]["selected"]["leaves"] == 31


def test_supporting_metric_cannot_replace_primary_selection_metric():
    all_records = records()
    for record in all_records:
        if (record["trees"], record["leaves"]) == (512, 31):
            record["scores"]["RMSE"] = 0
    assert build_selection(all_records, CELLS)["objectives"]["l2"]["selected"]["trees"] == 32


@pytest.mark.parametrize("change", ["missing", "duplicate", "missing_objective", "nan", "extra_cell"])
def test_incomplete_or_bad_candidate_never_wins_on_successful_subset(change):
    all_records = records()
    if change == "missing":
        all_records.pop()
    elif change == "duplicate":
        all_records.append(copy.deepcopy(all_records[0]))
    elif change == "missing_objective":
        all_records = [r for r in all_records if r["objective"] == "l2"]
    elif change == "nan":
        all_records[0]["scores"]["MASE"] = np.nan
    else:
        all_records[0]["cell"] = "C"
    with pytest.raises(ValueError):
        build_selection(all_records, CELLS)


def test_equal_cell_then_equal_fold_metrics_not_pooled_targets():
    all_records = records()
    for record in all_records:
        error = 2 if record["cell"] == "A" else 4
        record["scores"] = {"MAE": error, "RMSE": error, "MASE": error / 2}
        record["target_count"] = 1 if record["cell"] == "A" else 100
    selected = build_selection(all_records, CELLS)
    assert selected["objectives"]["l2"]["selected"]["scores"]["MAE"] == 3


def test_two_equal_blends_are_both_retained_and_overall_tie_prefers_l2():
    result = replay_study(cohorts(), CELLS)
    selection = result["selection"]
    assert selection["selected_weights"] == {"l2": .5, "l1": .5}
    assert selection["primary_candidate"] == "ensemble_l2_r0p5"
    assert {c["id"] for c in selection["conditions"]} == set(STANDALONES) | {
        "ensemble_l2_r0p5", "ensemble_l1_r0p5"}
    assert len(result["weight_records"]) == 2 * 8 * 2 * 3
    assert result["new_model_fits"] == 0


def test_losing_equal_blends_not_eliminated_and_at_most_eleven_conditions():
    result = replay_study(cohorts(direct=(10.5, 10.5)), CELLS)
    selection = result["selection"]
    assert selection["selected_weights"] == {"l2": .2, "l1": .2}
    assert len(selection["conditions"]) == 11
    assert {"ensemble_l2_r0p5", "ensemble_l1_r0p5"} <= {c["id"] for c in selection["conditions"]}
    assert selection["primary_is_interior_ensemble"]


def test_direct_endpoint_can_win_without_forcing_ensemble_improvement():
    selection = replay_study(cohorts(direct=(10., 10.)), CELLS)["selection"]
    assert selection["selected_weights"] == {"l2": 0., "l1": 0.}
    assert selection["primary_candidate"] == "lightgbm_l2"
    assert not selection["primary_is_interior_ensemble"]
    assert len(selection["conditions"]) == 9


def test_ridge_endpoint_is_one_shared_condition_for_both_objectives():
    selection = replay_study(cohorts(ridge=(10., 10.)), CELLS)["selection"]
    assert selection["selected_weights"] == {"l2": 1., "l1": 1.}
    assert selection["primary_candidate"] == "ridge"
    assert sum(c["id"] == "ridge" for c in selection["conditions"]) == 1
    assert condition_for_weight("l1", 32, 7, 1) == condition_for_weight("l2", 512, 31, 1)


def test_endpoint_tie_precedes_smaller_interior_weight():
    # For the selected component, r=.5,.75,1 all give MAE=1. Endpoint r=1 wins.
    selection = replay_study(cohorts(actual=(0., 2.), ridge=(0., 0.), direct=(3., 3.)), CELLS)["selection"]
    assert selection["selected_weights"] == {"l2": 1., "l1": 1.}


def test_tied_endpoints_prefer_smaller_weight_zero():
    data = cohorts(ridge=(10., 10.), direct=(10., 10.))
    selection = replay_study(data, CELLS)["selection"]
    assert selection["selected_weights"] == {"l2": 0., "l1": 0.}


def test_overall_tie_smaller_weight_precedes_objective_preference():
    data = cohorts()
    for (objective, trees, leaves, cell, fold), entry in data.items():
        if objective == "l1":
            entry["direct_raw"] = np.ones(2) * (10.5 + CAPACITIES.index((trees, leaves)) * .1)
    selection = replay_study(data, CELLS)["selection"]
    assert selection["selected_weights"] == {"l2": .5, "l1": .2}
    assert selection["primary_candidate"] == "ensemble_l1_r0p2"


def test_overall_tie_endpoint_precedes_objective_preference():
    data = cohorts()
    for (objective, trees, leaves, cell, fold), entry in data.items():
        if objective == "l1":
            entry["direct_raw"] = np.ones(2) * (10 + CAPACITIES.index((trees, leaves)) * .1)
    selection = replay_study(data, CELLS)["selection"]
    assert selection["selected_weights"] == {"l2": .5, "l1": 0.}
    assert selection["primary_candidate"] == "lightgbm_l1"


def test_combines_raw_predictions_before_clipping():
    result = replay_study(cohorts(actual=(1., 1.), ridge=(-10., -10.), direct=(10., 10.)), CELLS)
    half = next(r for r in result["weight_records"] if r["ridge_weight"] == .5)
    assert half["scores"]["MAE"] == 1.  # raw blend=0; clipping each component first would give MAE4.


@pytest.mark.parametrize("field,value", [
    ("actual", [11., 10.]), ("ridge_raw", [7., 8.]),
    ("weekly_scale", 3.), ("targets", [4117, 4118]),
])
def test_replay_rejects_noncommon_targets_actuals_ridge_or_denominator(field, value):
    data = cohorts()
    data[("l1", 32, 7, "A", 0)][field] = value
    with pytest.raises(ValueError, match="share targets"):
        replay_study(data, CELLS)


def test_replay_rejects_missing_capacity_and_holdout_hours():
    data = cohorts()
    data.pop(next(iter(data)))
    with pytest.raises(ValueError, match="complete six-capacity"):
        replay_study(data, CELLS)
    data = cohorts()
    data[("l1", 32, 7, "A", 0)]["targets"] = [6132, 6133]
    with pytest.raises(ValueError, match="development fold"):
        replay_study(data, CELLS)


def test_weight_search_rejects_changed_component_capacity_or_prior_selection():
    data = cohorts()
    selected = build_selection(capacity_records(data, CELLS), CELLS)
    weights = replay_weights(data, selected, CELLS)
    weights[0]["trees"] = 128
    with pytest.raises(ValueError, match="already selected"):
        select_ensembles(selected, weights, CELLS)
    selected["objectives"]["l1"]["selected"]["trees"] = 128
    with pytest.raises(ValueError, match="exactly reproduce"):
        replay_weights(data, selected, CELLS)


def test_weight_zero_exactly_reproduces_standalone_and_one_ridge():
    data = cohorts()
    selected = build_selection(capacity_records(data, CELLS), CELLS)
    weights = replay_weights(data, selected, CELLS)
    weights[0]["scores"]["MASE"] += .01
    with pytest.raises(ValueError, match="Weight zero"):
        select_ensembles(selected, weights, CELLS)
    weights = replay_weights(data, selected, CELLS)
    last = next(r for r in weights if r["objective"] == "l1" and r["ridge_weight"] == 1)
    last["scores"]["MASE"] += .01
    with pytest.raises(ValueError, match="same standalone ridge"):
        select_ensembles(selected, weights, CELLS)


def test_lstm_seed_scores_not_forecast_ensemble():
    result = score_raw_predictions([10, 10], {2026: [8, 10], 2027: [12, 10], 2028: [10, 10]},
                                   2, expected_seeds=(2026, 2027, 2028))
    assert result["scores"]["MAE"] == pytest.approx(2 / 3)
    assert result["scores"]["RMSE"] == pytest.approx(2 * np.sqrt(2) / 3)
    assert result["seed_count"] == 3
    with pytest.raises(ValueError, match="declared seeds"):
        score_raw_predictions([10], {2026: [10]}, 2, expected_seeds=(2026, 2027, 2028))


def test_condition_schema_is_canonical_and_rejects_false_endpoint_encodings():
    assert condition_for_weight("l2", 128, 7, .05)["id"] == "ensemble_l2_r0p05"
    for identifier in (*BASELINES, "ridge", "lstm"):
        value = make_condition(identifier)
        assert validate_condition(value) == value
    for weight in WEIGHTS:
        assert validate_condition(condition_for_weight("l1", 32, 31, weight))
    with pytest.raises(ValueError):
        make_condition("blend", objective="l2", trees=32, leaves=7, ridge_weight=1)
    with pytest.raises(ValueError):
        make_condition("ridge", ridge_weight=True)
    bad = make_condition("ridge")
    bad["id"] = "ensemble_l1_r1"
    with pytest.raises(ValueError):
        validate_condition(bad)


def test_replay_does_not_mutate_supplied_arrays():
    data = cohorts()
    before = copy.deepcopy(data)
    replay_study(data, CELLS)
    for key in data:
        for field in data[key]:
            np.testing.assert_array_equal(data[key][field], before[key][field])
