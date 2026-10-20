"""Causal fitting, independent tree traversal and artifact checks on toy data."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation.metrics import finalize_predictions
from forecasting.data import TrafficScaler, make_examples, tabular_inputs
from models.lightgbm import fit_lightgbm, predict_lightgbm
from tools.check_lightgbm import check_module, fit_prefix, native_parameters, predict_dump


@pytest.fixture
def settings():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "run_config.yaml").read_text())["models"]["trees"]


def test_future_traffic_cannot_change_scaler_bins_trees_or_its_own_prediction(settings):
    index = np.arange(420)
    values = 100 + 20 * np.sin(index / 5) + index / 10
    observed = np.ones(420, dtype=bool)
    clock = pd.date_range("2020-01-01", periods=420, freq="h")
    model, scaler, train, _, _ = fit_prefix(values, observed, clock, 350, 2, 7, settings)
    changed = values.copy()
    changed[350:] += 10000
    other, other_scaler, other_train, _, _ = fit_prefix(changed, observed, clock, 350, 2, 7, settings)
    assert scaler == other_scaler
    assert model.model_to_string() == other.model_to_string()
    np.testing.assert_array_equal(train.actual, other_train.actual)
    x = tabular_inputs(make_examples(values, observed, clock, 350, 352), scaler)
    other_x = tabular_inputs(make_examples(changed, observed, clock, 350, 352), scaler)
    np.testing.assert_array_equal(x[0], other_x[0])
    np.testing.assert_array_equal(predict_lightgbm(model, x[:1]), predict_lightgbm(other, other_x[:1]))
    assert x[1, 167] != other_x[1, 167]  # The observation becomes history one hour later.


def test_exported_tree_walk_and_native_defaults_agree(settings):
    features = np.repeat([[-1.], [1.]], 64, axis=0)
    model = fit_lightgbm(features, np.repeat([0., 2.], 64), 2, 2, settings)
    np.testing.assert_allclose(predict_dump(model.dump_model(), features),
                               predict_lightgbm(model, features), atol=1e-14, rtol=1e-14)
    parameters = native_parameters(model.model_to_string())
    assert parameters["objective"] == "regression"
    assert parameters["boosting"] == "gbdt"
    assert parameters["num_threads"] == "1"
    assert parameters["boost_from_average"] == "1"
    assert parameters["early_stopping_round"] == "0"
    assert parameters["linear_tree"] == "0"
    assert parameters["zero_as_missing"] == "0"
    assert parameters["max_bin"] == "255"  # Recorded installed default, not a new tuning value.


def small_dump():
    # Hand-written leaves already contain initial-mean/shrinkage contributions.
    return {"num_class": 1, "num_tree_per_iteration": 1, "average_output": False,
            "objective": "regression", "max_feature_idx": 0, "tree_info": [
        {"shrinkage": .05, "tree_structure": {"split_index": 0, "split_feature": 0,
            "threshold": 0., "decision_type": "<=", "missing_type": "None",
            "left_child": {"leaf_value": -2.}, "right_child": {"leaf_value": 3.}}},
        {"shrinkage": .05, "tree_structure": {"leaf_value": .5}}]}


def test_manual_tree_boundary_addition_and_final_only_clipping():
    predicted_z = predict_dump(small_dump(), [[-1], [0], [1]])
    np.testing.assert_array_equal(predicted_z, [-1.5, -1.5, 3.5])
    raw = TrafficScaler(2, 2, 3, 3).inverse(predicted_z)
    final, clipping = finalize_predictions(raw)
    np.testing.assert_array_equal(raw, [-1, -1, 9])
    np.testing.assert_array_equal(final, [0, 0, 9])
    assert clipping["clipped_count"] == 2


@pytest.mark.parametrize("key,value", [("decision_type", "=="), ("missing_type", "Zero")])
def test_tree_walk_rejects_unsupported_split_types(key, value):
    dump = deepcopy(small_dump())
    dump["tree_info"][0]["tree_structure"][key] = value
    with pytest.raises(ValueError, match="Unexpected"):
        predict_dump(dump, [[0]])


def test_existing_evidence_is_rejected_before_loading_or_fitting(tmp_path):
    evidence = tmp_path / "checks.json"
    evidence.write_text("existing evidence")
    with pytest.raises(ValueError, match="new or empty"):
        check_module("data-that-does-not-exist", tmp_path)
    assert evidence.read_text() == "existing evidence"
