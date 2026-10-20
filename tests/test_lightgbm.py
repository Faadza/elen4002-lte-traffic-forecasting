"""Hand-calculated boosting steps and contracts for the standalone tree model."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from models.lightgbm import build_parameters, fit_lightgbm, predict_lightgbm


@pytest.fixture
def settings():
    config = Path(__file__).resolve().parents[1] / "run_config.yaml"
    return yaml.safe_load(config.read_text())["models"]["trees"]


def separated_groups():
    return np.repeat([[-1.], [1.]], 64, axis=0), np.repeat([0., 2.], 64)


def test_declared_settings_map_to_native_single_thread_regression(settings):
    original = settings.copy()
    parameters = build_parameters(settings, 32, 7)
    assert parameters == {
        "objective": "regression", "boosting_type": "gbdt", "device_type": "cpu",
        "deterministic": True, "force_col_wise": True, "feature_fraction": 1.0,
        "bagging_fraction": 1.0, "bagging_freq": 0, "learning_rate": .05,
        "min_data_in_leaf": 48, "lambda_l2": 1., "seed": 2026,
        "num_iterations": 32, "num_leaves": 7, "num_threads": 1, "verbosity": -1,
    }
    assert settings == original


def test_one_tree_matches_mean_plus_shrunk_leaf_update(settings):
    features, actual = separated_groups()
    model = fit_lightgbm(features, actual, 1, 2, settings)
    # Initial mean = 1. Each leaf has 64 residuals of +/-1 and L2 penalty 1.
    update = .05 * 64 / 65
    expected = np.repeat([1 - update, 1 + update], 64)
    predicted = predict_lightgbm(model, features)
    np.testing.assert_allclose(predicted, expected, rtol=1e-14)
    assert predicted.dtype == np.float64
    assert model.num_trees() == 1
    assert model.params["num_threads"] == 1
    tree = model.dump_model()["tree_info"][0]
    assert tree["num_leaves"] == 2
    assert tree["tree_structure"]["left_child"]["leaf_value"] == pytest.approx(1 - update)
    assert tree["tree_structure"]["right_child"]["leaf_value"] == pytest.approx(1 + update)


def test_second_tree_corrects_remaining_error_and_exported_leaves_sum(settings):
    features, actual = separated_groups()
    model = fit_lightgbm(features, actual, 2, 2, settings)
    first_update = .05 * 64 / 65
    # Ideal arithmetic; the tolerance allows standard float32-gradient rounding.
    second_update = .05 * 64 * (1 - first_update) / 65
    expected = np.repeat([1 - first_update - second_update,
                          1 + first_update + second_update], 64)
    predicted = predict_lightgbm(model, features)
    np.testing.assert_allclose(predicted, expected, rtol=1e-12, atol=1e-8)
    trees = model.dump_model()["tree_info"]
    assert len(trees) == 2
    # Exported leaf values already include shrinkage and the first tree's mean.
    left_sum = sum(tree["tree_structure"]["left_child"]["leaf_value"] for tree in trees)
    right_sum = sum(tree["tree_structure"]["right_child"]["leaf_value"] for tree in trees)
    np.testing.assert_allclose(predicted, np.repeat([left_sum, right_sum], 64), rtol=1e-14)


def test_constant_negative_targets_keep_raw_values_and_can_use_fewer_trees(settings):
    features, _ = separated_groups()
    model = fit_lightgbm(features, np.full(128, -2.), 8, 7, settings)
    np.testing.assert_array_equal(predict_lightgbm(model, features), np.full(128, -2.))
    assert model.num_trees() == 1
    assert model.params["num_iterations"] == 8


def test_identical_fits_repeat_and_neither_fit_nor_predict_mutates_inputs(settings):
    features, actual = separated_groups()
    original_features, original_actual = features.copy(), actual.copy()
    original_settings = settings.copy()
    first = fit_lightgbm(features, actual, 2, 2, settings)
    second = fit_lightgbm(features, actual, 2, 2, settings)
    first_predictions = predict_lightgbm(first, features)
    np.testing.assert_array_equal(first_predictions, predict_lightgbm(second, features))
    np.testing.assert_array_equal(first_predictions, predict_lightgbm(first, features))
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(actual, original_actual)
    assert settings == original_settings


def test_prediction_explicitly_requests_one_thread_and_all_raw_trees(settings, monkeypatch):
    features, actual = separated_groups()
    model = fit_lightgbm(features, actual, 1, 2, settings)
    original_predict = model.predict
    seen = {}

    def capture(data, **kwargs):
        seen.update(kwargs)
        return original_predict(data, **kwargs)

    monkeypatch.setattr(model, "predict", capture)
    predict_lightgbm(model, features)
    assert seen == {"raw_score": True, "num_iteration": -1, "num_threads": 1}


@pytest.mark.parametrize("name,value", [
    ("objective", "binary"), ("boosting_type", "dart"), ("device_type", "gpu"),
    ("deterministic", False), ("force_col_wise", False), ("feature_fraction", .9),
    ("bagging_fraction", .9), ("bagging_freq", 1), ("learning_rate", 0),
    ("learning_rate", np.nan), ("learning_rate", True), ("lambda_l2", -1),
    ("lambda_l2", np.inf), ("min_data_in_leaf", 0), ("min_data_in_leaf", 1.5),
    ("seed", True), ("seed", -1), ("seed", 2 ** 31), ("num_threads", 8),
])
def test_incompatible_or_invalid_settings_are_rejected(settings, name, value):
    settings[name] = value
    with pytest.raises(ValueError):
        build_parameters(settings, 2, 2)


def test_missing_settings_are_rejected(settings):
    del settings["lambda_l2"]
    with pytest.raises(ValueError, match="Missing"):
        build_parameters(settings, 2, 2)


@pytest.mark.parametrize("tree_count,leaf_cap", [(0, 2), (1.5, 2), (True, 2),
                                               (1, 1), (1, 2.5), (1, True)])
def test_invalid_tree_budgets_are_rejected(settings, tree_count, leaf_cap):
    with pytest.raises(ValueError):
        build_parameters(settings, tree_count, leaf_cap)


@pytest.mark.parametrize("features", [[], [1, 2], np.empty((0, 1)), np.empty((3, 0)),
                                     [[[1]]], [[np.nan]], [[np.inf]], [[1j]], [["1"]]])
def test_invalid_fitting_features_are_rejected(settings, features):
    with pytest.raises(ValueError):
        fit_lightgbm(features, [1], 1, 2, settings)


@pytest.mark.parametrize("targets", [[], [1], [[1], [2]], [np.nan, 2], [1, np.inf],
                                    [1j, 2], ["1", "2"], [1e300, -1e300]])
def test_invalid_targets_are_rejected_before_native_fitting(settings, targets):
    with pytest.raises(ValueError):
        fit_lightgbm([[0], [1]], targets, 1, 2, settings)


@pytest.mark.parametrize("features", [[[0, 1]], [[np.nan]], [[np.inf]], [0], [[1j]]])
def test_invalid_prediction_features_are_rejected(settings, features):
    matrix, actual = separated_groups()
    model = fit_lightgbm(matrix, actual, 1, 2, settings)
    with pytest.raises(ValueError):
        predict_lightgbm(model, features)


def test_empty_prediction_preserves_the_shape_contract(settings):
    features, actual = separated_groups()
    model = fit_lightgbm(features, actual, 1, 2, settings)
    predicted = predict_lightgbm(model, np.empty((0, 1)))
    assert predicted.shape == (0,) and predicted.dtype == np.float64
    with pytest.raises(ValueError, match="column count"):
        predict_lightgbm(model, np.empty((0, 2)))
    with pytest.raises(ValueError, match="fitted native"):
        predict_lightgbm(None, [[0]])


def test_nonfinite_predictions_are_an_explicit_failure(settings, monkeypatch):
    features, actual = separated_groups()
    model = fit_lightgbm(features, actual, 1, 2, settings)
    monkeypatch.setattr(model, "predict", lambda *args, **kwargs: np.full(128, np.inf))
    with pytest.raises(ValueError, match="predictions must be finite"):
        predict_lightgbm(model, features)
