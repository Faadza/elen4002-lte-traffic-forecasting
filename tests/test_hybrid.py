"""Check residual direction, weighted addition and reuse of the ridge model."""

from pathlib import Path

import numpy as np
import pytest
import yaml

import models.hybrid as hybrid
from models.lightgbm import predict_lightgbm
from models.ridge import fit_ridge, predict_ridge


@pytest.fixture
def settings():
    config = Path(__file__).resolve().parents[1] / "run_config.yaml"
    return yaml.safe_load(config.read_text())["models"]["trees"]


def toy_ridge():
    features = np.repeat([[-1.], [1.]], 64, axis=0)
    targets = np.repeat([-2., 2.], 64)
    # X'X = 128, X'y = 256, alpha = 128: slope = 1, intercept = 0.
    ridge = fit_ridge(features, targets, alpha=128.)
    return features, targets, ridge


def test_residuals_use_actual_minus_unclipped_prediction_without_recentering():
    actual = np.array([-1., 2., 4.])
    predicted = np.array([-2., 3., 1.])
    np.testing.assert_array_equal(hybrid.training_residuals(actual, predicted), [1., -1., 3.])
    np.testing.assert_array_equal(actual, [-1., 2., 4.])
    np.testing.assert_array_equal(predicted, [-2., 3., 1.])


@pytest.mark.parametrize("weight,expected", [
    (.25, [-2.5, -.5, 4.25]), (1, [-4., 1., 2.]),
])
def test_weighted_addition_keeps_negative_corrections_and_predictions(weight, expected):
    ridge_z = np.array([-2., -1., 5.])
    correction_z = np.array([-2., 2., -3.])
    predicted = hybrid.combine_predictions(ridge_z, correction_z, weight)
    np.testing.assert_array_equal(predicted, expected)
    np.testing.assert_array_equal(ridge_z, [-2., -1., 5.])
    np.testing.assert_array_equal(correction_z, [-2., 2., -3.])


def test_hand_calculated_ridge_plus_one_residual_tree(settings):
    features, targets, ridge = toy_ridge()
    correction = hybrid.fit_correction(ridge, features, targets, 1, 2, settings)
    expected_ridge = np.repeat([-1., 1.], 64)
    # Residuals are -1/+1. Each leaf corrects .05 * 64 / (64 + 1).
    update = .05 * 64 / 65
    expected_correction = np.repeat([-update, update], 64)
    np.testing.assert_allclose(predict_ridge(ridge, features), expected_ridge, atol=1e-14)
    np.testing.assert_allclose(predict_lightgbm(correction, features), expected_correction,
                               rtol=1e-14, atol=1e-14)
    for weight in (.25, 1.):
        predicted = hybrid.predict_hybrid(ridge, correction, features, weight)
        np.testing.assert_allclose(predicted, expected_ridge + weight * expected_correction,
                                   rtol=1e-14, atol=1e-14)


def test_fit_passes_exact_residuals_and_original_features_without_refitting(settings, monkeypatch):
    features, targets, ridge = toy_ridge()
    expected = targets - predict_ridge(ridge, features)
    captured = {}
    sentinel = object()

    def capture(matrix, residuals, tree_count, leaf_cap, tree_settings):
        captured.update(matrix=matrix, residuals=residuals, trees=tree_count,
                        leaves=leaf_cap, settings=tree_settings)
        return sentinel

    def forbid_refit(*args, **kwargs):
        pytest.fail("Fitting the correction must reuse, not refit, ridge")

    monkeypatch.setattr(ridge, "fit", forbid_refit)
    monkeypatch.setattr(hybrid, "fit_lightgbm", capture)
    result = hybrid.fit_correction(ridge, features, targets, 32, 7, settings)
    assert result is sentinel
    assert captured["matrix"] is features
    np.testing.assert_array_equal(captured["residuals"], expected)
    assert captured["trees"] == 32 and captured["leaves"] == 7
    assert captured["settings"] is settings


def test_zero_weight_never_reads_or_predicts_correction(monkeypatch):
    features, _, ridge = toy_ridge()
    expected = predict_ridge(ridge, features)

    def forbid(*args, **kwargs):
        pytest.fail("Zero weight must not invoke the correction model")

    class Unreadable:
        def __array__(self, *args, **kwargs):
            forbid()

    monkeypatch.setattr(hybrid, "predict_lightgbm", forbid)
    np.testing.assert_array_equal(hybrid.predict_hybrid(ridge, None, features, 0), expected)
    combined = hybrid.combine_predictions(expected, Unreadable(), 0)
    np.testing.assert_array_equal(combined, expected)
    assert not np.shares_memory(combined, expected)


def test_fitting_and_prediction_leave_components_and_shared_inputs_unchanged(settings):
    features, targets, ridge = toy_ridge()
    original_features, original_targets = features.copy(), targets.copy()
    original_coef, original_intercept = ridge.coef_.copy(), ridge.intercept_.copy()
    original_settings = settings.copy()
    correction = hybrid.fit_correction(ridge, features, targets, 2, 2, settings)
    model_text = correction.model_to_string()
    first = hybrid.predict_hybrid(ridge, correction, features, .5)
    np.testing.assert_array_equal(first, hybrid.predict_hybrid(ridge, correction, features, .5))
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(targets, original_targets)
    np.testing.assert_array_equal(ridge.coef_, original_coef)
    assert ridge.intercept_ == original_intercept
    assert settings == original_settings and correction.model_to_string() == model_text


@pytest.mark.parametrize("weight", [-.1, 1.1, np.nan, np.inf, True, np.bool_(False), "1", [1]])
def test_invalid_weights_are_rejected_before_model_prediction(weight):
    with pytest.raises(ValueError, match="weight"):
        hybrid.combine_predictions([1.], [2.], weight)
    with pytest.raises(ValueError, match="weight"):
        hybrid.predict_hybrid(None, None, [[1.]], weight)


@pytest.mark.parametrize("first,second", [
    ([1., 2.], [1.]), ([[1.], [2.]], [1., 2.]),
    ([1., 2.], [[1.], [2.]]), ([1., np.nan], [1., 2.]),
    ([1., 2.], [1., np.inf]), ([1j], [1.]), (["1"], [1.]),
])
def test_malformed_vectors_cannot_broadcast_or_enter_arithmetic(first, second):
    with pytest.raises(ValueError):
        hybrid.training_residuals(first, second)
    with pytest.raises(ValueError):
        hybrid.combine_predictions(first, second, .5)


def test_overflow_in_residual_or_combination_is_an_explicit_failure():
    largest = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="Residual targets must be finite"):
        hybrid.training_residuals([largest], [-largest])
    with pytest.raises(ValueError, match="Hybrid predictions must be finite"):
        hybrid.combine_predictions([largest], [largest], 1.)


@pytest.mark.parametrize("targets", [[[1.]] * 128, [1.], np.full(128, np.nan)])
def test_bad_training_targets_fail_before_native_correction_fit(settings, monkeypatch, targets):
    features, _, ridge = toy_ridge()

    def forbid(*args, **kwargs):
        pytest.fail("Malformed residual targets must not enter native fitting")

    monkeypatch.setattr(hybrid, "fit_lightgbm", forbid)
    with pytest.raises(ValueError):
        hybrid.fit_correction(ridge, features, targets, 1, 2, settings)


def test_empty_prediction_and_wrong_column_contract(settings):
    features, targets, ridge = toy_ridge()
    correction = hybrid.fit_correction(ridge, features, targets, 1, 2, settings)
    for weight, model in [(0, None), (.5, correction)]:
        result = hybrid.predict_hybrid(ridge, model, np.empty((0, 1)), weight)
        assert result.shape == (0,) and result.dtype == np.float64
        with pytest.raises(ValueError, match="column count"):
            hybrid.predict_hybrid(ridge, model, np.empty((0, 2)), weight)
    with pytest.raises(ValueError, match="fitted native"):
        hybrid.predict_hybrid(ridge, None, features, 1.)
    with pytest.raises(ValueError, match="at least one"):
        hybrid.fit_correction(ridge, np.empty((0, 1)), [], 1, 2, settings)
