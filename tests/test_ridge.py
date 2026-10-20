"""Independent small calculations and boundary checks for ridge regression."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge

from models.ridge import fit_ridge, predict_ridge


def test_hand_calculated_slope_intercept_predictions_and_objective():
    features = np.array([[-1.], [0.], [1.]])
    actual = np.array([8., 10., 12.])
    model = fit_ridge(features, actual, alpha=1)
    predicted = predict_ridge(model, features)
    # Centering gives slope 4 / (2 + 1); the intercept remains the target mean.
    np.testing.assert_allclose(model.coef_, [4 / 3], rtol=1e-14)
    assert model.intercept_ == pytest.approx(10)
    np.testing.assert_allclose(predicted, [26 / 3, 10, 34 / 3], rtol=1e-14)
    objective = np.sum((actual - predicted) ** 2) + np.sum(model.coef_ ** 2)
    assert objective == pytest.approx(8 / 3)
    assert predicted.dtype == np.float64
    assert model.solver == "svd"
    assert model.fit_intercept and model.copy_X and not model.positive


def test_duplicating_rows_changes_sum_loss_balance():
    model = fit_ridge([[-1], [0], [1], [-1], [0], [1]],
                      [8, 10, 12, 8, 10, 12], alpha=1)
    # Sum loss gives 8/(4+1); averaging the loss would not give this answer.
    np.testing.assert_allclose(model.coef_, [8 / 5], rtol=1e-14)
    assert model.intercept_ == pytest.approx(10)


def test_zero_features_do_not_penalize_intercept():
    model = fit_ridge(np.zeros((3, 1)), [8, 10, 12], alpha=1000)
    np.testing.assert_array_equal(model.coef_, [0])
    assert model.intercept_ == pytest.approx(10)
    np.testing.assert_allclose(predict_ridge(model, [[0], [0]]), [10, 10])


def test_collinear_inputs_and_unbounded_standardized_predictions():
    features = [[-1, -1], [0, 0], [1, 1]]
    model = fit_ridge(features, [-2, 0, 2], alpha=1)
    np.testing.assert_allclose(model.coef_, [.8, .8], rtol=1e-14)
    assert model.intercept_ == pytest.approx(0, abs=1e-14)
    np.testing.assert_allclose(predict_ridge(model, features), [-1.6, 0, 1.6], atol=1e-14)
    # Both negative predictions and extrapolation beyond fitted targets survive.
    np.testing.assert_allclose(predict_ridge(model, [[-3, -3], [3, 3]]),
                               [-4.8, 4.8], rtol=1e-14)


def test_fit_and_prediction_preserve_input_arrays():
    features = np.array([[1., 2.], [3., 4.], [5., 6.]])
    actual = np.array([2., 4., 6.])
    original_features = features.copy()
    original_actual = actual.copy()
    model = fit_ridge(features, actual, np.float64(.1))
    predict_ridge(model, features)
    np.testing.assert_array_equal(features, original_features)
    np.testing.assert_array_equal(actual, original_actual)


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf, -np.inf,
                                  True, np.bool_(False), "1", [1], 1 + 0j])
def test_invalid_penalty_rejected(alpha):
    with pytest.raises(ValueError, match="Alpha"):
        fit_ridge([[-1], [0], [1]], [-2, 0, 2], alpha)


@pytest.mark.parametrize("features", [[], [1, 2], np.empty((0, 1)),
                                     np.empty((3, 0)), [[[1]]],
                                     [[np.nan]], [[np.inf]], [[1 + 1j]], [["1"]]])
def test_invalid_fitting_features_rejected(features):
    with pytest.raises(ValueError):
        fit_ridge(features, [1], 1)


@pytest.mark.parametrize("targets", [[], [1], [[1], [2]], [np.nan, 2],
                                    [1, np.inf], [1 + 1j, 2], ["1", "2"]])
def test_invalid_targets_rejected(targets):
    with pytest.raises(ValueError):
        fit_ridge([[0], [1]], targets, 1)


def test_only_fitted_single_output_ridge_is_accepted():
    with pytest.raises(ValueError, match="fitted sklearn Ridge"):
        predict_ridge(None, [[0]])
    with pytest.raises(NotFittedError):
        predict_ridge(Ridge(), [[0]])
    multi_output = Ridge().fit([[0], [1]], [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="one output"):
        predict_ridge(multi_output, [[0]])


@pytest.mark.parametrize("features", [[[0, 1]], [[np.nan]], [[np.inf]], [0], [[1j]]])
def test_invalid_prediction_features_rejected(features):
    model = fit_ridge([[0], [1]], [0, 1], 1)
    with pytest.raises(ValueError):
        predict_ridge(model, features)


def test_empty_prediction_preserves_shape_contract():
    model = fit_ridge([[0], [1]], [0, 1], 1)
    predicted = predict_ridge(model, np.empty((0, 1)))
    assert predicted.shape == (0,)
    assert predicted.dtype == np.float64
    with pytest.raises(ValueError, match="column count"):
        predict_ridge(model, np.empty((0, 2)))


def test_nonfinite_model_output_is_an_explicit_failure():
    model = fit_ridge([[0], [1]], [0, 1], 1)
    model.coef_[0] = np.inf
    with pytest.raises(ValueError, match="predictions must be finite"):
        predict_ridge(model, [[1]])
