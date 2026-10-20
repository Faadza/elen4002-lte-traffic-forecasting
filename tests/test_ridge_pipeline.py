"""Check the learned model's causal boundary and final-unit conversion."""
import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import finalize_predictions
from forecasting.data import TrafficScaler, make_examples, tabular_inputs
from models.ridge import fit_ridge, predict_ridge
from tools.check_ridge import check_module, fit_prefix


def test_evaluation_values_cannot_change_fit_or_their_own_prediction():
    index = np.arange(420)
    values = 100 + 20 * np.sin(index / 5) + index / 10
    observed = np.ones(420, dtype=bool)
    timestamps = pd.date_range("2020-01-01", periods=420, freq="h")
    fit_end = 350
    model, scaler, train, _, _ = fit_prefix(values, observed, timestamps, fit_end, 1)
    altered = values.copy()
    altered[fit_end:] += 10000
    other, other_scaler, other_train, _, _ = fit_prefix(altered, observed, timestamps, fit_end, 1)
    assert other_scaler == scaler
    np.testing.assert_array_equal(other.coef_, model.coef_)
    assert other.intercept_ == model.intercept_
    np.testing.assert_array_equal(other_train.actual, train.actual)
    original_score = make_examples(values, observed, timestamps, fit_end, fit_end + 2)
    altered_score = make_examples(altered, observed, timestamps, fit_end, fit_end + 2)
    x = tabular_inputs(original_score, scaler)
    altered_x = tabular_inputs(altered_score, scaler)
    np.testing.assert_array_equal(x[0], altered_x[0])
    # The newly observed first target may enter the NEXT hour's history.
    assert x[1, 167] != altered_x[1, 167]
    first = predict_ridge(model, x[:1])
    np.testing.assert_array_equal(first, predict_ridge(other, altered_x[:1]))


def test_inverse_once_then_clip_preserves_raw_and_future_hybrid_residuals():
    features = [[-1, -1], [0, 0], [1, 1]]
    actual_z = np.array([-2., 0., 2.])
    model = fit_ridge(features, actual_z, 1)
    residual = actual_z - predict_ridge(model, features)
    np.testing.assert_allclose(residual, [-.4, 0, .4], atol=1e-14)
    predicted_z = predict_ridge(model, [[-3, -3], [3, 3]])
    scaler = TrafficScaler(mean=2, sd=1, fit_end=3, observed_count=3)
    raw = scaler.inverse(predicted_z)
    final, clipping = finalize_predictions(raw)
    np.testing.assert_allclose(predicted_z, [-4.8, 4.8], atol=1e-14)
    np.testing.assert_allclose(raw, [-2.8, 6.8], atol=1e-14)
    np.testing.assert_allclose(final, [0, 6.8], atol=1e-14)
    assert clipping["clipped_count"] == 1


def test_existing_evidence_is_rejected_before_loading_or_fitting(tmp_path):
    evidence = tmp_path / "checks.json"
    evidence.write_text("existing evidence")
    with pytest.raises(ValueError, match="new or empty"):
        check_module("data-that-does-not-exist", tmp_path)
    assert evidence.read_text() == "existing evidence"
