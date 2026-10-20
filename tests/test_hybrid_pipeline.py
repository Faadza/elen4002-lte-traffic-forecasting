"""Check hybrid chronology, final-only clipping and component reconstruction."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
import yaml

from evaluation.metrics import finalize_predictions
from forecasting.data import TrafficScaler, make_examples, tabular_inputs
from models.hybrid import combine_predictions, predict_hybrid, training_residuals
from models.ridge import predict_ridge
from tools.check_hybrid import check_module, fit_prefix, manual_ridge, save_table
from tools.check_lightgbm import predict_dump


@pytest.fixture
def settings():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "run_config.yaml").read_text())["models"]["trees"]


def sample():
    index = np.arange(420)
    values = 100 + 20 * np.sin(index / 5) + index / 10
    observed = np.ones(420, dtype=bool)
    clock = pd.date_range("2020-01-01", periods=420, freq="h")
    return values, observed, clock


def test_future_values_and_masks_cannot_change_either_fitted_component(settings):
    values, observed, clock = sample()
    first = fit_prefix(values, observed, clock, 350, .1, 2, 7, settings)
    changed, changed_mask = values.copy(), observed.copy()
    changed[350:] += 10000
    changed_mask[400:] = False
    second = fit_prefix(changed, changed_mask, clock, 350, .1, 2, 7, settings)
    ridge, correction, scaler, train, train_x, train_y = first
    other, other_correction, other_scaler, other_train, other_x, other_y = second
    assert scaler == other_scaler
    np.testing.assert_array_equal(ridge.coef_, other.coef_)
    assert ridge.intercept_ == other.intercept_
    assert correction.model_to_string() == other_correction.model_to_string()
    np.testing.assert_array_equal(train.targets, other_train.targets)
    np.testing.assert_array_equal(train_x, other_x)
    np.testing.assert_array_equal(train_y, other_y)
    np.testing.assert_array_equal(training_residuals(train_y, predict_ridge(ridge, train_x)),
                                  training_residuals(other_y, predict_ridge(other, other_x)))
    x = tabular_inputs(make_examples(values, observed, clock, 350, 352), scaler)
    changed_x = tabular_inputs(make_examples(changed, changed_mask, clock, 350, 352), scaler)
    np.testing.assert_array_equal(x[0], changed_x[0])
    np.testing.assert_array_equal(predict_hybrid(ridge, correction, x[:1], 1),
                                  predict_hybrid(other, other_correction, changed_x[:1], 1))
    assert x[1, 167] != changed_x[1, 167]  # The new observation enters the next history.


def test_prefix_fits_each_component_once_on_identical_eligible_training_rows(settings, monkeypatch):
    import tools.check_hybrid as checker
    values, observed, clock = sample()
    observed[200] = False  # A causally prepared input may be present, but never a training label.
    values[200] = values[199]
    calls = {"ridge": 0, "correction": 0}
    original_base, original_correction = checker.fit_ridge_prefix, checker.fit_correction
    captured = {}

    def base(*args):
        calls["ridge"] += 1
        result = original_base(*args)
        captured["base"] = result
        return result

    def correction(model, features, targets, *args):
        calls["correction"] += 1
        assert model is captured["base"][0]
        assert features is captured["base"][3] and targets is captured["base"][4]
        assert 200 not in captured["base"][2].targets
        assert max(captured["base"][2].targets) < 350
        return original_correction(model, features, targets, *args)

    monkeypatch.setattr(checker, "fit_ridge_prefix", base)
    monkeypatch.setattr(checker, "fit_correction", correction)
    checker.fit_prefix(values, observed, clock, 350, .1, 2, 7, settings)
    assert calls == {"ridge": 1, "correction": 1}


def test_hybrid_inverts_once_and_clips_only_the_final_original_units():
    scaler = TrafficScaler(mean=10., sd=2., fit_end=10, observed_count=10)
    combined = combine_predictions([-6., -6., 2.], [2., -.5, -3.], 1.)
    raw = scaler.inverse(combined)
    final, diagnostics = finalize_predictions(raw)
    np.testing.assert_array_equal(combined, [-4., -6.5, -1.])
    np.testing.assert_array_equal(raw, [2., -3., 8.])
    np.testing.assert_array_equal(final, [2., 0., 8.])
    assert diagnostics["clipped_count"] == 1
    # Inverting each component would add the mean twice: that is not this model.
    assert not np.array_equal(raw, scaler.inverse([-6., -6., 2.]) + scaler.inverse([2., -.5, -3.]))


def test_saved_components_reconstruct_weighted_forecasts_and_ridge_identity(settings):
    values, observed, clock = sample()
    ridge, correction, scaler, _, _, _ = fit_prefix(values, observed, clock, 350, .1, 2, 7, settings)
    x = tabular_inputs(make_examples(values, observed, clock, 350, 355), scaler)
    metadata = json.loads(json.dumps({"coefficient": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)}))
    dump = json.loads(json.dumps(correction.dump_model()))
    restored = lgb.Booster(model_str=correction.model_to_string())
    for weight in (0., .25, 1.):
        expected = manual_ridge(x, metadata["coefficient"], metadata["intercept"]) + weight * predict_dump(dump, x)
        actual = predict_hybrid(ridge, restored, x, weight)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(predict_hybrid(ridge, None, x, 0), predict_ridge(ridge, x))


def test_saved_component_columns_and_residuals_round_trip_exactly(tmp_path):
    table = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=2, freq="h"),
                          "residual_z": [np.nextafter(.1, 1), -1.2345678901234567],
                          "target_index": [168, 169], "cell": ["Cell_A", "Cell_A"]})
    save_table(table, tmp_path / "training.csv")


def test_existing_evidence_is_rejected_before_loading_or_fitting(tmp_path):
    evidence = tmp_path / "checks.json"
    evidence.write_text("existing evidence")
    with pytest.raises(ValueError, match="new or empty"):
        check_module("data-that-does-not-exist", tmp_path)
    assert evidence.read_text() == "existing evidence"
