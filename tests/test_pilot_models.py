"""Invented-data checks for the common pilot adapter and deployment artifacts."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from forecasting import pilot_models as pilot
from forecasting.data import make_examples, tabular_inputs, lstm_inputs
from forecasting.lstm_training import state_fingerprint
from models.ridge import predict_ridge


@pytest.fixture(scope="module")
def invented():
    config = yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())
    # Smaller budgets apply only to this invented fixture, never to measured data.
    config["models"]["trees"].update(tree_counts=[2], leaf_caps=[3], min_data_in_leaf=4)
    config["models"]["lstm"].update(maximum_epochs=1, patience_epochs=1,
                                      inner_stopping_tail_hours=24)
    positions = np.arange(320, dtype=np.float64)
    values = 20 + 2*np.sin(2*np.pi*positions/24) + positions/100
    observed = np.ones(320, dtype=bool)
    observed[200] = False
    clock = pd.date_range("2020-01-06", periods=len(values), freq="h")
    return config, values, observed, clock, 280


@pytest.fixture(scope="module")
def fitted(invented):
    config, values, observed, clock, fit_end = invented
    return {method: pilot.fit_model(method, values, observed, clock, fit_end, config)
            for method in pilot.METHODS}


@pytest.mark.parametrize("method", pilot.METHODS)
def test_all_methods_predict_save_load_and_preserve_state(invented, fitted, tmp_path, method):
    _, values, observed, clock, fit_end = invented
    bundle = fitted[method]
    targets = np.arange(fit_end, fit_end+3)
    before = pilot.model_fingerprint(bundle)
    raw, final = pilot.predict_model(bundle, values, observed, clock, targets)
    assert raw.shape == final.shape == (3,)
    np.testing.assert_array_equal(final, np.maximum(0, raw))
    assert pilot.model_fingerprint(bundle) == before
    saved = pilot.save_model(bundle, tmp_path / method)
    reloaded = pilot.load_model(saved)
    assert pilot.model_fingerprint(reloaded) == before
    raw_again, final_again = pilot.predict_model(reloaded, values, observed, clock, targets)
    np.testing.assert_array_equal(raw_again, raw)
    np.testing.assert_array_equal(final_again, final)
    assert reloaded.training_metadata == {}
    manifest = json.loads((saved / "model.json").read_text())
    assert manifest["settings"]["fit_end"] == fit_end
    assert set(manifest) == {"schema_version", "method", "settings", "scaler"}
    assert not any("history" in path.name or "selection" in path.name for path in saved.iterdir())
    if method == "lstm":
        assert {path.name for path in saved.iterdir()} == {"model.json", "weights.pt"}
        assert "stopping_history" in bundle.training_metadata
        assert not hasattr(bundle, "selection_model")
        weights = torch.load(saved / "weights.pt", map_location="cpu", weights_only=True)
        assert all(isinstance(value, torch.Tensor) for value in weights.values())


@pytest.mark.parametrize("method", pilot.METHODS)
def test_no_prediction_target_or_future_label_dependency(invented, fitted, method):
    _, values, observed, clock, fit_end = invented
    expected = pilot.predict_model(fitted[method], values, observed, clock, [fit_end])
    poisoned, flags = values.copy(), observed.astype(int)
    poisoned[fit_end:] = np.inf
    flags[fit_end:] = 9
    # Neither the current target label nor later traffic/masks are inspected.
    actual = pilot.predict_model(fitted[method], poisoned, flags, clock, [fit_end])
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("method", ["ridge", "direct_lightgbm", "ridge_lightgbm_hybrid", "lstm"])
def test_fitting_physically_excludes_later_values(invented, fitted, method):
    config, values, observed, clock, fit_end = invented
    poisoned, flags = values.copy(), observed.astype(int)
    poisoned[fit_end:] = np.inf
    flags[fit_end:] = 99
    # Invalid later values would fail existing full-array validators if exposed.
    actual = pilot.fit_model(method, poisoned, flags, clock, fit_end, config)
    assert pilot.model_fingerprint(actual) == pilot.model_fingerprint(fitted[method])
    assert actual.training_metadata == fitted[method].training_metadata


def test_prediction_features_exactly_match_existing_builder(invented, fitted):
    _, values, observed, clock, fit_end = invented
    expected = make_examples(values, observed, clock, fit_end, fit_end+3)
    actual = pilot.prediction_examples(values, observed, clock, expected.targets)
    for name in ("targets", "history", "missing", "calendar"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert actual.timestamps.equals(expected.timestamps)
    assert actual.actual.size == 0
    scaler = fitted["ridge"].scaler
    np.testing.assert_array_equal(tabular_inputs(actual, scaler), tabular_inputs(expected, scaler))
    for before, after in zip(lstm_inputs(actual, scaler), lstm_inputs(expected, scaler)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("method", pilot.METHODS)
def test_single_origin_and_batch_forecasts_agree(invented, fitted, method):
    _, values, observed, clock, fit_end = invented
    targets = np.arange(fit_end, fit_end+3)
    batch = pilot.predict_model(fitted[method], values, observed, clock, targets)[0]
    individual = np.array([pilot.predict_model(fitted[method], values, observed, clock, [int(t)])[0][0]
                           for t in targets])
    # Changing BLAS/native batch shapes may change floating-point accumulation.
    tolerance = 1e-5 if method == "lstm" else 1e-12
    np.testing.assert_allclose(individual, batch, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("method,lag", [("persistence", 1), ("daily_seasonal_naive", 24),
                                        ("weekly_seasonal_naive", 168)])
def test_baselines_need_only_their_named_lag(invented, fitted, method, lag):
    _, values, observed, clock, fit_end = invented
    sparse = np.full_like(values, np.nan)
    sparse[fit_end-lag] = values[fit_end-lag]
    raw, final = pilot.predict_model(fitted[method], sparse, observed, clock, [fit_end])
    np.testing.assert_array_equal(raw, [values[fit_end-lag]])
    np.testing.assert_array_equal(final, raw)
    assert fitted[method].model is None and fitted[method].scaler is None
    assert fitted[method].training_metadata["component_fit_counts"] == {}


def test_first_candidates_and_hybrid_includes_both_fits(invented, fitted):
    config, values, observed, clock, fit_end = invented
    hybrid = fitted["ridge_lightgbm_hybrid"]
    assert hybrid.settings["alpha"] == config["models"]["ridge"]["alphas"][0]
    assert hybrid.settings["requested_trees"] == config["models"]["trees"]["tree_counts"][0]
    assert hybrid.settings["leaf_cap"] == config["models"]["trees"]["leaf_caps"][0]
    assert hybrid.settings["weight"] == 1.0
    assert hybrid.training_metadata["component_fit_counts"] == {"ridge": 1, "correction": 1}
    assert fitted["lstm"].settings["hidden_units"] == 16
    assert fitted["lstm"].settings["dropout"] == 0
    assert fitted["lstm"].settings["seed"] == 2026
    examples = pilot.prediction_examples(values, observed, clock, [fit_end])
    features = tabular_inputs(examples, hybrid.scaler)
    np.testing.assert_array_equal(predict_ridge(hybrid.model, features),
                                  predict_ridge(fitted["ridge"].model, features))


def test_lstm_load_preserves_initialisation_rng(invented, fitted, tmp_path):
    pilot.save_model(fitted["lstm"], tmp_path)
    before = torch.get_rng_state().clone()
    restored = pilot.load_model(tmp_path)
    assert torch.equal(before, torch.get_rng_state())
    assert state_fingerprint(restored.model) == state_fingerprint(fitted["lstm"].model)


@pytest.mark.parametrize("bad_targets", [[279.0], [True], [[280]], [279, 279], [281, 280], [167], [320]])
def test_unsafe_learned_target_indices_are_rejected(invented, fitted, bad_targets):
    _, values, observed, clock, _ = invented
    with pytest.raises(ValueError, match="Targets"):
        pilot.predict_model(fitted["ridge"], values, observed, clock, bad_targets)


def test_missing_prediction_history_is_rejected(invented, fitted):
    _, values, observed, clock, fit_end = invented
    broken = values.copy()
    broken[fit_end-1] = np.nan
    with pytest.raises(ValueError, match="Requested histories"):
        pilot.predict_model(fitted["ridge"], broken, observed, clock, [fit_end])


def test_unsupported_array_conversion_is_not_hidden_in_prediction(invented, fitted):
    _, values, observed, clock, fit_end = invented
    with pytest.raises(ValueError, match="float64 NumPy"):
        pilot.predict_model(fitted["persistence"], values.tolist(), observed, clock, [fit_end])
    with pytest.raises(ValueError, match="float64 NumPy"):
        pilot.predict_model(fitted["ridge"], values.astype(np.float32), observed, clock, [fit_end])


def test_save_refuses_to_overwrite_existing_evidence(fitted, tmp_path):
    pilot.save_model(fitted["ridge"], tmp_path)
    original = (tmp_path / "model.json").read_bytes()
    with pytest.raises(ValueError, match="new or empty"):
        pilot.save_model(fitted["lstm"], tmp_path)
    assert (tmp_path / "model.json").read_bytes() == original


def test_load_refuses_extra_training_files(fitted, tmp_path):
    pilot.save_model(fitted["ridge"], tmp_path)
    (tmp_path / "training_history.json").write_text("{}")
    with pytest.raises(ValueError, match="model-only payload"):
        pilot.load_model(tmp_path)


def test_load_refuses_nonfinite_ridge_state(fitted, tmp_path):
    pilot.save_model(fitted["ridge"], tmp_path)
    path = tmp_path / "ridge.json"
    payload = json.loads(path.read_text())
    payload["intercept"] = float("inf")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite"):
        pilot.load_model(tmp_path)


def test_load_refuses_inconsistent_scaler_boundary(fitted, tmp_path):
    pilot.save_model(fitted["ridge"], tmp_path)
    path = tmp_path / "model.json"
    payload = json.loads(path.read_text())
    payload["settings"]["fit_end"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="boundary"):
        pilot.load_model(tmp_path)


@pytest.mark.parametrize("fit_end", [True, 1.5, 0, 321])
def test_invalid_fit_boundary_rejected(invented, fit_end):
    config, values, observed, clock, _ = invented
    with pytest.raises(ValueError, match="fit_end"):
        pilot.fit_model("ridge", values, observed, clock, fit_end, config)


def test_unknown_method_and_invalid_seed_rejected(invented):
    config, values, observed, clock, fit_end = invented
    with pytest.raises(ValueError, match="Unknown pilot method"):
        pilot.fit_model("other", values, observed, clock, fit_end, config)
    with pytest.raises(ValueError, match="only for"):
        pilot.fit_model("ridge", values, observed, clock, fit_end, config, seed=2026)
    with pytest.raises(ValueError, match="declared seed"):
        pilot.fit_model("lstm", values, observed, clock, fit_end, config, seed=999)
