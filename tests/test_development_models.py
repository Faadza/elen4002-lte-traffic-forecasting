"""Invented-data candidate routing, cached-ridge and causality checks."""
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from forecasting import development_models as development
from forecasting.data import fit_scaler, make_examples, scaled_targets, tabular_inputs
from forecasting.pilot_models import load_model, model_fingerprint, save_model
from models.hybrid import training_residuals
from models.ridge import fit_ridge, predict_ridge


@pytest.fixture(scope="module")
def invented():
    config = yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())
    # Short training is confined to these invented fixtures; grids stay declared.
    config["models"]["lstm"].update(maximum_epochs=1, patience_epochs=1, inner_stopping_tail_hours=24)
    t = np.arange(320, dtype=np.float64)
    values = 20 + 2*np.sin(2*np.pi*t/24) + t/100
    observed = np.ones(len(t), dtype=bool)
    observed[200] = False
    clock = pd.date_range("2020-01-06", periods=len(t), freq="h")
    return config, values, observed, clock


def job(method, parameters=None, seed=None, **changes):
    defaults = {"ridge": {"alpha": 100}, "direct_lightgbm": {"trees": 128, "leaves": 31},
                "ridge_lightgbm_hybrid": {"trees": 32, "leaves": 7},
                "lstm": {"hidden_units": 32, "dropout": .2}}
    row = {"id": f"{method}-fixture", "stage": development.STAGES[method], "method": method,
           "cell": "invented-cell", "fold_index": 0, "fit_end": 280,
           "score_start": 280, "score_end": 310, "candidate_id": f"{method}-settings",
           "parameters": defaults.get(method, {}) if parameters is None else parameters, "seed": seed}
    row.update(changes)
    return row


def fit(invented, candidate, **kwargs):
    config, values, observed, clock = invented
    return development.fit_candidate(candidate, values, observed, clock, config, **kwargs)


@pytest.fixture(scope="module")
def fitted(invented):
    ridge = fit(invented, job("ridge"))
    bundles = {method: fit(invented, job(method)) for method in development.BASELINES}
    bundles.update(ridge=ridge,
                   direct_lightgbm=fit(invented, job("direct_lightgbm")),
                   ridge_lightgbm_hybrid=fit(invented, job("ridge_lightgbm_hybrid"), ridge_bundle=ridge),
                   lstm=fit(invented, job("lstm", seed=2027)))
    return bundles


@pytest.mark.parametrize("method", development.STAGES)
def test_candidate_roundtrip_preserves_predictions_and_provenance(invented, fitted, tmp_path, method):
    _, values, observed, clock = invented
    model = fitted[method]
    before = model_fingerprint(model)
    expected = development.predict_candidate(model, values, observed, clock, [280, 281])
    save_model(model, tmp_path)
    restored = load_model(tmp_path)
    actual = development.predict_candidate(restored, values, observed, clock, [280, 281])
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    assert model_fingerprint(restored) == before == model_fingerprint(model)
    assert restored.settings["development_job"]["cell"] == "invented-cell"
    assert restored.settings["development_job"]["fold_index"] == 0
    if method not in development.BASELINES:
        assert restored.settings["training_target_count"] == 111


def test_explicit_ridge_candidate_matches_primitive_not_first_alpha(invented, fitted):
    _, values, observed, clock = invented
    model = fitted["ridge"]
    examples = make_examples(values[:280], observed[:280], clock[:280], 0, 280)
    scaler = fit_scaler(values[:280], observed[:280], 280)
    expected = fit_ridge(tabular_inputs(examples, scaler), scaled_targets(examples, scaler), 100)
    np.testing.assert_array_equal(model.model.coef_, expected.coef_)
    assert model.model.intercept_ == expected.intercept_
    assert model.settings["alpha"] == 100
    assert fitted["direct_lightgbm"].settings["requested_trees"] == 128
    assert fitted["direct_lightgbm"].settings["leaf_cap"] == 31
    assert fitted["lstm"].settings["hidden_units"] == 32
    assert fitted["lstm"].settings["dropout"] == .2
    assert fitted["lstm"].settings["seed"] == 2027
    assert fitted["lstm"].training_metadata["phase_timings"]["selection_seconds"] > 0
    assert fitted["lstm"].training_metadata["phase_timings"]["refit_seconds"] > 0


def test_hybrid_reuses_loaded_ridge_without_any_ridge_fit(invented, fitted, monkeypatch, tmp_path):
    original = fitted["ridge"]
    save_model(original, tmp_path)
    cached = load_model(tmp_path)
    before = model_fingerprint(cached)
    def forbidden(*args, **kwargs):
        raise AssertionError("The residual stage must not refit ridge")
    monkeypatch.setattr(development, "fit_ridge", forbidden)
    hybrid = fit(invented, job("ridge_lightgbm_hybrid"), ridge_bundle=cached)
    assert hybrid.model is cached.model
    assert hybrid.scaler == cached.scaler
    assert model_fingerprint(cached) == before
    assert hybrid.training_metadata["component_fit_counts"] == {"ridge": 0, "correction": 1}
    assert hybrid.training_metadata["ridge_reused"] is True
    assert hybrid.settings["ridge_dependency_job"] == cached.settings["development_job"]


def test_residual_fit_sees_unclipped_ridge_predictions(invented, fitted, monkeypatch):
    config, values, observed, clock = invented
    captured = {}
    original = development.fit_correction
    def spy(model, features, targets, trees, leaves, settings):
        captured["targets"] = targets.copy()
        captured["ridge_z"] = predict_ridge(model, features)
        captured["residuals"] = training_residuals(targets, captured["ridge_z"])
        return original(model, features, targets, trees, leaves, settings)
    monkeypatch.setattr(development, "fit_correction", spy)
    fit(invented, job("ridge_lightgbm_hybrid"), ridge_bundle=fitted["ridge"])
    assert (captured["ridge_z"] < 0).any()
    expected = captured["targets"] - captured["ridge_z"]
    np.testing.assert_array_equal(captured["residuals"], expected)
    assert not np.array_equal(expected, captured["targets"] - np.maximum(0, captured["ridge_z"]))


def test_all_weight_predictions_and_zero_weight_exact_identity(invented, fitted):
    config, values, observed, clock = invented
    hybrid = fitted["ridge_lightgbm_hybrid"]
    z, correction = development.predict_hybrid_components(hybrid, values, observed, clock, [280, 281])
    ridge_raw, ridge_final = development.predict_candidate(fitted["ridge"], values, observed, clock, [280, 281])
    for weight in config["models"]["hybrid"]["weights"]:
        raw, final = development.finalize_hybrid_components(hybrid, z, correction, weight, config)
        np.testing.assert_array_equal(raw, hybrid.scaler.inverse(z + weight*correction))
        np.testing.assert_array_equal(final, np.maximum(0, raw))
        if weight == 0:
            np.testing.assert_array_equal(raw, ridge_raw)
            np.testing.assert_array_equal(final, ridge_final)
        if weight == 1:
            expected = development.predict_candidate(hybrid, values, observed, clock, [280, 281])
            np.testing.assert_array_equal(raw, expected[0])
            np.testing.assert_array_equal(final, expected[1])


def test_no_intermediate_clipping_and_zero_needs_no_correction(invented, fitted):
    config = invented[0]
    hybrid = fitted["ridge_lightgbm_hybrid"]
    z, correction = np.array([-20., 1.]), np.array([10., -20.])
    raw, final = development.finalize_hybrid_components(hybrid, z, correction, .5, config)
    np.testing.assert_array_equal(raw, hybrid.scaler.inverse(np.array([-15., -9.])))
    np.testing.assert_array_equal(final, np.maximum(0, raw))
    raw0, _ = development.finalize_hybrid_components(hybrid, z, None, 0, config)
    np.testing.assert_array_equal(raw0, hybrid.scaler.inverse(z))


@pytest.mark.parametrize("method", ["ridge", "direct_lightgbm", "ridge_lightgbm_hybrid", "lstm"])
def test_no_future_fitting_dependency_or_input_mutation(invented, fitted, method):
    config, values, observed, clock = invented
    snapshot = copy.deepcopy(config)
    bad, flags = values.copy(), observed.astype(int)
    bad[280:], flags[280:] = np.inf, 99
    altered = config, bad, flags, clock
    candidate = job(method, seed=2027 if method == "lstm" else None)
    bundle = fit(altered, candidate, ridge_bundle=fitted["ridge"] if method == "ridge_lightgbm_hybrid" else None)
    assert model_fingerprint(bundle) == model_fingerprint(fitted[method])
    assert config == snapshot
    assert np.isinf(bad[280:]).all() and (flags[280:] == 99).all()
    assert bundle.scaler.mean == np.mean(values[:280][observed[:280]])


def test_hybrid_component_prediction_does_not_read_current_or_future_target(invented, fitted):
    _, values, observed, clock = invented
    bundle = fitted["ridge_lightgbm_hybrid"]
    expected = development.predict_hybrid_components(bundle, values, observed, clock, [280])
    bad, flags = values.copy(), observed.astype(int)
    bad[280:], flags[280:] = np.inf, 99
    actual = development.predict_hybrid_components(bundle, bad, flags, clock, [280])
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("edit", [
    {"parameters": {"alpha": True}}, {"parameters": {"alpha": "100"}},
    {"parameters": {"alpha": np.nan}}, {"parameters": {"alpha": 2}},
    {"parameters": {"alpha": 100, "extra": 1}}, {"parameters": {}},
    {"seed": 2026}, {"stage": "hybrid"}, {"fold_index": True},
    {"fit_end": 280.0}, {"score_start": 281}, {"score_end": 280},
    {"id": ""}, {"cell": ""}, {"unexpected": True},
])
def test_invalid_job_rejected_before_fitting(invented, monkeypatch, edit):
    candidate = job("ridge", **edit)
    monkeypatch.setattr(development, "fit_ridge", lambda *args: pytest.fail("Invalid candidate fitted"))
    with pytest.raises(ValueError):
        fit(invented, candidate)


@pytest.mark.parametrize("method,parameters,seed", [
    ("direct_lightgbm", {"trees": True, "leaves": 7}, None),
    ("direct_lightgbm", {"trees": 32., "leaves": 7}, None),
    ("direct_lightgbm", {"trees": 33, "leaves": 7}, None),
    ("direct_lightgbm", {"trees": 32, "leaves": 8}, None),
    ("lstm", {"hidden_units": 17, "dropout": 0.}, 2026),
    ("lstm", {"hidden_units": 16, "dropout": True}, 2026),
    ("lstm", {"hidden_units": 16, "dropout": .3}, 2026),
    ("lstm", {"hidden_units": 16, "dropout": 0.}, None),
    ("lstm", {"hidden_units": 16, "dropout": 0.}, 2026.),
    ("lstm", {"hidden_units": 16, "dropout": 0.}, 999),
    ("persistence", {"lag": 2}, None),
])
def test_undeclared_candidates_rejected(invented, method, parameters, seed):
    with pytest.raises(ValueError):
        fit(invented, job(method, parameters, seed))


@pytest.mark.parametrize("change", ["cell", "fold", "prefix", "scaler", "alpha", "count", "report", "method"])
def test_incompatible_cached_ridge_rejected(invented, fitted, change):
    original = fitted["ridge"]
    settings, metadata = copy.deepcopy(original.settings), copy.deepcopy(original.training_metadata)
    bundle = replace(original, settings=settings, training_metadata=metadata)
    if change == "cell":
        settings["development_job"]["cell"] = "another-cell"
    elif change == "fold":
        settings["development_job"]["fold_index"] = 1
    elif change == "prefix":
        settings["fit_end"] = 279
    elif change == "scaler":
        bundle = replace(bundle, scaler=replace(bundle.scaler, mean=bundle.scaler.mean+1))
    elif change == "alpha":
        settings["alpha"] = .1
    elif change == "count":
        settings["training_target_count"] -= 1
    elif change == "report":
        metadata["training_target_count"] -= 1
    else:
        bundle = replace(bundle, method="direct_lightgbm")
    with pytest.raises(ValueError, match="ridge"):
        fit(invented, job("ridge_lightgbm_hybrid"), ridge_bundle=bundle)


def test_hybrid_dependency_is_required_and_exclusive(invented, fitted):
    with pytest.raises(ValueError, match="ridge"):
        fit(invented, job("ridge_lightgbm_hybrid"))
    with pytest.raises(ValueError, match="Only the hybrid"):
        fit(invented, job("ridge"), ridge_bundle=fitted["ridge"])


@pytest.mark.parametrize("weight", [True, .1, "0", np.inf, -1])
def test_weight_must_belong_to_declared_grid(invented, fitted, weight):
    with pytest.raises(ValueError):
        development.finalize_hybrid_components(fitted["ridge_lightgbm_hybrid"], [1.], [2.], weight, invented[0])


def test_lstm_epoch_cap_cannot_be_enlarged(invented):
    config, *arrays = invented
    changed = copy.deepcopy(config)
    changed["models"]["lstm"]["maximum_epochs"] = 101
    with pytest.raises(ValueError, match="maximum_epochs"):
        fit((changed, *arrays), job("lstm", seed=2026))
