"""Invented-input checks for frozen complete fits; no LTE data are accessed."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from blend_revision.model import blend_raw_predictions
from final_evaluation import holdout_models as selected
from forecasting.data import scaled_targets, tabular_inputs
from forecasting.pilot_models import predict_model
from models.hybrid import training_residuals
from models.ridge import predict_ridge


@pytest.fixture(scope="module")
def invented():
    config = yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())
    t = np.arange(360, dtype=np.float64)
    values = 30 + 3*np.sin(2*np.pi*t/24) + 2*np.cos(2*np.pi*t/168) + t/100
    observed = np.ones(len(t), dtype=bool)
    observed[200] = False
    clock = pd.date_range("2020-01-06", periods=len(t), freq="h")
    return config, values, observed, clock, 330


@pytest.fixture(scope="module")
def fitted(invented):
    config, values, observed, clock, end = invented
    return {method: selected.fit_selected(method, values, observed, clock, end, config,
                                          seed=2026 if method == "lstm" else None)
            for method in selected.METHODS}


@pytest.mark.parametrize("method", selected.METHODS)
def test_complete_fit_roundtrip_and_single_origin_equivalence(invented, fitted, tmp_path, method):
    _, values, observed, clock, end = invented
    bundle, stats = fitted[method]
    targets = np.arange(end, end+3)
    before = selected.selected_fingerprint(bundle)
    expected = selected.predict_selected(bundle, values, observed, clock, targets)
    np.testing.assert_array_equal(expected[1], np.maximum(0, expected[0]))
    singles = np.array([selected.predict_selected(bundle, values, observed, clock, [int(t)])[0][0]
                        for t in targets])
    tolerance = 1e-5 if method == "lstm" else 1e-12
    np.testing.assert_allclose(singles, expected[0], rtol=tolerance, atol=tolerance)
    saved = selected.save_selected(bundle, tmp_path / method)
    restored = selected.load_selected(saved)
    assert selected.selected_fingerprint(restored) == before
    assert selected.selected_fingerprint(bundle) == before
    for actual, wanted in zip(selected.predict_selected(restored, values, observed, clock, targets), expected):
        np.testing.assert_array_equal(actual, wanted)
    assert stats["fit_seconds"] >= 0
    assert stats["fit_interval"] == [0, end]
    assert stats["cached_component_reused"] is False
    assert "feature construction" in stats["fit_scope"]


@pytest.mark.parametrize("method", selected.METHODS)
def test_current_and_future_targets_never_enter_prediction(invented, fitted, method):
    _, values, observed, clock, end = invented
    bundle, _ = fitted[method]
    expected = selected.predict_selected(bundle, values, observed, clock, [end])
    poison, flags = values.copy(), observed.astype(int)
    poison[end:], flags[end:] = np.inf, 99
    for actual, wanted in zip(selected.predict_selected(bundle, poison, flags, clock, [end]), expected):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("method", ["ridge", "direct_lightgbm", selected.ORIGINAL_HYBRID,
                                     selected.BLEND_METHOD, "lstm"])
def test_later_values_physically_excluded_from_all_learned_fits(invented, fitted, method):
    config, values, observed, clock, end = invented
    poison, flags = values.copy(), observed.astype(int)
    poison[end:], flags[end:] = np.inf, 99
    config_before = copy.deepcopy(config)
    refit, stats = selected.fit_selected(method, poison, flags, clock, end, config,
                                        seed=2026 if method == "lstm" else None)
    assert selected.selected_fingerprint(refit) == selected.selected_fingerprint(fitted[method][0])
    assert config == config_before
    assert np.isinf(poison[end:]).all() and (flags[end:] == 99).all()
    assert stats["component_fit_counts"] == fitted[method][1]["component_fit_counts"]


def test_blend_fresh_components_match_standalones_and_raw_arithmetic(invented, fitted):
    _, values, observed, clock, end = invented
    blend, stats = fitted[selected.BLEND_METHOD]
    ridge, _ = fitted["ridge"]
    direct, _ = fitted["direct_lightgbm"]
    assert blend.ridge.model is not ridge.model
    assert blend.direct.model is not direct.model
    assert stats["component_fit_counts"] == {"ridge": 1, "direct_lightgbm": 1}
    assert blend.ridge.scaler is blend.direct.scaler
    indices = np.arange(end, end+3)
    ridge_raw = predict_model(ridge, values, observed, clock, indices)[0]
    direct_raw = predict_model(direct, values, observed, clock, indices)[0]
    expected = blend_raw_predictions(ridge_raw, direct_raw, .2)
    for actual, wanted in zip(selected.predict_selected(blend, values, observed, clock, indices), expected):
        np.testing.assert_array_equal(actual, wanted)


def test_original_hybrid_fits_fresh_ridge_then_true_residuals_inside_fit_timer(invented):
    config, values, observed, clock, end = invented
    events, correction_inputs = [], []
    actual_scaler, actual_examples = selected.fit_scaler, selected.make_examples
    actual_ridge, actual_correction = selected.fit_ridge, selected.fit_correction

    def timed_clock():
        events.append("timer")
        return 100 if events.count("timer") == 1 else 200

    def scaler(*args, **kwargs):
        events.append("scaler")
        return actual_scaler(*args, **kwargs)

    def examples(*args, **kwargs):
        events.append("features")
        return actual_examples(*args, **kwargs)

    def ridge(*args, **kwargs):
        events.append("ridge")
        return actual_ridge(*args, **kwargs)

    def correction(model, features, targets, trees, leaves, settings):
        events.append("correction")
        correction_inputs.append((model, features.copy(), targets.copy(), trees, leaves))
        return actual_correction(model, features, targets, trees, leaves, settings)

    with patch.object(selected, "perf_counter_ns", side_effect=timed_clock), \
            patch.object(selected, "fit_scaler", side_effect=scaler), \
            patch.object(selected, "make_examples", side_effect=examples), \
            patch.object(selected, "fit_ridge", side_effect=ridge), \
            patch.object(selected, "fit_correction", side_effect=correction):
        bundle, stats = selected.fit_selected(selected.ORIGINAL_HYBRID, values, observed, clock, end, config)
    assert events == ["timer", "scaler", "features", "ridge", "correction", "timer"]
    assert stats["fit_seconds"] == 1e-7
    assert stats["component_fit_counts"] == {"ridge": 1, "correction": 1}
    model, features, targets, trees, leaves = correction_inputs[0]
    assert (trees, leaves, bundle.settings["weight"]) == (32, 31, 1.0)
    assert model is bundle.model
    train = actual_examples(values[:end], observed[:end], clock[:end], 0, end)
    np.testing.assert_array_equal(features, tabular_inputs(train, bundle.scaler))
    np.testing.assert_array_equal(targets, scaled_targets(train, bundle.scaler))
    # The residual fitting primitive must receive the unmodified full target;
    # it computes residuals itself rather than learning traffic directly.
    residuals = training_residuals(targets, predict_ridge(model, features))
    assert not np.array_equal(residuals, targets)


def test_blend_feature_and_both_fits_are_inside_one_timer(invented):
    import forecasting.data as data
    import models.lightgbm as lightgbm
    import models.ridge as ridge
    config, values, observed, clock, end = invented
    events = []
    originals = data.fit_scaler, data.make_examples, ridge.fit_ridge, lightgbm.fit_lightgbm

    def called(label, function):
        def wrapper(*args, **kwargs):
            events.append(label)
            return function(*args, **kwargs)
        return wrapper

    def timer():
        events.append("timer")
        return 100 if events.count("timer") == 1 else 200

    with patch.object(selected, "perf_counter_ns", side_effect=timer), \
            patch.object(data, "fit_scaler", side_effect=called("scaler", originals[0])), \
            patch.object(data, "make_examples", side_effect=called("features", originals[1])), \
            patch.object(ridge, "fit_ridge", side_effect=called("ridge", originals[2])), \
            patch.object(lightgbm, "fit_lightgbm", side_effect=called("direct", originals[3])):
        _, stats = selected.fit_selected(selected.BLEND_METHOD, values, observed, clock, end, config)
    assert events == ["timer", "scaler", "features", "ridge", "direct", "timer"]
    assert stats["fit_seconds"] == 1e-7


def test_lstm_runs_exactly23_epochs_without_inner_selection(invented, fitted):
    bundle, stats = fitted["lstm"]
    assert bundle.settings["selected_epochs"] == 23
    assert bundle.settings["hidden_units"] == 32
    assert bundle.settings["dropout"] == .2
    assert stats["early_stopping"] is False
    assert stats["component_fit_counts"] == {"full_prefix_fit": 1}
    assert [row["epoch"] for row in stats["history"]] == list(range(1, 24))
    assert not any("tail" in key or "selection" in key for key in stats if key != "candidate_selection")
    assert stats["fit_count"] == bundle.settings["training_target_count"]


@pytest.mark.parametrize("seed", selected.LSTM_SEEDS)
def test_every_declared_seed_uses_identical_frozen_architecture_and_epochs(invented, fitted, seed):
    config, values, observed, clock, end = invented
    # Spy on routing without three redundant full numerical fits in this test.
    with patch.object(selected, "_fit_complete", return_value=fitted["lstm"][0]) as fit:
        selected.fit_selected("lstm", values, observed, clock, end, config, seed=seed)
    assert fit.call_args.args[6] == seed


@pytest.mark.parametrize("method,seed", [("lstm", None), ("lstm", 2029), ("lstm", True),
                                         ("ridge", 2026), ("unknown", None)])
def test_invalid_methods_and_seeds_rejected_before_fitting(invented, method, seed):
    config, values, observed, clock, end = invented
    with patch.object(selected, "_fit_complete") as fitting, pytest.raises(ValueError):
        selected.fit_selected(method, values, observed, clock, end, config, seed=seed)
    fitting.assert_not_called()


@pytest.mark.parametrize("method,change", [("direct_lightgbm", "objective"),
                                            (selected.ORIGINAL_HYBRID, "capacity"),
                                            (selected.BLEND_METHOD, "learning_rate"),
                                            ("ridge", "alpha"), ("lstm", "early_stopping")])
def test_changed_scientific_settings_fail_before_any_fit(invented, method, change):
    config, values, observed, clock, end = invented
    bad = copy.deepcopy(config)
    if change == "objective":
        bad["models"]["trees"]["objective"] = "regression_l1"
    elif change == "capacity":
        bad["models"]["trees"]["leaf_caps"] = [7]
    elif change == "learning_rate":
        bad["models"]["trees"]["learning_rate"] = .1
    elif change == "alpha":
        bad["models"]["ridge"]["alphas"] = [1]
    else:
        bad["models"]["lstm"]["confirmation_final_early_stopping"] = True
    with patch.object(selected, "_fit_complete") as fitting, pytest.raises(ValueError):
        selected.fit_selected(method, values, observed, clock, end, bad,
                              seed=2026 if method == "lstm" else None)
    fitting.assert_not_called()


@pytest.mark.parametrize("method,field,value", [("ridge", "alpha", 100),
                                                  (selected.ORIGINAL_HYBRID, "weight", .5),
                                                  ("lstm", "selected_epochs", 24)])
def test_reload_rejects_frozen_setting_tampering(fitted, tmp_path, method, field, value):
    selected.save_selected(fitted[method][0], tmp_path)
    manifest = tmp_path / "model.json"
    state = json.loads(manifest.read_text())
    state["settings"][field] = value
    manifest.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        selected.load_selected(tmp_path)


def test_save_rejects_alternate_blend_weight(fitted, tmp_path):
    changed = replace(fitted[selected.BLEND_METHOD][0], ridge_weight=.3)
    with pytest.raises(ValueError, match="weight0.2"):
        selected.save_selected(changed, tmp_path)
