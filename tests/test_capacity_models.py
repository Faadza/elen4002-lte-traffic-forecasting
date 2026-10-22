"""Invented-data checks of candidate routing, causality and stored models."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from forecasting import capacity_models as capacity
from forecasting.data import fit_scaler, make_examples, tabular_inputs, scaled_targets
from forecasting.pilot_models import (predict_model, save_model, load_model,
                                     model_fingerprint)
from models.ridge import fit_ridge
from models.lightgbm import fit_lightgbm
from models.hybrid import fit_correction


@pytest.fixture(scope="module")
def invented():
    config = yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())
    # Only the invented LSTM fixture shortens training; scientific grids stay intact.
    config["models"]["lstm"].update(maximum_epochs=1, patience_epochs=1,
                                      inner_stopping_tail_hours=24)
    positions = np.arange(320, dtype=np.float64)
    values = 20 + 2*np.sin(2*np.pi*positions/24) + positions/100
    observed = np.ones(320, dtype=bool)
    observed[200] = False
    clock = pd.date_range("2020-01-06", periods=len(values), freq="h")
    return config, values, observed, clock, 280


CASES = [("ridge", None), ("direct_lightgbm", None), ("ridge_lightgbm_hybrid", None),
         ("lstm", 0.), ("lstm", .2)]


@pytest.fixture(scope="module")
def fitted(invented):
    config, values, observed, clock, fit_end = invented
    return {(method, dropout): capacity.fit_model(method, values, observed, clock, fit_end,
             config, dropout=dropout, seed=2026 if method == "lstm" else None)
            for method, dropout in CASES}


@pytest.mark.parametrize("method,dropout", CASES)
def test_capacity_settings_survive_saved_model_roundtrip(invented, fitted, tmp_path, method, dropout):
    _, values, observed, clock, fit_end = invented
    model = fitted[(method, dropout)]
    expected = predict_model(model, values, observed, clock, np.arange(fit_end, fit_end+3))
    save_model(model, tmp_path)
    restored = load_model(tmp_path)
    actual = predict_model(restored, values, observed, clock, np.arange(fit_end, fit_end+3))
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert model_fingerprint(restored) == model_fingerprint(model)
    if method == "lstm":
        assert restored.settings["hidden_units"] == 32
        assert restored.settings["dropout"] == dropout
        assert model.training_metadata["phase_timings"]["selection_seconds"] > 0
        assert model.training_metadata["phase_timings"]["refit_seconds"] > 0
        assert restored.training_metadata == {}
    if method in ("direct_lightgbm", "ridge_lightgbm_hybrid"):
        assert restored.settings["requested_trees"] == 512
        assert restored.settings["leaf_cap"] == 31


def test_tabular_candidates_match_direct_checked_primitives(invented, fitted):
    config, values, observed, clock, fit_end = invented
    scaler = fit_scaler(values[:fit_end], observed[:fit_end], fit_end)
    examples = make_examples(values[:fit_end], observed[:fit_end], clock[:fit_end], 0, fit_end)
    x, y = tabular_inputs(examples, scaler), scaled_targets(examples, scaler)
    ridge = fit_ridge(x, y, .1)
    direct = fit_lightgbm(x, y, 512, 31, config["models"]["trees"])
    correction = fit_correction(ridge, x, y, 512, 31, config["models"]["trees"])
    np.testing.assert_array_equal(fitted[("ridge", None)].model.coef_, ridge.coef_)
    assert fitted[("direct_lightgbm", None)].model.model_to_string() == direct.model_to_string()
    hybrid = fitted[("ridge_lightgbm_hybrid", None)]
    assert hybrid.correction_model.model_to_string() == correction.model_to_string()
    np.testing.assert_array_equal(hybrid.model.coef_, ridge.coef_)
    assert hybrid.settings["weight"] == 1.
    assert hybrid.training_metadata["component_fit_counts"] == {"ridge": 1, "correction": 1}


@pytest.mark.parametrize("method,dropout", CASES)
def test_fitting_ignores_future_values_and_does_not_mutate_configuration(invented, fitted, method, dropout):
    config, values, observed, clock, fit_end = invented
    original_config = copy.deepcopy(config)
    bad_values, bad_observed = values.copy(), observed.astype(int)
    bad_values[fit_end:], bad_observed[fit_end:] = np.inf, 99
    model = capacity.fit_model(method, bad_values, bad_observed, clock, fit_end, config,
                               seed=2026 if method == "lstm" else None, dropout=dropout)
    assert model_fingerprint(model) == model_fingerprint(fitted[(method, dropout)])
    assert config == original_config


@pytest.mark.parametrize("section,key", [("ridge", "alphas"), ("trees", "tree_counts"),
    ("trees", "leaf_caps"), ("hybrid", "weights"), ("lstm", "hidden_units"),
    ("lstm", "output_dropout"), ("lstm", "seeds_all_formal_periods")])
def test_changed_candidate_grids_are_rejected(invented, section, key):
    config = copy.deepcopy(invented[0])
    config["models"][section][key].reverse()
    with pytest.raises(ValueError, match="unchanged declared grids"):
        capacity.chosen_settings(config)


@pytest.mark.parametrize("method,seed,dropout", [("persistence", None, None), ("ridge", 2026, None),
    ("direct_lightgbm", None, .2), ("lstm", 2026, None), ("lstm", 999, .2),
    ("lstm", None, 0.), ("lstm", 2026, .3)])
def test_undeclared_or_ambiguous_candidates_rejected(invented, method, seed, dropout):
    config, values, observed, clock, fit_end = invented
    with pytest.raises(ValueError):
        capacity.fit_model(method, values, observed, clock, fit_end, config, seed=seed, dropout=dropout)
