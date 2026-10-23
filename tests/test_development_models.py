"""Invented-data checks for baseline and LSTM development candidates."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from forecasting import development_models as development
from forecasting.pilot_models import load_model, model_fingerprint, save_model


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
    defaults = {"lstm": {"hidden_units": 32, "dropout": .2}}
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
    bundles = {method: fit(invented, job(method)) for method in development.BASELINES}
    bundles["lstm"] = fit(invented, job("lstm", seed=2027))
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


def test_explicit_lstm_candidate_and_phase_timings(fitted):
    assert fitted["lstm"].settings["hidden_units"] == 32
    assert fitted["lstm"].settings["dropout"] == .2
    assert fitted["lstm"].settings["seed"] == 2027
    assert fitted["lstm"].training_metadata["phase_timings"]["selection_seconds"] > 0
    assert fitted["lstm"].training_metadata["phase_timings"]["refit_seconds"] > 0


@pytest.mark.parametrize("method", ["lstm"])
def test_no_future_fitting_dependency_or_input_mutation(invented, fitted, method):
    config, values, observed, clock = invented
    snapshot = copy.deepcopy(config)
    bad, flags = values.copy(), observed.astype(int)
    bad[280:], flags[280:] = np.inf, 99
    altered = config, bad, flags, clock
    candidate = job(method, seed=2027)
    bundle = fit(altered, candidate)
    assert model_fingerprint(bundle) == model_fingerprint(fitted[method])
    assert config == snapshot
    assert np.isinf(bad[280:]).all() and (flags[280:] == 99).all()
    assert bundle.scaler.mean == np.mean(values[:280][observed[:280]])


@pytest.mark.parametrize("edit", [
    {"parameters": {}}, {"stage": "baselines"}, {"fold_index": True},
    {"fit_end": 280.0}, {"score_start": 281}, {"score_end": 280},
    {"id": ""}, {"cell": ""}, {"unexpected": True},
])
def test_invalid_job_rejected_before_fitting(invented, monkeypatch, edit):
    candidate = job("lstm", seed=2027, **edit)
    monkeypatch.setattr(development, "fit_development", lambda *args, **kwargs: pytest.fail("Invalid candidate fitted"))
    with pytest.raises(ValueError):
        fit(invented, candidate)


@pytest.mark.parametrize("method,parameters,seed", [
    ("lstm", {"hidden_units": 17, "dropout": 0.}, 2026),
    ("lstm", {"hidden_units": 16, "dropout": True}, 2026),
    ("lstm", {"hidden_units": 16, "dropout": .3}, 2026),
    ("lstm", {"hidden_units": 16, "dropout": 0.}, None),
    ("lstm", {"hidden_units": 16, "dropout": 0.}, 2026.),
    ("lstm", {"hidden_units": 16, "dropout": 0.}, 999),
    ("persistence", {"lag": 2}, None),
    ("persistence", {}, 2026),
])
def test_undeclared_candidates_rejected(invented, method, parameters, seed):
    with pytest.raises(ValueError):
        fit(invented, job(method, parameters, seed))


def test_lstm_epoch_cap_cannot_be_enlarged(invented):
    config, *arrays = invented
    changed = copy.deepcopy(config)
    changed["models"]["lstm"]["maximum_epochs"] = 101
    with pytest.raises(ValueError, match="maximum_epochs"):
        fit((changed, *arrays), job("lstm", seed=2026))
