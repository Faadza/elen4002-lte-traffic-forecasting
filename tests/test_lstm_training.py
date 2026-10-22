"""Check stopping arithmetic, fresh refitting and chronological isolation."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import forecasting.lstm_training as training
from forecasting.data import fit_scaler, lstm_inputs, make_examples, scaled_targets
from models.lstm import predict_lstm


@pytest.fixture
def settings():
    result = yaml.safe_load((Path(__file__).resolve().parents[1] / "run_config.yaml").read_text())["models"]["lstm"]
    result.update(maximum_epochs=2, patience_epochs=1, inner_stopping_tail_hours=24)
    return result


def invented_data():
    index = np.arange(240)
    values = 100 + 20 * np.sin(index / 9) + index / 10
    observed = np.ones(len(index), dtype=bool)
    observed[10] = False  # An already prepared input is not a genuine scaler hour.
    return values, observed, pd.date_range("2020-01-01", periods=len(index), freq="h")


def fit_development(data, settings, **kwargs):
    options = dict(hidden_units=16, dropout=0., seed=2026, settings=settings)
    options.update(kwargs)
    return training.fit_development(*data, 224, **options)


def test_stopping_ties_keep_earliest_epoch_and_patience_counts_epochs():
    state = training.EarlyStopping()
    for epoch, score in enumerate([4., 3., 3., 5.], start=1):
        state = training.update_early_stopping(state, epoch, score, patience=2)
    assert state.best_epoch == 2
    assert state.best_score == 3.
    assert state.epochs_since_improvement == 2
    assert state.should_stop
    with pytest.raises(ValueError, match="advance in order"):
        training.update_early_stopping(state, 5, 2., 2)


def test_any_strict_improvement_resets_patience_without_rounding():
    state = training.EarlyStopping()
    for epoch, score in enumerate([1., 1., np.nextafter(1., 0.), 1.], start=1):
        state = training.update_early_stopping(state, epoch, score, patience=2)
    assert state.best_epoch == 3
    assert state.epochs_since_improvement == 1
    assert not state.should_stop


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1., True])
def test_invalid_tail_scores_fail_explicitly(score):
    with pytest.raises(ValueError, match="tail MAE"):
        training.update_early_stopping(training.EarlyStopping(), 1, score, 2)


def test_two_phase_fit_is_repeatable_and_ignores_every_future_value(settings):
    original = invented_data()
    first = fit_development(original, settings)
    altered_values, altered_flags, altered_times = [x.copy() for x in original]
    altered_values[224:] = np.inf
    altered_flags[224:] = False
    # Even invalid outer traffic must not reach either fitting phase.
    second = fit_development((altered_values, altered_flags, altered_times), settings)
    assert first.scaler == second.scaler
    assert first.metadata == second.metadata
    assert training.state_fingerprint(first.model) == training.state_fingerprint(second.model)
    assert first.metadata["initial_state_sha256"]["inner_selection"] == first.metadata["initial_state_sha256"]["refit"]
    assert first.metadata["inner_fit_interval"] == [0, 200]
    assert first.metadata["inner_tail_interval"] == [200, 224]
    assert first.metadata["inner_fit_count"] == 32
    assert first.metadata["inner_tail_count"] == 24
    assert first.metadata["refit_count"] == 56
    assert first.metadata["inner_scaler"]["observed_count"] == 199
    assert first.scaler.observed_count == 223
    assert first.selection_model is not first.model

    # The current target never enters its own forecast history.
    score = make_examples(*original, 224, 226)
    changed_values = original[0].copy()
    changed_values[224:] += 10000
    changed = make_examples(changed_values, original[1], original[2], 224, 226)
    x, clock = lstm_inputs(score, first.scaler)
    other_x, other_clock = lstm_inputs(changed, second.scaler)
    np.testing.assert_array_equal(x[0], other_x[0])
    assert x[1, -1, 0] != other_x[1, -1, 0]
    np.testing.assert_array_equal(predict_lstm(first.model, x[:1], clock[:1]),
                                  predict_lstm(second.model, other_x[:1], other_clock[:1]))


def test_selected_epoch_count_fresh_optimizer_and_full_prefix_scaler(settings, monkeypatch):
    settings["maximum_epochs"] = 4
    data = invented_data()
    inner_scaler = fit_scaler(data[0], data[1], 200)
    tail = make_examples(*data, 200, 224)
    calls, optimizers, callbacks = [], [], []
    real_train = training.train_epoch

    def inspect_train(model, optimizer, sequence, calendar, targets, batch_size):
        if not any(optimizer is previous for previous in optimizers):
            assert not optimizer.state
            optimizers.append(optimizer)
        calls.append((id(model), targets.copy()))
        return real_train(model, optimizer, sequence, calendar, targets, batch_size=batch_size)

    scores_requested = []

    def controlled_tail(model, sequence, calendar, batch_size):
        scores_requested.append(len(sequence))
        if len(scores_requested) == 1:
            return np.full(len(sequence), -100.)  # Inversion then lower bound yields zero.
        return scaled_targets(tail, inner_scaler)

    monkeypatch.setattr(training, "train_epoch", inspect_train)
    monkeypatch.setattr(training, "predict_lstm", controlled_tail)
    result = fit_development(data, settings, progress=callbacks.append)
    meta = result.metadata
    assert meta["selected_epochs"] == 2
    assert meta["selection_epochs_run"] == 3
    assert meta["stop_reason"] == "patience"
    assert not meta["maximum_epochs_reached"]
    assert len(optimizers) == 2
    assert len(calls) == 5
    assert calls[0][0] == calls[2][0] != calls[3][0] == calls[4][0]
    assert meta["stopping_history"][0]["tail_mae"] == np.mean(tail.actual)
    assert meta["stopping_history"][1]["tail_mae"] == meta["stopping_history"][2]["tail_mae"]
    np.testing.assert_array_equal(calls[0][1], scaled_targets(make_examples(*data, 0, 200), inner_scaler).astype(np.float32))
    np.testing.assert_array_equal(calls[3][1], scaled_targets(make_examples(*data, 0, 224), result.scaler).astype(np.float32))
    assert [row["phase"] for row in callbacks] == ["selection"] * 3 + ["refit"] * 2
    assert meta["optimizer_defaults"]["lr"] == .001
    assert meta["optimizer_defaults"]["weight_decay"] == 0
    assert meta["fresh_optimizer_state_empty"] == {"inner_selection": True, "refit": True}


def test_fixed_epochs_never_invokes_tail_prediction(settings, monkeypatch):
    data = invented_data()
    altered = (data[0].copy(), data[1], data[2])
    altered[0][224:] = np.nan

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed-epoch fitting must not score a stopping tail")

    monkeypatch.setattr(training, "predict_lstm", forbidden)
    result = training.fit_fixed_epochs(*altered, 224, hidden_units=16, dropout=0.,
                                       seed=2026, settings=settings, epochs=2)
    assert result.selection_model is None
    assert result.metadata["epochs"] == 2
    assert len(result.metadata["history"]) == 2
    assert result.metadata["early_stopping"] is False
    assert result.scaler == fit_scaler(*data[:2], 224)


@pytest.mark.parametrize("name,value", [("shuffle_windows", True), ("stateless_windows", False),
    ("layers", 2), ("batch_size", True), ("learning_rate", float("nan")),
    ("maximum_epochs", 0), ("patience_epochs", 11), ("inner_stopping_tail_hours", 337)])
def test_invalid_protocol_settings_rejected_before_model_build(settings, monkeypatch, name, value):
    settings[name] = value
    monkeypatch.setattr(training, "build_model", lambda **kwargs: pytest.fail("Model should not be built"))
    with pytest.raises(ValueError):
        fit_development(invented_data(), settings)


@pytest.mark.parametrize("candidate", [{"hidden_units": 8}, {"dropout": .3}, {"seed": 1}])
def test_candidate_must_be_in_declared_grid(settings, candidate):
    with pytest.raises(ValueError):
        fit_development(invented_data(), settings, **candidate)


def test_no_eligible_inner_tail_fails_before_model_fitting(settings, monkeypatch):
    data = invented_data()
    data[1][200:224] = False
    monkeypatch.setattr(training, "build_model", lambda **kwargs: pytest.fail("Model should not be built"))
    with pytest.raises(ValueError, match="tail has no eligible"):
        fit_development(data, settings)
