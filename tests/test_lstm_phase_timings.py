"""Optional LSTM timings preserve fitting and enclose both preparation phases."""

from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
import pytest
import yaml

import forecasting.lstm_training as training
from forecasting.data import lstm_inputs, make_examples
from models.lstm import predict_lstm


@pytest.fixture
def settings():
    config = Path(__file__).resolve().parents[1] / "run_config.yaml"
    result = yaml.safe_load(config.read_text())["models"]["lstm"]
    result.update(maximum_epochs=2, patience_epochs=1, inner_stopping_tail_hours=24)
    return result


def invented_data():
    index = np.arange(240)
    values = 100 + 20 * np.sin(index / 9) + index / 10
    observed = np.ones(len(index), dtype=bool)
    observed[10] = False
    return values, observed, pd.date_range("2020-01-01", periods=len(index), freq="h")


def fit(data, settings, **options):
    kwargs = dict(hidden_units=16, dropout=0., seed=2026, settings=settings)
    kwargs.update(options)
    return training.fit_development(*data, 224, **kwargs)


@pytest.mark.parametrize("value", [0, 1, None, "yes", np.bool_(True)])
def test_flag_is_strictly_boolean_before_any_training(settings, monkeypatch, value):
    def forbidden(*args, **kwargs):
        pytest.fail("An invalid timing flag must fail before training or clock reads")

    monkeypatch.setattr(training, "_settings", forbidden)
    monkeypatch.setattr(training, "perf_counter_ns", forbidden)
    with pytest.raises(ValueError, match="record_phase_times must be a boolean"):
        fit(invented_data(), settings, record_phase_times=value)


def test_default_never_reads_clock_or_adds_timing_metadata(settings, monkeypatch):
    def forbidden():
        pytest.fail("The existing default fitting path must not read a clock")

    monkeypatch.setattr(training, "perf_counter_ns", forbidden)
    result = fit(invented_data(), settings)
    assert "phase_timings" not in result.metadata


@pytest.mark.parametrize("dropout", [0., .2])
def test_timing_preserves_histories_parameters_and_predictions(settings, dropout):
    data = invented_data()
    untimed = fit(data, settings, dropout=dropout)
    outer_start = perf_counter_ns()
    timed = fit(data, settings, dropout=dropout, record_phase_times=True)
    outer_seconds = (perf_counter_ns() - outer_start) / 1e9
    phases = timed.metadata["phase_timings"]
    assert phases["selection_seconds"] > 0
    assert phases["refit_seconds"] > 0
    assert phases["selection_seconds"] + phases["refit_seconds"] <= outer_seconds
    assert {k: v for k, v in timed.metadata.items() if k != "phase_timings"} == untimed.metadata
    assert timed.scaler == untimed.scaler
    for name in ("model", "selection_model"):
        assert training.state_fingerprint(getattr(timed, name)) == training.state_fingerprint(getattr(untimed, name))
    examples = make_examples(*data, 224, 230)
    sequence, calendar = lstm_inputs(examples, timed.scaler)
    np.testing.assert_array_equal(predict_lstm(timed.model, sequence, calendar),
                                  predict_lstm(untimed.model, sequence, calendar))


def test_clock_boundaries_include_scalers_features_models_and_training(settings, monkeypatch):
    events = []
    ticks = iter([1_000_000_000, 3_000_000_000, 4_000_000_000, 7_000_000_000])

    def clock():
        tick = next(ticks)
        events.append(("clock", tick))
        return tick

    def watch(name):
        original = getattr(training, name)

        def call(*args, **kwargs):
            events.append((name, None))
            return original(*args, **kwargs)

        monkeypatch.setattr(training, name, call)

    monkeypatch.setattr(training, "perf_counter_ns", clock)
    for name in ("_settings", "_prefix", "fit_scaler", "make_examples", "lstm_inputs",
                 "_new_model", "train_epoch", "predict_lstm"):
        watch(name)
    callbacks = []

    def progress(row):
        events.append(("progress_" + row["phase"], None))
        callbacks.append(row)

    result = fit(invented_data(), settings, record_phase_times=True, progress=progress)
    clock_positions = [i for i, row in enumerate(events) if row[0] == "clock"]
    start, end, refit_start, refit_end = clock_positions
    assert [name for name, _ in events[:start]] == ["_settings", "_prefix"]
    selection = [name for name, _ in events[start + 1:end]]
    refit = [name for name, _ in events[refit_start + 1:refit_end]]
    for segment in (selection, refit):
        assert segment.count("fit_scaler") == 1
        assert segment.count("_new_model") == 1
        assert segment.index("fit_scaler") < segment.index("make_examples") < segment.index("_new_model") < segment.index("train_epoch")
    assert selection.count("make_examples") == 2  # Inner fit and stopping tail.
    assert selection.count("lstm_inputs") == 2
    assert selection.count("predict_lstm") == result.metadata["selection_epochs_run"]
    assert selection.count("progress_selection") == result.metadata["selection_epochs_run"]
    assert refit.count("make_examples") == 1
    assert refit.count("lstm_inputs") == 1
    assert refit.count("train_epoch") == result.metadata["selected_epochs"]
    assert refit.count("progress_refit") == result.metadata["selected_epochs"]
    assert "predict_lstm" not in refit
    phases = result.metadata["phase_timings"]
    assert phases["clock"] == "perf_counter_ns"
    assert phases["selection_seconds"] == 2.
    assert phases["refit_seconds"] == 3.
    assert "scaler/features" in phases["selection_scope"]
    assert "scaler/features" in phases["refit_scope"]


@pytest.mark.parametrize("ticks", [[1, 1, 2, 3], [2, 1, 3, 4], [1, 3, 2, 4], [1, 2, 3, 3]])
def test_nonadvancing_or_overlapping_phase_clocks_are_rejected(settings, monkeypatch, ticks):
    values = iter(ticks)
    monkeypatch.setattr(training, "perf_counter_ns", lambda: next(values))
    with pytest.raises(RuntimeError, match="advancing monotonic clock"):
        fit(invented_data(), settings, record_phase_times=True)
