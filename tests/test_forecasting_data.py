"""Hand-checkable examples; no measured LTE traffic or fitted prediction models."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from forecasting.data import (
    TrafficScaler, fit_scaler, lstm_inputs, make_examples, scaled_targets,
    tabular_inputs,
)
from preprocessing.prepare import causal_fill


def series(n=400):
    return (
        np.arange(n, dtype=np.float64),
        np.ones(n, dtype=bool),
        pd.date_range("2017-10-23", periods=n, freq="h"),
    )


def test_first_window_and_half_open_end_are_hand_aligned():
    values, observed, clock = series()
    examples = make_examples(values, observed, clock, 0, 170)
    np.testing.assert_array_equal(examples.targets, [168, 169])
    np.testing.assert_array_equal(examples.history[0], np.arange(168))
    np.testing.assert_array_equal(examples.history[1], np.arange(1, 169))
    np.testing.assert_array_equal(examples.actual, [168, 169])
    assert examples.timestamps.equals(clock[168:170])


def test_evaluation_window_reuses_history_from_before_split():
    values, observed, clock = series()
    examples = make_examples(values, observed, clock, 200, 202)
    np.testing.assert_array_equal(examples.targets, [200, 201])
    np.testing.assert_array_equal(examples.history[0], np.arange(32, 200))
    assert examples.history[1, -1] == 200


def test_target_and_future_traffic_never_enter_current_input():
    values, observed, clock = series()
    before = make_examples(values, observed, clock, 200, 201)
    values[200:] += 10000
    after = make_examples(values, observed, clock, 200, 201)
    for name in ("targets", "history", "missing", "calendar"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    assert after.actual[0] == before.actual[0] + 10000
    assert before.actual[0] == 200  # Returned examples do not alias the source.


def test_causal_filled_input_keeps_flag_and_cannot_be_a_target():
    values, _, clock = series()
    values[198:200] = np.nan
    filled, observed = causal_fill(values)
    examples = make_examples(filled, observed, clock, 198, 201)
    np.testing.assert_array_equal(examples.targets, [200])
    np.testing.assert_array_equal(examples.history[0, -3:], [197, 197, 197])
    np.testing.assert_array_equal(examples.missing[0, -3:], [0, 1, 1])


def test_unresolved_hour_invalidates_window_without_compressing_time():
    values, observed, clock = series()
    values[200] = np.nan
    observed[200] = False
    examples = make_examples(values, observed, clock, 199, 371)
    np.testing.assert_array_equal(examples.targets, [199, 369, 370])
    np.testing.assert_array_equal(examples.history[1], np.arange(201, 369))


def test_zero_is_a_genuine_observation():
    values = np.zeros(170)
    examples = make_examples(values, np.ones(170), series(170)[2], 0, 170)
    np.testing.assert_array_equal(examples.targets, [168, 169])
    assert not examples.missing.any()
    assert not examples.actual.any()


@pytest.mark.parametrize("start,end", [(0, 0), (0, 168), (200, 200)])
def test_empty_examples_and_feature_shapes(start, end):
    values, observed, clock = series()
    examples = make_examples(values, observed, clock, start, end)
    assert examples.targets.shape == examples.actual.shape == (0,)
    assert examples.history.shape == examples.missing.shape == (0, 168)
    assert examples.calendar.shape == (0, 4)
    assert len(examples.timestamps) == 0
    scaler = fit_scaler(values, observed, 200)
    assert tabular_inputs(examples, scaler).shape == (0, 340)
    sequence, calendar = lstm_inputs(examples, scaler)
    assert sequence.shape == (0, 168, 2)
    assert calendar.shape == (0, 4)
    assert scaled_targets(examples, scaler).shape == (0,)


def test_entire_empty_series_is_well_defined():
    examples = make_examples([], [], [], 0, 0)
    assert examples.history.shape == (0, 168)
    assert examples.calendar.shape == (0, 4)


@pytest.mark.parametrize("start,end", [(-1, 200), (200, 401), (201, 200), (True, 200), (0, 200.0)])
def test_bad_intervals_are_rejected(start, end):
    with pytest.raises(ValueError):
        make_examples(*series(), start, end)


def test_bad_array_shapes_and_observation_values_are_rejected():
    values, observed, clock = series()
    for bad_values, bad_mask in [
        (values[:, None], observed), (values, observed[:-1]),
        (values, np.full(400, 2)),
    ]:
        with pytest.raises(ValueError):
            make_examples(bad_values, bad_mask, clock, 168, 170)
    values[169] = np.nan
    with pytest.raises(ValueError, match="Observed positions"):
        make_examples(values, observed, clock, 168, 170)


def test_calendar_rejects_missing_duplicate_unordered_zoned_and_fractional_hours():
    values, observed, clock = series()
    duplicate = clock.to_numpy(copy=True)
    duplicate[200] = duplicate[199]
    missing = clock.to_numpy(copy=True)
    missing[200] = np.datetime64("NaT")
    skipped = clock.to_numpy(copy=True)
    skipped[200:] += np.timedelta64(1, "h")
    for bad_clock in (
        clock[:-1], duplicate, missing, skipped, clock[::-1],
        clock.tz_localize("UTC"), clock + pd.Timedelta(minutes=30), np.arange(400),
    ):
        with pytest.raises(ValueError):
            make_examples(values, observed, bad_clock, 168, 170)


def test_target_calendar_uses_monday_zero_and_daily_weekly_cycles():
    values, observed, clock = series(505)
    examples = make_examples(values, observed, clock, 168, 505)
    # Index 168 is Monday 00:00, with sine 0 and cosine 1 for both cycles.
    np.testing.assert_allclose(examples.calendar[0], [0, 1, 0, 1], atol=1e-15)
    np.testing.assert_allclose(examples.calendar[6, :2], [1, 0], atol=1e-15)
    np.testing.assert_allclose(examples.calendar[24, :2], examples.calendar[0, :2], atol=1e-15)
    np.testing.assert_allclose(examples.calendar[168], examples.calendar[0], atol=1e-15)
    # Sunday 23:00 wraps to Monday 00:00 on the next row.
    angle = 2 * np.pi * 167 / 168
    np.testing.assert_allclose(examples.calendar[167, 2:], [np.sin(angle), np.cos(angle)])
    np.testing.assert_allclose(examples.calendar[168, 2:], [0, 1], atol=1e-15)


def test_scaler_hand_calculation_ignores_fill_and_evaluation_values():
    # Genuine fit values are exactly [1, 2, 3, 4]: mean 2.5, variance 1.25.
    values = np.array([1, 1, 2, 3, 4, 100000], dtype=float)
    observed = np.array([1, 0, 1, 1, 1, 1])
    scaler = fit_scaler(values, observed, 5)
    assert scaler.mean == 2.5
    assert scaler.sd == np.sqrt(1.25)
    assert scaler.fit_end == 5 and scaler.observed_count == 4
    values[1] = 999999  # A filled value is never fitted as a genuine observation.
    values[5] = 888888
    assert fit_scaler(values, observed, 5) == scaler
    with pytest.raises(FrozenInstanceError):
        scaler.mean = 0


def test_scaler_counts_calendar_hours_once_not_repeated_lag_entries():
    values, observed, clock = series(172)
    scaler = fit_scaler(values, observed, 172)
    examples = make_examples(values, observed, clock, 0, 172)
    assert scaler.observed_count == 172
    assert scaler.mean == 85.5
    assert scaler.sd == np.sqrt((172 ** 2 - 1) / 12)
    assert examples.history.mean() == 85  # Flattening windows would bias the fit.


def test_scaler_inverse_is_shared_and_does_not_clip():
    scaler = fit_scaler([2, 4], [1, 1], 2)
    np.testing.assert_array_equal(scaler.transform([0, 2, 3, 4]), [-3, -1, 0, 1])
    np.testing.assert_array_equal(scaler.inverse([-4, -1, 0, 1]), [-1, 2, 3, 4])
    np.testing.assert_array_equal(scaler.inverse(scaler.transform([2, 4, 10])), [2, 4, 10])


@pytest.mark.parametrize("values,observed,end", [
    ([], [], 0), ([1], [1], 1), ([2, 2], [1, 1], 2),
    ([1, 2], [0, 0], 2), ([1, np.nan], [1, 1], 2),
    ([1, np.inf], [1, 1], 2), ([1, 2], [1], 2),
    ([1, 2], [1, 1], True), ([1, 2], [1, 1], 3),
])
def test_unavailable_or_invalid_scaler_is_explicit(values, observed, end):
    with pytest.raises(ValueError):
        fit_scaler(values, observed, end)


def test_scaler_rejects_nonfinite_transform_or_inverse():
    scaler = fit_scaler([2, 4], [1, 1], 2)
    for method in (scaler.transform, scaler.inverse):
        with pytest.raises(ValueError):
            method([1, np.nan])
        with pytest.raises(ValueError):
            method([1, np.inf])


def test_tabular_and_lstm_features_share_values_with_declared_precision():
    values, observed, clock = series()
    observed[199] = False
    values[199] = 198
    examples = make_examples(values, observed, clock, 200, 202)
    scaler = fit_scaler(values, observed, 200)
    tabular = tabular_inputs(examples, scaler)
    sequence, calendar = lstm_inputs(examples, scaler)
    targets = scaled_targets(examples, scaler)
    assert tabular.shape == (2, 340) and tabular.dtype == np.float64
    assert sequence.shape == (2, 168, 2) and sequence.dtype == np.float32
    assert calendar.shape == (2, 4) and calendar.dtype == np.float32
    assert targets.shape == (2,) and targets.dtype == np.float64
    np.testing.assert_array_equal(tabular[:, :168], scaler.transform(examples.history))
    np.testing.assert_array_equal(tabular[:, 168:336], examples.missing)
    np.testing.assert_array_equal(tabular[:, 336:], examples.calendar)
    np.testing.assert_array_equal(sequence[:, :, 0], tabular[:, :168].astype(np.float32))
    np.testing.assert_array_equal(sequence[:, :, 1], examples.missing.astype(np.float32))
    np.testing.assert_array_equal(calendar, examples.calendar.astype(np.float32))
    np.testing.assert_allclose(scaler.inverse(targets), examples.actual)


def test_lstm_float32_overflow_is_rejected():
    values, observed, clock = series()
    values[190] = 1e40
    examples = make_examples(values, observed, clock, 200, 201)
    scaler = TrafficScaler(0.0, 1.0, 2, 2)
    assert np.isfinite(tabular_inputs(examples, scaler)).all()
    with pytest.raises(ValueError, match="float32"):
        lstm_inputs(examples, scaler)
