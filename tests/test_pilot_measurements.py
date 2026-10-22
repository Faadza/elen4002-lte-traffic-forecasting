"""Check measurement boundaries and units with invented outputs and clocks."""

from types import SimpleNamespace

import numpy as np
import psutil
import pytest

import tools.pilot_measurements as measure


def test_elapsed_call_times_only_callback_and_retains_its_result(monkeypatch):
    events = []
    ticks = iter([1_000_000_000, 1_250_000_000])

    def clock():
        events.append("clock")
        return next(ticks)

    def call():
        events.append("callback")
        return "result"

    monkeypatch.setattr(measure.time, "perf_counter_ns", clock)
    assert measure.elapsed_call(call) == ("result", .25)
    assert events == ["clock", "callback", "clock"]


def test_backwards_clock_fails(monkeypatch):
    ticks = iter([2, 1])
    monkeypatch.setattr(measure.time, "perf_counter_ns", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="backwards"):
        measure.elapsed_call(lambda: None)


def test_origins_are_selected_from_masks_without_forecast_values():
    eligible = np.array([168, 170, 172, 181, 190, 200, 210])
    np.testing.assert_array_equal(measure.select_origins(eligible, maximum=3), [168, 181, 210])
    np.testing.assert_array_equal(measure.select_origins(eligible), eligible)
    assert len(measure.select_origins([])) == 0
    selected = measure.select_origins(np.arange(168, 840))
    assert len(selected) == len(np.unique(selected)) == 100
    assert selected[0] == 168 and selected[-1] == 839


@pytest.mark.parametrize("targets", [[3, 2], [2, 2], [True, False], [-1, 2], [1., 2.], [[1, 2]]])
def test_invalid_target_lists_rejected(targets):
    with pytest.raises(ValueError):
        measure.select_origins(targets)


def test_prediction_pass_counts_timings_and_known_percentile(monkeypatch):
    calls = []
    durations = iter([1, 2, 3, 4, 5, 6, 7, 8, 9])
    elapsed = 0
    started = False

    def clock():
        nonlocal elapsed, started
        if started:
            elapsed += next(durations) * 1_000_000_000
        started = not started
        return elapsed

    def predict(targets):
        calls.append(targets.tolist())
        raw = targets.astype(float) - 3
        return raw, np.maximum(0., raw)

    monkeypatch.setattr(measure.time, "perf_counter_ns", clock)
    result = measure.measure_predictions(predict, np.array([2, 4, 6]), np.array([2, 6]), rtol=0., atol=0.)
    timings = result["timings"]
    assert calls == [[2, 4, 6]] * 4 + [[2], [6]] * 4
    assert timings["batch_seconds"] == [1., 2., 3.]
    assert timings["single_origin_seconds"] == [[4., 5.], [6., 7.], [8., 9.]]
    assert timings["single_origin_summary"]["median_seconds"] == 6.5
    assert timings["single_origin_summary"]["p95_seconds"] == 8.75
    np.testing.assert_array_equal(result["raw"], [-1., 1., 3.])
    np.testing.assert_array_equal(result["final"], [0., 1., 3.])


def test_repeated_outputs_must_be_exact_even_if_batch_tolerance_large():
    count = 0

    def predict(targets):
        nonlocal count
        count += 1
        raw = np.ones(len(targets)) + count * 1e-8
        return raw, raw.copy()

    with pytest.raises(AssertionError, match="changed its outputs"):
        measure.measure_predictions(predict, [2, 4], [2], rtol=1., atol=1.)


def test_single_vs_batch_uses_explicit_tolerance():
    def predict(targets):
        raw = np.ones(len(targets)) + (1e-6 if len(targets) == 1 else 0.)
        return raw, raw.copy()

    result = measure.measure_predictions(predict, [2, 4], [2], rtol=0., atol=1e-5)
    assert result["timings"]["single_vs_batch_tolerance"]["atol"] == 1e-5
    with pytest.raises(AssertionError):
        measure.measure_predictions(predict, [2, 4], [2], rtol=0., atol=1e-8)


@pytest.mark.parametrize("bad_result", [(np.ones((2, 1)), np.ones(2)),
    (np.ones(2), np.array([1., np.nan])), (np.ones(2), np.array([1., 2.])),
    (np.array([-1., 1.]), np.array([-1., 1.]))])
def test_invalid_prediction_outputs_rejected(bad_result):
    with pytest.raises(ValueError):
        measure.measure_predictions(lambda _: bad_result, [2, 4], [2], rtol=0., atol=0.)


def test_origin_membership_and_tolerance_fail_before_prediction():
    def forbidden(_):
        pytest.fail("Invalid workload must fail before model execution")

    with pytest.raises(ValueError, match="members"):
        measure.measure_predictions(forbidden, [2, 4], [3], rtol=0., atol=0.)
    with pytest.raises(ValueError, match="rtol"):
        measure.measure_predictions(forbidden, [2, 4], [2], rtol=float("nan"), atol=0.)


def test_platform_peak_rss_units_and_snapshot_scope(monkeypatch):
    assert measure.peak_rss_bytes(42, "Linux") == 42 * 1024
    assert measure.peak_rss_bytes(42, "Darwin") == 42
    monkeypatch.setattr(measure.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(measure.resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=4096))
    snapshot = measure.memory_snapshot()
    assert snapshot["process_peak_rss_bytes"] == 4096
    assert snapshot["native_peak_unit"] == "bytes"
    assert snapshot["incremental_model_memory"] is False


def test_required_peak_rss_does_not_query_pid_namespace_or_current_rss(monkeypatch):
    def unavailable_process(*args, **kwargs):
        raise psutil.NoSuchProcess(11)

    requested = []
    monkeypatch.setattr(psutil, "Process", unavailable_process)
    monkeypatch.setattr(measure.platform, "system", lambda: "Linux")

    def usage(who):
        requested.append(who)
        return SimpleNamespace(ru_maxrss=4096)

    monkeypatch.setattr(measure.resource, "getrusage", usage)
    snapshot = measure.memory_snapshot()
    assert requested == [measure.resource.RUSAGE_SELF]
    assert snapshot["process_peak_rss_bytes"] == 4096 * 1024
    assert "current_rss_bytes" not in snapshot


@pytest.mark.parametrize("value,system", [(3, "Windows"), (-1, "Linux"), (1.5, "Darwin"), (float("inf"), "Linux")])
def test_unknown_memory_units_or_invalid_values_fail(value, system):
    with pytest.raises(ValueError):
        measure.peak_rss_bytes(value, system)
