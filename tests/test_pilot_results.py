"""Pilot scoring arithmetic and saved evidence on invented hourly sequences."""

import json

import numpy as np
import pandas as pd
import pytest

from tools.pilot_results import save_predictions, score_predictions


def sample():
    values = np.arange(240, dtype=float)
    observed = np.ones(240, dtype=bool)
    clock = pd.date_range("2020-01-01", periods=240, freq="h")
    targets = np.arange(216, 240)
    return values, observed, clock, targets


def score(values, observed, clock, targets, raw, final=None, start=216, end=240):
    return score_predictions(values, observed, clock, 216, start, end,
                             targets, raw, np.maximum(0, raw) if final is None else final)


def test_hand_errors_clipping_and_complete_daily_maximum():
    values, observed, clock, targets = sample()
    raw = values[targets] + 2
    raw[0] = -1
    result = score(values, observed, clock, targets, raw)
    assert result["scores"]["MAE"] == pytest.approx((216 + 23 * 2) / 24)
    assert result["scores"]["RMSE"] == pytest.approx(np.sqrt((216**2 + 23 * 4) / 24))
    assert result["weekly_scale"] == 168
    assert result["scores"]["MASE"] == pytest.approx(result["scores"]["MAE"] / 168)
    assert result["clipping"]["clipped_count"] == 1
    assert result["clipping"]["mean_upward_adjustment"] == pytest.approx(1 / 24)
    assert result["busiest_hour"]["qualifying_days"] == 1
    assert result["busiest_hour"]["absolute_error"] == 2
    assert result["busiest_hour"]["signed_error"] == 2
    assert result["counts"]["fitting_observed_weekly_pairs"] == 48
    assert result["coverage"]["fraction_calendar_scored"] == 1


def test_weekly_denominator_excludes_missing_and_all_later_values():
    values, observed, clock, targets = sample()
    observed[0] = False
    values[0] = np.nan
    raw = values[targets] + 1
    first = score(values, observed, clock, targets, raw)
    changed = values.copy()
    changed[216:] += 10000
    second = score(changed, observed, clock, targets, raw)
    assert first["weekly_scale"] == second["weekly_scale"] == 168
    assert first["counts"]["fitting_observed_weekly_pairs"] == 47
    assert first["scores"]["MAE"] != second["scores"]["MAE"]
    # Unused data after the scoring interval is never validated or scored.
    first = score(values, observed, clock, targets[:12], raw[:12], end=228)
    changed[228:] = np.inf
    changed[:228] = values[:228]
    assert score(changed, observed, clock, targets[:12], raw[:12], end=228) == first


@pytest.mark.parametrize("missing_pairs", [False, True])
def test_unavailable_mase_is_explicit_json_and_preserves_original_unit_errors(missing_pairs):
    values, observed, clock, targets = sample()
    values[:216] = 7
    if missing_pairs:
        observed[:216] = False
        values[:216] = np.nan
    result = score(values, observed, clock, targets, values[targets] + 2)
    assert result["scores"] == {"MAE": 2, "RMSE": 2, "MASE": None}
    assert result["weekly_scale"] == (None if missing_pairs else 0)
    assert result["weekly_scale_status"] == ("no_observed_weekly_pairs" if missing_pairs else "zero_scale")
    assert result["mase_status"] == "unavailable"
    json.dumps(result, allow_nan=False)


def test_partial_day_and_ineligible_maximum_are_not_replaced():
    values, observed, clock, targets = sample()
    partial = score(values, observed, clock, targets[:-1], values[targets[:-1]], end=239)
    assert partial["busiest_hour"]["qualifying_days"] == 0
    assert partial["busiest_hour"]["excluded_incomplete_calendar_days"] == 1
    missing_max = score(values, observed, clock, targets[:-1], values[targets[:-1]])
    assert missing_max["busiest_hour"]["qualifying_days"] == 0
    assert missing_max["busiest_hour"]["excluded_ineligible_maximum_days"] == 1


@pytest.mark.parametrize("change", ["column_raw", "nonfinite", "unclipped", "overclipped",
    "duplicate", "unsorted", "fractional_target", "unobserved", "outside", "bad_clock", "negative_actual"])
def test_invalid_shapes_targets_values_and_prediction_rules_rejected(change):
    values, observed, clock, targets = sample()
    raw = values[targets].copy()
    final = raw.copy()
    if change == "column_raw":
        raw = raw[:, None]
    elif change == "nonfinite":
        raw[0] = np.inf
    elif change == "unclipped":
        raw[0] = final[0] = -1
    elif change == "overclipped":
        final[0] = 0
    elif change == "duplicate":
        targets[1] = targets[0]
    elif change == "unsorted":
        targets[:2] = targets[1::-1]
    elif change == "fractional_target":
        targets = targets.astype(float)
    elif change == "unobserved":
        observed[targets[0]] = False
    elif change == "outside":
        targets[0] = 215
    elif change == "bad_clock":
        clock = clock + pd.Timedelta(minutes=1)
    else:
        values[0] = -1
    with pytest.raises(ValueError):
        score(values, observed, clock, targets, raw, final)


@pytest.mark.parametrize("seed", [None, 2026])
def test_csv_preserves_ids_seed_and_difficult_floats_without_overwrite(tmp_path, seed):
    targets = np.array([216, 218, 219])
    clock = pd.date_range("2020-01-01", periods=240, freq="h")[targets]
    actual = np.array([np.nextafter(1., 2.), .1, 1e-100])
    raw = np.array([-1e-100, np.nextafter(.1, 1.), 1e100])
    final = np.maximum(0, raw)
    path = tmp_path / "nested" / "predictions.csv"
    result = save_predictions(path, "Cell_001", "NA", seed, targets, clock, actual, raw, final)
    assert result["rows"] == 3 and result["round_trip_checked"]
    assert len(result["sha256"]) == 64
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_predictions(path, "Cell_001", "NA", seed, targets, clock, actual, raw, final)
    assert path.read_bytes() == before


def test_csv_rejects_misaligned_hours_before_creating_output(tmp_path):
    path = tmp_path / "predictions.csv"
    with pytest.raises(ValueError, match="gaps"):
        save_predictions(path, "Cell_001", "ridge", None, [1, 3],
                         pd.date_range("2020-01-01", periods=2, freq="h"), [1, 2], [1, 2], [1, 2])
    assert not path.exists()
