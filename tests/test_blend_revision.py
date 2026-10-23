"""Invented-data checks for the isolated direct/ridge blend revision."""
from dataclasses import replace
from pathlib import Path
import copy
import json
import hashlib

import numpy as np
import pandas as pd
import pytest
import yaml

from blend_revision.model import (BlendModel, blend_raw_predictions, checked_model,
                                  fit_blend, load_blend, predict_blend, save_blend)
from forecasting.pilot_models import model_fingerprint, predict_model
from blend_revision.replay import (InputReader, WEIGHTS, aggregate_scores,
                                   select_weight, validate_output)


def test_hand_arithmetic_blends_raw_before_final_clipping():
    ridge, direct = np.array([-10., 10., -20., 8.]), np.array([4., -2., 3., 4.])
    before = ridge.copy(), direct.copy()
    raw, final = blend_raw_predictions(ridge, direct, .2)
    np.testing.assert_allclose(raw, [1.2, .4, -1.6, 4.8], rtol=0, atol=1e-14)
    np.testing.assert_allclose(final, [1.2, .4, 0., 4.8], rtol=0, atol=1e-14)
    assert not np.allclose(final, .2*np.maximum(ridge, 0)+.8*np.maximum(direct, 0))
    np.testing.assert_array_equal(ridge, before[0])
    np.testing.assert_array_equal(direct, before[1])


@pytest.mark.parametrize("weight", [0, 1])
def test_endpoints_are_bitwise_exact_copies(weight):
    ridge, direct = np.array([-.125, 1e200, 1e-150]), np.array([2e200, -.25, 2e-150])
    raw, final = blend_raw_predictions(ridge, direct, weight)
    wanted = ridge if weight else direct
    np.testing.assert_array_equal(raw, wanted)
    np.testing.assert_array_equal(final, np.maximum(wanted, 0))
    assert not np.shares_memory(raw, wanted)


@pytest.mark.parametrize("weight", [True, np.bool_(False), -.01, 1.01, np.nan, np.inf, "0.2", None])
def test_invalid_weights_fail(weight):
    with pytest.raises(ValueError):
        blend_raw_predictions([1.], [2.], weight)


@pytest.mark.parametrize("ridge,direct", [([1.], [2., 3.]), ([[1.]], [2.]),
                                        ([np.inf], [2.]), ([1.], [np.nan]),
                                        ([1+2j], [2.]), (["1"], [2.])])
def test_invalid_components_fail_even_at_endpoint(ridge, direct):
    with pytest.raises(ValueError):
        blend_raw_predictions(ridge, direct, 0)


def test_empty_predictions_rejected_consistently_with_shared_metrics():
    with pytest.raises(ValueError, match="nonempty"):
        blend_raw_predictions(np.empty(0), np.empty(0))


@pytest.fixture(scope="module")
def invented():
    config = yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())
    t = np.arange(330, dtype=np.float64)
    values = 30 + 3*np.sin(2*np.pi*t/24) + 2*np.cos(2*np.pi*t/168) + t/100
    observed = np.ones(len(t), dtype=bool)
    observed[200] = False
    clock = pd.date_range("2020-01-06", periods=len(t), freq="h")
    return config, values, observed, clock


@pytest.fixture(scope="module")
def fitted(invented):
    config, values, observed, clock = invented
    return fit_blend(values, observed, clock, 300, config)


def test_selected_components_independent_prediction_and_no_mutation(invented, fitted):
    _, values, observed, clock = invented
    before = model_fingerprint(fitted.ridge), model_fingerprint(fitted.direct)
    ridge, _ = predict_model(fitted.ridge, values, observed, clock, [300, 301])
    direct, _ = predict_model(fitted.direct, values, observed, clock, [300, 301])
    expected = blend_raw_predictions(ridge, direct, .2)
    actual = predict_blend(fitted, values, observed, clock, [300, 301])
    for left, right in zip(expected, actual):
        np.testing.assert_array_equal(left, right)
    assert before == (model_fingerprint(fitted.ridge), model_fingerprint(fitted.direct))
    assert fitted.ridge.settings["alpha"] == 1000
    assert fitted.direct.settings["requested_trees"] == 128
    assert fitted.direct.settings["leaf_cap"] == 7
    assert fitted.ridge.scaler is fitted.direct.scaler
    assert fitted.ridge.training_metadata["component_fit_counts"] == {"ridge": 1}
    assert fitted.direct.training_metadata["component_fit_counts"] == {"direct_lightgbm": 1}


def test_fitting_prefix_isolation_and_input_preservation(invented, fitted):
    config, values, observed, clock = invented
    bad, flags = values.copy(), observed.astype(int)
    bad[300:], flags[300:] = np.inf, 99
    config_before, values_before, flags_before = copy.deepcopy(config), bad.copy(), flags.copy()
    actual = fit_blend(bad, flags, clock, 300, config)
    assert model_fingerprint(actual.ridge) == model_fingerprint(fitted.ridge)
    assert model_fingerprint(actual.direct) == model_fingerprint(fitted.direct)
    assert config == config_before
    np.testing.assert_array_equal(bad, values_before)
    np.testing.assert_array_equal(flags, flags_before)


def test_prediction_does_not_read_current_or_future_targets(invented, fitted):
    _, values, observed, clock = invented
    expected = predict_blend(fitted, values, observed, clock, [300])
    bad, flags = values.copy(), observed.astype(int)
    bad[300:], flags[300:] = np.inf, 99
    for left, right in zip(expected, predict_blend(fitted, bad, flags, clock, [300])):
        np.testing.assert_array_equal(left, right)


def test_serialization_preserves_both_components_and_predictions(invented, fitted, tmp_path):
    _, values, observed, clock = invented
    before = model_fingerprint(fitted.ridge), model_fingerprint(fitted.direct)
    expected = predict_blend(fitted, values, observed, clock, [300, 301])
    save_blend(fitted, tmp_path)
    loaded = load_blend(tmp_path)
    for left, right in zip(expected, predict_blend(loaded, values, observed, clock, [300, 301])):
        np.testing.assert_array_equal(left, right)
    assert loaded.ridge_weight == .2
    assert before == (model_fingerprint(loaded.ridge), model_fingerprint(loaded.direct))
    assert before == (model_fingerprint(fitted.ridge), model_fingerprint(fitted.direct))
    assert set(json.loads((tmp_path / "blend.json").read_text())["files"]) == {
        "ridge/model.json", "ridge/ridge.json", "direct/model.json", "direct/trees.txt"}
    with pytest.raises(ValueError, match="new or empty"):
        save_blend(fitted, tmp_path)


def test_component_file_tampering_rejected_before_loading(fitted, tmp_path):
    save_blend(fitted, tmp_path)
    with (tmp_path / "direct" / "trees.txt").open("a") as stream:
        stream.write("\nchanged\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_blend(tmp_path)


@pytest.mark.parametrize("change", ["method", "prefix", "scaler", "alpha", "count", "weight"])
def test_mismatched_components_rejected(fitted, change):
    ridge, direct, weight = fitted.ridge, fitted.direct, fitted.ridge_weight
    if change == "method":
        direct = replace(direct, method="ridge_lightgbm_hybrid")
    elif change == "prefix":
        direct = replace(direct, settings={**direct.settings, "fit_end": 299})
    elif change == "scaler":
        direct = replace(direct, scaler=replace(direct.scaler, mean=direct.scaler.mean+1))
    elif change == "alpha":
        ridge = replace(ridge, settings={**ridge.settings, "alpha": 100})
    elif change == "count":
        ridge = replace(ridge, settings={**ridge.settings, "training_target_count": 0})
    else:
        weight = True
    with pytest.raises(ValueError):
        checked_model(BlendModel(ridge, direct, weight))


def test_changed_tree_training_rule_rejected_before_fits(invented, monkeypatch):
    from models import ridge
    config, values, observed, clock = invented
    config = copy.deepcopy(config)
    config["models"]["trees"]["learning_rate"] = .1
    monkeypatch.setattr(ridge, "fit_ridge", lambda *a: pytest.fail("Invalid configuration fitted"))
    with pytest.raises(ValueError, match="frozen"):
        fit_blend(values, observed, clock, 300, config)


def test_replay_equal_cell_then_fold_aggregation():
    rows = [{"cell": cell, "fold_index": fold, "MAE": float(value),
             "RMSE": float(value), "MASE": float(value)}
            for fold in range(3) for cell, value in [("a", 1+fold), ("b", 3+fold)]]
    scores, folds = aggregate_scores(rows, ["a", "b"])
    assert scores == {"MAE": 3., "RMSE": 3., "MASE": 3.}
    assert [row["MASE"] for row in folds] == [2., 3., 4.]
    with pytest.raises(ValueError, match="cohort"):
        aggregate_scores(rows[:-1], ["a", "b"])
    with pytest.raises(ValueError, match="cohort"):
        aggregate_scores(rows[:-1] + [rows[0]], ["a", "b"])


def test_replay_exact_ties_and_unrounded_selection():
    rows = [{"ridge_proportion": weight, "scores": {"MASE": 1.}} for weight in WEIGHTS]
    assert select_weight(rows)["ridge_proportion"] == 0
    rows[3]["scores"]["MASE"] -= 1e-15
    assert select_weight(rows)["ridge_proportion"] == .2
    rows[2]["scores"]["MASE"] = rows[3]["scores"]["MASE"]
    assert select_weight(rows)["ridge_proportion"] == .1
    with pytest.raises(ValueError, match="grid"):
        select_weight(rows[:-1])


def test_replay_hash_drift_and_unbound_reads_rejected(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    payload = b'{"status":"saved"}'
    (original / "input.json").write_bytes(payload)
    reader = InputReader(original, {"input.json": hashlib.sha256(payload).hexdigest()})
    assert reader.json("input.json") == {"status": "saved"}
    (original / "unknown.json").write_bytes(payload)
    with pytest.raises(ValueError, match="Unbound"):
        reader.read("unknown.json")
    (original / "input.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        reader.verify_unchanged()


def test_replay_output_cannot_overwrite_inputs_or_prior_output(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    with pytest.raises(ValueError, match="outside"):
        validate_output(original, original / "new-output")
    with pytest.raises(ValueError, match="already exists"):
        validate_output(original, tmp_path)
    assert validate_output(original, tmp_path / "new-output") == tmp_path / "new-output"
