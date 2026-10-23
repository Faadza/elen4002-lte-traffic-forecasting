"""Invented-data verification of independent full-target L1/L2 components."""
import copy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from ensemble_study import model as study
from forecasting.data import make_examples, scaled_targets
from forecasting.pilot_models import model_fingerprint, predict_model


def condition(kind, objective=None, trees=None, leaves=None, weight=None, identifier=None):
    return {"id": identifier or kind, "kind": kind, "objective": objective,
            "trees": trees, "leaves": leaves, "ridge_weight": weight}


CONDITIONS = [
    condition("ridge", weight=1),
    *[condition(kind, objective, 32, 7, .2 if kind == "blend" else 0, kind+"-"+objective)
      for kind in ("direct", "blend") for objective in ("l1", "l2")],
]


@pytest.fixture(scope="module")
def invented():
    config = yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())
    t = np.arange(380, dtype=np.float64)
    traffic = 35 + 5*np.sin(t*.2) + 3*np.cos(t*.043) + t*.025
    observed = np.ones(len(t), dtype=bool)
    observed[190] = False
    clock = pd.date_range("2020-01-06", periods=len(t), freq="h")
    return config, traffic, observed, clock, 350


@pytest.fixture(scope="module")
def fitted(invented):
    config, values, observed, clock, end = invented
    return {c["id"]: study.fit_selected(c, values, observed, clock, end, config,
                                       seed=2026 if c["kind"] == "lstm" else None)
            for c in CONDITIONS}


@pytest.mark.parametrize("c", CONDITIONS, ids=[c["id"] for c in CONDITIONS])
def test_roundtrip_batch_single_and_immutable_state(invented, fitted, tmp_path, c):
    _, values, observed, clock, end = invented
    bundle, stats = fitted[c["id"]]
    targets = np.arange(end, end+3)
    before = study.fingerprint(bundle)
    output = study.predict_with_components(bundle, values, observed, clock, targets)
    np.testing.assert_array_equal(output["final"], np.maximum(0, output["raw"]))
    np.testing.assert_array_equal(study.predict_raw(bundle, values, observed, clock, targets), output["raw"])
    single = np.array([study.predict(bundle, values, observed, clock, [int(t)])[0][0] for t in targets])
    tol = 1e-5 if c["kind"] == "lstm" else 1e-12
    np.testing.assert_allclose(single, output["raw"], rtol=tol, atol=tol)
    saved = study.save_model(bundle, tmp_path / c["id"])
    restored = study.load_model(saved)
    assert study.fingerprint(bundle) == study.fingerprint(restored) == before
    again = study.predict_with_components(restored, values, observed, clock, targets)
    for key in output:
        if output[key] is None:
            assert again[key] is None
        else:
            np.testing.assert_array_equal(output[key], again[key])
    assert stats["fit_interval"] == [0, end]
    assert stats["cached_component_reused"] is False
    assert stats["fit_seconds"] >= 0
    assert stats["condition"] == c
    if c["kind"] == "blend":
        assert stats["component_fit_counts"] == {"ridge": 1, "direct_lightgbm": 1}
        assert {p.name for p in saved.iterdir()} == {"ensemble.json", "ridge", "direct"}
    if c["objective"]:
        metric = restored.direct.model.params["metric"]
        assert metric == [c["objective"]]


@pytest.mark.parametrize("objective", ["l1", "l2"])
def test_blend_components_equal_independently_fitted_full_target_models(invented, fitted, objective):
    _, values, observed, clock, end = invented
    blend, _ = fitted["blend-"+objective]
    direct, _ = fitted["direct-"+objective]
    ridge, _ = fitted["ridge"]
    assert blend.ridge.model is not ridge.ridge.model
    assert blend.direct.model is not direct.direct.model
    assert model_fingerprint(blend.ridge) == model_fingerprint(ridge.ridge)
    assert model_fingerprint(blend.direct) == model_fingerprint(direct.direct)
    targets = np.arange(end, end+3)
    result = study.predict_with_components(blend, values, observed, clock, targets)
    np.testing.assert_array_equal(result["ridge_raw"], study.predict_raw(ridge, values, observed, clock, targets))
    np.testing.assert_array_equal(result["direct_raw"], study.predict_raw(direct, values, observed, clock, targets))
    np.testing.assert_array_equal(result["raw"], .2*result["ridge_raw"] + .8*result["direct_raw"])
    assert blend.ridge.scaler is blend.direct.scaler


@pytest.mark.parametrize("objective", ["l1", "l2"])
def test_direct_and_ridge_endpoints_do_not_include_unused_components(invented, fitted, objective):
    _, values, observed, clock, end = invented
    direct, _ = fitted["direct-"+objective]
    ridge, _ = fitted["ridge"]
    assert direct.ridge is None and ridge.direct is None
    for bundle, name in ((direct, "direct_raw"), (ridge, "ridge_raw")):
        result = study.predict_with_components(bundle, values, observed, clock, [end])
        np.testing.assert_array_equal(result["raw"], result[name])
        assert not np.shares_memory(result["raw"], result[name])


def test_raw_original_unit_blend_occurs_before_single_clipping(invented, fitted):
    _, values, observed, clock, end = invented
    bundle, _ = fitted["blend-l1"]
    desired_ridge = np.array([-10., 10., -20.])
    desired_direct = np.array([4., -2., 3.])
    rz, dz = bundle.scaler.transform(desired_ridge), bundle.scaler.transform(desired_direct)
    with patch.object(study, "predict_ridge", return_value=rz), \
            patch.object(study, "predict_lightgbm", return_value=dz):
        result = study.predict_with_components(bundle, values, observed, clock, np.arange(end, end+3))
    np.testing.assert_allclose(result["raw"], [1.2, .4, -1.6], atol=1e-13)
    np.testing.assert_allclose(result["final"], [1.2, .4, 0.], atol=1e-13)
    assert not np.allclose(result["final"], .2*np.maximum(desired_ridge, 0)+.8*np.maximum(desired_direct, 0))


@pytest.mark.parametrize("c", CONDITIONS, ids=[c["id"] for c in CONDITIONS])
def test_current_future_values_do_not_enter_prediction(invented, fitted, c):
    _, values, observed, clock, end = invented
    bundle, _ = fitted[c["id"]]
    expected = study.predict(bundle, values, observed, clock, [end])
    poison, flags = values.copy(), observed.astype(int)
    poison[end:], flags[end:] = np.inf, 99
    for actual, wanted in zip(study.predict(bundle, poison, flags, clock, [end]), expected):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("c", [c for c in CONDITIONS if c["kind"] != "baseline"],
                         ids=[c["id"] for c in CONDITIONS if c["kind"] != "baseline"])
def test_all_fits_physically_exclude_future_and_preserve_input(invented, fitted, c):
    config, values, observed, clock, end = invented
    before = copy.deepcopy(config), copy.deepcopy(c)
    poison, flags = values.copy(), observed.astype(int)
    poison[end:], flags[end:] = np.inf, 99
    again, stats = study.fit_selected(c, poison, flags, clock, end, config,
                                     seed=2026 if c["kind"] == "lstm" else None)
    assert study.fingerprint(again) == study.fingerprint(fitted[c["id"]][0])
    assert config == before[0] and c == before[1]
    assert np.isinf(poison[end:]).all() and (flags[end:] == 99).all()
    assert stats["cached_component_reused"] is False


@pytest.mark.parametrize("objective", ["l1", "l2"])
def test_timer_includes_shared_preparation_and_both_full_target_fits(invented, objective):
    config, values, observed, clock, end = invented
    events, labels = [], []
    originals = study.fit_scaler, study.make_examples, study.fit_ridge, study._fit_tree
    def wrapped(label, original):
        def call(*args, **kwargs):
            events.append(label)
            if label in ("ridge", "direct"):
                labels.append(args[1].copy())
            return original(*args, **kwargs)
        return call
    def timer():
        events.append("timer")
        return 10 if events.count("timer") == 1 else 110
    c = condition("blend", objective, 32, 7, .2)
    with patch.object(study, "perf_counter_ns", side_effect=timer), \
            patch.object(study, "fit_scaler", side_effect=wrapped("scaler", originals[0])), \
            patch.object(study, "make_examples", side_effect=wrapped("features", originals[1])), \
            patch.object(study, "fit_ridge", side_effect=wrapped("ridge", originals[2])), \
            patch.object(study, "_fit_tree", side_effect=wrapped("direct", originals[3])):
        bundle, stats = study.fit_selected(c, values, observed, clock, end, config)
    assert events == ["timer", "scaler", "features", "ridge", "direct", "timer"]
    expected = scaled_targets(make_examples(values[:end], observed[:end], clock[:end], 0, end), bundle.scaler)
    np.testing.assert_array_equal(labels[0], expected)
    np.testing.assert_array_equal(labels[1], expected)
    assert stats["fit_seconds"] == 1e-7


@pytest.mark.parametrize("trees,leaves", [(t,l) for t in study.TREE_COUNTS for l in study.LEAF_CAPS])
@pytest.mark.parametrize("objective", ["l1", "l2"])
def test_all_declared_capacity_and_objective_settings(invented, trees, leaves, objective):
    config, values, observed, clock, end = invented
    bundle, _ = study.fit_selected(condition("direct", objective, trees, leaves, 0),
                                   values, observed, clock, end, config)
    assert bundle.direct.model.params["objective"] == ("regression_l1" if objective == "l1" else "regression")
    assert bundle.direct.settings["requested_trees"] == trees
    assert bundle.direct.settings["leaf_cap"] == leaves


@pytest.mark.parametrize("change", ["extra", "missing", "weight", "endpoint", "bool", "capacity", "objective", "id", "residual"])
def test_invalid_conditions_rejected_before_fitting(invented, change):
    config, values, observed, clock, end = invented
    c = condition("blend", "l1", 32, 7, .2)
    if change == "extra": c["extra"] = 1
    elif change == "missing": del c["trees"]
    elif change == "weight": c["ridge_weight"] = .25
    elif change == "endpoint": c["ridge_weight"] = 0
    elif change == "bool": c["ridge_weight"] = True
    elif change == "capacity": c["trees"] = 64
    elif change == "objective": c["objective"] = "huber"
    elif change == "id": c["id"] = "../x"
    else: c["kind"] = "residual"
    with patch.object(study, "_fit_complete") as fitting, pytest.raises(ValueError):
        study.fit_selected(c, values, observed, clock, end, config)
    fitting.assert_not_called()


@pytest.mark.parametrize("kind,seed", [("ridge", 2026)])
def test_invalid_seed_is_rejected(invented, kind, seed):
    config, values, observed, clock, end = invented
    c = condition(kind, weight=1 if kind == "ridge" else None)
    with patch.object(study, "_fit_complete") as fitting, pytest.raises(ValueError):
        study.fit_selected(c, values, observed, clock, end, config, seed=seed)
    fitting.assert_not_called()


def test_component_tampering_is_detected_before_load(fitted, tmp_path):
    study.save_model(fitted["blend-l1"][0], tmp_path)
    with (tmp_path / "direct" / "trees.txt").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="inventory"):
        study.load_model(tmp_path)


def test_manifest_objective_tampering_is_detected(fitted, tmp_path):
    study.save_model(fitted["blend-l1"][0], tmp_path)
    path = tmp_path / "ensemble.json"
    payload = json.loads(path.read_text())
    payload["condition"]["objective"] = "l2"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="parameters"):
        study.load_model(tmp_path)


def test_mismatched_component_prefix_or_target_count_is_rejected(fitted):
    bundle, _ = fitted["blend-l1"]
    for field, value in (("fit_end", 349), ("training_target_count", 100)):
        direct = replace(bundle.direct, settings={**bundle.direct.settings, field: value})
        with pytest.raises(ValueError):
            study.checked_model(replace(bundle, direct=direct))
