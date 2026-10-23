"""Independent full-target ridge/LightGBM ensembles under two training losses.

The runner owns CPU controls, chronology, selection and scoring. Import this
numerical module after applying startup CPU controls. Each fit builds the same
causal features once and freshly trains every required component; no residual
targets, cached model dependencies, early stopping or parameter search occur.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path
import re
from time import perf_counter_ns

import lightgbm as lgb
import numpy as np

from blend_revision.model import FROZEN_TREE_PARAMETERS, blend_raw_predictions
from evaluation.metrics import finalize_predictions
from forecasting.data import fit_scaler, make_examples, scaled_targets, tabular_inputs
from forecasting.pilot_models import (PilotModel, _prefix,
    load_model as load_component, model_fingerprint,
    prediction_examples, save_model as save_component)
from models.lightgbm import build_parameters, fit_lightgbm, predict_lightgbm
from models.ridge import fit_ridge, predict_ridge
from preprocessing.splits import LOOKBACK
from tools.development_store import _inventory, _read_json, verify_inventory, write_json_atomic

CONDITION_KEYS = {"id", "kind", "objective", "trees", "leaves", "ridge_weight"}
TREE_COUNTS, LEAF_CAPS = (32, 128, 512), (7, 31)
INTERIOR_WEIGHTS = (.05, .1, .2, .3, .5, .75)
RIDGE_ALPHA = 1000
FIT_SCOPE = (
    "Loaded arrays to freshly fitted complete model: prefix slicing, scaler, "
    "features and all component fits; excludes imports, disk I/O, logging, "
    "search, prediction, scoring and serialization."
)


def checked_condition(condition):
    """Validate the exact six-field scientific condition; return a private copy.

    Protocol-level canonical IDs are checked by the runner. This adapter accepts
    safe IDs so protocol naming cannot silently change the numerical conditions.
    Endpoint weight zero belongs to a standalone tree; weight one to ridge.
    """
    if not isinstance(condition, Mapping) or set(condition) != CONDITION_KEYS:
        raise ValueError("A condition requires exactly id, kind, objective, trees, leaves and ridge_weight")
    c = deepcopy(dict(condition))
    if not isinstance(c["id"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", c["id"]) is None:
        raise ValueError("Condition id must be a safe nonempty identifier")
    kind = c["kind"]
    if kind == "ridge":
        if any(c[k] is not None for k in ("objective", "trees", "leaves")):
            raise ValueError("Ridge conditions do not accept tree settings")
        weight = c["ridge_weight"]
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real) or weight != 1:
            raise ValueError("Standalone ridge requires endpoint weight1")
        c["ridge_weight"] = 1.
    elif kind in ("direct", "blend"):
        if c["objective"] not in ("l1", "l2"):
            raise ValueError("Tree objective must be l1 or l2")
        for key, choices in (("trees", TREE_COUNTS), ("leaves", LEAF_CAPS)):
            if isinstance(c[key], (bool, np.bool_)) or not isinstance(c[key], Integral) or c[key] not in choices:
                raise ValueError(f"Condition {key} is outside the declared tree capacities")
            c[key] = int(c[key])
        weight = c["ridge_weight"]
        choices = (0,) if kind == "direct" else INTERIOR_WEIGHTS
        if (isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real)
                or not np.isfinite(weight) or weight not in choices):
            raise ValueError("Direct trees require weight0; blends require a declared interior weight")
        c["ridge_weight"] = float(weight)
    else:
        raise ValueError("Unknown condition kind")
    return c


def _expected_tree(condition):
    return {**FROZEN_TREE_PARAMETERS, "num_iterations": condition["trees"],
            "num_leaves": condition["leaves"],
            "objective": "regression_l1" if condition["objective"] == "l1" else "regression"}


def _arguments(condition, config, seed):
    c = checked_condition(condition)
    kind, models = c["kind"], config["models"]
    if seed is not None:
        raise ValueError("These conditions do not accept a per-fit seed")
    if kind in ("ridge", "blend"):
        spec = models["ridge"]
        if (RIDGE_ALPHA not in spec["alphas"] or spec.get("fit_intercept") is not True
                or spec.get("intercept_penalized") is not False or spec.get("solver") != "svd"):
            raise ValueError("Ridge requires alpha1000, SVD and an unpenalized intercept")
    if kind in ("direct", "blend"):
        spec = models["trees"]
        if c["trees"] not in spec["tree_counts"] or c["leaves"] not in spec["leaf_caps"]:
            raise ValueError("The condition capacity must belong to the supplied configuration")
        actual = build_parameters(spec, c["trees"], c["leaves"])
        actual["objective"] = _expected_tree(c)["objective"]
        if any(actual.get(k) != v for k, v in _expected_tree(c).items()):
            raise ValueError("Tree settings differ from the fixed study parameters")
    return c


@dataclass(frozen=True)
class StudyModel:
    condition: dict
    scaler: object
    settings: dict
    training_metadata: dict
    ridge: PilotModel | None = None
    direct: PilotModel | None = None
    single: PilotModel | None = None


def _fit_tree(features, target, condition, settings):
    """Train trees directly on the full standardized traffic target."""
    trees, leaves = condition["trees"], condition["leaves"]
    if condition["objective"] == "l2":
        return fit_lightgbm(features, target, trees, leaves, settings)
    parameters = build_parameters(settings, trees, leaves)
    parameters["objective"] = "regression_l1"
    if (target.ndim != 1 or len(target) != len(features) or not len(target)
            or not np.isfinite(target).all() or np.any(np.abs(target) > np.finfo(np.float32).max)):
        raise ValueError("L1 full-target labels must be finite nonempty float32-representable values")
    dataset = lgb.Dataset(features, label=target, free_raw_data=True)
    return lgb.train(parameters, dataset, num_boost_round=trees)


def _fit_complete(condition, values, observed, timestamps, fit_end, config, seed):
    past, flags, clock = _prefix(values, observed, timestamps, fit_end)
    kind = condition["kind"]
    settings = {"fit_end": int(fit_end), "lookback": LOOKBACK}
    metadata = {"fit_interval": [0, int(fit_end)], "cached_component_reused": False,
                "target_design": "independent full traffic target", "component_fit_counts": {}}
    scaler = fit_scaler(past, flags, fit_end)
    examples = make_examples(past, flags, clock, 0, fit_end)
    features, target = tabular_inputs(examples, scaler), scaled_targets(examples, scaler)
    settings["training_target_count"] = metadata["training_target_count"] = len(target)
    ridge = direct = None
    if kind in ("ridge", "blend"):
        ridge_model = fit_ridge(features, target, RIDGE_ALPHA)
        ridge = PilotModel("ridge", ridge_model, scaler, {**settings, "alpha": float(RIDGE_ALPHA)},
                           {"component_fit_counts": {"ridge": 1}})
        metadata["component_fit_counts"]["ridge"] = 1
        metadata["ridge_parameters"] = ridge_model.get_params(deep=False)
    if kind in ("direct", "blend"):
        tree = _fit_tree(features, target, condition, config["models"]["trees"])
        direct = PilotModel("direct_lightgbm", tree, scaler,
            {**settings, "requested_trees": condition["trees"], "leaf_cap": condition["leaves"],
             "tree_parameters": dict(tree.params)}, {"component_fit_counts": {"direct_lightgbm": 1}})
        metadata["component_fit_counts"]["direct_lightgbm"] = 1
        metadata.update(tree_parameters=dict(tree.params), actual_trees=tree.current_iteration(),
                        native_label_conversion="float32")
    return StudyModel(condition, scaler, settings, metadata, ridge=ridge, direct=direct)


def fit_selected(condition, values, observed, timestamps, fit_end, config, seed=None):
    """Return (StudyModel, stats); measure features plus every fresh component fit."""
    condition = _arguments(condition, config, seed)
    start = perf_counter_ns()
    bundle = _fit_complete(condition, values, observed, timestamps, fit_end, config, seed)
    stop = perf_counter_ns()
    if stop < start:
        raise RuntimeError("The fitting timer moved backwards")
    checked_model(bundle)
    return bundle, {**deepcopy(bundle.training_metadata), "condition": deepcopy(condition),
                    "fit_seconds": (stop-start)/1e9, "fit_clock": "perf_counter_ns", "fit_scope": FIT_SCOPE}


def _tree_check(component, condition):
    tree, settings = component.model, component.settings
    if (component.method != "direct_lightgbm" or component.correction_model is not None
            or settings.get("requested_trees") != condition["trees"]
            or settings.get("leaf_cap") != condition["leaves"] or tree.num_feature() != 340
            or not 1 <= tree.current_iteration() <= condition["trees"]
            or tree.num_model_per_iteration() != 1):
        raise ValueError("Direct-tree identity or capacity differs from its study condition")
    for parameters in (settings.get("tree_parameters", {}), tree.params):
        normalized = {**parameters, "boosting_type": parameters.get("boosting_type", parameters.get("boosting"))}
        if any(normalized.get(key) != value for key, value in _expected_tree(condition).items()):
            raise ValueError("Native tree parameters differ from the requested training objective or capacity")
        # Native text reload exposes the automatically inferred objective metric.
        # It must agree with the objective; it is not an early-stopping rule.
        metric = parameters.get("metric")
        if metric is not None:
            values = metric if isinstance(metric, (list, tuple)) else [metric]
            if list(values) != [condition["objective"]]:
                raise ValueError("Effective native metric differs from the tree objective")


def checked_model(bundle):
    """Verify deployed components match the independent full-target condition."""
    if not isinstance(bundle, StudyModel):
        raise ValueError("Expected a StudyModel")
    condition = checked_condition(bundle.condition)
    kind, settings = condition["kind"], bundle.settings
    expected = ({"ridge"} if kind == "ridge" else {"direct"} if kind == "direct" else {"ridge", "direct"})
    if {name for name in ("ridge", "direct", "single") if getattr(bundle, name) is not None} != expected:
        raise ValueError("Model components do not match the requested study condition")
    end = settings.get("fit_end")
    if (isinstance(end, (bool, np.bool_)) or not isinstance(end, Integral) or end < 1
            or settings.get("lookback") != LOOKBACK):
        raise ValueError("Invalid fitted prefix or feature schema")
    if bundle.scaler is None or bundle.scaler.fit_end != end:
        raise ValueError("Study model scaler belongs to a different fitting prefix")
    for name in expected:
        component = getattr(bundle, name)
        if (not isinstance(component, PilotModel) or component.correction_model is not None
                or component.scaler != bundle.scaler or component.settings.get("fit_end") != end
                or component.settings.get("lookback") != LOOKBACK):
            raise ValueError("Component scaler, feature schema or prefix mismatch")
        count = settings.get("training_target_count")
        if (isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 1
                or component.settings.get("training_target_count") != count):
            raise ValueError("Components must use the same nonempty supervised training rows")
    if bundle.ridge is not None:
        component = bundle.ridge
        native = component.model.get_params(deep=False)
        if (component.method != "ridge" or component.settings.get("alpha") != RIDGE_ALPHA
                or native.get("alpha") != RIDGE_ALPHA or native.get("solver") != "svd"
                or native.get("fit_intercept") is not True or native.get("positive") is not False
                or component.model.n_features_in_ != 340):
            raise ValueError("Ridge component differs from the selected alpha1000 model")
    if bundle.direct is not None:
        _tree_check(bundle.direct, condition)
    return bundle


def predict_with_components(bundle, values, observed, timestamps, targets):
    """Expose raw original-unit component outputs before the one final bound."""
    checked_model(bundle)
    kind = bundle.condition["kind"]
    ridge_raw = direct_raw = None
    examples = prediction_examples(values, observed, timestamps, targets)
    features = tabular_inputs(examples, bundle.scaler)
    if bundle.ridge is not None:
        ridge_raw = bundle.scaler.inverse(predict_ridge(bundle.ridge.model, features))
    if bundle.direct is not None:
        direct_raw = bundle.scaler.inverse(predict_lightgbm(bundle.direct.model, features))
    if kind == "blend":
        raw, final = blend_raw_predictions(ridge_raw, direct_raw, bundle.condition["ridge_weight"])
    else:
        raw = ridge_raw.copy() if kind == "ridge" else direct_raw.copy()
        final, _ = finalize_predictions(raw)
    return {"raw": raw, "final": final, "ridge_raw": ridge_raw, "direct_raw": direct_raw}


def predict(bundle, values, observed, timestamps, targets):
    """Return raw and zero-bounded predictions; both model stages are included."""
    result = predict_with_components(bundle, values, observed, timestamps, targets)
    return result["raw"], result["final"]


def predict_raw(bundle, values, observed, timestamps, targets):
    return predict_with_components(bundle, values, observed, timestamps, targets)["raw"]


predict_selected = predict


def fingerprint(bundle):
    """Hash complete deployment state outside fitting/prediction timers."""
    checked_model(bundle)
    payload = {"condition": bundle.condition, "settings": bundle.settings,
               "components": {name: model_fingerprint(getattr(bundle, name))
                              for name in ("ridge", "direct", "single") if getattr(bundle, name) is not None}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_model(bundle, directory):
    """Serialize all required components and a complete hashed deployment manifest."""
    checked_model(bundle)
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Use a new or empty study-model directory")
    directory.mkdir(parents=True, exist_ok=True)
    names = [name for name in ("ridge", "direct", "single") if getattr(bundle, name) is not None]
    for name in names:
        save_component(getattr(bundle, name), directory / name)
    payload = {"schema_version": 1, "condition": deepcopy(bundle.condition),
               "settings": deepcopy(bundle.settings), "components": names,
               "files_sha256": _inventory(directory)}
    write_json_atomic(directory / "ensemble.json", payload)
    return directory


def load_model(directory):
    """Verify immutable component files before native model deserialization."""
    directory = Path(directory)
    payload = _read_json(directory / "ensemble.json")
    if (set(payload) != {"schema_version", "condition", "settings", "components", "files_sha256"}
            or payload["schema_version"] != 1):
        raise ValueError("Unsupported study-model deployment manifest")
    condition = checked_condition(payload["condition"])
    expected = (["ridge"] if condition["kind"] == "ridge" else
                ["direct"] if condition["kind"] == "direct" else ["ridge", "direct"])
    if payload["components"] != expected or {p.name for p in directory.iterdir()} != {"ensemble.json", *expected}:
        raise ValueError("Unexpected study-model component inventory")
    verify_inventory(directory, payload["files_sha256"], exclude=("ensemble.json",))
    components = {name: load_component(directory / name) for name in expected}
    scaler = components[expected[0]].scaler
    return checked_model(StudyModel(condition, scaler, payload["settings"], {}, **components))
