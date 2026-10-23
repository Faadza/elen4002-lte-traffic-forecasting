"""Fit the frozen original models and the independent 20/80 blend.

Import this numerical module inside a CPU-configured worker, after startup
thread controls. The driver owns period authorization, common targets, repeated
timing runs and scoring. This adapter never searches settings or reads results.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

from blend_revision.model import (BlendModel, FROZEN_TREE_PARAMETERS,
                                  checked_model, fit_blend, load_blend,
                                  predict_blend, save_blend)
from forecasting.data import fit_scaler, make_examples, scaled_targets, tabular_inputs
from forecasting.pilot_models import (PilotModel, _prefix, load_model,
                                      model_fingerprint, predict_model, save_model)
from models.hybrid import fit_correction
from models.lightgbm import build_parameters, fit_lightgbm
from models.ridge import fit_ridge
from preprocessing.splits import LOOKBACK

BLEND_METHOD = "ridge_direct_lightgbm_blend"
ORIGINAL_HYBRID = "ridge_lightgbm_hybrid"
METHODS = ("ridge", "direct_lightgbm", ORIGINAL_HYBRID, BLEND_METHOD)
RIDGE_ALPHA = 1000
DIRECT_TREES, DIRECT_LEAVES = 128, 7
RESIDUAL_TREES, RESIDUAL_LEAVES, RESIDUAL_WEIGHT = 32, 31, 1.0
RIDGE_WEIGHT = .2
FIT_SCOPE = (
    "Loaded arrays to fitted model: prefix slicing, scaler and feature construction, "
    "fresh complete fitting including every hybrid component; excludes disk I/O, "
    "imports, search, prediction, metric scoring and serialization."
)


def _selected_arguments(method, config, seed):
    if method not in METHODS:
        raise ValueError(f"Unknown frozen method: {method!r}")
    if seed is not None:
        raise ValueError("These frozen methods do not accept a seed argument")
    models = config["models"]
    if method in ("ridge", ORIGINAL_HYBRID, BLEND_METHOD):
        ridge = models["ridge"]
        if (RIDGE_ALPHA not in ridge["alphas"] or ridge.get("fit_intercept") is not True
                or ridge.get("intercept_penalized") is not False or ridge.get("solver") != "svd"):
            raise ValueError("Frozen ridge requires alpha1000, SVD and an unpenalized intercept")
    if method in ("direct_lightgbm", ORIGINAL_HYBRID, BLEND_METHOD):
        trees, leaves = ((RESIDUAL_TREES, RESIDUAL_LEAVES) if method == ORIGINAL_HYBRID
                         else (DIRECT_TREES, DIRECT_LEAVES))
        tree_spec = models["trees"]
        if trees not in tree_spec["tree_counts"] or leaves not in tree_spec["leaf_caps"]:
            raise ValueError("Frozen tree capacity is absent from the supplied specification")
        expected = {**FROZEN_TREE_PARAMETERS, "num_iterations": trees, "num_leaves": leaves}
        native = build_parameters(tree_spec, trees, leaves)
        if any(native.get(key) != value for key, value in expected.items()):
            raise ValueError("Tree parameters differ from the frozen squared-error configuration")
    if method == ORIGINAL_HYBRID and RESIDUAL_WEIGHT not in models["hybrid"]["weights"]:
        raise ValueError("Frozen original hybrid requires full residual correction weight1")


def _fit_complete(method, values, observed, timestamps, fit_end, config, seed, progress):
    """The entire measured fit; all model dependencies are already imported."""
    past, flags, clock = _prefix(values, observed, timestamps, fit_end)
    settings = {"fit_end": int(fit_end), "lookback": LOOKBACK}
    metadata = {"fit_interval": [0, int(fit_end)],
                "candidate_selection": "frozen development-selected configuration",
                "cached_component_reused": False}
    if method == BLEND_METHOD:
        # fit_blend creates one shared feature/scaler representation and fits
        # BOTH independent predictors afresh within this timing boundary.
        return fit_blend(past, flags, clock, fit_end, config, ridge_weight=RIDGE_WEIGHT)

    scaler = fit_scaler(past, flags, fit_end)
    examples = make_examples(past, flags, clock, 0, fit_end)
    features, target = tabular_inputs(examples, scaler), scaled_targets(examples, scaler)
    settings["training_target_count"] = metadata["training_target_count"] = len(examples.targets)
    ridge = correction = None
    if method in ("ridge", ORIGINAL_HYBRID):
        ridge = fit_ridge(features, target, RIDGE_ALPHA)
        settings["alpha"] = float(RIDGE_ALPHA)
        metadata["ridge_parameters"] = ridge.get_params(deep=False)
    if method == "ridge":
        metadata["component_fit_counts"] = {"ridge": 1}
        return PilotModel(method, ridge, scaler, settings, metadata)

    trees, leaves = ((RESIDUAL_TREES, RESIDUAL_LEAVES) if method == ORIGINAL_HYBRID
                     else (DIRECT_TREES, DIRECT_LEAVES))
    settings.update(requested_trees=trees, leaf_cap=leaves)
    if method == "direct_lightgbm":
        model = fit_lightgbm(features, target, trees, leaves, config["models"]["trees"])
        metadata["component_fit_counts"] = {"direct_lightgbm": 1}
    else:
        model = ridge
        correction = fit_correction(ridge, features, target, trees, leaves, config["models"]["trees"])
        settings["weight"] = RESIDUAL_WEIGHT
        metadata["component_fit_counts"] = {"ridge": 1, "correction": 1}
    tree = correction if correction is not None else model
    settings["tree_parameters"] = dict(tree.params)
    metadata.update(actual_trees=tree.current_iteration(), tree_parameters=dict(tree.params),
                    native_label_conversion="float32")
    return PilotModel(method, model, scaler, settings, metadata, correction)


def fit_selected(method, values, observed, timestamps, fit_end, config, seed=None, progress=None):
    """Return (bundle, stats), fitting only [0, fit_end) with frozen settings.

    There is no cached ridge dependency and no early stopping or reselection.
    One call means one complete selected fitting procedure; the experiment
    driver schedules deterministic repeats.
    """
    _selected_arguments(method, config, seed)
    start = perf_counter_ns()
    bundle = _fit_complete(method, values, observed, timestamps, fit_end, config, seed, progress)
    stop = perf_counter_ns()
    if stop < start:
        raise RuntimeError("The fitting performance counter moved backwards")
    _checked_selected(bundle)
    if isinstance(bundle, BlendModel):
        metadata = {
            "fit_interval": [0, int(fit_end)],
            "candidate_selection": "frozen development-selected configuration",
            "component_fit_counts": {"ridge": 1, "direct_lightgbm": 1},
            "training_target_count": bundle.ridge.settings["training_target_count"],
            "ridge_parameters": bundle.ridge.model.get_params(deep=False),
            "tree_parameters": dict(bundle.direct.model.params),
            "actual_trees": bundle.direct.model.current_iteration(),
            "native_label_conversion": "float32",
        }
    else:
        metadata = deepcopy(bundle.training_metadata)
    return bundle, {**metadata, "method": method, "fit_seconds": (stop-start)/1e9,
                    "fit_clock": "perf_counter_ns", "fit_scope": FIT_SCOPE,
                    "cached_component_reused": False}


def _checked_selected(bundle):
    """Validate a fitted or reloaded model before deployment; never alter it."""
    if isinstance(bundle, BlendModel):
        checked_model(bundle)
        if bundle.ridge_weight != RIDGE_WEIGHT:
            raise ValueError("The final blend requires frozen ridge weight0.2")
        return bundle
    if not isinstance(bundle, PilotModel) or bundle.method not in METHODS or bundle.method == BLEND_METHOD:
        raise ValueError("Expected one frozen selected model")
    method, settings = bundle.method, bundle.settings
    if settings.get("lookback") != LOOKBACK:
        raise ValueError("Selected model has an incompatible feature schema")
    if bundle.scaler is None or settings.get("fit_end") != bundle.scaler.fit_end:
        raise ValueError("Selected model has an inconsistent fitting prefix")
    if method in ("ridge", ORIGINAL_HYBRID):
        native = bundle.model.get_params(deep=False)
        if (settings.get("alpha") != RIDGE_ALPHA or native.get("alpha") != RIDGE_ALPHA
                or native.get("solver") != "svd" or native.get("fit_intercept") is not True
                or native.get("positive") is not False):
            raise ValueError("Selected ridge differs from the frozen alpha1000 model")
    if method in ("direct_lightgbm", ORIGINAL_HYBRID):
        trees, leaves = ((RESIDUAL_TREES, RESIDUAL_LEAVES) if method == ORIGINAL_HYBRID
                         else (DIRECT_TREES, DIRECT_LEAVES))
        if settings.get("requested_trees") != trees or settings.get("leaf_cap") != leaves:
            raise ValueError("Selected tree capacity differs from the frozen configuration")
        tree = bundle.correction_model if method == ORIGINAL_HYBRID else bundle.model
        expected = {**FROZEN_TREE_PARAMETERS, "num_iterations": trees, "num_leaves": leaves}
        for parameters in (settings.get("tree_parameters", {}), tree.params):
            normalized = {**parameters, "boosting_type": parameters.get("boosting_type", parameters.get("boosting"))}
            if any(normalized.get(key) != value for key, value in expected.items()):
                raise ValueError("Selected native trees differ from the frozen squared-error model")
        if not 1 <= tree.current_iteration() <= trees:
            raise ValueError("Selected tree count exceeds its frozen round budget")
        if method == ORIGINAL_HYBRID and settings.get("weight") != RESIDUAL_WEIGHT:
            raise ValueError("Original hybrid requires full residual correction weight1")
    return bundle


def predict_selected(bundle, values, observed, timestamps, targets):
    """Predict batch or one origin with all feature/model stages and final bound."""
    if isinstance(bundle, BlendModel):
        return predict_blend(bundle, values, observed, timestamps, targets)
    return predict_model(bundle, values, observed, timestamps, targets)


def save_selected(bundle, directory):
    """Save only a selected deployment payload; training histories remain in stats."""
    _checked_selected(bundle)
    return save_blend(bundle, directory) if isinstance(bundle, BlendModel) else save_model(bundle, directory)


def load_selected(directory):
    """Load supported model files, then reject settings outside the frozen set."""
    directory = Path(directory)
    bundle = load_blend(directory) if (directory / "blend.json").exists() else load_model(directory)
    return _checked_selected(bundle)


def selected_fingerprint(bundle):
    """Stable deployment-state digest for mutation/repetition checks, outside timers."""
    _checked_selected(bundle)
    if not isinstance(bundle, BlendModel):
        return model_fingerprint(bundle)
    state = {"method": BLEND_METHOD, "ridge_weight": bundle.ridge_weight,
             "ridge": model_fingerprint(bundle.ridge), "direct": model_fingerprint(bundle.direct)}
    return hashlib.sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()
