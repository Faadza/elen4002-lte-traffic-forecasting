"""A fixed convex blend of two independently fitted traffic forecasters.

The original residual-correction model is unchanged. This module shares its
checked feature/scaler/model primitives, but the tree target is traffic itself.
Fit functions are library interfaces, not permission to open held-out periods.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

from evaluation.metrics import finalize_predictions
from preprocessing.splits import LOOKBACK

if TYPE_CHECKING:
    from forecasting.pilot_models import PilotModel

RIDGE_ALPHA = 1000
TREE_COUNT = 128
LEAF_CAP = 7
SELECTED_RIDGE_WEIGHT = 0.2
MODEL_FILES = ("ridge/model.json", "ridge/ridge.json", "direct/model.json", "direct/trees.txt")
FROZEN_TREE_PARAMETERS = {"objective": "regression", "boosting_type": "gbdt",
                          "device_type": "cpu", "deterministic": True,
                          "force_col_wise": True, "feature_fraction": 1.0,
                          "bagging_fraction": 1.0, "bagging_freq": 0,
                          "learning_rate": .05, "lambda_l2": 1.0,
                          "min_data_in_leaf": 48, "seed": 2026,
                          "num_iterations": TREE_COUNT, "num_leaves": LEAF_CAP,
                          "num_threads": 1}


def checked_weight(value):
    """Reject booleans, strings and nonfinite or nonconvex weights."""
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Ridge weight must be a finite real scalar in [0, 1]")
    return float(value)


def _vector(values):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "biuf" or array.size == 0:
        raise ValueError("Raw component predictions must be nonempty one-dimensional real arrays")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("Raw component predictions must be finite")
    return array


def blend_raw_predictions(ridge_raw, direct_raw, ridge_weight=SELECTED_RIDGE_WEIGHT):
    """Blend unclipped ORIGINAL-unit predictions; apply the final bound once.

    Endpoints return exact copies of their selected component. Both input arrays
    must still be finite and shape-compatible, including at endpoints.
    """
    weight = checked_weight(ridge_weight)
    ridge, direct = _vector(ridge_raw), _vector(direct_raw)
    if ridge.shape != direct.shape:
        raise ValueError("Component prediction shapes must match")
    if weight == 0:
        raw = direct.copy()
    elif weight == 1:
        raw = ridge.copy()
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            raw = weight * ridge + (1 - weight) * direct
    if not np.isfinite(raw).all():
        raise ValueError("Combined raw predictions must remain finite")
    final, _ = finalize_predictions(raw)
    return raw, final


@dataclass(frozen=True)
class BlendModel:
    ridge: PilotModel
    direct: PilotModel
    ridge_weight: float = SELECTED_RIDGE_WEIGHT


def checked_model(bundle):
    from forecasting.pilot_models import PilotModel

    if not isinstance(bundle, BlendModel):
        raise ValueError("Expected a BlendModel")
    checked_weight(bundle.ridge_weight)
    ridge, direct = bundle.ridge, bundle.direct
    if (not isinstance(ridge, PilotModel) or not isinstance(direct, PilotModel)
            or ridge.method != "ridge" or direct.method != "direct_lightgbm"
            or ridge.correction_model is not None or direct.correction_model is not None):
        raise ValueError("Blend needs independent ridge and direct LightGBM components")
    if ridge.scaler is None or ridge.scaler != direct.scaler:
        raise ValueError("Components must share the same fitting-prefix scaler")
    for component in (ridge, direct):
        if (component.settings.get("fit_end") != ridge.scaler.fit_end
                or component.settings.get("lookback") != LOOKBACK):
            raise ValueError("Component fitting prefix or feature schema differs")
    if (ridge.settings.get("alpha") != RIDGE_ALPHA
            or direct.settings.get("requested_trees") != TREE_COUNT
            or direct.settings.get("leaf_cap") != LEAF_CAP):
        raise ValueError("Components must use frozen alpha1000 / 128-tree / 7-leaf settings")
    native = ridge.model.get_params(deep=False)
    if (native.get("alpha") != RIDGE_ALPHA or native.get("solver") != "svd"
            or native.get("fit_intercept") is not True or native.get("positive") is not False):
        raise ValueError("Native ridge differs from frozen configuration")
    for parameters in (direct.settings.get("tree_parameters", {}), direct.model.params):
        # Native text reload canonicalizes boosting_type to boosting.
        native_parameters = {**parameters, "boosting_type": parameters.get(
            "boosting_type", parameters.get("boosting"))}
        if any(native_parameters.get(key) != value for key, value in FROZEN_TREE_PARAMETERS.items()):
            raise ValueError("Native or recorded tree parameters differ from frozen configuration")
    if not 1 <= direct.model.current_iteration() <= TREE_COUNT:
        raise ValueError("Actual tree count must respect the frozen round budget")
    if (ridge.settings.get("training_target_count") != direct.settings.get("training_target_count")
            or not isinstance(ridge.settings.get("training_target_count"), int)
            or ridge.settings["training_target_count"] <= 0):
        raise ValueError("Components must share a nonempty training target set")
    return bundle


def fit_blend(values, observed, timestamps, fit_end, config,
              ridge_weight=SELECTED_RIDGE_WEIGHT):
    """Fit the frozen independent components on ONE permitted prefix/scaler.

    Only [0, fit_end) is passed to feature construction or either estimator.
    Prefix authorization and the evaluation mask belong to the experiment
    driver; this function never selects hyperparameters or reads result files.
    """
    from forecasting.data import fit_scaler, make_examples, scaled_targets, tabular_inputs
    from forecasting.pilot_models import PilotModel, _prefix
    from models.lightgbm import build_parameters, fit_lightgbm
    from models.ridge import fit_ridge

    weight = checked_weight(ridge_weight)
    # Validate the fixed selected candidates before any fitting.
    models = config["models"]
    if (RIDGE_ALPHA not in models["ridge"]["alphas"]
            or TREE_COUNT not in models["trees"]["tree_counts"]
            or LEAF_CAP not in models["trees"]["leaf_caps"]):
        raise ValueError("Frozen blend components are absent from the supplied specification")
    tree_settings = models["trees"]
    parameters = build_parameters(tree_settings, TREE_COUNT, LEAF_CAP)
    if any(parameters.get(key) != value for key, value in FROZEN_TREE_PARAMETERS.items()):
        raise ValueError("Supplied tree settings differ from the frozen selected configuration")
    past, flags, clock = _prefix(values, observed, timestamps, fit_end)
    scaler = fit_scaler(past, flags, fit_end)
    examples = make_examples(past, flags, clock, 0, fit_end)
    features, target = tabular_inputs(examples, scaler), scaled_targets(examples, scaler)
    ridge = fit_ridge(features, target, RIDGE_ALPHA)
    direct = fit_lightgbm(features, target, TREE_COUNT, LEAF_CAP, tree_settings)
    common = {"fit_end": int(fit_end), "lookback": LOOKBACK,
              "training_target_count": len(examples.targets)}
    metadata = {"fit_interval": [0, int(fit_end)],
                "candidate_selection": "fixed development-selected convex blend components"}
    ridge_bundle = PilotModel("ridge", ridge, scaler, {**common, "alpha": float(RIDGE_ALPHA)},
                              {**metadata, "component_fit_counts": {"ridge": 1}})
    direct_bundle = PilotModel("direct_lightgbm", direct, scaler,
                               {**common, "requested_trees": TREE_COUNT, "leaf_cap": LEAF_CAP,
                                "tree_parameters": dict(direct.params)},
                               {**metadata, "component_fit_counts": {"direct_lightgbm": 1}})
    return checked_model(BlendModel(ridge_bundle, direct_bundle, weight))


def predict_blend(bundle, values, observed, timestamps, targets):
    """Build causal features once and combine raw outputs without refitting."""
    from forecasting.data import tabular_inputs
    from forecasting.pilot_models import prediction_examples
    from models.lightgbm import predict_lightgbm
    from models.ridge import predict_ridge

    checked_model(bundle)
    examples = prediction_examples(values, observed, timestamps, targets)
    features = tabular_inputs(examples, bundle.ridge.scaler)
    ridge_raw = bundle.ridge.scaler.inverse(predict_ridge(bundle.ridge.model, features))
    direct_raw = bundle.direct.scaler.inverse(predict_lightgbm(bundle.direct.model, features))
    return blend_raw_predictions(ridge_raw, direct_raw, bundle.ridge_weight)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_blend(bundle, directory):
    """Save two checked model-only payloads and their SHA256 manifest."""
    from forecasting.pilot_models import save_model

    checked_model(bundle)
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Use a new or empty blend-model directory")
    directory.mkdir(parents=True, exist_ok=True)
    save_model(bundle.ridge, directory / "ridge")
    save_model(bundle.direct, directory / "direct")
    payload = {"schema_version": 1, "method": "ridge_direct_lightgbm_blend",
               "ridge_weight": checked_weight(bundle.ridge_weight),
               "combination": "raw_original_units_then_zero_bound",
               "files": {name: _sha256(directory / name) for name in MODEL_FILES}}
    (directory / "blend.json").write_text(json.dumps(payload, indent=2, sort_keys=True,
                                                    allow_nan=False) + "\n")
    return directory


def load_blend(directory):
    """Verify hashes before loading native JSON/tree payloads (no pickle)."""
    from forecasting.pilot_models import load_model

    directory = Path(directory)
    payload = json.loads((directory / "blend.json").read_text())
    if (set(payload) != {"schema_version", "method", "ridge_weight", "combination", "files"}
            or payload["schema_version"] != 1
            or payload["method"] != "ridge_direct_lightgbm_blend"
            or payload["combination"] != "raw_original_units_then_zero_bound"
            or set(payload["files"]) != set(MODEL_FILES)
            or {p.name for p in directory.iterdir()} != {"blend.json", "ridge", "direct"}):
        raise ValueError("Invalid blend deployment manifest or file set")
    weight = checked_weight(payload["ridge_weight"])
    for name, expected in payload["files"].items():
        if _sha256(directory / name) != expected:
            raise ValueError(f"Blend component hash mismatch: {name}")
    return checked_model(BlendModel(load_model(directory / "ridge"),
                                    load_model(directory / "direct"), weight))
