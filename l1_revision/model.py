"""Fit L1 trees against traffic minus a fixed partial ridge initialization.

Beta zero is a real standalone L1 tree model: it never reads or requires ridge.
For positive beta, a verified alpha-1000 ridge dependency is reused, not fitted.
The experiment runner owns CPU controls, artifact provenance and authorization.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
import hashlib
import json

import lightgbm as lgb
import numpy as np

from evaluation.metrics import finalize_predictions
from forecasting.data import (TrafficScaler, fit_scaler, make_examples,
                              scaled_targets, tabular_inputs)
from forecasting.pilot_models import (PilotModel, _prefix, load_model as load_ridge_bundle,
                                     model_fingerprint as ridge_fingerprint,
                                     prediction_examples, save_model as save_ridge_bundle)
from models.lightgbm import build_parameters, predict_lightgbm
from models.ridge import predict_ridge
from preprocessing.splits import LOOKBACK

BETAS = (0, .25, .5, 1)
CAPACITIES = ((128, 7), (32, 31))
RIDGE_ALPHA = 1000


def checked_beta(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value not in BETAS):
        raise ValueError("beta must be one of 0, 0.25, 0.5, 1")
    return float(value)


def _vector(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "biuf":
        raise ValueError("Predictions/targets must be one-dimensional real arrays")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Predictions/targets must be finite")
    return values


def residual_targets(target_z, ridge_z, beta):
    """No centering, clipping or extra scaling; beta zero never reads ridge_z."""
    beta, target = checked_beta(beta), _vector(target_z)
    if beta == 0:
        return target.copy()
    ridge = _vector(ridge_z)
    if target.shape != ridge.shape:
        raise ValueError("Target and ridge prediction shapes must match")
    with np.errstate(over="ignore", invalid="ignore"):
        result = target - beta * ridge
    return _vector(result)


def combine_predictions(ridge_z, tree_z, beta):
    """Return beta*ridge + tree in standardized units, with no clipping."""
    beta, tree = checked_beta(beta), _vector(tree_z)
    if beta == 0:
        return tree.copy()
    ridge = _vector(ridge_z)
    if ridge.shape != tree.shape:
        raise ValueError("Ridge and tree prediction shapes must match")
    with np.errstate(over="ignore", invalid="ignore"):
        return _vector(beta * ridge + tree)


def _job(job):
    required = {"id", "cell", "fold_index", "fit_end", "score_start", "score_end",
                "candidate_id", "parameters"}
    if not isinstance(job, Mapping) or not required <= set(job):
        raise ValueError("Candidate job lacks identity, chronology or parameters")
    for key in ("id", "cell", "candidate_id"):
        if not isinstance(job[key], str) or not job[key]:
            raise ValueError(f"Job {key} must be a nonempty string")
    for key in ("fold_index", "fit_end", "score_start", "score_end"):
        if (isinstance(job[key], (bool, np.bool_)) or not isinstance(job[key], Integral)
                or job[key] < (0 if key == "fold_index" else 1)):
            raise ValueError(f"Job {key} must be a valid nonnegative calendar integer")
    if job["score_start"] != job["fit_end"] or job["score_end"] <= job["fit_end"]:
        raise ValueError("Scoring must immediately follow the fitting prefix")
    if job.get("seed") is not None:
        raise ValueError("The deterministic L1 candidate has no per-job seed")
    p = job["parameters"]
    if not isinstance(p, Mapping) or set(p) != {"beta", "trees", "leaves"}:
        raise ValueError("Expected exactly beta, trees and leaves parameters")
    beta = checked_beta(p["beta"])
    if any(isinstance(p[k], (bool, np.bool_)) or not isinstance(p[k], Integral)
           for k in ("trees", "leaves")) or (p["trees"], p["leaves"]) not in CAPACITIES:
        raise ValueError("Tree configuration is outside the two declared capacities")
    return beta, int(p["trees"]), int(p["leaves"])


def _cached_ridge(bundle, job, scaler, count):
    if (not isinstance(bundle, PilotModel) or bundle.method != "ridge"
            or bundle.correction_model is not None or bundle.scaler != scaler):
        raise ValueError("Positive beta requires a same-prefix standalone ridge/scaler")
    s, identity = bundle.settings, bundle.settings.get("development_job", {})
    if (s.get("fit_end") != job["fit_end"] or s.get("lookback") != LOOKBACK
            or s.get("alpha") != RIDGE_ALPHA or s.get("training_target_count") != count
            or identity.get("cell") != job["cell"]
            or identity.get("fold_index") != job["fold_index"]
            or identity.get("stage") != "ridge"
            or identity.get("candidate_id") != "ridge-a1000" or not identity.get("id")):
        raise ValueError("Cached ridge identity, prefix, training count or alpha differs")
    p = bundle.model.get_params(deep=False)
    if (p.get("alpha") != RIDGE_ALPHA or p.get("solver") != "svd"
            or p.get("fit_intercept") is not True or p.get("positive") is not False
            or bundle.model.n_features_in_ != 340):
        raise ValueError("Cached ridge native model differs from declared alpha1000 model")
    return bundle


@dataclass(frozen=True)
class L1Model:
    model: object
    scaler: TrafficScaler
    beta: float
    settings: dict
    training_metadata: dict
    ridge_bundle: PilotModel | None = None


def fit_candidate(job, values, observed, timestamps, config, *, ridge_bundle=None):
    """Fit one declared L1 tree, excluding every value at/after fit_end."""
    beta, trees, leaves = _job(job)
    if beta == 0 and ridge_bundle is not None:
        raise ValueError("Standalone beta zero must not receive a ridge dependency")
    parameters = build_parameters(config["models"]["trees"], trees, leaves)
    parameters["objective"] = "regression_l1"
    past, flags, clock = _prefix(values, observed, timestamps, job["fit_end"])
    scaler = fit_scaler(past, flags, job["fit_end"])
    examples = make_examples(past, flags, clock, 0, job["fit_end"])
    features, target = tabular_inputs(examples, scaler), scaled_targets(examples, scaler)
    ridge_z = None
    metadata = {"fit_interval": [0, int(job["fit_end"])],
                "training_target_count": len(target), "native_label_conversion": "float32",
                "component_fit_counts": {"direct_lightgbm_l1" if beta == 0 else "correction_l1": 1}}
    settings = {"job": deepcopy(dict(job)), "fit_end": int(job["fit_end"]), "lookback": LOOKBACK,
                "training_target_count": len(target), "requested_trees": trees, "leaf_cap": leaves,
                "tree_parameters": parameters}
    if beta != 0:
        _cached_ridge(ridge_bundle, job, scaler, len(target))
        ridge_z = predict_ridge(ridge_bundle.model, features)
        settings["ridge_state_sha256"] = ridge_fingerprint(ridge_bundle)
        settings["ridge_dependency_job"] = deepcopy(ridge_bundle.settings["development_job"])
        metadata["ridge_reused"] = True
    labels = residual_targets(target, ridge_z, beta)
    if not len(labels) or np.any(np.abs(labels) > np.finfo(np.float32).max):
        raise ValueError("L1 labels must be nonempty and representable as finite float32")
    dataset = lgb.Dataset(features, label=labels, free_raw_data=True)
    model = lgb.train(parameters, dataset, num_boost_round=trees)
    if not 1 <= model.current_iteration() <= trees or model.num_model_per_iteration() != 1:
        raise ValueError("L1 tree must be a fitted single-output model within the round budget")
    metadata["actual_trees"] = model.current_iteration()
    return L1Model(model, scaler, beta, settings, metadata, ridge_bundle)


def _checked_model(bundle, *, check_dependency_hash=False):
    if not isinstance(bundle, L1Model):
        raise ValueError("Expected an L1Model")
    beta, trees, leaves = _job(bundle.settings["job"])
    if (beta != checked_beta(bundle.beta) or bundle.scaler.fit_end != bundle.settings["fit_end"]
            or bundle.settings["fit_end"] != bundle.settings["job"]["fit_end"]
            or bundle.settings["lookback"] != LOOKBACK
            or bundle.settings["requested_trees"] != trees or bundle.settings["leaf_cap"] != leaves
            or bundle.model.num_feature() != 340
            or not 1 <= bundle.model.current_iteration() <= trees
            or bundle.model.num_model_per_iteration() != 1):
        raise ValueError("L1 model schema, prefix, capacity or beta is inconsistent")
    expected = bundle.settings["tree_parameters"]
    if expected.get("objective") != "regression_l1":
        raise ValueError("Expected L1 objective")
    actual = dict(bundle.model.params)
    actual["boosting_type"] = actual.get("boosting_type", actual.get("boosting"))
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"Native L1 parameter differs: {key}")
    if beta == 0:
        if bundle.ridge_bundle is not None or any(key.startswith("ridge_") for key in bundle.settings):
            raise ValueError("Standalone L1 must not contain ridge dependency state")
    else:
        _cached_ridge(bundle.ridge_bundle, bundle.settings["job"], bundle.scaler,
                      bundle.settings["training_target_count"])
        if (bundle.settings["ridge_dependency_job"] != bundle.ridge_bundle.settings["development_job"]
                or (check_dependency_hash and bundle.settings["ridge_state_sha256"]
                    != ridge_fingerprint(bundle.ridge_bundle))):
            raise ValueError("Ridge dependency identity or state hash differs")
    return bundle


def predict_components(bundle, values, observed, timestamps, targets):
    """Return (ridge_z, tree_z); beta zero uses zeros without ridge inference."""
    _checked_model(bundle)
    examples = prediction_examples(values, observed, timestamps, targets)
    features = tabular_inputs(examples, bundle.scaler)
    tree_z = predict_lightgbm(bundle.model, features)
    ridge_z = np.zeros_like(tree_z) if bundle.beta == 0 else predict_ridge(bundle.ridge_bundle.model, features)
    return ridge_z, tree_z


def predict_candidate(bundle, values, observed, timestamps, targets):
    """Predict from common causal features; inverse once, then bound once."""
    ridge_z, tree_z = predict_components(bundle, values, observed, timestamps, targets)
    raw = bundle.scaler.inverse(combine_predictions(ridge_z, tree_z, bundle.beta))
    final, _ = finalize_predictions(raw)
    return raw, final


def _json_bytes(payload):
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _payload(bundle):
    return {"schema_version": 1, "method": "direct_lightgbm_l1" if bundle.beta == 0 else "partial_ridge_lightgbm_l1",
            "beta": bundle.beta, "scaler": asdict(bundle.scaler), "settings": bundle.settings}


def model_fingerprint(bundle):
    """Hash deployment state; caller may compare before/after predictions."""
    _checked_model(bundle, check_dependency_hash=True)
    payload = {**_payload(bundle), "trees": bundle.model.model_to_string()}
    if bundle.beta != 0:
        payload["ridge_state_sha256"] = ridge_fingerprint(bundle.ridge_bundle)
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def save_model(bundle, directory):
    """Save native model-only files plus a manifest; standalone has no ridge."""
    _checked_model(bundle, check_dependency_hash=True)
    directory = Path(directory)
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError("Use a new or empty model directory")
    directory.mkdir(parents=True, exist_ok=True)
    bundle.model.save_model(str(directory / "trees.txt"))
    (directory / "model.json").write_bytes(_json_bytes(_payload(bundle)))
    names = ["model.json", "trees.txt"]
    if bundle.beta != 0:
        save_ridge_bundle(bundle.ridge_bundle, directory / "ridge")
        names += ["ridge/model.json", "ridge/ridge.json"]
    manifest = {"schema_version": 1, "files_sha256": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}}
    (directory / "manifest.json").write_bytes(_json_bytes(manifest))
    return directory


def load_model(directory):
    """Verify exact native payload inventory and hashes before deserialization."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if set(manifest) != {"schema_version", "files_sha256"} or manifest["schema_version"] != 1:
        raise ValueError("Unsupported L1 deployment manifest")
    names = set(manifest["files_sha256"])
    if names not in ({"model.json", "trees.txt"},
                     {"model.json", "trees.txt", "ridge/model.json", "ridge/ridge.json"}):
        raise ValueError("Unexpected L1 deployment inventory")
    present = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if present != names | {"manifest.json"}:
        raise ValueError("Deployment file inventory differs")
    for name, digest in manifest["files_sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Deployment hash mismatch: {name}")
    payload = json.loads((directory / "model.json").read_text())
    if set(payload) != {"schema_version", "method", "beta", "scaler", "settings"} or payload["schema_version"] != 1:
        raise ValueError("Unsupported L1 model schema")
    beta = checked_beta(payload["beta"])
    expected_method = "direct_lightgbm_l1" if beta == 0 else "partial_ridge_lightgbm_l1"
    if payload["method"] != expected_method or (len(names) == 4) != (beta != 0):
        raise ValueError("L1 method, beta and dependency inventory differ")
    ridge = load_ridge_bundle(directory / "ridge") if beta != 0 else None
    bundle = L1Model(lgb.Booster(model_file=str(directory / "trees.txt")),
                     TrafficScaler(**payload["scaler"]), beta, payload["settings"], {}, ridge)
    return _checked_model(bundle, check_dependency_hash=True)
