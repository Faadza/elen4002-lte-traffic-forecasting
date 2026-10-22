"""Common fit/predict/artifact interface for fixed-candidate development pilots.

These adapters reuse the separately checked models. They do not choose targets,
score models, search settings, or authorise an experiment. Callers supply the
validated hourly arrays and the same preselected target indices for every method.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from evaluation.metrics import finalize_predictions
from forecasting.data import (Examples, TrafficScaler, fit_scaler, lstm_inputs,
                              make_examples, scaled_targets, tabular_inputs)
from forecasting.lstm_training import fit_development, state_fingerprint
from models.baselines import persistence, daily_naive, weekly_naive
from models.hybrid import fit_correction, predict_hybrid
from models.lightgbm import fit_lightgbm, predict_lightgbm
from models.lstm import TrafficLSTM, predict_lstm
from models.ridge import fit_ridge, predict_ridge
from preprocessing.splits import LOOKBACK

BASELINES = {"persistence": (persistence, 1), "daily_seasonal_naive": (daily_naive, 24),
             "weekly_seasonal_naive": (weekly_naive, 168)}
METHODS = (*BASELINES, "ridge", "direct_lightgbm", "ridge_lightgbm_hybrid", "lstm")


@dataclass(frozen=True)
class PilotModel:
    method: str
    model: object
    scaler: TrafficScaler | None
    settings: dict
    training_metadata: dict
    correction_model: object = None


def _method(method):
    if method not in METHODS:
        raise ValueError(f"Unknown pilot method: {method!r}")
    return method


def _prefix(values, observed, timestamps, fit_end):
    if (isinstance(fit_end, (bool, np.bool_)) or not isinstance(fit_end, (int, np.integer))
            or fit_end <= 0 or fit_end > len(values)):
        raise ValueError("fit_end must be a positive integer within the hourly calendar")
    if not len(values) == len(observed) == len(timestamps):
        raise ValueError("Traffic, observation mask and clock must have matching lengths")
    # Nothing at or after fit_end is converted, checked or supplied to fitting.
    return values[:fit_end], observed[:fit_end], timestamps[:fit_end]


def fit_model(method, values, observed, timestamps, fit_end, config, seed=None, progress=None):
    """Fit only the first declared candidate; a hybrid includes its own ridge fit."""
    method = _method(method)
    past, flags, clock = _prefix(values, observed, timestamps, fit_end)
    settings = {"fit_end": int(fit_end), "lookback": LOOKBACK}
    metadata = {"fit_interval": [0, int(fit_end)], "candidate_selection": "first declared settings"}
    if method != "lstm" and seed is not None:
        raise ValueError("The seed argument is only for the declared LSTM seeds")
    if method in BASELINES:
        settings["lag"] = BASELINES[method][1]
        return PilotModel(method, None, None, settings, {**metadata, "component_fit_counts": {}})
    if method == "lstm":
        spec = config["models"]["lstm"]
        seed = spec["seeds_all_formal_periods"][0] if seed is None else seed
        hidden, dropout = spec["hidden_units"][0], spec["output_dropout"][0]
        fitted = fit_development(past, flags, clock, fit_end, hidden_units=hidden,
                                 dropout=dropout, seed=seed, settings=spec, progress=progress)
        fitted.model.eval()
        settings.update(hidden_units=int(hidden), dropout=float(dropout), seed=int(seed),
                        batch_size=int(spec["batch_size"]), selected_epochs=fitted.metadata["selected_epochs"])
        # Keep training traces separate; the discarded inner model is not deployed.
        return PilotModel(method, fitted.model, fitted.scaler, settings,
                          {**metadata, **fitted.metadata})

    scaler = fit_scaler(past, flags, fit_end)
    examples = make_examples(past, flags, clock, 0, fit_end)
    features, targets = tabular_inputs(examples, scaler), scaled_targets(examples, scaler)
    metadata["training_target_count"] = len(examples.targets)
    ridge = correction = None
    if method in ("ridge", "ridge_lightgbm_hybrid"):
        alpha = config["models"]["ridge"]["alphas"][0]
        ridge = fit_ridge(features, targets, alpha)
        settings["alpha"] = float(alpha)
        metadata["ridge_parameters"] = ridge.get_params(deep=False)
    if method == "ridge":
        metadata["component_fit_counts"] = {"ridge": 1}
        return PilotModel(method, ridge, scaler, settings, metadata)

    tree_spec = config["models"]["trees"]
    trees, leaves = tree_spec["tree_counts"][0], tree_spec["leaf_caps"][0]
    settings.update(requested_trees=int(trees), leaf_cap=int(leaves))
    if method == "direct_lightgbm":
        model = fit_lightgbm(features, targets, trees, leaves, tree_spec)
        metadata["component_fit_counts"] = {"direct_lightgbm": 1}
    else:
        correction = fit_correction(ridge, features, targets, trees, leaves, tree_spec)
        model = ridge
        settings["weight"] = 1.0
        metadata["component_fit_counts"] = {"ridge": 1, "correction": 1}
    tree = correction if correction is not None else model
    settings["tree_parameters"] = dict(tree.params)
    metadata.update(actual_trees=tree.current_iteration(), tree_parameters=dict(tree.params),
                    native_label_conversion="float32")
    return PilotModel(method, model, scaler, settings, metadata, correction)


def _prediction_indices(values, observed, timestamps, targets, minimum):
    # Requiring already loaded arrays avoids a hidden whole-series copy per origin.
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype != np.float64:
        raise ValueError("Prediction traffic must be a one-dimensional float64 NumPy array")
    if not isinstance(observed, np.ndarray) or observed.ndim != 1:
        raise ValueError("Prediction observation mask must be a one-dimensional NumPy array")
    if not isinstance(timestamps, pd.DatetimeIndex) or timestamps.tz is not None:
        raise ValueError("Prediction clock must be the recorded timezone-naive DatetimeIndex")
    if not len(values) == len(observed) == len(timestamps):
        raise ValueError("Prediction arrays and clock must have matching lengths")
    positions = np.asarray(targets)
    if positions.ndim != 1 or (positions.size and positions.dtype.kind not in "iu"):
        raise ValueError("Targets must be one-dimensional integer calendar indices")
    if positions.size and ((positions < minimum).any() or (positions >= len(timestamps)).any()
                           or (np.diff(positions.astype(np.int64)) <= 0).any()):
        raise ValueError("Targets must be increasing, unique and have sufficient history")
    return positions.astype(np.int64, copy=False)


def prediction_examples(values, observed, timestamps, targets):
    """Construct only requested causal features, with no read of target labels.

    The checked data loader already establishes the complete hourly calendar.
    The empty actual field is deliberate: prediction does not require labels.
    """
    positions = _prediction_indices(values, observed, timestamps, targets, LOOKBACK)
    history_positions = positions[:, None] + np.arange(-LOOKBACK, 0)
    history, flags = values[history_positions], observed[history_positions]
    if not np.isfinite(history).all() or (history < 0).any() or not np.isin(flags, [0, 1]).all():
        raise ValueError("Requested histories must be finite nonnegative traffic with binary masks")
    clock = timestamps[positions]
    if clock.hasnans or not clock.equals(clock.floor("h")):
        raise ValueError("Requested target clocks must be valid whole hours")
    hour = clock.hour.to_numpy(dtype=np.float64)
    week_hour = 24 * clock.dayofweek.to_numpy(dtype=np.float64) + hour
    calendar = np.column_stack((np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                                np.sin(2*np.pi*week_hour/168), np.cos(2*np.pi*week_hour/168)))
    return Examples(positions, clock, history, (1-flags.astype(bool)).astype(np.float64),
                    calendar, np.empty(0, dtype=np.float64))


def predict_model(bundle, values, observed, timestamps, targets):
    """Include causal features, scaling, all model stages and final lower bound."""
    method = _method(bundle.method)
    if method in BASELINES:
        predictor, lag = BASELINES[method]
        positions = _prediction_indices(values, observed, timestamps, targets, lag)
        raw = predictor(values, positions)  # Validated float64 input: no whole-array conversion/copy.
    else:
        examples = prediction_examples(values, observed, timestamps, targets)
        if method == "lstm":
            sequence, calendar = lstm_inputs(examples, bundle.scaler)
            predicted = predict_lstm(bundle.model, sequence, calendar, bundle.settings["batch_size"])
        else:
            features = tabular_inputs(examples, bundle.scaler)
            if method == "ridge":
                predicted = predict_ridge(bundle.model, features)
            elif method == "direct_lightgbm":
                predicted = predict_lightgbm(bundle.model, features)
            else:
                predicted = predict_hybrid(bundle.model, bundle.correction_model, features,
                                           bundle.settings["weight"])
        raw = bundle.scaler.inverse(predicted)
    final, _ = finalize_predictions(raw)
    return raw, final


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _ridge_state(model):
    return {"parameters": model.get_params(deep=False), "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_), "feature_count": int(model.n_features_in_)}


def save_model(bundle, directory):
    """Save one deployed model and its scaler/config; exclude training histories/data."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Use a new or empty model directory")
    directory.mkdir(parents=True, exist_ok=True)
    method = _method(bundle.method)
    payload = {"schema_version": 1, "method": method, "settings": bundle.settings,
               "scaler": asdict(bundle.scaler) if bundle.scaler is not None else None}
    if method in ("ridge", "ridge_lightgbm_hybrid"):
        _write_json(directory / "ridge.json", _ridge_state(bundle.model))
    if method in ("direct_lightgbm", "ridge_lightgbm_hybrid"):
        tree = bundle.correction_model if method == "ridge_lightgbm_hybrid" else bundle.model
        tree.save_model(str(directory / "trees.txt"))
    if method == "lstm":
        # Tensor-only state; never serialize a Python model or optimizer object.
        torch.save(bundle.model.state_dict(), directory / "weights.pt")
    _write_json(directory / "model.json", payload)
    return directory


def load_model(directory):
    """Restore only supported deployment payloads, without general object unpickling."""
    directory = Path(directory)
    payload = json.loads((directory / "model.json").read_text())
    if payload.get("schema_version") != 1 or set(payload) != {"schema_version", "method", "settings", "scaler"}:
        raise ValueError("Unsupported deployment manifest")
    method, settings = _method(payload["method"]), payload["settings"]
    expected = {"model.json"}
    if method in ("ridge", "ridge_lightgbm_hybrid"):
        expected.add("ridge.json")
    if method in ("direct_lightgbm", "ridge_lightgbm_hybrid"):
        expected.add("trees.txt")
    if method == "lstm":
        expected.add("weights.pt")
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("Deployment files differ from the expected model-only payload")
    scaler = TrafficScaler(**payload["scaler"]) if payload["scaler"] is not None else None
    if (method in BASELINES) != (scaler is None):
        raise ValueError("Only learned deployment models require a traffic scaler")
    if settings.get("lookback") != LOOKBACK or (scaler is not None and settings.get("fit_end") != scaler.fit_end):
        raise ValueError("Deployment feature schema or fitting boundary is inconsistent")
    model = correction = None
    if method in ("ridge", "ridge_lightgbm_hybrid"):
        state = json.loads((directory / "ridge.json").read_text())
        model = Ridge(**state["parameters"])
        model.coef_ = np.asarray(state["coefficients"], dtype=np.float64)
        model.intercept_, model.n_features_in_ = float(state["intercept"]), int(state["feature_count"])
        if model.coef_.shape != (340,) or model.n_features_in_ != 340:
            raise ValueError("Deployment ridge must have the approved 340-feature layout")
        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_):
            raise ValueError("Deployment ridge parameters must be finite")
    if method in ("direct_lightgbm", "ridge_lightgbm_hybrid"):
        tree = lgb.Booster(model_file=str(directory / "trees.txt"))
        if tree.num_feature() != 340 or tree.current_iteration() < 1 or tree.num_model_per_iteration() != 1:
            raise ValueError("Deployment trees must be fitted single-output 340-feature models")
        if method == "direct_lightgbm":
            model = tree
        else:
            correction = tree
    if method == "lstm":
        # Loading a model does not consume the caller's initialization RNG sequence.
        with torch.random.fork_rng(devices=[]):
            model = TrafficLSTM(settings["hidden_units"], settings["dropout"])
        state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("Deployment LSTM weights must be finite float32 tensors")
        model.load_state_dict(state, strict=True)
        model.eval()
    return PilotModel(method, model, scaler, settings, {}, correction)


def model_fingerprint(bundle):
    """Hash deployment state outside measurements to detect prediction-time mutation."""
    payload = {"method": bundle.method, "settings": bundle.settings,
               "scaler": asdict(bundle.scaler) if bundle.scaler is not None else None}
    if bundle.method in ("ridge", "ridge_lightgbm_hybrid"):
        payload["ridge"] = _ridge_state(bundle.model)
    if bundle.method in ("direct_lightgbm", "ridge_lightgbm_hybrid"):
        tree = bundle.correction_model if bundle.method == "ridge_lightgbm_hybrid" else bundle.model
        payload["trees"] = tree.model_to_string()
    if bundle.method == "lstm":
        payload["weights_sha256"] = state_fingerprint(bundle.model)
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
