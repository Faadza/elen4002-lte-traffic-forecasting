"""Fit explicit declared development candidates using the checked model modules.

The runner chooses candidates and verifies saved dependency provenance. This
adapter only fits the requested candidate on the supplied fitting prefix. The
residual stage reuses a selected ridge model; it never refits that component.
"""
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real

import numpy as np

from evaluation.metrics import finalize_predictions
from forecasting.data import fit_scaler, make_examples, scaled_targets, tabular_inputs
from forecasting.lstm_training import fit_development
from forecasting.pilot_models import (BASELINES, PilotModel, _prefix,
                                     prediction_examples, predict_model)
from models.hybrid import combine_predictions, fit_correction
from models.lightgbm import fit_lightgbm, predict_lightgbm
from models.ridge import fit_ridge, predict_ridge
from preprocessing.splits import LOOKBACK

JOB_KEYS = {"id", "stage", "method", "cell", "fold_index", "fit_end",
            "score_start", "score_end", "candidate_id", "parameters", "seed"}
STAGES = {**{method: "baselines" for method in BASELINES}, "ridge": "ridge",
          "direct_lightgbm": "direct", "ridge_lightgbm_hybrid": "hybrid", "lstm": "lstm"}
PARAMETERS = {**{method: set() for method in BASELINES}, "ridge": {"alpha"},
              "direct_lightgbm": {"trees", "leaves"},
              "ridge_lightgbm_hybrid": {"trees", "leaves"},
              "lstm": {"hidden_units", "dropout"}}


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _candidate(value, choices, name, *, integer=False):
    """Membership never silently accepts booleans or string representations."""
    if integer:
        _integer(value, name)
    elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric candidate")
    if value not in choices:
        raise ValueError(f"{name} must be in the declared grid")
    return value


def _checked_job(job, config):
    if not isinstance(job, Mapping) or set(job) != JOB_KEYS:
        raise ValueError("Candidate job must contain exactly the development job keys")
    for key in ("id", "cell", "candidate_id", "stage", "method"):
        if not isinstance(job[key], str) or not job[key]:
            raise ValueError(f"Job {key} must be a nonempty string")
    method = job["method"]
    if method not in STAGES or job["stage"] != STAGES[method]:
        raise ValueError("Candidate method and stage are inconsistent")
    _integer(job["fold_index"], "fold_index")
    fit_end = _integer(job["fit_end"], "fit_end", 1)
    start = _integer(job["score_start"], "score_start", 1)
    end = _integer(job["score_end"], "score_end", 1)
    if start != fit_end or end <= start:
        raise ValueError("Development scoring must follow the fitting prefix")
    parameters = job["parameters"]
    if not isinstance(parameters, Mapping) or set(parameters) != PARAMETERS[method]:
        raise ValueError("Parameters must contain exactly the keys for this candidate method")
    models = config["models"]
    if method == "ridge":
        _candidate(parameters["alpha"], models["ridge"]["alphas"], "alpha")
    elif method in ("direct_lightgbm", "ridge_lightgbm_hybrid"):
        _candidate(parameters["trees"], models["trees"]["tree_counts"], "trees", integer=True)
        _candidate(parameters["leaves"], models["trees"]["leaf_caps"], "leaves", integer=True)
    elif method == "lstm":
        _candidate(parameters["hidden_units"], models["lstm"]["hidden_units"], "hidden_units", integer=True)
        _candidate(parameters["dropout"], models["lstm"]["output_dropout"], "dropout")
        _candidate(job["seed"], models["lstm"]["seeds_all_formal_periods"], "seed", integer=True)
    if method != "lstm" and job["seed"] is not None:
        raise ValueError("Only an LSTM candidate accepts a seed")
    return method, parameters


def _cached_ridge(bundle, job, config, scaler, training_count):
    """Check numerical compatibility; the runner separately checks file hashes."""
    if not isinstance(bundle, PilotModel) or bundle.method != "ridge" or bundle.correction_model is not None:
        raise ValueError("The hybrid requires a previously fitted standalone ridge bundle")
    settings = bundle.settings
    alpha = _candidate(settings.get("alpha"), config["models"]["ridge"]["alphas"], "cached ridge alpha")
    identity = settings.get("development_job", {})
    if (identity.get("cell") != job["cell"] or identity.get("fold_index") != job["fold_index"]
            or identity.get("stage") != "ridge" or not identity.get("id")
            or not identity.get("candidate_id")):
        raise ValueError("Cached ridge must belong to this development cell and fold")
    if (bundle.scaler != scaler or settings.get("fit_end") != job["fit_end"]
            or settings.get("lookback") != LOOKBACK
            or settings.get("training_target_count") != training_count):
        raise ValueError("Cached ridge scaler, prefix or training targets are inconsistent")
    native = bundle.model.get_params(deep=False)
    if (native.get("alpha") != alpha or native.get("solver") != "svd"
            or native.get("fit_intercept") is not True or native.get("positive") is not False):
        raise ValueError("Cached ridge parameters differ from the declared model")
    # Also reject a contradictory report if the runner attached one after loading.
    reported = bundle.training_metadata.get("training_target_count", training_count)
    if reported != training_count:
        raise ValueError("Cached ridge report has an inconsistent training target count")
    return float(alpha)


def fit_candidate(job, values, observed, timestamps, config, *, ridge_bundle=None, progress=None):
    """Fit one grid candidate, physically excluding traffic at/after fit_end."""
    method, parameters = _checked_job(job, config)
    if ridge_bundle is not None and method != "ridge_lightgbm_hybrid":
        raise ValueError("Only the hybrid candidate accepts a cached ridge dependency")
    past, flags, clock = _prefix(values, observed, timestamps, job["fit_end"])
    identity = {key: deepcopy(job[key]) for key in ("id", "stage", "cell", "fold_index", "candidate_id")}
    settings = {"fit_end": int(job["fit_end"]), "lookback": LOOKBACK, "development_job": identity}
    metadata = {"fit_interval": [0, int(job["fit_end"])],
                "candidate_selection": "explicit declared development candidate"}
    if method in BASELINES:
        settings["lag"] = BASELINES[method][1]
        return PilotModel(method, None, None, settings, {**metadata, "component_fit_counts": {}})
    if method == "lstm":
        spec = config["models"]["lstm"]
        fitted = fit_development(past, flags, clock, job["fit_end"],
                                 hidden_units=parameters["hidden_units"], dropout=parameters["dropout"],
                                 seed=job["seed"], settings=spec, progress=progress, record_phase_times=True)
        fitted.model.eval()
        settings.update(hidden_units=int(parameters["hidden_units"]), dropout=float(parameters["dropout"]),
                        seed=int(job["seed"]), batch_size=int(spec["batch_size"]),
                        selected_epochs=fitted.metadata["selected_epochs"],
                        training_target_count=fitted.metadata["refit_count"])
        return PilotModel(method, fitted.model, fitted.scaler, settings, {**metadata, **fitted.metadata})

    scaler = fit_scaler(past, flags, job["fit_end"])
    examples = make_examples(past, flags, clock, 0, job["fit_end"])
    features, targets = tabular_inputs(examples, scaler), scaled_targets(examples, scaler)
    training_count = len(examples.targets)
    settings["training_target_count"] = metadata["training_target_count"] = training_count
    if method == "ridge":
        model = fit_ridge(features, targets, parameters["alpha"])
        settings["alpha"] = float(parameters["alpha"])
        metadata.update(ridge_parameters=model.get_params(deep=False), component_fit_counts={"ridge": 1})
        return PilotModel(method, model, scaler, settings, metadata)

    trees, leaves = parameters["trees"], parameters["leaves"]
    settings.update(requested_trees=int(trees), leaf_cap=int(leaves))
    correction = None
    if method == "direct_lightgbm":
        model = fit_lightgbm(features, targets, trees, leaves, config["models"]["trees"])
        metadata["component_fit_counts"] = {"direct_lightgbm": 1}
    else:
        settings["alpha"] = _cached_ridge(ridge_bundle, job, config, scaler, training_count)
        model = ridge_bundle.model
        correction = fit_correction(model, features, targets, trees, leaves, config["models"]["trees"])
        settings["weight"] = 1.0  # One fitted correction supports every declared weight.
        settings["ridge_dependency_job"] = deepcopy(ridge_bundle.settings["development_job"])
        metadata.update(component_fit_counts={"ridge": 0, "correction": 1},
                        ridge_reused=True, ridge_parameters=model.get_params(deep=False),
                        ridge_dependency_job=deepcopy(settings["ridge_dependency_job"]))
    tree = correction if correction is not None else model
    settings["tree_parameters"] = dict(tree.params)
    metadata.update(actual_trees=tree.current_iteration(), tree_parameters=dict(tree.params),
                    native_label_conversion="float32")
    return PilotModel(method, model, scaler, settings, metadata, correction)


def predict_candidate(bundle, values, observed, timestamps, targets):
    """Reuse checked causal prediction and final clipping for every model family."""
    return predict_model(bundle, values, observed, timestamps, targets)


def predict_hybrid_components(bundle, values, observed, timestamps, targets):
    """Return the two unclipped standardized outputs on identical inputs."""
    if not isinstance(bundle, PilotModel) or bundle.method != "ridge_lightgbm_hybrid":
        raise ValueError("Hybrid component prediction requires a hybrid bundle")
    examples = prediction_examples(values, observed, timestamps, targets)
    features = tabular_inputs(examples, bundle.scaler)
    return predict_ridge(bundle.model, features), predict_lightgbm(bundle.correction_model, features)


def finalize_hybrid_components(bundle, ridge_z, correction_z, weight, config):
    """Combine in z units, invert once and only then bound final traffic at zero."""
    if not isinstance(bundle, PilotModel) or bundle.method != "ridge_lightgbm_hybrid":
        raise ValueError("Hybrid component combination requires a hybrid bundle")
    _candidate(weight, config["models"]["hybrid"]["weights"], "weight")
    raw = bundle.scaler.inverse(combine_predictions(ridge_z, correction_z, weight))
    final, _ = finalize_predictions(raw)
    return raw, final
