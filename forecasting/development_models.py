"""Fit explicit declared baseline and LSTM development candidates.

The runner chooses candidates and verifies saved dependency provenance. This
adapter only fits the requested candidate on the supplied fitting prefix.
"""
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real

import numpy as np

from forecasting.lstm_training import fit_development
from forecasting.pilot_models import (BASELINES, PilotModel, _prefix,
                                     predict_model)
from preprocessing.splits import LOOKBACK

JOB_KEYS = {"id", "stage", "method", "cell", "fold_index", "fit_end",
            "score_start", "score_end", "candidate_id", "parameters", "seed"}
STAGES = {**{method: "baselines" for method in BASELINES}, "lstm": "lstm"}
PARAMETERS = {**{method: set() for method in BASELINES},
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
    if method == "lstm":
        _candidate(parameters["hidden_units"], models["lstm"]["hidden_units"], "hidden_units", integer=True)
        _candidate(parameters["dropout"], models["lstm"]["output_dropout"], "dropout")
        _candidate(job["seed"], models["lstm"]["seeds_all_formal_periods"], "seed", integer=True)
    if method != "lstm" and job["seed"] is not None:
        raise ValueError("Only an LSTM candidate accepts a seed")
    return method, parameters


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


def predict_candidate(bundle, values, observed, timestamps, targets):
    """Reuse checked causal prediction and final clipping for every model family."""
    return predict_model(bundle, values, observed, timestamps, targets)
