"""Native LightGBM fitting and raw prediction; shared code owns scaling."""

from collections.abc import Mapping
from numbers import Integral, Real

import lightgbm as lgb
import numpy as np


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def build_parameters(settings, tree_count, leaf_cap):
    """Map the declared tree settings to native LightGBM parameters.

    Tree count is a boosting-round budget, not a guarantee of that many trees:
    LightGBM may stop when no useful split remains. Grid membership is checked
    by experiment selection, so small invented examples may use fewer rounds.
    """
    if not isinstance(settings, Mapping):
        raise ValueError("Tree settings must be a mapping")
    fixed = {"objective": "squared_error", "boosting_type": "gbdt",
             "device_type": "cpu", "deterministic": True,
             "force_col_wise": True, "feature_fraction": 1.0,
             "bagging_fraction": 1.0, "bagging_freq": 0}
    numeric = {"learning_rate", "min_data_in_leaf", "lambda_l2", "seed"}
    metadata = {"tree_counts", "leaf_caps", "automatic_capacity_extension",
                "remaining_library_defaults"}
    unknown = set(settings) - set(fixed) - numeric - metadata
    if unknown:
        raise ValueError(f"Unknown tree settings: {sorted(unknown)}")
    missing = (set(fixed) | numeric) - set(settings)
    if missing:
        raise ValueError(f"Missing tree settings: {sorted(missing)}")
    for name, expected in fixed.items():
        value = settings[name]
        if isinstance(expected, bool):
            valid = isinstance(value, (bool, np.bool_)) and bool(value) == expected
        elif isinstance(expected, str):
            valid = isinstance(value, str) and value == expected
        else:
            valid = (not isinstance(value, (bool, np.bool_))
                     and isinstance(value, Real) and value == expected)
        if not valid:
            raise ValueError(f"{name} must be {expected!r} for this experiment")
    for name in ("learning_rate", "lambda_l2"):
        value = settings[name]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0
                or (name == "learning_rate" and value == 0)):
            raise ValueError(f"{name} must be finite and "
                             + ("positive" if name == "learning_rate" else "nonnegative"))
    parameters = dict(fixed)
    parameters.update(
        objective="regression",  # Native name for the approved squared-error loss.
        learning_rate=float(settings["learning_rate"]),
        lambda_l2=float(settings["lambda_l2"]),
        min_data_in_leaf=_integer(settings["min_data_in_leaf"], "min_data_in_leaf", 1),
        seed=_integer(settings["seed"], "seed", 0),
        num_iterations=_integer(tree_count, "tree_count", 1),
        num_leaves=_integer(leaf_cap, "leaf_cap", 2),
        num_threads=1,
        verbosity=-1,
    )
    if parameters["seed"] > np.iinfo(np.int32).max:
        raise ValueError("seed must fit a nonnegative signed 32-bit integer")
    return parameters


def _features(values, allow_empty=False):
    """Require finite dense real values, with one example per row."""
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Features must be a two-dimensional matrix with columns")
    if not allow_empty and matrix.shape[0] == 0:
        raise ValueError("Fitting requires at least one example")
    if matrix.dtype.kind not in "biuf":
        raise ValueError("Features must contain real numeric values")
    matrix = matrix.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Features must be finite")
    return matrix


def fit_lightgbm(features, targets, tree_count, leaf_cap, settings):
    """Fit one squared-error tree ensemble without searching or scaling.

    Traffic columns and targets arrive standardized from the shared pipeline;
    missingness flags and clock features remain unchanged. Arrays are float64,
    but LightGBM's Python Dataset interface converts labels to float32.
    No validation set, early-stopping callback, weights or warm start is used.
    """
    parameters = build_parameters(settings, tree_count, leaf_cap)
    matrix = _features(features)
    values = np.asarray(targets)
    if values.ndim != 1 or len(values) != len(matrix):
        raise ValueError("Targets must be one-dimensional and match feature rows")
    if values.dtype.kind not in "biuf":
        raise ValueError("Targets must contain real numeric values")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Targets must be finite")
    if np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError("Targets must be representable as finite native float32 labels")

    dataset = lgb.Dataset(matrix, label=values, free_raw_data=True)
    model = lgb.train(parameters, dataset, num_boost_round=parameters["num_iterations"])
    if model.current_iteration() < 1 or model.num_model_per_iteration() != 1:
        raise ValueError("LightGBM must produce at least one single-output tree")
    return model


def predict_lightgbm(model, features):
    """Return finite raw standardized predictions, including negative values.

    The shared pipeline inverts scaling and bounds final traffic predictions
    after any hybrid addition. Prediction also explicitly requests one thread.
    """
    if not isinstance(model, lgb.Booster):
        raise ValueError("Model must be a fitted native LightGBM Booster")
    if model.current_iteration() < 1 or model.num_model_per_iteration() != 1:
        raise ValueError("Model must contain fitted single-output trees")
    matrix = _features(features, allow_empty=True)
    if matrix.shape[1] != model.num_feature():
        raise ValueError("Prediction features must match the fitted column count")
    if len(matrix) == 0:
        return np.empty(0, dtype=np.float64)
    predictions = np.asarray(model.predict(matrix, raw_score=True,
                                          num_iteration=-1, num_threads=1),
                             dtype=np.float64)
    if predictions.shape != (len(matrix),) or not np.isfinite(predictions).all():
        raise ValueError("LightGBM predictions must be finite and one-dimensional")
    return predictions
