"""Add a LightGBM correction to an already fitted ridge regression model."""

from numbers import Real

import numpy as np

from models.lightgbm import fit_lightgbm, predict_lightgbm
from models.ridge import predict_ridge


def _vector(values, name):
    """Keep arithmetic one-dimensional so broadcasting cannot change targets."""
    vector = np.asarray(values)
    if vector.ndim != 1 or vector.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric vector")
    vector = vector.astype(np.float64, copy=False)
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must be finite")
    return vector


def _weight(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Correction weight must be a finite scalar between 0 and 1")
    return float(value)


def training_residuals(targets, ridge_predictions):
    """Return actual minus raw ridge fitted prediction, both in z units.

    Use the eligible training rows from the same fitting prefix as ridge.
    There is no extra centering, scaling, clipping or cross-fitting here.
    """
    actual = _vector(targets, "Targets")
    predicted = _vector(ridge_predictions, "Ridge predictions")
    if actual.shape != predicted.shape:
        raise ValueError("Targets and ridge predictions must have identical shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = actual - predicted
    if not np.isfinite(residuals).all():
        raise ValueError("Residual targets must be finite")
    return residuals


def combine_predictions(ridge_z, correction_z, weight):
    """Add a weighted correction in standardized units, retaining negatives.

    At zero weight, return ridge exactly without inspecting the correction.
    The shared pipeline converts to traffic units and applies its lower bound
    only after this addition.
    """
    weight = _weight(weight)
    ridge_z = _vector(ridge_z, "Ridge predictions")
    if weight == 0:
        return ridge_z.copy()
    correction_z = _vector(correction_z, "Correction predictions")
    if correction_z.shape != ridge_z.shape:
        raise ValueError("Ridge and correction predictions must have identical shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        combined = ridge_z + weight * correction_z
    if not np.isfinite(combined).all():
        raise ValueError("Hybrid predictions must be finite")
    return combined


def fit_correction(ridge_model, features, targets, tree_count, leaf_cap, settings):
    """Fit one residual ensemble while preserving the supplied fitted ridge.

    Inputs and targets must already come from the permitted fitting prefix.
    Residuals retain the shared traffic scale; LightGBM's Dataset interface
    converts these labels to float32 just as for the standalone tree model.
    A zero-weight experiment needs only ridge and should skip this function.
    """
    fitted_z = predict_ridge(ridge_model, features)
    residuals = training_residuals(targets, fitted_z)
    return fit_lightgbm(features, residuals, tree_count, leaf_cap, settings)


def predict_hybrid(ridge_model, correction_model, features, weight):
    """Predict without refitting; zero weight does not require a tree model."""
    weight = _weight(weight)
    ridge_z = predict_ridge(ridge_model, features)
    if weight == 0:
        return ridge_z
    correction_z = predict_lightgbm(correction_model, features)
    return combine_predictions(ridge_z, correction_z, weight)
