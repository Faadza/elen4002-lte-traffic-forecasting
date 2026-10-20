"""Ridge fitting and raw prediction; shared pipeline code owns scaling."""

from numbers import Real

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted


def _features(values, allow_empty=False):
    """Require a finite dense matrix, with one example per row."""
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


def fit_ridge(features, targets, alpha):
    """Fit sum squared errors + alpha * squared coefficient norm.

    Traffic columns and targets are already standardized by the shared pipeline;
    missingness flags and calendar features remain unchanged.
    The intercept is fitted but not penalized. No parameter search occurs here.
    """
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("Alpha must be a positive finite scalar")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Alpha must be a positive finite scalar")
    matrix = _features(features)
    values = np.asarray(targets)
    if values.ndim != 1 or len(values) != len(matrix):
        raise ValueError("Targets must be one-dimensional and match feature rows")
    if values.dtype.kind not in "biuf":
        raise ValueError("Targets must contain real numeric values")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Targets must be finite")

    # SVD handles correlated lag inputs; copy_X preserves shared input arrays.
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver="svd",
                  copy_X=True, positive=False)
    model.fit(matrix, values)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_):
        raise ValueError("Ridge fitting produced nonfinite parameters")
    return model


def predict_ridge(model, features):
    """Return raw standardized predictions, including negative values.

    Conversion to original traffic units and the final zero bound happen in
    the shared pipeline, after any future hybrid correction has been added.
    An empty matrix with the fitted number of columns returns an empty vector.
    """
    if not isinstance(model, Ridge):
        raise ValueError("Model must be a fitted sklearn Ridge")
    check_is_fitted(model, ["coef_", "intercept_", "n_features_in_"])
    if np.ndim(model.coef_) != 1 or np.ndim(model.intercept_) != 0:
        raise ValueError("Ridge must have one output target")
    matrix = _features(features, allow_empty=True)
    if matrix.shape[1] != model.n_features_in_:
        raise ValueError("Prediction features must match the fitted column count")
    if len(matrix) == 0:
        return np.empty(0, dtype=np.float64)
    predictions = np.asarray(model.predict(matrix), dtype=np.float64)
    if predictions.shape != (len(matrix),) or not np.isfinite(predictions).all():
        raise ValueError("Ridge predictions must be finite and one-dimensional")
    return predictions
