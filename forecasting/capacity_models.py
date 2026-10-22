"""Fit the declared larger candidates without changing the scientific grid.

Prediction and model storage use the checked pilot interface. Only candidate
routing differs: no new algorithms, target selection or hyperparameter search.
"""
import numpy as np

from forecasting.data import fit_scaler, make_examples, scaled_targets, tabular_inputs
from forecasting.lstm_training import fit_development
from forecasting.pilot_models import PilotModel, _prefix
from models.hybrid import fit_correction
from models.lightgbm import fit_lightgbm
from models.ridge import fit_ridge
from preprocessing.splits import LOOKBACK

METHODS = ("ridge", "direct_lightgbm", "ridge_lightgbm_hybrid", "lstm")


def chosen_settings(config):
    """Check the declared positions before reading the fixed capacity candidates."""
    models = config["models"]
    expected = ((models["ridge"]["alphas"], [0.1, 1, 10, 100, 1000]),
                (models["trees"]["tree_counts"], [32, 128, 512]),
                (models["trees"]["leaf_caps"], [7, 31]),
                (models["hybrid"]["weights"], [0, 0.25, 0.5, 0.75, 1]),
                (models["lstm"]["hidden_units"], [16, 32]),
                (models["lstm"]["output_dropout"], [0.0, 0.2]),
                (models["lstm"]["seeds_all_formal_periods"], [2026, 2027, 2028]))
    if any(actual != declared for actual, declared in expected):
        raise ValueError("Capacity candidates require the unchanged declared grids")
    return {"alpha": models["ridge"]["alphas"][0],
            "trees": models["trees"]["tree_counts"][2],
            "leaves": models["trees"]["leaf_caps"][1],
            "weight": models["hybrid"]["weights"][4],
            "hidden_units": models["lstm"]["hidden_units"][1]}


def fit_model(method, values, observed, timestamps, fit_end, config, *, seed=None,
              dropout=None):
    """Fit one explicit capacity candidate; the hybrid includes its own ridge."""
    if method not in METHODS:
        raise ValueError("Unsupported capacity method")
    candidates = chosen_settings(config)
    past, flags, clock = _prefix(values, observed, timestamps, fit_end)
    settings = {"fit_end": int(fit_end), "lookback": LOOKBACK}
    metadata = {"fit_interval": [0, int(fit_end)],
                "candidate_selection": "fixed declared capacity settings; no tuning"}
    if method != "lstm" and (seed is not None or dropout is not None):
        raise ValueError("Only LSTM accepts seed and dropout arguments")
    if method == "lstm":
        spec = config["models"]["lstm"]
        if (isinstance(dropout, (bool, np.bool_)) or dropout not in spec["output_dropout"]
                or isinstance(seed, (bool, np.bool_)) or seed not in spec["seeds_all_formal_periods"]):
            raise ValueError("LSTM requires an explicit declared dropout and seed")
        fitted = fit_development(past, flags, clock, fit_end,
                                 hidden_units=candidates["hidden_units"], dropout=dropout,
                                 seed=seed, settings=spec, record_phase_times=True)
        fitted.model.eval()
        settings.update(hidden_units=int(candidates["hidden_units"]), dropout=float(dropout),
                        seed=int(seed), batch_size=int(spec["batch_size"]),
                        selected_epochs=fitted.metadata["selected_epochs"])
        return PilotModel(method, fitted.model, fitted.scaler, settings,
                          {**metadata, **fitted.metadata})

    scaler = fit_scaler(past, flags, fit_end)
    examples = make_examples(past, flags, clock, 0, fit_end)
    features, targets = tabular_inputs(examples, scaler), scaled_targets(examples, scaler)
    metadata["training_target_count"] = len(examples.targets)
    ridge = correction = None
    if method in ("ridge", "ridge_lightgbm_hybrid"):
        ridge = fit_ridge(features, targets, candidates["alpha"])
        settings["alpha"] = float(candidates["alpha"])
        metadata["ridge_parameters"] = ridge.get_params(deep=False)
    if method == "ridge":
        metadata["component_fit_counts"] = {"ridge": 1}
        return PilotModel(method, ridge, scaler, settings, metadata)

    spec = config["models"]["trees"]
    trees, leaves = candidates["trees"], candidates["leaves"]
    settings.update(requested_trees=int(trees), leaf_cap=int(leaves))
    if method == "direct_lightgbm":
        model = fit_lightgbm(features, targets, trees, leaves, spec)
        metadata["component_fit_counts"] = {"direct_lightgbm": 1}
    else:
        correction = fit_correction(ridge, features, targets, trees, leaves, spec)
        model = ridge
        settings["weight"] = float(candidates["weight"])
        metadata["component_fit_counts"] = {"ridge": 1, "correction": 1}
    tree = correction if correction is not None else model
    settings["tree_parameters"] = dict(tree.params)
    metadata.update(actual_trees=tree.current_iteration(), tree_parameters=dict(tree.params),
                    native_label_conversion="float32")
    return PilotModel(method, model, scaler, settings, metadata, correction)
