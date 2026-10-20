"""Verify a fixed ridge–LightGBM residual hybrid, without tuning or ranking.

Run with tools/run_cpu.py. One ridge fit and one correction fit use only the
first development prefix. Weight one checks addition; zero checks ridge reuse.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
import platform
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from evaluation.metrics import mae, rmse, mase, finalize_predictions, busiest_hour_measures
from forecasting.data import tabular_inputs
from models.hybrid import fit_correction, training_residuals, predict_hybrid
from models.ridge import predict_ridge
from models.lightgbm import predict_lightgbm
from preprocessing.splits import TRAIN_END
from preprocessing.validate_data import sha256, write_json
from tools.check_forecasting_pipeline import check_shared, check_thread_controls
from tools.check_lightgbm import predict_dump, native_parameters
from tools.check_prepared_data import load_validated_data
from tools.check_ridge import fit_prefix as fit_ridge_prefix
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]


def fit_prefix(values, observed, timestamps, fit_end, alpha, tree_count, leaf_cap, settings):
    """Reuse one prefix-only ridge fit and fit trees to its raw training errors."""
    ridge, scaler, train, features, targets = fit_ridge_prefix(
        values, observed, timestamps, fit_end, alpha)
    coefficients, intercept = ridge.coef_.copy(), float(ridge.intercept_)
    correction = fit_correction(ridge, features, targets, tree_count, leaf_cap, settings)
    np.testing.assert_array_equal(ridge.coef_, coefficients)
    if float(ridge.intercept_) != intercept:
        raise AssertionError("Correction fitting changed the supplied ridge model")
    return ridge, correction, scaler, train, features, targets


def manual_ridge(features, coefficients, intercept):
    """Rebuild the weighted sum without sklearn prediction or matrix multiply."""
    return np.array([float(intercept) + math.fsum(float(x) * float(w)
        for x, w in zip(row, coefficients)) for row in features], dtype=np.float64)


def save_table(table, path):
    """Preserve numeric values and verify the complete CSV round trip."""
    table.to_csv(path, index=False, float_format="%.17g")
    saved = pd.read_csv(path, float_precision="round_trip")
    for column in table.columns:
        if column == "timestamp":
            np.testing.assert_array_equal(pd.to_datetime(saved[column]).to_numpy(),
                                          table[column].to_numpy())
        else:
            np.testing.assert_array_equal(saved[column].to_numpy(), table[column].to_numpy())


def check_module(data_directory, output_directory):
    """Verify one bounded fit pair and save both components and their sum."""
    output = Path(output_directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new or empty output directory to preserve previous checks")
    controls = check_thread_controls()
    config = yaml.safe_load((ROOT / "run_config.yaml").read_text())
    ridge_settings = config["models"]["ridge"]
    hybrid = config["models"]["hybrid"]
    if (ridge_settings["fit_intercept"] is not True
            or ridge_settings["intercept_penalized"] is not False
            or ridge_settings["solver"] != "svd"
            or ridge_settings["objective"] != "sum squared residuals + alpha * squared coefficient norm"
            or hybrid["residual_cross_fitting"] is not False
            or hybrid["residual_extra_centering"] is not False
            or hybrid["residual_target"] != "z_actual - unclipped ridge fitted z prediction on eligible training rows"
            or hybrid["weight_global_and_fixed"] is not True
            or hybrid["weights"] != [0, .25, .5, .75, 1]):
        raise ValueError("Review the checker after changing the approved hybrid specification")
    alpha = float(ridge_settings["alphas"][0])
    settings = config["models"]["trees"]
    tree_count, leaf_cap = settings["tree_counts"][0], settings["leaf_caps"][0]
    weight = 1.0  # Full correction exercises both components; this is not selection.
    cell = config["tuning_panel"]["ids"][0]
    fold = config["chronology"]["folds"][0]
    fit_start, fit_end = fold["fit"]
    start, end = fold["score"][0], fold["score"][0] + 72
    if not (fit_start == 0 and 168 <= fit_end == start < end <= fold["score"][1] <= TRAIN_END):
        raise ValueError("The fixed correctness check must stay inside the first development fold")
    hourly, data_report = load_validated_data(data_directory)
    group = hourly.loc[hourly.cell == cell].reset_index(drop=True).iloc[:end]
    values, observed = group.traffic.to_numpy(), group.observed.to_numpy()
    timestamps = pd.DatetimeIndex(group.ts)
    score, scale, shared = check_shared(values, observed, timestamps, fit_end, start, end)
    ridge, correction, scaler, train, train_x, train_y = fit_prefix(
        values, observed, timestamps, fit_end, alpha, tree_count, leaf_cap, settings)
    np.testing.assert_array_equal(train.targets, np.flatnonzero(group.eligible_target.to_numpy()[:fit_end]))
    np.testing.assert_array_equal(score.targets,
        np.flatnonzero(group.eligible_target.to_numpy()[start:end]) + start)
    score_x = tabular_inputs(score, scaler)
    coefficients, intercept = ridge.coef_.copy(), float(ridge.intercept_)
    model_text = correction.model_to_string(num_iteration=-1)
    dump = correction.dump_model(num_iteration=-1)
    actual_trees = correction.num_trees()
    if not (1 <= actual_trees <= tree_count and len(dump["tree_info"]) == actual_trees
            and all(1 <= tree["num_leaves"] <= leaf_cap for tree in dump["tree_info"])):
        raise AssertionError("Fitted correction capacity is outside the declared bounds")
    parameters = native_parameters(model_text)
    for key, expected in {"objective": "regression", "boosting": "gbdt", "device_type": "cpu",
                          "num_threads": "1", "boost_from_average": "1", "zero_as_missing": "0",
                          "linear_tree": "0", "early_stopping_round": "0"}.items():
        if parameters.get(key) != expected:
            raise AssertionError(f"Unexpected native correction parameter: {key}")

    fitted_z = predict_ridge(ridge, train_x)
    residual = training_residuals(train_y, fitted_z)
    manual_fitted_z = manual_ridge(train_x, coefficients, intercept)
    np.testing.assert_allclose(residual, train_y - manual_fitted_z, rtol=1e-12, atol=1e-12)
    # The base still obeys sum-SSE ridge with an unpenalised intercept.
    normal_error = train_x.T @ residual - alpha * coefficients
    normal_scale = max(1.0, float(np.max(np.abs(train_x.T @ train_y))))
    relative_normal_error = float(np.max(np.abs(normal_error))) / normal_scale
    relative_intercept_error = abs(math.fsum(map(float, residual))) / max(
        1.0, math.fsum(abs(float(value)) for value in train_y))
    if max(relative_normal_error, relative_intercept_error) > 1e-10:
        raise AssertionError("Reused ridge base fails its first-order equations")

    ridge_z = predict_ridge(ridge, score_x)
    correction_z = predict_lightgbm(correction, score_x)
    predicted_z = predict_hybrid(ridge, correction, score_x, weight)
    np.testing.assert_array_equal(predicted_z, ridge_z + weight * correction_z)
    np.testing.assert_array_equal(predicted_z, predict_hybrid(ridge, correction, score_x, weight))
    zero_z = predict_hybrid(ridge, None, score_x, 0.0)
    np.testing.assert_array_equal(zero_z, ridge_z)
    manual_base = manual_ridge(score_x, coefficients, intercept)
    manual_correction = predict_dump(dump, score_x)
    manual_z = manual_base + weight * manual_correction
    np.testing.assert_allclose(ridge_z, manual_base, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(correction_z, manual_correction, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(predicted_z, manual_z, rtol=1e-12, atol=1e-12)
    raw = scaler.inverse(predicted_z)
    final, clipping = finalize_predictions(raw)
    zero_final, _ = finalize_predictions(scaler.inverse(zero_z))
    ridge_final, _ = finalize_predictions(scaler.inverse(ridge_z))
    np.testing.assert_array_equal(zero_final, ridge_final)
    manual_final = np.array([max(0.0, scaler.mean + scaler.sd * z) for z in manual_z])
    np.testing.assert_allclose(final, manual_final, rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(ridge.coef_, coefficients)
    if float(ridge.intercept_) != intercept or correction.model_to_string(num_iteration=-1) != model_text:
        raise AssertionError("Correction fitting or prediction changed a frozen component")

    errors = [float(p) - float(a) for p, a in zip(final, score.actual)]
    manual_mae = math.fsum(abs(e) for e in errors) / len(errors)
    manual_rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The fixed sample has no usable weekly MASE denominator")
    scores = {"MAE": mae(score.actual, final), "RMSE": rmse(score.actual, final),
              "MASE": mase(score.actual, final, scale)}
    np.testing.assert_allclose(list(scores.values()),
        [manual_mae, manual_rmse, manual_mae / scale], rtol=1e-12, atol=1e-12)
    full_prediction = np.full(end - start, np.nan)
    common_mask = np.zeros(end - start, dtype=bool)
    full_prediction[score.targets - start] = final
    common_mask[score.targets - start] = True
    busy = busiest_hour_measures(timestamps[start:end], group.traffic_observed.to_numpy()[start:end],
                                observed[start:end], common_mask, full_prediction)

    metadata = {"method": "ridge_lightgbm_hybrid", "purpose": "development correctness check only",
        "cell": cell, "fit_interval": [0, fit_end], "weight": weight, "scaler": asdict(scaler),
        "ridge": {"params": ridge.get_params(deep=False), "effective_solver": ridge.solver_,
                  "coefficient": coefficients.tolist(), "intercept": intercept},
        "correction": {"explicit_params": correction.params, "native_parameters_as_strings": parameters,
                       "requested_trees": tree_count, "actual_trees": actual_trees, "leaf_cap": leaf_cap},
        "residual_target": "scaled actual minus raw ridge fitted prediction; same training rows",
        "residual_cross_fitting": False, "residual_extra_centering_or_scaling": False,
        "prediction_rule": "max(0, mean + sd * (ridge_z + weight * correction_z))",
        "feature_order": ["168 scaled traffic lags oldest first", "168 original missing flags oldest first",
                          "daily sine", "daily cosine", "weekly sine", "weekly cosine"],
        "python_features_dtype": str(train_x.dtype), "python_residual_dtype": str(residual.dtype),
        "label_conversion": "Python Dataset converts residual labels to float32; features use native histogram binning",
        "label_cast_max_absolute_change": float(np.max(np.abs(residual.astype(np.float32).astype(np.float64) - residual)))}
    output.mkdir(parents=True, exist_ok=True)
    (output / "correction_model.txt").write_text(model_text)
    write_json(output / "correction_tree_dump.json", dump)
    write_json(output / "model_metadata.json", metadata)
    restored = lgb.Booster(model_file=str(output / "correction_model.txt"))
    np.testing.assert_array_equal(predict_hybrid(ridge, restored, score_x, weight), predicted_z)
    saved_model = json.loads((output / "model_metadata.json").read_text())
    saved_dump = json.loads((output / "correction_tree_dump.json").read_text())
    restored_base = manual_ridge(score_x, saved_model["ridge"]["coefficient"], saved_model["ridge"]["intercept"])
    restored_z = restored_base + saved_model["weight"] * predict_dump(saved_dump, score_x)
    restored_raw = saved_model["scaler"]["mean"] + saved_model["scaler"]["sd"] * restored_z
    np.testing.assert_allclose(np.maximum(0.0, restored_raw), final, rtol=1e-12, atol=1e-9)
    predictions = pd.DataFrame({"cell": cell, "method": "ridge_lightgbm_hybrid",
        "target_index": score.targets, "timestamp": score.timestamps, "actual": score.actual,
        "ridge_z": ridge_z, "correction_z": correction_z, "weight": weight,
        "prediction_z": predicted_z, "prediction_raw": raw, "prediction": final,
        "zero_weight_prediction": zero_final})
    save_table(predictions, output / "predictions.csv")
    training = pd.DataFrame({"cell": cell, "target_index": train.targets, "timestamp": train.timestamps,
        "actual": train.actual, "actual_z": train_y, "ridge_fitted_z": fitted_z,
        "residual_z": residual, "native_label_float32": residual.astype(np.float32).astype(np.float64)})
    save_table(training, output / "training_residuals.csv")
    result = {"status": "passed", "scope": "development correctness check only; no tuning or model ranking",
        "module": "ridge_lightgbm_hybrid", "cell": cell, "fit_interval": [0, fit_end],
        "score_interval": [start, end], "alpha": alpha, "weight": weight,
        "requested_trees": tree_count, "actual_trees": actual_trees, "leaf_cap": leaf_cap,
        "candidate_rule": "first configured alpha/trees/leaves; weight one for addition, zero for identity; no selection",
        "fit_count": 2, "component_fit_counts": {"ridge": 1, "correction": 1},
        "zero_weight_requires_correction_model": False, "zero_weight_matches_ridge_exactly": True,
        "shared_checks": shared, "scores": scores, "clipping": clipping, "busiest_hour": busy,
        "stationarity_relative_errors": {"coefficients": relative_normal_error, "intercept": relative_intercept_error},
        "manual_max_absolute_differences": {"training_residual": float(np.max(np.abs(residual - (train_y - manual_fitted_z)))),
            "ridge_z": float(np.max(np.abs(ridge_z - manual_base))),
            "correction_z": float(np.max(np.abs(correction_z - manual_correction))),
            "combined_z": float(np.max(np.abs(predicted_z - manual_z)))},
        "independent_prediction_and_error_checks": True, "saved_component_reload_checked": True,
        "saved_complete_model_reconstruction_checked": True, "saved_tables_round_trip_checked": True,
        "repeat_prediction_equal": True, "parameters_unchanged_during_prediction": True,
        "source_sha256": data_report["source_sha256"], "config_sha256": sha256(ROOT / "run_config.yaml"),
        "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
        "code_sha256": {name: sha256(ROOT / name) for name in ["models/hybrid.py", "models/ridge.py",
            "models/lightgbm.py", "forecasting/data.py", "evaluation/metrics.py", "tools/check_hybrid.py",
            "tools/check_ridge.py", "tools/check_lightgbm.py", "tools/check_forecasting_pipeline.py",
            "tools/check_prepared_data.py", "tools/run_cpu.py"]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
            "system": platform.system(), "machine": platform.machine()},
        "startup_controls": {key: os.environ.get(key) for key in CPU_ENV},
        "native_pools": [{key: pool.get(key) for key in ["prefix", "internal_api", "version", "num_threads"]}
                         for pool in threadpool_info()], "thread_controls": controls,
        "formal_timing_measured": False, "holdout_forecasts_evaluated": False,
        "output_sha256": {name: sha256(output / name) for name in ["correction_model.txt",
            "correction_tree_dump.json", "model_metadata.json", "predictions.csv", "training_residuals.csv"]}}
    write_json(output / "checks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = check_module(args.data_directory, args.output_dir)
    print(json.dumps({key: result[key] for key in ["status", "scope", "module", "cell", "fit_interval",
        "score_interval", "alpha", "weight", "requested_trees", "actual_trees", "leaf_cap",
        "component_fit_counts", "zero_weight_matches_ridge_exactly", "thread_controls"]}, indent=2))


if __name__ == "__main__":
    main()
