"""Verify standalone LightGBM on one fixed development sample, without tuning.

Run with tools/run_cpu.py. Use the first configured tree/leaf counts, not values
selected from scores. This is correctness evidence, never a timing benchmark.
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
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from evaluation.metrics import mae, rmse, mase, finalize_predictions, busiest_hour_measures
from forecasting.data import make_examples, fit_scaler, tabular_inputs, scaled_targets
from models.lightgbm import fit_lightgbm, predict_lightgbm
from preprocessing.splits import TRAIN_END
from preprocessing.validate_data import sha256, write_json
from tools.check_forecasting_pipeline import check_shared, check_thread_controls
from tools.check_prepared_data import load_validated_data
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]


def fit_prefix(values, observed, timestamps, fit_end, tree_count, leaf_cap, settings):
    """Fit scaler, feature bins and trees using only the permitted past."""
    past, flags, clock = values[:fit_end], observed[:fit_end], timestamps[:fit_end]
    scaler = fit_scaler(past, flags, fit_end)
    train = make_examples(past, flags, clock, 0, fit_end)
    features = tabular_inputs(train, scaler)
    targets = scaled_targets(train, scaler)
    model = fit_lightgbm(features, targets, tree_count, leaf_cap, settings)
    return model, scaler, train, features, targets


def predict_dump(dump, features):
    """Independently follow numeric splits and add saved leaf values.

    LightGBM has already put shrinkage and the initial mean into exported leaf
    values. Multiplying shrinkage or adding the mean again would double-count.
    This checker deliberately supports only our finite, numeric regression case.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Tree reconstruction requires finite two-dimensional features")
    if (dump["num_class"] != 1 or dump["num_tree_per_iteration"] != 1
            or dump["average_output"] or dump["objective"] != "regression"
            or matrix.shape[1] != dump["max_feature_idx"] + 1):
        raise ValueError("Only ordinary single-output regression trees are supported")
    predictions = []
    for row in matrix:
        leaves = []
        for tree in dump["tree_info"]:
            node = tree["tree_structure"]
            while "split_index" in node:
                if node["decision_type"] != "<=" or node.get("missing_type") not in ("None", "NaN"):
                    raise ValueError("Unexpected categorical or zero-as-missing split")
                branch = "left_child" if row[node["split_feature"]] <= node["threshold"] else "right_child"
                node = node[branch]
            if "leaf_coeff" in node:
                raise ValueError("Linear leaves are outside this checker")
            leaves.append(float(node["leaf_value"]))
        predictions.append(math.fsum(leaves))
    result = np.array(predictions, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Tree reconstruction produced nonfinite predictions")
    return result


def native_parameters(model_text):
    """Keep the native parameter block, including defaults, as reported strings."""
    block = model_text.split("parameters:\n", 1)[1].split("end of parameters", 1)[0]
    parameters = {}
    for line in block.splitlines():
        if line.startswith("[") and line.endswith("]"):
            key, value = line[1:-1].split(":", 1)
            parameters[key] = value.strip()
    if not parameters:
        raise ValueError("Native LightGBM parameter record is missing")
    return parameters


def check_module(data_directory, output_directory):
    """Fit once, verify independent predictions, and preserve all evidence."""
    output = Path(output_directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new or empty output directory to preserve previous checks")
    controls = check_thread_controls()
    config = yaml.safe_load((ROOT / "run_config.yaml").read_text())
    settings = config["models"]["trees"]
    tree_count, leaf_cap = settings["tree_counts"][0], settings["leaf_caps"][0]
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
    model, scaler, train, train_x, train_y = fit_prefix(
        values, observed, timestamps, fit_end, tree_count, leaf_cap, settings)
    np.testing.assert_array_equal(train.targets, np.flatnonzero(group.eligible_target.to_numpy()[:fit_end]))
    np.testing.assert_array_equal(score.targets,
        np.flatnonzero(group.eligible_target.to_numpy()[start:end]) + start)
    score_x = tabular_inputs(score, scaler)
    model_text = model.model_to_string(num_iteration=-1)
    dump = model.dump_model(num_iteration=-1)
    actual_trees = model.num_trees()
    if not (1 <= actual_trees <= tree_count and len(dump["tree_info"]) == actual_trees
            and all(1 <= tree["num_leaves"] <= leaf_cap for tree in dump["tree_info"])):
        raise AssertionError("Fitted tree capacity is outside the declared bounds")
    parameters = native_parameters(model_text)
    for key, expected in {"objective": "regression", "boosting": "gbdt", "device_type": "cpu",
                          "num_threads": "1", "boost_from_average": "1", "zero_as_missing": "0",
                          "linear_tree": "0", "early_stopping_round": "0"}.items():
        if parameters.get(key) != expected:
            raise AssertionError(f"Unexpected native parameter: {key}")
    predicted_z = predict_lightgbm(model, score_x)
    np.testing.assert_array_equal(predicted_z, predict_lightgbm(model, score_x))
    if model.model_to_string(num_iteration=-1) != model_text:
        raise AssertionError("Prediction changed the fitted model")
    manual_z = predict_dump(dump, score_x)
    np.testing.assert_allclose(predicted_z, manual_z, rtol=1e-12, atol=1e-12)
    raw = scaler.inverse(predicted_z)
    final, clipping = finalize_predictions(raw)
    manual_final = np.array([max(0.0, scaler.mean + scaler.sd * z) for z in manual_z])
    np.testing.assert_allclose(final, manual_final, rtol=1e-12, atol=1e-9)
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
    busy = busiest_hour_measures(timestamps[start:end],
        group.traffic_observed.to_numpy()[start:end], observed[start:end], common_mask, full_prediction)

    metadata = {"method": "direct_lightgbm", "purpose": "development correctness check only",
        "cell": cell, "fit_interval": [0, fit_end], "explicit_params": model.params,
        "native_parameters_as_strings": parameters, "requested_trees": tree_count,
        "actual_trees": actual_trees, "leaf_cap": leaf_cap, "scaler": asdict(scaler),
        "feature_order": ["168 scaled traffic lags oldest first", "168 original missing flags oldest first",
                          "daily sine", "daily cosine", "weekly sine", "weekly cosine"],
        "python_features_dtype": str(train_x.dtype), "python_targets_dtype": str(train_y.dtype),
        "label_conversion": "Python Dataset converts labels to float32; features use native histogram binning",
        "label_cast_max_absolute_change": float(np.max(np.abs(train_y.astype(np.float32).astype(np.float64) - train_y)))}
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.txt").write_text(model_text)
    write_json(output / "tree_dump.json", dump)
    write_json(output / "model_metadata.json", metadata)
    restored = lgb.Booster(model_file=str(output / "model.txt"))
    np.testing.assert_array_equal(predict_lightgbm(restored, score_x), predicted_z)
    saved_dump = json.loads((output / "tree_dump.json").read_text())
    saved_scaler = json.loads((output / "model_metadata.json").read_text())["scaler"]
    restored_raw = saved_scaler["mean"] + saved_scaler["sd"] * predict_dump(saved_dump, score_x)
    np.testing.assert_allclose(np.maximum(0.0, restored_raw), final, rtol=1e-12, atol=1e-9)
    predictions = pd.DataFrame({"cell": cell, "method": "direct_lightgbm", "target_index": score.targets,
        "timestamp": score.timestamps, "actual": score.actual, "prediction_z": predicted_z,
        "prediction_raw": raw, "prediction": final})
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    saved = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    for column in predictions.columns:
        if column != "timestamp":
            np.testing.assert_array_equal(saved[column].to_numpy(), predictions[column].to_numpy())
    np.testing.assert_array_equal(pd.to_datetime(saved.timestamp).to_numpy(), score.timestamps.to_numpy())
    result = {"status": "passed", "scope": "development correctness check only; no tuning or model ranking",
        "module": "direct_lightgbm", "cell": cell, "fit_interval": [0, fit_end], "score_interval": [start, end],
        "requested_trees": tree_count, "actual_trees": actual_trees, "leaf_cap": leaf_cap,
        "capacity_rule": "first configured tree/leaf candidates, fixed before results; not selected",
        "fit_count": 1, "shared_checks": shared, "scores": scores, "clipping": clipping, "busiest_hour": busy,
        "tree_walk_max_absolute_difference": float(np.max(np.abs(predicted_z - manual_z))),
        "independent_prediction_and_error_checks": True, "saved_model_reload_checked": True,
        "saved_dump_reconstruction_checked": True, "saved_predictions_round_trip_checked": True,
        "repeat_prediction_equal": True, "parameters_unchanged_during_prediction": True,
        "source_sha256": data_report["source_sha256"], "config_sha256": sha256(ROOT / "run_config.yaml"),
        "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
        "code_sha256": {name: sha256(ROOT / name) for name in ["models/lightgbm.py", "forecasting/data.py",
            "evaluation/metrics.py", "tools/check_lightgbm.py", "tools/check_forecasting_pipeline.py",
            "tools/check_prepared_data.py", "tools/run_cpu.py"]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "lightgbm": lgb.__version__,
            "system": platform.system(), "machine": platform.machine()},
        "startup_controls": {key: os.environ.get(key) for key in CPU_ENV},
        "native_pools": [{key: pool.get(key) for key in ["prefix", "internal_api", "version", "num_threads"]}
                         for pool in threadpool_info()], "thread_controls": controls,
        "formal_timing_measured": False, "holdout_forecasts_evaluated": False,
        "output_sha256": {name: sha256(output / name) for name in
            ["model.txt", "model_metadata.json", "tree_dump.json", "predictions.csv"]}}
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
        "score_interval", "requested_trees", "actual_trees", "leaf_cap", "fit_count", "thread_controls"]}, indent=2))


if __name__ == "__main__":
    main()
