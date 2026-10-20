"""Check ridge on one fixed development sample; no tuning or model ranking.

Run with tools/run_cpu.py. The first configured alpha is an engineering test
setting, not a selected experiment configuration. Save evidence separately.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from evaluation.metrics import mae, rmse, mase, finalize_predictions, busiest_hour_measures
from forecasting.data import make_examples, fit_scaler, tabular_inputs, scaled_targets
from models.ridge import fit_ridge, predict_ridge
from preprocessing.splits import TRAIN_END
from preprocessing.validate_data import sha256, write_json
from tools.check_forecasting_pipeline import check_shared, check_thread_controls
from tools.check_prepared_data import load_validated_data
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]


def fit_prefix(values, observed, timestamps, fit_end, alpha):
    """Physically restrict both scaler and model fitting to the permitted past."""
    past, flags, clock = values[:fit_end], observed[:fit_end], timestamps[:fit_end]
    scaler = fit_scaler(past, flags, fit_end)
    train = make_examples(past, flags, clock, 0, fit_end)
    features = tabular_inputs(train, scaler)
    targets = scaled_targets(train, scaler)
    model = fit_ridge(features, targets, alpha)
    return model, scaler, train, features, targets


def check_module(data_directory, output_directory):
    """Fit once, keep weights/scaler fixed, and verify every saved prediction."""
    output = Path(output_directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new or empty output directory to preserve previous checks")
    controls = check_thread_controls()
    config = yaml.safe_load((ROOT / "run_config.yaml").read_text())
    ridge = config["models"]["ridge"]
    if (ridge["fit_intercept"] is not True or ridge["intercept_penalized"] is not False
            or ridge["solver"] != "svd"
            or ridge["objective"] != "sum squared residuals + alpha * squared coefficient norm"):
        raise ValueError("Review the checker after changing the approved ridge specification")
    alpha = float(ridge["alphas"][0])
    cell = config["tuning_panel"]["ids"][0]
    fold = config["chronology"]["folds"][0]
    fit_start, fit_end = fold["fit"]
    start, end = fold["score"][0], fold["score"][0] + 72
    if not (fit_start == 0 and 168 <= fit_end == start < end <= fold["score"][1] <= TRAIN_END):
        raise ValueError("The fixed correctness check must stay inside the first development fold")
    hourly, data_report = load_validated_data(data_directory)
    # Loading verifies bundle integrity; model work sees only this development prefix.
    group = hourly.loc[hourly.cell == cell].reset_index(drop=True).iloc[:end]
    values, observed = group.traffic.to_numpy(), group.observed.to_numpy()
    timestamps = pd.DatetimeIndex(group.ts)
    score, scale, shared = check_shared(values, observed, timestamps, fit_end, start, end)
    model, scaler, train, train_x, train_y = fit_prefix(values, observed, timestamps, fit_end, alpha)
    expected_train = np.flatnonzero(group.eligible_target.to_numpy()[:fit_end])
    expected_score = np.flatnonzero(group.eligible_target.to_numpy()[start:end]) + start
    np.testing.assert_array_equal(train.targets, expected_train)
    np.testing.assert_array_equal(score.targets, expected_score)
    score_x = tabular_inputs(score, scaler)
    weights_before, intercept_before = model.coef_.copy(), float(model.intercept_)
    fitted_z = predict_ridge(model, train_x)
    predicted_z = predict_ridge(model, score_x)
    np.testing.assert_array_equal(model.coef_, weights_before)
    if float(model.intercept_) != intercept_before:
        raise AssertionError("Prediction changed the fitted intercept")

    # Independent first-order equations for sum-SSE ridge, including the intercept.
    residual = train_y - fitted_z
    normal_error = train_x.T @ residual - alpha * model.coef_
    normal_scale = max(1.0, float(np.max(np.abs(train_x.T @ train_y))))
    relative_normal_error = float(np.max(np.abs(normal_error))) / normal_scale
    relative_intercept_error = abs(math.fsum(map(float, residual))) / max(
        1.0, math.fsum(abs(float(value)) for value in train_y))
    if not (relative_normal_error <= 1e-10 and relative_intercept_error <= 1e-10):
        raise AssertionError("Fitted weights do not satisfy the ridge equations within tolerance")
    manual_z = np.array([float(model.intercept_) + math.fsum(
        float(x) * float(w) for x, w in zip(row, model.coef_)) for row in score_x])
    np.testing.assert_allclose(predicted_z, manual_z, rtol=1e-12, atol=1e-12)
    raw = scaler.inverse(predicted_z)
    final, clipping = finalize_predictions(raw)
    manual_final = np.array([max(0.0, scaler.mean + scaler.sd * z) for z in manual_z])
    np.testing.assert_allclose(final, manual_final, rtol=1e-12, atol=1e-9)
    errors = [float(p) - float(a) for p, a in zip(final, score.actual)]
    manual_mae = math.fsum(abs(e) for e in errors) / len(errors)
    manual_rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
    scores = {"MAE": mae(score.actual, final), "RMSE": rmse(score.actual, final),
              "MASE": mase(score.actual, final, scale)}
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The fixed sample has no usable weekly MASE denominator")
    np.testing.assert_allclose(list(scores.values()),
        [manual_mae, manual_rmse, manual_mae / scale], rtol=1e-12, atol=1e-12)
    full_prediction = np.full(end - start, np.nan)
    common_mask = np.zeros(end - start, dtype=bool)
    full_prediction[score.targets - start] = final
    common_mask[score.targets - start] = True
    busy = busiest_hour_measures(timestamps[start:end],
        group.traffic_observed.to_numpy()[start:end], observed[start:end], common_mask, full_prediction)

    # Plain JSON makes the learned weighted sum inspectable without loading a pickle.
    artifact = {"method": "ridge", "purpose": "development correctness check only",
        "cell": cell, "fit_interval": [0, fit_end], "params": model.get_params(deep=False),
        "effective_solver": model.solver_, "scaler": asdict(scaler),
        "feature_order": ["168 scaled traffic lags oldest first", "168 original missing flags oldest first",
                          "daily sine", "daily cosine", "weekly sine", "weekly cosine"],
        "coefficient": model.coef_.tolist(), "intercept": float(model.intercept_)}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model.json", artifact)
    saved_model = json.loads((output / "model.json").read_text())
    restored_z = score_x @ np.array(saved_model["coefficient"]) + saved_model["intercept"]
    restored_raw = saved_model["scaler"]["mean"] + saved_model["scaler"]["sd"] * restored_z
    np.testing.assert_allclose(np.maximum(0.0, restored_raw), final, rtol=1e-12, atol=1e-9)
    predictions = pd.DataFrame({"cell": cell, "method": "ridge", "target_index": score.targets,
        "timestamp": score.timestamps, "actual": score.actual, "prediction_z": predicted_z,
        "prediction_raw": raw, "prediction": final})
    predictions.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
    saved = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    for column in predictions.columns:
        if column != "timestamp":
            np.testing.assert_array_equal(saved[column].to_numpy(), predictions[column].to_numpy())
    np.testing.assert_array_equal(pd.to_datetime(saved.timestamp).to_numpy(), score.timestamps.to_numpy())
    result = {"status": "passed", "scope": "development correctness check only; no tuning or model ranking",
        "module": "ridge", "cell": cell, "fit_interval": [0, fit_end], "score_interval": [start, end],
        "alpha": alpha, "alpha_rule": "first configured candidate, fixed before results; not selected",
        "fit_count": 1, "shared_checks": shared, "scores": scores, "clipping": clipping,
        "busiest_hour": busy, "stationarity_relative_errors": {
            "weights": relative_normal_error, "intercept": relative_intercept_error, "tolerance": 1e-10},
        "independent_prediction_and_error_checks": True, "saved_model_reconstruction_checked": True,
        "saved_predictions_round_trip_checked": True, "parameters_unchanged_during_prediction": True,
        "source_sha256": data_report["source_sha256"], "config_sha256": sha256(ROOT / "run_config.yaml"),
        "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
        "code_sha256": {name: sha256(ROOT / name) for name in ["models/ridge.py", "forecasting/data.py",
            "evaluation/metrics.py", "tools/check_ridge.py", "tools/check_forecasting_pipeline.py",
            "tools/check_prepared_data.py", "tools/run_cpu.py"]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "system": platform.system(), "machine": platform.machine()},
        "startup_controls": {key: os.environ.get(key) for key in CPU_ENV},
        "native_pools": [{key: pool.get(key) for key in ["prefix", "internal_api", "version", "num_threads"]}
                         for pool in threadpool_info()], "thread_controls": controls,
        "formal_timing_measured": False, "holdout_forecasts_evaluated": False,
        "output_sha256": {name: sha256(output / name) for name in ["model.json", "predictions.csv"]}}
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
        "score_interval", "alpha", "fit_count", "thread_controls"]}, indent=2))


if __name__ == "__main__":
    main()
