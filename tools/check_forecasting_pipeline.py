"""Check one module on a fixed development sample; never run a model search.

Use tools/run_cpu.py as the launcher. Choose shared, persistence,
daily_seasonal_naive or weekly_seasonal_naive explicitly. Reports are preliminary
correctness evidence, not a model ranking, benchmark or final project result.
"""
import argparse
import json
import math
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from evaluation.metrics import (mae, rmse, mase, seasonal_scale,
    finalize_predictions, busiest_hour_measures)
from forecasting.data import make_examples, fit_scaler, tabular_inputs, lstm_inputs, scaled_targets
from models.baselines import persistence, daily_naive, weekly_naive
from preprocessing.splits import LOOKBACK, TRAIN_END
from preprocessing.validate_data import sha256, write_json
from tools.check_prepared_data import load_validated_data
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"persistence": (persistence, 1), "daily_seasonal_naive": (daily_naive, 24),
           "weekly_seasonal_naive": (weekly_naive, 168)}


def check_thread_controls():
    """Inspect controls inside threadpool_limits; this is not a timing certificate."""
    wrong = [key for key, value in CPU_ENV.items() if os.environ.get(key) != value]
    if wrong:
        raise RuntimeError("Use tools/run_cpu.py; missing CPU controls: " + ", ".join(wrong))
    pools = threadpool_info()
    invalid = [pool for pool in pools if pool.get("num_threads") != 1]
    if invalid:
        details = [{key: pool.get(key) for key in ("prefix", "num_threads")} for pool in invalid]
        raise RuntimeError("Detected numerical pools do not report one thread: " + json.dumps(details))

    blas = np.show_config(mode="dicts").get("Build Dependencies", {}).get("blas", {})
    accelerate = (platform.system() == "Darwin" and blas.get("name") == "accelerate"
                  and blas.get("found") is True)
    # An empty list is expected for Accelerate with the pinned threadpoolctl.
    # Allow correctness checks, but never present an unobserved limit as verified.
    if not pools and not accelerate:
        raise RuntimeError("No supported numerical pool detected and NumPy is not confirmed "
                           "as Apple Accelerate on macOS; inspect the numerical backend")
    limitations = []
    if accelerate:
        limitations.append("Apple Accelerate is not inspected by threadpoolctl. "
                           "VECLIB_MAXIMUM_THREADS=1 is requested, but its runtime limit "
                           "is not independently verified. Resolve before formal timing.")
    return {
        "status": "partial_accelerate_visibility" if accelerate else "detected_pools_checked",
        "requested_threads": 1,
        "startup_controls_verified": True,
        "detected_pool_count": len(pools),
        "detected_pool_limits_verified": bool(pools),
        "numpy_blas_build": {key: blas.get(key) for key in ("name", "found", "version")},
        "accelerate_runtime_limit_verified": False if accelerate else None,
        "limitations": limitations,
        "formal_timing_eligible": False,
    }


def check_shared(values, observed, timestamps, fit_end, start, end):
    """Check the connected data path against separate elementary calculations."""
    scaler = fit_scaler(values, observed, fit_end)
    train = make_examples(values, observed, timestamps, 0, fit_end)
    score = make_examples(values, observed, timestamps, start, end)
    if not len(train.targets) or not len(score.targets):
        raise ValueError("The fixed development sample has no usable examples")
    genuine = [float(values[i]) for i in range(fit_end) if observed[i]]
    mean = math.fsum(genuine) / len(genuine)
    sd = math.sqrt(math.fsum((v-mean)**2 for v in genuine) / len(genuine))
    np.testing.assert_allclose([scaler.mean, scaler.sd], [mean, sd], rtol=1e-13, atol=1e-13)
    tabular = tabular_inputs(score, scaler)
    sequence, calendar = lstm_inputs(score, scaler)
    np.testing.assert_allclose(scaler.inverse(scaled_targets(score, scaler)), score.actual,
                               rtol=1e-13, atol=1e-13)
    for row, target in enumerate(score.targets):
        np.testing.assert_array_equal(score.history[row], values[target-LOOKBACK:target])
        np.testing.assert_array_equal(score.missing[row], 1-observed[target-LOOKBACK:target])
        if not observed[target] or score.actual[row] != values[target]:
            raise ValueError("A scoring target is not a genuine observation")
        stamp = pd.Timestamp(timestamps[target])
        hour, week_hour = stamp.hour, 24*stamp.dayofweek+stamp.hour
        expected_calendar = [math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24),
                             math.sin(2*math.pi*week_hour/168), math.cos(2*math.pi*week_hour/168)]
        np.testing.assert_allclose(score.calendar[row], expected_calendar, atol=1e-14)
    pairs = [abs(float(values[t])-float(values[t-168])) for t in range(168, fit_end)
             if observed[t] and observed[t-168]]
    scale = seasonal_scale(values, observed, fit_end)
    np.testing.assert_allclose(scale, math.fsum(pairs)/len(pairs), rtol=1e-13)
    expected_targets = [t for t in range(start, end) if t >= LOOKBACK and observed[t]
                        and all(math.isfinite(v) for v in values[t-LOOKBACK:t])]
    np.testing.assert_array_equal(score.targets, expected_targets)
    if tabular.shape != (len(score.targets), 340) or tabular.dtype != np.float64:
        raise ValueError("Incorrect tabular feature layout")
    if sequence.shape != (len(score.targets), 168, 2) or sequence.dtype != np.float32:
        raise ValueError("Incorrect LSTM feature layout")
    if calendar.shape != (len(score.targets), 4) or calendar.dtype != np.float32:
        raise ValueError("Incorrect LSTM calendar layout")
    return score, scale, {"fit_end": fit_end, "mean": scaler.mean, "sd": scaler.sd,
        "observed_scaling_hours": scaler.observed_count, "fitting_targets": len(train.targets),
        "scoring_targets": len(score.targets), "tabular_shape": list(tabular.shape),
        "sequence_shape": list(sequence.shape), "weekly_observed_pairs": len(pairs),
        "weekly_scale": scale}


def check_module(data_directory, module, output_directory):
    """Verify one explicit comparator; keep each module's evidence separate."""
    config = yaml.safe_load((ROOT / "run_config.yaml").read_text())
    hourly, data_report = load_validated_data(data_directory)
    cell = config["tuning_panel"]["ids"][0]  # Fixed before examining any forecasts.
    group = hourly.loc[hourly.cell == cell].reset_index(drop=True).iloc[:TRAIN_END]
    fit_end = config["chronology"]["folds"][0]["fit"][1]
    start = config["chronology"]["folds"][0]["score"][0]
    end = start + 72  # A small integration check, not a complete fold experiment.
    if end > TRAIN_END:
        raise ValueError("Correctness checks must stay inside development")
    values, observed = group.traffic.to_numpy(), group.observed.to_numpy()
    timestamps = pd.DatetimeIndex(group.ts)
    examples, scale, shared = check_shared(values, observed, timestamps, fit_end, start, end)
    checked_targets = np.flatnonzero(group.eligible_target.to_numpy()[start:end]) + start
    np.testing.assert_array_equal(examples.targets, checked_targets)
    result = {"status": "passed", "scope": "development correctness check only; no model ranking",
        "module": module, "cell": cell, "score_interval": [start, end], "shared_checks": shared,
        "source_sha256": data_report["source_sha256"],
        "config_sha256": sha256(ROOT / "run_config.yaml"),
        "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
        "code_sha256": {name: sha256(ROOT / name) for name in ["forecasting/data.py",
            "models/baselines.py", "evaluation/metrics.py", "tools/check_forecasting_pipeline.py"]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "system": platform.system(), "machine": platform.machine()},
        "startup_controls": {key: os.environ.get(key) for key in CPU_ENV},
        "native_pools": [{key: pool.get(key) for key in ["prefix", "internal_api", "version", "num_threads"]}
                         for pool in threadpool_info()],
        "thread_controls": check_thread_controls(),
        "formal_timing_measured": False, "holdout_forecasts_evaluated": False}
    predictions = None
    if module != "shared":
        method, lag = METHODS[module]
        raw = method(values, examples.targets)
        expected = np.array([values[int(t)-lag] for t in examples.targets])
        np.testing.assert_array_equal(raw, expected)
        final, clipping = finalize_predictions(raw)
        # Independently recompute errors with Python arithmetic, not the metric functions.
        errors = [float(p)-float(a) for p, a in zip(final, examples.actual)]
        manual_mae = math.fsum(abs(e) for e in errors)/len(errors)
        manual_rmse = math.sqrt(math.fsum(e*e for e in errors)/len(errors))
        scores = {"MAE": mae(examples.actual, final), "RMSE": rmse(examples.actual, final),
                  "MASE": mase(examples.actual, final, scale)}
        np.testing.assert_allclose(list(scores.values()),
            [manual_mae, manual_rmse, manual_mae/scale], rtol=1e-13, atol=1e-13)
        full_forecast = np.full(end-start, np.nan)
        common_mask = np.zeros(end-start, dtype=bool)
        full_forecast[examples.targets-start] = final
        common_mask[examples.targets-start] = True
        busy = busiest_hour_measures(timestamps[start:end],
            group.traffic_observed.to_numpy()[start:end], observed[start:end], common_mask, full_forecast)
        result.update(scores=scores, clipping=clipping, busiest_hour=busy,
            independent_forecast_and_error_checks=True)
        predictions = pd.DataFrame({"cell": cell, "method": module, "target_index": examples.targets,
            "timestamp": examples.timestamps, "actual": examples.actual,
            "prediction_raw": raw, "prediction": final})
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory to preserve previous checks")
    output.mkdir(parents=True, exist_ok=True)
    if predictions is not None:
        path = output / "predictions.csv"
        predictions.to_csv(path, index=False, float_format="%.17g")
        saved = pd.read_csv(path, float_precision="round_trip")
        np.testing.assert_array_equal(saved.prediction.to_numpy(), predictions.prediction.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), predictions.actual.to_numpy())
        result["predictions_sha256"] = sha256(path)
    write_json(output / "checks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory")
    parser.add_argument("--module", required=True, choices=["shared", *METHODS])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        check_thread_controls()  # Reject bad controls before reading the prepared data.
        result = check_module(args.data_directory, args.module, args.output_dir)
    print(json.dumps({key: result[key] for key in
        ["status", "scope", "module", "cell", "score_interval", "thread_controls"]}, indent=2))


if __name__ == "__main__":
    main()
