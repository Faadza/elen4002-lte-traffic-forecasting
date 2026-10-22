"""Check one fixed LSTM architecture/seed inside development, without ranking.

The declared inner-tail rule selects epochs, then a fresh same-seed model is
refitted on the full prefix. No architecture search or holdout forecasts occur.
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
import torch
import yaml
from threadpoolctl import threadpool_info, threadpool_limits

from evaluation.metrics import mae, rmse, mase, finalize_predictions, busiest_hour_measures
from forecasting.data import TrafficScaler, make_examples, lstm_inputs
from forecasting.lstm_training import fit_development, state_fingerprint
from models.lstm import build_model, configure_cpu, predict_lstm
from preprocessing.splits import TRAIN_END
from preprocessing.validate_data import sha256, write_json
from tools.check_forecasting_pipeline import check_shared, check_thread_controls
from tools.check_hybrid import save_table
from tools.check_prepared_data import load_validated_data
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]
# Fixed before reading predictions: float32 recurrent kernels differ from the
# independent float64 gate arithmetic. These are numerical, not accuracy limits.
GATE_ATOL = 1e-5
GATE_RTOL = 1e-5


def model_weights(model):
    """Export named parameter values as readable JSON-compatible lists."""
    return {name: value.detach().cpu().tolist() for name, value in model.state_dict().items()}


def restore_weights(weights, hidden_units, dropout, seed):
    """Load explicit parameters into the declared architecture, with no fitting."""
    model = build_model(hidden_units, dropout, seed)
    state = {name: torch.tensor(value, dtype=torch.float32, device="cpu")
             for name, value in weights.items()}
    model.load_state_dict(state, strict=True)
    return model


def predict_saved_weights(weights, sequence, calendar):
    """Independently evaluate i/f/g/o gates using NumPy, without torch LSTM.

    Each window starts with zero cell/hidden states. Evaluation dropout is the
    identity. Inputs have already undergone the specified float32 conversion.
    """
    expected = {"lstm.weight_ih_l0", "lstm.weight_hh_l0", "lstm.bias_ih_l0",
                "lstm.bias_hh_l0", "head.weight", "head.bias"}
    if set(weights) != expected:
        raise ValueError("Expected exactly one unidirectional LSTM and scalar head")
    arrays = {key: np.asarray(value, dtype=np.float64) for key, value in weights.items()}
    wih, whh = arrays["lstm.weight_ih_l0"], arrays["lstm.weight_hh_l0"]
    bih, bhh = arrays["lstm.bias_ih_l0"], arrays["lstm.bias_hh_l0"]
    head, bias = arrays["head.weight"], arrays["head.bias"]
    if whh.ndim != 2:
        raise ValueError("Invalid recurrent weight shape")
    hidden = whh.shape[1]
    if (hidden < 1 or whh.shape != (4 * hidden, hidden) or wih.shape != (4 * hidden, 2)
            or bih.shape != (4 * hidden,) or bhh.shape != (4 * hidden,)
            or head.shape != (1, hidden + 4) or bias.shape != (1,)
            or not all(np.isfinite(a).all() for a in arrays.values())):
        raise ValueError("Invalid saved LSTM parameters")
    x, clock = np.asarray(sequence, dtype=np.float64), np.asarray(calendar, dtype=np.float64)
    if (x.ndim != 3 or x.shape[1:] != (168, 2) or clock.shape != (len(x), 4)
            or not np.isfinite(x).all() or not np.isfinite(clock).all()):
        raise ValueError("Invalid saved-model inputs")
    h = np.zeros((len(x), hidden), dtype=np.float64)
    c = np.zeros_like(h)
    for hour in range(168):
        gates = x[:, hour, :] @ wih.T + h @ whh.T + bih + bhh
        i, f, g, o = np.split(gates, 4, axis=1)
        # exp(-abs(v)) gives stable sigmoid values without overflow.
        def sigmoid(v):
            exp = np.exp(-np.abs(v))
            return np.where(v >= 0, 1 / (1 + exp), exp / (1 + exp))
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
        h = sigmoid(o) * np.tanh(c)
    result = (np.column_stack((h, clock)) @ head.T + bias).ravel()
    if not np.isfinite(result).all():
        raise ValueError("Independent gate evaluation produced nonfinite predictions")
    return result


def verify_stopping_history(metadata, settings):
    """Recompute strict best-epoch and patience decisions from the saved scores."""
    best, best_epoch, stale = math.inf, 0, 0
    history = metadata["stopping_history"]
    if not 1 <= len(history) <= settings["maximum_epochs"]:
        raise AssertionError("Unexpected selection history length")
    for epoch, row in enumerate(history, 1):
        if row["epoch"] != epoch or not math.isfinite(row["tail_mae"]) or row["tail_mae"] < 0:
            raise AssertionError("Invalid stopping history")
        if row["tail_mae"] < best:
            best, best_epoch, stale = row["tail_mae"], epoch, 0
        else:
            stale += 1
        if row["best_epoch"] != best_epoch or row["epochs_since_improvement"] != stale:
            raise AssertionError("Recorded stopping decision differs from independent arithmetic")
        if stale >= settings["patience_epochs"] and epoch != len(history):
            raise AssertionError("Training continued beyond the patience stop")
    if metadata["selected_epochs"] != best_epoch or metadata["best_tail_mae"] != best:
        raise AssertionError("Selected epoch is not the earliest strict minimum")
    if metadata["selection_epochs_run"] != len(history):
        raise AssertionError("Selection epoch count differs from the trace")
    expected_reason = "patience" if stale >= settings["patience_epochs"] else "maximum_epochs"
    if expected_reason == "maximum_epochs" and len(history) != settings["maximum_epochs"]:
        raise AssertionError("Selection stopped before patience or the declared cap")
    if metadata["stop_reason"] != expected_reason:
        raise AssertionError("Stopping reason differs from the trace")
    if [row["epoch"] for row in metadata["refit_history"]] != list(range(1, best_epoch + 1)):
        raise AssertionError("Fresh refit must use exactly the selected epoch count")


def check_module(data_directory, output_directory, progress=None):
    """Run the fixed development check and retain inspectable parameters/results."""
    output = Path(output_directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new or empty output directory to preserve previous checks")
    config = yaml.safe_load((ROOT / "run_config.yaml").read_text())
    settings = config["models"]["lstm"]
    hidden, dropout, seed = settings["hidden_units"][0], settings["output_dropout"][0], settings["seeds_all_formal_periods"][0]
    if (settings["maximum_epochs"], settings["patience_epochs"], settings["inner_stopping_tail_hours"]) != (100, 10, 336):
        raise ValueError("The data check must use the complete approved stopping budget")
    configure_cpu(seed)
    controls = check_thread_controls()
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
    fit = fit_development(values, observed, timestamps, fit_end, hidden_units=hidden,
                          dropout=dropout, seed=seed, settings=settings, progress=progress)
    model, scaler, training = fit.model, fit.scaler, fit.metadata
    verify_stopping_history(training, settings)
    if fit.selection_model is None:
        raise AssertionError("Retain the last inner model to audit its stopping-tail score")
    if training["initial_state_sha256"]["inner_selection"] != training["initial_state_sha256"]["refit"]:
        raise AssertionError("Fresh refit did not reproduce the same initial weights")
    np.testing.assert_array_equal(score.targets, np.flatnonzero(group.eligible_target.to_numpy()[start:end]) + start)
    train = make_examples(values[:fit_end], observed[:fit_end], timestamps[:fit_end], 0, fit_end)
    np.testing.assert_array_equal(train.targets, np.flatnonzero(group.eligible_target.to_numpy()[:fit_end]))
    if len(train.targets) != training["refit_count"] or asdict(scaler) != training["refit_scaler"]:
        raise AssertionError("Refit rows/scaler do not match the common pipeline")
    sequence, calendar = lstm_inputs(score, scaler)
    weights, selection_weights = model_weights(model), model_weights(fit.selection_model)
    state_before = state_fingerprint(model)
    predicted_z = predict_lstm(model, sequence, calendar, batch_size=settings["batch_size"])
    np.testing.assert_array_equal(predicted_z, predict_lstm(model, sequence, calendar, batch_size=settings["batch_size"]))
    manual_z = predict_saved_weights(weights, sequence, calendar)
    np.testing.assert_allclose(predicted_z, manual_z, rtol=GATE_RTOL, atol=GATE_ATOL)
    raw = scaler.inverse(predicted_z)
    final, clipping = finalize_predictions(raw)
    if state_fingerprint(model) != state_before:
        raise AssertionError("Prediction changed learned parameters")
    if not (all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
            and torch.get_num_threads() == torch.get_num_interop_threads() == 1
            and torch.are_deterministic_algorithms_enabled()):
        raise AssertionError("LSTM CPU precision/thread/determinism controls are missing")

    # Check the last actual stopping-tail score from its preserved inner model.
    inner_end = training["inner_fit_interval"][1]
    inner_scaler = TrafficScaler(**training["inner_scaler"])
    tail = make_examples(values[:fit_end], observed[:fit_end], timestamps[:fit_end], inner_end, fit_end)
    tail_sequence, tail_calendar = lstm_inputs(tail, inner_scaler)
    tail_z = predict_lstm(fit.selection_model, tail_sequence, tail_calendar, batch_size=settings["batch_size"])
    manual_tail_z = predict_saved_weights(selection_weights, tail_sequence, tail_calendar)
    np.testing.assert_allclose(tail_z, manual_tail_z, rtol=GATE_RTOL, atol=GATE_ATOL)
    tail_raw = inner_scaler.inverse(tail_z)
    tail_final, _ = finalize_predictions(tail_raw)
    np.testing.assert_equal(mae(tail.actual, tail_final), training["stopping_history"][-1]["tail_mae"])
    errors = [float(p) - float(a) for p, a in zip(final, score.actual)]
    manual_mae = math.fsum(abs(e) for e in errors) / len(errors)
    manual_rmse = math.sqrt(math.fsum(e * e for e in errors) / len(errors))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The fixed sample has no usable weekly MASE denominator")
    scores = {"MAE": mae(score.actual, final), "RMSE": rmse(score.actual, final),
              "MASE": mase(score.actual, final, scale)}
    np.testing.assert_allclose(list(scores.values()), [manual_mae, manual_rmse, manual_mae / scale], rtol=1e-12, atol=1e-12)
    full_prediction = np.full(end - start, np.nan)
    common_mask = np.zeros(end - start, dtype=bool)
    full_prediction[score.targets - start] = final
    common_mask[score.targets - start] = True
    busy = busiest_hour_measures(timestamps[start:end], group.traffic_observed.to_numpy()[start:end],
                                observed[start:end], common_mask, full_prediction)

    metadata = {"method": "lstm", "purpose": "development correctness check only",
        "cell": cell, "fit_interval": [0, fit_end], "seed": seed, "scaler": asdict(scaler),
        "architecture": {"hidden_units": hidden, "output_dropout": dropout, "layers": 1,
            "input_size": 2, "sequence_length": 168, "calendar_features": 4, "bias": True,
            "bidirectional": False, "internal_dropout": 0.0, "proj_size": 0, "batch_first": True,
            "dtype": "float32", "device": "cpu", "head": "last hidden -> output dropout -> calendar -> linear"},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "state_sha256": state_before, "selection_state_sha256": state_fingerprint(fit.selection_model),
        "selection_scaler": asdict(inner_scaler), "selection_saved_epoch": training["selection_epochs_run"],
        "selected_epochs": training["selected_epochs"], "optimizer_defaults": training["optimizer_defaults"],
        "input_order": ["168 hours oldest first: scaled traffic, original missingness flag",
                        "daily sine", "daily cosine", "weekly sine", "weekly cosine"],
        "prediction_rule": "max(0, mean + sd * predicted_z); no output activation",
        "inference_batch_size": settings["batch_size"], "dropout_during_inference": False,
        "manual_gate_tolerances": {"absolute_z": GATE_ATOL, "relative_z": GATE_RTOL,
            "reason": "float32 recurrent kernels versus independent float64 gate arithmetic; fixed before outputs"}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "model_metadata.json", metadata)
    write_json(output / "model_weights.json", weights)
    write_json(output / "selection_weights.json", selection_weights)
    write_json(output / "training_history.json", training)
    # State dictionaries contain parameters only, not a pickled model object.
    torch.save(model.state_dict(), output / "model_state.pt")
    restored = build_model(hidden, dropout, seed)
    restored.load_state_dict(torch.load(output / "model_state.pt", map_location="cpu", weights_only=True))
    np.testing.assert_array_equal(predict_lstm(restored, sequence, calendar, settings["batch_size"]), predicted_z)
    json_model = restore_weights(json.loads((output / "model_weights.json").read_text()), hidden, dropout, seed)
    np.testing.assert_array_equal(predict_lstm(json_model, sequence, calendar, settings["batch_size"]), predicted_z)
    predictions = pd.DataFrame({"cell": cell, "method": "lstm", "seed": seed, "target_index": score.targets,
        "timestamp": score.timestamps, "actual": score.actual, "prediction_z": predicted_z,
        "prediction_raw": raw, "prediction": final})
    save_table(predictions, output / "predictions.csv")
    tail_predictions = pd.DataFrame({"cell": cell, "epoch": training["selection_epochs_run"],
        "target_index": tail.targets, "timestamp": tail.timestamps, "actual": tail.actual,
        "prediction_z": tail_z, "prediction_raw": tail_raw, "prediction": tail_final})
    save_table(tail_predictions, output / "tail_predictions.csv")
    result = {"status": "passed", "scope": "development correctness check only; one fixed architecture/seed; no model ranking",
        "module": "lstm", "cell": cell, "fit_interval": [0, fit_end], "score_interval": [start, end],
        "hidden_units": hidden, "dropout": dropout, "seed": seed, "selected_epochs": training["selected_epochs"],
        "selection_epochs_run": training["selection_epochs_run"], "stop_reason": training["stop_reason"],
        "maximum_epochs_reached": training["maximum_epochs_reached"], "component_fit_counts": training["component_fit_counts"],
        "candidate_rule": "first declared hidden width/dropout/seed; complete inner-tail stopping protocol; no architecture or seed search",
        "shared_checks": shared, "scores": scores, "clipping": clipping, "busiest_hour": busy,
        "manual_gate_max_absolute_differences": {"outer_score_z": float(np.max(np.abs(predicted_z - manual_z))),
                                                "last_inner_tail_z": float(np.max(np.abs(tail_z - manual_tail_z)))},
        "manual_gate_tolerances": metadata["manual_gate_tolerances"],
        "stopping_trace_independently_checked": True, "last_tail_score_recomputed": True,
        "saved_tensor_and_json_reload_equal": True, "saved_tables_round_trip_checked": True,
        "repeat_prediction_equal": True, "parameters_unchanged_during_prediction": True,
        "same_seed_fresh_initialisation_verified": training["same_seed_initialisation_verified"],
        "source_sha256": data_report["source_sha256"], "config_sha256": sha256(ROOT / "run_config.yaml"),
        "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
        "code_sha256": {name: sha256(ROOT / name) for name in ["models/lstm.py", "forecasting/lstm_training.py",
            "forecasting/data.py", "evaluation/metrics.py", "tools/check_lstm.py", "tools/check_hybrid.py",
            "tools/check_forecasting_pipeline.py", "tools/check_prepared_data.py", "tools/run_cpu.py"]},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "system": platform.system(), "machine": platform.machine()},
        "torch_controls": {"device": "cpu", "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "mps_available_but_unused": bool(torch.backends.mps.is_available()), "mkldnn_enabled": torch.backends.mkldnn.enabled},
        "startup_controls": {key: os.environ.get(key) for key in CPU_ENV},
        "native_pools": [{key: pool.get(key) for key in ["prefix", "internal_api", "version", "num_threads"]}
                         for pool in threadpool_info()], "thread_controls": controls,
        "formal_timing_measured": False, "holdout_forecasts_evaluated": False,
        "output_sha256": {name: sha256(output / name) for name in ["model_metadata.json", "model_weights.json",
            "selection_weights.json", "training_history.json", "model_state.pt", "predictions.csv", "tail_predictions.csv"]}}
    write_json(output / "checks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    def progress(event):
        if event["epoch"] == 1 or event["epoch"] % 10 == 0:
            print(f"LSTM {event['phase']}: completed epoch {event['epoch']}", flush=True)
    with threadpool_limits(limits=1):
        result = check_module(args.data_directory, args.output_dir, progress)
    print(json.dumps({key: result[key] for key in ["status", "scope", "module", "cell", "fit_interval",
        "score_interval", "hidden_units", "dropout", "seed", "selected_epochs", "selection_epochs_run",
        "stop_reason", "maximum_epochs_reached", "component_fit_counts", "thread_controls"]}, indent=2))


if __name__ == "__main__":
    main()
