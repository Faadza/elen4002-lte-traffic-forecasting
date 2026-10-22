"""Check declared larger candidates in sequential, fresh CPU processes.

This command checks the complete workflow. It cannot tune models, choose cells,
change the scoring period, or enable confirmation/final forecasting.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.cpu_controls import single_thread_cpu
from tools.run_cpu import CPU_ENV
from tools.run_development_pilot import (sha256, write_json, code_hashes,
                                         load_series, file_hashes)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = "421a4a6742cec01bde5a756927db9cca7fcf76efb683e040e3b0f2e4f2fe5911"
METHODS = ("ridge", "direct_lightgbm", "ridge_lightgbm_hybrid", "lstm")
GROUPS = (*METHODS[:3], "lstm-h32-d0", "lstm-h32-d0p2")
FIT_END, SCORE_END = 5460, 6132


def read_config():
    import yaml
    if sha256(ROOT / "run_config.yaml") != CONFIG_SHA256:
        raise ValueError("Review this bounded capacity check after changing run_config.yaml")
    return yaml.safe_load((ROOT / "run_config.yaml").read_text())


def capacity_jobs():
    jobs = []
    for group in GROUPS:
        method = "lstm" if group.startswith("lstm-") else group
        dropout = (0.0 if group == "lstm-h32-d0" else 0.2) if method == "lstm" else None
        for repeat in range(1, 4):
            seed = 2025 + repeat if method == "lstm" else None
            jobs.append({"id": f"{group}-{repeat}", "group": group, "method": method,
                         "repeat": repeat, "seed": seed, "dropout": dropout})
    return jobs


def validate_fit_evidence(job, settings, training, elapsed):
    """Reject a saved fit whose actual settings differ from this fixed workload."""
    import math
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Complete fitting time must be positive and finite")
    if settings.get("fit_end") != FIT_END or settings.get("lookback") != 168:
        raise ValueError("Unexpected fitted history or prefix")
    method = job["method"]
    if method in ("ridge", "ridge_lightgbm_hybrid") and settings.get("alpha") != 0.1:
        raise ValueError("Capacity ridge must use alpha 0.1")
    if method in ("direct_lightgbm", "ridge_lightgbm_hybrid"):
        if settings.get("requested_trees") != 512 or settings.get("leaf_cap") != 31:
            raise ValueError("Capacity trees must use 512 rounds and a 31-leaf cap")
        if not 1 <= training.get("actual_trees", 0) <= 512:
            raise ValueError("Actual tree count is outside the requested budget")
    if method == "ridge_lightgbm_hybrid" and settings.get("weight") != 1.0:
        raise ValueError("Capacity hybrid requires full residual correction")
    expected_counts = {"ridge": {"ridge": 1}, "direct_lightgbm": {"direct_lightgbm": 1},
                       "ridge_lightgbm_hybrid": {"ridge": 1, "correction": 1},
                       "lstm": {"inner_selection": 1, "full_prefix_refit": 1}}
    if training.get("component_fit_counts") != expected_counts[method]:
        raise ValueError("Component fitting phases differ from the declared workload")
    if method == "lstm":
        required = {"hidden_units": 32, "dropout": job["dropout"], "seed": job["seed"],
                    "batch_size": 32}
        if any(settings.get(name) != value for name, value in required.items()):
            raise ValueError("Unexpected fitted LSTM architecture or seed")
        if training.get("effective_budgets") != {"maximum_epochs": 100, "patience_epochs": 10,
                                                  "inner_stopping_tail_hours": 336}:
            raise ValueError("Capacity check cannot change the LSTM training budget")
        selected, ran = training.get("selected_epochs", 0), training.get("selection_epochs_run", 0)
        if not 1 <= selected <= ran <= 100 or settings.get("selected_epochs") != selected:
            raise ValueError("Invalid LSTM epoch accounting")
        if len(training.get("stopping_history", [])) != ran or len(training.get("refit_history", [])) != selected:
            raise ValueError("LSTM training histories do not cover the saved epoch counts")
        reason = training.get("stop_reason")
        if reason not in ("patience", "maximum_epochs") or (reason == "maximum_epochs" and ran != 100):
            raise ValueError("Invalid LSTM stopping reason")
        if training.get("maximum_epochs_reached") is not (ran == 100):
            raise ValueError("LSTM cap flag disagrees with the epoch count")
        times = training.get("phase_timings", {})
        phases = [times.get("selection_seconds"), times.get("refit_seconds")]
        if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in phases):
            raise ValueError("LSTM phase times must be positive and finite")
        if sum(phases) > elapsed:
            raise ValueError("LSTM phase times exceed the complete fitting time")


def verify_control_report(report):
    if report.get("runtime_library_controls_verified") is not True:
        raise ValueError("Saved worker runtime controls are not verified")


def verify_inputs(directory, plan):
    if plan["code_sha256"] != code_hashes() or plan["config_sha256"] != CONFIG_SHA256:
        raise ValueError("Code/configuration changed since this capacity check was prepared")
    if sha256(directory / "series.npz") != plan["series_sha256"]:
        raise ValueError("Capacity check input bytes changed")
    if plan["jobs"] != capacity_jobs() or plan["cell"] != "Cell_003791":
        raise ValueError("Unexpected capacity workload")
    if plan["fit_interval"] != [0, 5460] or plan["score_interval"] != [5460, 6132]:
        raise ValueError("Capacity check must use the fixed last development fold")
    if sha256(ROOT / "run_config.yaml") != CONFIG_SHA256:
        raise ValueError("Scientific configuration bytes changed")


def verify_series(data, plan):
    """Cross-check the recorded workload against its physical worker arrays."""
    import numpy as np
    from preprocessing.windows import eligible_targets
    from tools.pilot_measurements import select_origins
    required = {"traffic", "traffic_observed", "observed", "timestamps", "targets", "origins"}
    if set(data) != required or any(len(data[name]) != SCORE_END for name in
                                   ("traffic", "traffic_observed", "observed", "timestamps")):
        raise ValueError("Capacity arrays must contain exactly the development prefix")
    if any(np.asarray(data[name]).ndim != 1 for name in required):
        raise ValueError("Capacity arrays must be one dimensional")
    if not np.array_equal(np.isfinite(data["traffic_observed"]), data["observed"]):
        raise ValueError("Genuine observations disagree with the recorded mask")
    flags = np.asarray(data["observed"], dtype=bool)
    np.testing.assert_array_equal(data["traffic"][flags], data["traffic_observed"][flags])
    if np.any(np.diff(data["timestamps"].asi8) != 3_600_000_000_000):
        raise ValueError("Capacity clock must contain consecutive hourly timestamps")
    eligible = eligible_targets(data["traffic"], data["observed"])
    counts = {"full_fit": int(eligible[:5460].sum()), "inner_fit": int(eligible[:5124].sum()),
              "inner_tail": int(eligible[5124:5460].sum()), "score": int(eligible[5460:6132].sum())}
    if counts != {"full_fit": 5281, "inner_fit": 4945, "inner_tail": 336, "score": 672}:
        raise ValueError("Capacity training/tail/score counts differ from the validated fixed cell")
    expected = np.flatnonzero(eligible[5460:6132]) + 5460
    np.testing.assert_array_equal(data["targets"], expected)
    np.testing.assert_array_equal(data["targets"], plan["targets"])
    np.testing.assert_array_equal(data["origins"], select_origins(expected))
    np.testing.assert_array_equal(data["origins"], plan["origins"])
    if len(data["traffic"]) != 6132 or len(expected) != 672:
        raise ValueError("Capacity check data exceed or do not cover the fixed development workload")


def fit_worker(directory, plan, job, config, controls):
    import numpy as np
    from forecasting.capacity_models import fit_model
    from forecasting.pilot_models import predict_model, save_model, model_fingerprint
    from tools.pilot_measurements import elapsed_call, memory_snapshot

    data = load_series(directory)
    verify_series(data, plan)
    output = directory / "jobs" / job["id"]
    output.mkdir(parents=True, exist_ok=False)
    before = controls.verify()
    def fit():
        return fit_model(job["method"], data["traffic"][:5460], data["observed"][:5460],
                         data["timestamps"][:5460], 5460, config,
                         seed=job["seed"], dropout=job["dropout"])
    bundle, seconds = elapsed_call(fit)
    memory = memory_snapshot()  # Capture before export or prediction audits.
    after = controls.verify()
    validate_fit_evidence(job, bundle.settings, bundle.training_metadata, seconds)
    state_before = model_fingerprint(bundle)
    save_model(bundle, output / "model")
    artifact_hashes = file_hashes(output / "model")
    # Reference outputs are made after the fitting timer and RSS observation.
    raw, final = predict_model(bundle, data["traffic"], data["observed"],
                               data["timestamps"], data["targets"])
    if model_fingerprint(bundle) != state_before:
        raise AssertionError("Reference prediction changed fitted parameters")
    np.savez(output / "reference_predictions.npz", raw=raw, final=final)
    write_json(output / "training.json", bundle.training_metadata)
    result = {"status": "passed", "job": job, "fit_seconds": seconds,
              "fit_memory": memory,
              "fit_kind": "development_epoch_selection_and_refit" if job["method"] == "lstm" else "complete_fit",
              "settings": bundle.settings, "state_sha256": state_before,
              "artifact_sha256": artifact_hashes,
              "artifact_bytes": sum(p.stat().st_size for p in (output / "model").rglob("*") if p.is_file()),
              "training_sha256": sha256(output / "training.json"),
              "reference_sha256": sha256(output / "reference_predictions.npz"),
              "controls_before": before, "controls_after": after,
              "formal_timing_eligible": False}
    return output / "fit.json", result


def predict_worker(directory, plan, job, config, controls):
    import numpy as np
    from forecasting.pilot_models import load_model, predict_model, model_fingerprint
    from tools.pilot_measurements import measure_predictions, memory_snapshot
    from tools.pilot_results import score_predictions, save_predictions

    data = load_series(directory)
    verify_series(data, plan)
    output = directory / "jobs" / job["id"]
    fitted = json.loads((output / "fit.json").read_text())
    if fitted["job"] != job or file_hashes(output / "model") != fitted["artifact_sha256"]:
        raise ValueError("Model artifact or job differs from its completed fit")
    if sha256(output / "reference_predictions.npz") != fitted["reference_sha256"]:
        raise ValueError("Reference predictions changed")
    bundle = load_model(output / "model")
    before = controls.verify()
    # Float32 batch kernels may round differently from single-origin kernels.
    sd = bundle.scaler.sd if bundle.scaler is not None else 1.0
    rtol, atol = (1e-5, 1e-5 * sd) if job["method"] == "lstm" else (1e-10, 1e-9 * max(1.0, sd))
    def predict(targets):
        return predict_model(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
    measured = measure_predictions(predict, data["targets"], data["origins"], rtol=rtol, atol=atol)
    memory = memory_snapshot()
    after = controls.verify()
    if model_fingerprint(bundle) != fitted["state_sha256"]:
        raise AssertionError("Saved-model loading or prediction changed fitted parameters")
    with np.load(output / "reference_predictions.npz", allow_pickle=False) as reference:
        np.testing.assert_array_equal(measured["raw"], reference["raw"])
        np.testing.assert_array_equal(measured["final"], reference["final"])
    scored = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
                               5460, 5460, 6132, data["targets"], measured["raw"], measured["final"])
    if scored["mase_status"] != "available":
        raise ValueError("The fixed capacity cell requires its validated positive weekly scale")
    save_predictions(output / "predictions.csv", plan["cell"], job["method"], job["seed"],
                     data["targets"], data["timestamps"][data["targets"]],
                     data["traffic_observed"][data["targets"]], measured["raw"], measured["final"])
    result = {"status": "passed", "job": job, "evaluation": scored,
              "timings": measured["timings"], "prediction_memory": memory,
              "reload_predictions_exact": True, "fitted_state_unchanged": True,
              "controls_before": before, "controls_after": after,
              "predictions_sha256": sha256(output / "predictions.csv"),
              "formal_timing_eligible": False}
    return output / "predict.json", result


def worker(directory, job_id, phase):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text())
    verify_inputs(directory, plan)
    job = next((item for item in plan["jobs"] if item["id"] == job_id), None)
    if job is None:
        raise ValueError("Unknown capacity job")
    # Each fresh process establishes its own caller-thread numerical controls.
    with single_thread_cpu() as controls:
        config = read_config()
        if phase == "fit":
            path, result = fit_worker(directory, plan, job, config, controls)
        else:
            path, result = predict_worker(directory, plan, job, config, controls)
    verify_inputs(directory, plan)
    write_json(path, result)  # A pass is saved only after successful cleanup.


def prepare(data_directory, directory, host_ready):
    import numpy as np
    from tools.check_prepared_data import load_validated_data
    from tools.check_cpu_controls import machine_information
    from tools.pilot_measurements import select_origins

    config = read_config()
    hourly, checked = load_validated_data(data_directory)
    cell = config["tuning_panel"]["ids"][0]
    if cell != "Cell_003791" or cell not in checked["tuning_cells"]:
        raise ValueError("Capacity check cell differs from the verified tuning panel")
    group = hourly.loc[hourly.cell == cell].iloc[:6132].reset_index(drop=True)
    targets = np.flatnonzero(group.eligible_target.to_numpy()[5460:6132]) + 5460
    if len(group) != 6132 or len(targets) != 672:
        raise ValueError("Capacity check calendar/mask differs from the verified last development fold")
    np.savez(directory / "series.npz", traffic=group.traffic.to_numpy(),
             traffic_observed=group.traffic_observed.to_numpy(), observed=group.observed.to_numpy(),
             timestamps=group.ts.to_numpy(), targets=targets, origins=select_origins(targets))
    plan = {"scope": "bounded development capacity and training-budget check; no tuning or model ranking",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cell": cell, "fit_interval": [0, 5460], "score_interval": [5460, 6132],
            "targets": targets.tolist(), "origins": select_origins(targets).tolist(), "jobs": capacity_jobs(),
            "configuration_count": len(GROUPS), "learned_fit_jobs": 15,
            "component_fitting_phases": 24, "fresh_worker_processes": 30,
            "maximum_lstm_epochs_unchanged": 100,
            "eligible_counts": {"full_fit": 5281, "inner_fit": 4945, "inner_tail": 336, "score": 672},
            "source_sha256": checked["source_sha256"], "config_sha256": CONFIG_SHA256,
            "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
            "series_sha256": sha256(directory / "series.npz"), "code_sha256": code_hashes(),
            "host_conditions_confirmed_by_operator": bool(host_ready),
            "machine_before": machine_information(),
            "versions": {name: metadata.version(name) for name in
                         ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch", "psutil", "threadpoolctl"]},
            "formal_timing_eligible": False, "holdout_forecasts_evaluated": False}
    verify_series(load_series(directory), plan)
    write_json(directory / "plan.json", plan)
    return plan


def launch(directory, job, phase):
    arguments = [sys.executable, str(ROOT / "tools/run_cpu.py"), "-m", "tools.run_development_capacity",
                 "--worker", phase, "--job", job["id"], "--output-dir", str(directory)]
    log = directory / f"{job['id']}-{phase}.log"
    print(f"Capacity check {phase}: {job['id']}", flush=True)
    with log.open("x") as stream:
        result = subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Capacity check job failed; retained log: {log}")


def summarize(directory, plan):
    import numpy as np
    import pandas as pd
    from statistics import median
    from tools.check_cpu_controls import machine_information

    from tools.pilot_results import score_predictions
    data = load_series(directory)
    verify_series(data, plan)
    rows, prediction_frames = [], {}
    for job in plan["jobs"]:
        output = directory / "jobs" / job["id"]
        fit = json.loads((output / "fit.json").read_text())
        prediction = json.loads((output / "predict.json").read_text())
        if fit["status"] != "passed" or prediction["status"] != "passed":
            raise ValueError("A required capacity job is incomplete")
        if fit["job"] != job or prediction["job"] != job:
            raise ValueError("Unexpected completed job identity")
        if sha256(output / "predictions.csv") != prediction["predictions_sha256"]:
            raise ValueError("Saved prediction file changed")
        if file_hashes(output / "model") != fit["artifact_sha256"]:
            raise ValueError("A saved model changed after fitting")
        for filename, field in (("training.json", "training_sha256"),
                                ("reference_predictions.npz", "reference_sha256")):
            if sha256(output / filename) != fit[field]:
                raise ValueError("Saved fitting evidence changed")
        training = json.loads((output / "training.json").read_text())
        validate_fit_evidence(job, fit["settings"], training, fit["fit_seconds"])
        for report in (fit, prediction):
            for field in ("controls_before", "controls_after"):
                verify_control_report(report[field])
        if prediction.get("reload_predictions_exact") is not True or prediction.get("fitted_state_unchanged") is not True:
            raise ValueError("Saved prediction audit did not confirm unchanged reloaded forecasts")
        frame = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
        if not (frame.cell.eq(plan["cell"]).all() and frame.method.eq(job["method"]).all()):
            raise ValueError("Saved predictions have the wrong cell or method")
        if job["seed"] is None:
            if not frame.seed.isna().all():
                raise ValueError("Unexpected seed on deterministic predictions")
        elif not frame.seed.eq(job["seed"]).all():
            raise ValueError("Saved predictions have the wrong LSTM seed")
        np.testing.assert_array_equal(frame.actual, data["traffic_observed"][data["targets"]])
        np.testing.assert_array_equal(pd.to_datetime(frame.timestamp).to_numpy(),
                                      data["timestamps"][data["targets"]].to_numpy())
        recomputed = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
                                       5460, 5460, 6132, frame.target_index.to_numpy(),
                                       frame.prediction_raw.to_numpy(), frame.prediction.to_numpy())
        if recomputed != prediction["evaluation"]:
            raise AssertionError("Saved metrics do not reproduce from saved forecasts")
        np.testing.assert_array_equal(frame.target_index, plan["targets"])
        with np.load(output / "reference_predictions.npz", allow_pickle=False) as reference:
            np.testing.assert_array_equal(frame.prediction_raw, reference["raw"])
            np.testing.assert_array_equal(frame.prediction, reference["final"])
        prediction_frames[job["id"]] = frame
        rows.append({"job": job, "fit": fit, "training": training, "prediction": prediction})
    first = next(iter(prediction_frames.values()))
    for frame in prediction_frames.values():
        for column in ("target_index", "timestamp", "actual"):
            np.testing.assert_array_equal(frame[column], first[column])
    for method in ("ridge", "direct_lightgbm", "ridge_lightgbm_hybrid"):
        for repeat in (2, 3):
            for column in ("prediction_raw", "prediction"):
                np.testing.assert_array_equal(prediction_frames[f"{method}-1"][column],
                                              prediction_frames[f"{method}-{repeat}"][column])
    summaries = []
    for group in GROUPS:
        items = [row for row in rows if row["job"]["group"] == group]
        times = [item["fit"]["fit_seconds"] for item in items]
        summaries.append({"group": group, "method": items[0]["job"]["method"],
                          "dropout": items[0]["job"]["dropout"],
                          "fit_seconds": {"observations": times, "median": median(times),
                                          "minimum": min(times), "maximum": max(times)} if times else None,
                          "representative_job": items[0]["job"]["id"],
                          "representative_artifact_bytes": items[0]["fit"]["artifact_bytes"],
                          "scores": {key: float(np.mean([x["prediction"]["evaluation"]["scores"][key]
                                                        for x in items])) for key in ("MAE", "RMSE", "MASE")}})
    return {"status": "passed", "scope": plan["scope"], "cell": plan["cell"],
            "fit_interval": plan["fit_interval"], "score_interval": plan["score_interval"],
            "method_count": len(METHODS), "configuration_count": len(GROUPS),
            "learned_fit_jobs": len(rows), "prediction_jobs": len(rows),
            "component_fitting_phases": sum(sum(row["training"]["component_fit_counts"].values()) for row in rows),
            "fresh_worker_processes": 2 * len(rows),
            "saved_prediction_rows": sum(len(frame) for frame in prediction_frames.values()),
            "common_target_count": len(plan["targets"]), "latency_origin_count": len(plan["origins"]),
            "deterministic_repeat_predictions_exact": True, "all_reload_predictions_exact": True,
            "same_targets_and_actuals": True, "configurations_in_declared_order": summaries, "jobs": rows,
            "lstm_training_diagnostics": [
                {"job": row["job"], **{name: row["training"][name] for name in
                    ("selected_epochs", "selection_epochs_run", "stop_reason", "maximum_epochs_reached", "phase_timings")}}
                for row in rows if row["job"]["method"] == "lstm"],
            "machine_after": machine_information(), "formal_timing_eligible": False,
            "formal_timing_measured": False, "capacity_timing_measured": True,
            "holdout_forecasts_evaluated": False, "model_ranking_performed": False,
            "limitations": ["One cell and fixed larger candidates only; no inference about best model or full-panel accuracy.",
                            "LSTM development fit includes epoch selection and fresh refitting, not fixed-epoch deployment fit.",
                            "RSS is total-process high-water memory including dependencies, data and model loading.",
                            "Other cells, earlier folds and longer final fitting prefixes are not measured by this check.",
                            "Epoch-cap hits are diagnostic evidence, not an automatic budget extension or failed fit.",
                            "Full-search scheduling requires review of the Mac outputs; no workload estimate is asserted here.",
                            "Machine snapshots and operator confirmation do not eliminate OS scheduling or thermal variability."]}


def run_capacity(data_directory, output_directory, host_ready=False):
    directory = Path(output_directory).resolve()
    if directory.exists():
        raise ValueError("Use a new output directory; completed or interrupted evidence is preserved")
    wrong = [key for key, value in CPU_ENV.items() if os.environ.get(key) != value]
    if wrong:
        raise RuntimeError("Launch with tools/run_cpu.py before importing numerical libraries")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        with single_thread_cpu() as controls:
            plan = prepare(data_directory, directory, host_ready)
            controls.verify()
        for job in plan["jobs"]:
            launch(directory, job, "fit")
            launch(directory, job, "predict")
        verify_inputs(directory, plan)
        result = summarize(directory, plan)
        result["capacity_total_wall_seconds_including_loading_and_exports"] = time.perf_counter() - started
        result["output_sha256"] = file_hashes(directory)
        write_json(directory / "checks.json", result)
        return result
    except BaseException as error:
        write_json(directory / "incomplete.json", {"status": "incomplete", "error_type": type(error).__name__,
                   "message": str(error), "automatic_retry": False, "formal_timing_eligible": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", nargs="?")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host-ready", action="store_true", help="Operator confirms AC power, low-power mode off, no heavy background work and an awake run")
    parser.add_argument("--worker", choices=["fit", "predict"], help=argparse.SUPPRESS)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.job or args.data_directory or args.host_ready:
            parser.error("Worker requires only phase, job and output directory")
        worker(args.output_dir, args.job, args.worker)
    else:
        if not args.data_directory or args.job:
            parser.error("Capacity check requires prepared data_directory")
        result = run_capacity(args.data_directory, args.output_dir, args.host_ready)
        print(json.dumps({key: result[key] for key in ["status", "scope", "cell", "fit_interval", "score_interval",
              "method_count", "configuration_count", "learned_fit_jobs", "component_fitting_phases",
              "prediction_jobs", "common_target_count", "latency_origin_count",
              "deterministic_repeat_predictions_exact", "all_reload_predictions_exact", "formal_timing_eligible"]}, indent=2))


if __name__ == "__main__":
    main()
