"""Run one fixed development-fold pilot in sequential, fresh CPU processes.

This command checks the complete workflow. It cannot tune models, choose cells,
change the scoring period, or enable confirmation/final forecasting.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.cpu_controls import single_thread_cpu
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = "421a4a6742cec01bde5a756927db9cca7fcf76efb683e040e3b0f2e4f2fe5911"
METHODS = ("persistence", "daily_seasonal_naive", "weekly_seasonal_naive",
           "ridge", "direct_lightgbm", "ridge_lightgbm_hybrid", "lstm")
BASELINES = METHODS[:3]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def code_hashes():
    """Include all execution modules, not only the new runner."""
    return {str(p.relative_to(ROOT)): sha256(p)
            for directory in ("tools", "models", "forecasting", "evaluation", "preprocessing")
            for p in sorted((ROOT / directory).glob("*.py"))}


def read_config():
    import yaml
    if sha256(ROOT / "run_config.yaml") != CONFIG_SHA256:
        raise ValueError("Review this bounded pilot after changing run_config.yaml")
    return yaml.safe_load((ROOT / "run_config.yaml").read_text())


def pilot_jobs():
    jobs = []
    for method in METHODS:
        repeats = range(1, 2) if method in BASELINES else range(1, 4)
        for repeat in repeats:
            seed = 2025 + repeat if method == "lstm" else None
            jobs.append({"id": f"{method}-{repeat}", "method": method,
                         "repeat": repeat, "seed": seed})
    return jobs


def verify_inputs(directory, plan):
    if plan["code_sha256"] != code_hashes() or plan["config_sha256"] != CONFIG_SHA256:
        raise ValueError("Code/configuration changed since this pilot was prepared")
    if sha256(directory / "series.npz") != plan["series_sha256"]:
        raise ValueError("Pilot input bytes changed")
    if plan["jobs"] != pilot_jobs() or plan["cell"] != "Cell_003791":
        raise ValueError("Unexpected pilot workload")
    if plan["fit_interval"] != [0, 4116] or plan["score_interval"] != [4116, 4788]:
        raise ValueError("Pilot must use the fixed first development fold")


def verify_series(data, plan):
    """Cross-check the recorded workload against its physical worker arrays."""
    import numpy as np
    from preprocessing.windows import eligible_targets
    from tools.pilot_measurements import select_origins
    expected = np.flatnonzero(eligible_targets(data["traffic"], data["observed"])[4116:4788]) + 4116
    np.testing.assert_array_equal(data["targets"], expected)
    np.testing.assert_array_equal(data["targets"], plan["targets"])
    np.testing.assert_array_equal(data["origins"], select_origins(expected))
    np.testing.assert_array_equal(data["origins"], plan["origins"])
    if len(data["traffic"]) != 4788 or len(expected) != 672:
        raise ValueError("Pilot data exceed or do not cover the fixed development workload")


def load_series(directory):
    import numpy as np
    import pandas as pd
    with np.load(directory / "series.npz", allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    data["timestamps"] = pd.DatetimeIndex(data["timestamps"])
    return data


def file_hashes(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def fit_worker(directory, plan, job, config, controls):
    import numpy as np
    from forecasting.pilot_models import fit_model, predict_model, save_model, model_fingerprint
    from tools.pilot_measurements import elapsed_call, memory_snapshot

    data = load_series(directory)
    verify_series(data, plan)
    output = directory / "jobs" / job["id"]
    output.mkdir(parents=True, exist_ok=False)
    before = controls.verify()
    def fit():
        return fit_model(job["method"], data["traffic"][:4116], data["observed"][:4116],
                         data["timestamps"][:4116], 4116, config, seed=job["seed"])
    if job["method"] in BASELINES:
        bundle, seconds, memory = fit(), None, None
    else:
        bundle, seconds = elapsed_call(fit)
        memory = memory_snapshot()  # Capture before export or prediction audits.
    after = controls.verify()
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
              "fit_memory": memory, "fit_kind": "not_applicable" if seconds is None else
              ("development_epoch_selection_and_refit" if job["method"] == "lstm" else "complete_fit"),
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
                               4116, 4116, 4788, data["targets"], measured["raw"], measured["final"])
    if scored["mase_status"] != "available":
        raise ValueError("The fixed pilot cell requires its validated positive weekly scale")
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
        raise ValueError("Unknown pilot job")
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
        raise ValueError("Pilot cell differs from the verified tuning panel")
    group = hourly.loc[hourly.cell == cell].iloc[:4788].reset_index(drop=True)
    targets = np.flatnonzero(group.eligible_target.to_numpy()[4116:4788]) + 4116
    if len(group) != 4788 or len(targets) != 672:
        raise ValueError("Pilot calendar/mask differs from the verified first fold")
    np.savez(directory / "series.npz", traffic=group.traffic.to_numpy(),
             traffic_observed=group.traffic_observed.to_numpy(), observed=group.observed.to_numpy(),
             timestamps=group.ts.to_numpy(), targets=targets, origins=select_origins(targets))
    plan = {"scope": "bounded development integration pilot; no tuning or model ranking",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "cell": cell, "fit_interval": [0, 4116], "score_interval": [4116, 4788],
            "targets": targets.tolist(), "origins": select_origins(targets).tolist(), "jobs": pilot_jobs(),
            "source_sha256": checked["source_sha256"], "config_sha256": CONFIG_SHA256,
            "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
            "series_sha256": sha256(directory / "series.npz"), "code_sha256": code_hashes(),
            "host_conditions_confirmed_by_operator": bool(host_ready),
            "machine_before": machine_information(),
            "versions": {name: metadata.version(name) for name in
                         ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch", "psutil", "threadpoolctl"]},
            "formal_timing_eligible": False, "holdout_forecasts_evaluated": False}
    write_json(directory / "plan.json", plan)
    return plan


def launch(directory, job, phase):
    arguments = [sys.executable, str(ROOT / "tools/run_cpu.py"), "-m", "tools.run_development_pilot",
                 "--worker", phase, "--job", job["id"], "--output-dir", str(directory)]
    log = directory / f"{job['id']}-{phase}.log"
    print(f"Pilot {phase}: {job['id']}", flush=True)
    with log.open("x") as stream:
        result = subprocess.run(arguments, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Pilot job failed; retained log: {log}")


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
            raise ValueError("A required pilot job is incomplete")
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
                                       4116, 4116, 4788, frame.target_index.to_numpy(),
                                       frame.prediction_raw.to_numpy(), frame.prediction.to_numpy())
        if recomputed != prediction["evaluation"]:
            raise AssertionError("Saved metrics do not reproduce from saved forecasts")
        np.testing.assert_array_equal(frame.target_index, plan["targets"])
        prediction_frames[job["id"]] = frame
        rows.append({"job": job, "fit_seconds": fit["fit_seconds"], "fit_kind": fit["fit_kind"],
                     "fit_memory": fit["fit_memory"], "artifact_bytes": fit["artifact_bytes"],
                     "prediction": prediction})
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
    for method in METHODS:
        items = [row for row in rows if row["job"]["method"] == method]
        times = [item["fit_seconds"] for item in items if item["fit_seconds"] is not None]
        summaries.append({"method": method,
                          "fit_seconds": {"observations": times, "median": median(times),
                                          "minimum": min(times), "maximum": max(times)} if times else None,
                          "representative_job": items[0]["job"]["id"],
                          "representative_artifact_bytes": items[0]["artifact_bytes"],
                          "scores": {key: float(np.mean([x["prediction"]["evaluation"]["scores"][key]
                                                        for x in items])) for key in ("MAE", "RMSE", "MASE")}})
    return {"status": "passed", "scope": plan["scope"], "cell": plan["cell"],
            "fit_interval": plan["fit_interval"], "score_interval": plan["score_interval"],
            "method_count": len(METHODS), "learned_fit_jobs": 12, "prediction_jobs": len(rows),
            "common_target_count": len(plan["targets"]), "latency_origin_count": len(plan["origins"]),
            "deterministic_repeat_predictions_exact": True, "all_reload_predictions_exact": True,
            "same_targets_and_actuals": True, "methods_in_declared_order": summaries, "jobs": rows,
            "machine_after": machine_information(), "formal_timing_eligible": False,
            "formal_timing_measured": False, "pilot_timing_measured": True,
            "holdout_forecasts_evaluated": False, "model_ranking_performed": False,
            "limitations": ["One cell and first candidates only; no inference about best model or full-panel accuracy.",
                            "LSTM development fit includes epoch selection and fresh refitting, not fixed-epoch deployment fit.",
                            "RSS is total-process high-water memory including dependencies, data and model loading.",
                            "Larger models, later prefixes and other cells need capacity checks before a full-search time estimate.",
                            "Machine snapshots and operator confirmation do not eliminate OS scheduling or thermal variability."]}


def run_pilot(data_directory, output_directory, host_ready=False):
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
        result["pilot_total_wall_seconds_including_loading_and_exports"] = time.perf_counter() - started
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
            parser.error("Pilot requires prepared data_directory")
        result = run_pilot(args.data_directory, args.output_dir, args.host_ready)
        print(json.dumps({key: result[key] for key in ["status", "scope", "cell", "fit_interval", "score_interval",
              "method_count", "learned_fit_jobs", "prediction_jobs", "common_target_count", "latency_origin_count",
              "deterministic_repeat_predictions_exact", "all_reload_predictions_exact", "formal_timing_eligible"]}, indent=2))


if __name__ == "__main__":
    main()
