"""Run one declared development stage at a time, retaining every attempt.

Prepare only records the fixed workload. Fitting needs an explicit stage and
--execute-development. Confirmation/final forecasting is not implemented here.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from tools.development_store import (begin_attempt, complete_attempt, load_success,
                                     run_lock, sha256, write_json_atomic)
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = "421a4a6742cec01bde5a756927db9cca7fcf76efb683e040e3b0f2e4f2fe5911"
DEVELOPMENT_END = 6132


def read_config():
    import yaml
    from forecasting.development_plan import validate_plan
    if sha256(ROOT / "run_config.yaml") != CONFIG_SHA256:
        raise ValueError("Review the development runner after changing the configuration")
    config = yaml.safe_load((ROOT / "run_config.yaml").read_text())
    validate_plan(config)
    return config


def environment_identity():
    """Stable restart identity; battery levels, PIDs and timestamps may vary."""
    from tools.check_cpu_controls import machine_information
    machine = machine_information()
    names = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch",
             "psutil", "threadpoolctl", "pyyaml")
    stable = {key: machine[key] for key in ("system", "machine", "python", "executable",
              "physical_cpus", "logical_cpus", "memory_bytes", "macos")}
    stable["cpu_model"] = machine.get("cpu_model", {}).get("stdout")
    stable["hostname"] = platform.node()
    stable["packages"] = {name: importlib.metadata.version(name) for name in names}
    return stable, machine


def startup_check():
    if any(os.environ.get(key) != value for key, value in CPU_ENV.items()):
        raise RuntimeError("Launch through tools/run_cpu.py before numerical imports")


def all_jobs(config):
    from forecasting.development_plan import STAGES, build_jobs
    return [job for stage in STAGES for job in build_jobs(config, stage)]


def targets_for(data, job):
    import numpy as np
    from preprocessing.windows import eligible_targets
    mask = eligible_targets(data["traffic"], data["observed"])
    start, end = job["score_start"], job["score_end"]
    return np.flatnonzero(mask[start:end]) + start


def load_series(directory, plan, cell):
    import numpy as np
    import pandas as pd
    entry = plan["series"][cell]
    path = directory / entry["path"]
    if path.is_symlink() or sha256(path) != entry["sha256"]:
        raise ValueError("Prepared development series changed")
    with np.load(path, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    if set(data) != {"traffic", "traffic_observed", "observed", "timestamps"}:
        raise ValueError("Unexpected development arrays")
    if any(array.ndim != 1 or len(array) != DEVELOPMENT_END for array in data.values()):
        raise ValueError("Worker arrays must stop at the development boundary")
    if not np.isin(data["observed"], [0, 1]).all():
        raise ValueError("Invalid original observation mask")
    data["observed"] = data["observed"].astype(bool)
    flags = data["observed"]
    if not np.array_equal(np.isfinite(data["traffic_observed"]), flags):
        raise ValueError("Genuine observations differ from the saved mask")
    np.testing.assert_array_equal(data["traffic"][flags], data["traffic_observed"][flags])
    data["timestamps"] = pd.DatetimeIndex(data["timestamps"])
    clock = data["timestamps"]
    if (clock.tz is not None or clock.hasnans or not clock.equals(clock.floor("h"))
            or not np.all(np.diff(clock.asi8) == 3_600_000_000_000)):
        raise ValueError("Development timestamps must form the complete hourly calendar")
    return data


def prepare(data_directory, directory):
    """Validate the prepared bundle and save only the nine development prefixes."""
    import numpy as np
    from forecasting.development_plan import STAGES, TUNING_CELLS, FOLDS, build_jobs
    from preprocessing.windows import eligible_targets
    from tools.check_prepared_data import load_validated_data
    from tools.run_development_pilot import code_hashes
    config = read_config()
    if directory.exists():
        raise ValueError("Use a new directory for a new development plan")
    hourly, report = load_validated_data(data_directory)
    if report["tuning_cells"] != list(TUNING_CELLS):
        raise ValueError("Prepared tuning cells differ from the approved panel")
    directory.mkdir(parents=True)
    (directory / "series").mkdir()
    series = {}
    for cell in TUNING_CELLS:
        group = hourly.loc[hourly.cell == cell].iloc[:DEVELOPMENT_END]
        if len(group) != DEVELOPMENT_END:
            raise ValueError("Missing development calendar")
        values, flags = group.traffic.to_numpy(), group.observed.to_numpy(dtype=bool)
        mask = eligible_targets(values, flags)
        cohorts = []
        for start, end in FOLDS:
            targets = np.flatnonzero(mask[start:end]) + start
            counts = {"fit": int(mask[:start].sum()), "inner_fit": int(mask[:start-336].sum()),
                      "inner_tail": int(mask[start-336:start].sum()), "score": len(targets)}
            if min(counts.values()) < 1:
                raise ValueError("Resolve an unscoreable cell/fold before fitting any candidate")
            cohorts.append({"targets": targets.tolist(), "counts": counts})
        path = directory / "series" / f"{cell}.npz"
        np.savez(path, traffic=values, traffic_observed=group.traffic_observed.to_numpy(),
                 observed=flags, timestamps=group.ts.to_numpy())
        series[cell] = {"path": f"series/{cell}.npz", "sha256": sha256(path), "folds": cohorts}
    identity, machine = environment_identity()
    plan = {"schema_version": 1, "scope": "declared development search only",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": CONFIG_SHA256, "code_sha256": code_hashes(),
            "source_sha256": report["source_sha256"],
            "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
            "series": series, "jobs": all_jobs(config),
            "stage_counts": {stage: len(build_jobs(config, stage)) for stage in STAGES},
            "environment_identity": identity, "machine_at_preparation": machine,
            "cpu_startup_profile": CPU_ENV, "holdout_forecasts_enabled": False,
            "formal_comparative_timing": False,
            "timing_scope": "search operations and consumed attempts; cached residual fits exclude ridge"}
    write_json_atomic(directory / "plan.json", plan)
    write_json_atomic(directory / "plan-checksum.json", {"sha256": sha256(directory / "plan.json")})
    return plan


def verify_plan(directory):
    from forecasting.development_plan import STAGES, TUNING_CELLS, build_jobs
    from tools.run_development_pilot import code_hashes
    config = read_config()
    plan = json.loads((directory / "plan.json").read_text())
    checksum = json.loads((directory / "plan-checksum.json").read_text())["sha256"]
    if sha256(directory / "plan.json") != checksum:
        raise ValueError("Run plan changed; do not mix experiments")
    if (plan["schema_version"] != 1 or plan["config_sha256"] != CONFIG_SHA256
            or plan["code_sha256"] != code_hashes() or plan["jobs"] != all_jobs(config)
            or set(plan["series"]) != set(TUNING_CELLS)
            or plan["source_sha256"] != config["data"]["raw_sha256"]
            or plan["prepared_checklist_sha256"] != sha256(ROOT / "docs/data-bundle-checksums.json")
            or plan["stage_counts"] != {s: len(build_jobs(config, s)) for s in STAGES}
            or plan["cpu_startup_profile"] != CPU_ENV or plan["holdout_forecasts_enabled"] is not False):
        raise ValueError("Run plan differs from the fixed code, data or development workload")
    identity, _ = environment_identity()
    if identity != plan["environment_identity"]:
        raise ValueError("Resume requires the same machine, interpreter and package versions")
    expected = {f"{cell}.npz" for cell in TUNING_CELLS}
    if {p.name for p in (directory / "series").iterdir()} != expected:
        raise ValueError("Unexpected development series files")
    for cell in TUNING_CELLS:
        if plan["series"][cell]["path"] != f"series/{cell}.npz":
            raise ValueError("Unexpected series path")
        data = load_series(directory, plan, cell)
        for job in build_jobs(config, "ridge"):
            if job["cell"] == cell and job["parameters"]["alpha"] == 0.1:
                actual = targets_for(data, job).tolist()
                if actual != plan["series"][cell]["folds"][job["fold_index"]]["targets"]:
                    raise ValueError("Shared target cohort changed")
    known = {job["id"] for job in plan["jobs"]}
    if (directory / "jobs").exists() and any(p.name not in known for p in (directory / "jobs").iterdir()):
        raise ValueError("Unexpected job directory; never silently ignore extra attempts")
    return plan, config, checksum


def verify_current_identity(directory, plan, checksum):
    """Reject code/input-plan drift before sealing the last completed job too."""
    from tools.run_development_pilot import code_hashes
    identity, _ = environment_identity()
    if (sha256(directory / "plan.json") != checksum or sha256(ROOT / "run_config.yaml") != CONFIG_SHA256
            or code_hashes() != plan["code_sha256"] or identity != plan["environment_identity"]):
        raise ValueError("Code, plan or environment changed while the stage was running")


def validate_training(result, job, config):
    """Retain explicit fit counts, effective settings and the stopping budget."""
    settings, training = result["settings"], result["training"]
    expected_counts = {"baselines": {}, "ridge": {"ridge": 1},
                       "direct": {"direct_lightgbm": 1}, "hybrid": {"ridge": 0, "correction": 1},
                       "lstm": {"inner_selection": 1, "full_prefix_refit": 1}}
    if (settings["fit_end"] != job["fit_end"] or settings["lookback"] != 168
            or settings["development_job"] != {k: job[k] for k in ("id", "stage", "cell", "fold_index", "candidate_id")}
            or training["fit_interval"] != [0, job["fit_end"]]
            or training["component_fit_counts"] != expected_counts[job["stage"]]):
        raise ValueError("Saved fitted settings or phase counts differ from the job")
    seconds = result["fit_seconds"]
    if isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Measured search fit duration must be positive and finite")
    if job["stage"] == "ridge" and settings["alpha"] != job["parameters"]["alpha"]:
        raise ValueError("Ridge alpha differs from the planned candidate")
    if job["stage"] in ("direct", "hybrid"):
        if (settings["requested_trees"] != job["parameters"]["trees"]
                or settings["leaf_cap"] != job["parameters"]["leaves"]
                or not 1 <= training["actual_trees"] <= settings["requested_trees"]):
            raise ValueError("Tree capacity differs from its declared fitting budget")
    if job["stage"] == "lstm":
        spec = config["models"]["lstm"]
        if any(settings[name] != value for name, value in {**job["parameters"], "seed": job["seed"], "batch_size": spec["batch_size"]}.items()):
            raise ValueError("LSTM architecture/seed differs from the planned job")
        selected, ran = training["selected_epochs"], training["selection_epochs_run"]
        if (not 1 <= selected <= ran <= spec["maximum_epochs"] or settings["selected_epochs"] != selected
                or len(training["stopping_history"]) != ran or len(training["refit_history"]) != selected
                or training["effective_budgets"] != {"maximum_epochs": spec["maximum_epochs"],
                    "patience_epochs": spec["patience_epochs"], "inner_stopping_tail_hours": spec["inner_stopping_tail_hours"]}):
            raise ValueError("LSTM epochs or effective budget are inconsistent")
        phase = training["phase_timings"]
        durations = [phase["selection_seconds"], phase["refit_seconds"]]
        if any(not math.isfinite(v) or v <= 0 for v in durations) or sum(durations) > seconds:
            raise ValueError("LSTM phases must fit inside the complete measured duration")


def expected_scores(config, job):
    from forecasting.development_plan import score_candidates
    candidates = score_candidates(config, job["stage"])
    if job["stage"] == "hybrid":
        return [c for c in candidates if c["parameters"].get("trees") == job["parameters"]["trees"]
                and c["parameters"].get("leaves") == job["parameters"]["leaves"]]
    return [c for c in candidates if c["candidate_id"] == job["candidate_id"]]


def validate_result(attempt, job, plan, checksum, config):
    """Recompute stored metrics before a job becomes eligible for selection."""
    import numpy as np
    import pandas as pd
    from tools.pilot_results import score_predictions
    result = json.loads((attempt / "result.json").read_text())
    if result["job"] != job or result["plan_sha256"] != checksum or result["status"] != "passed":
        raise ValueError("Worker result has the wrong identity or is incomplete")
    validate_training(result, job, config)
    if result["reload_predictions_exact"] is not True or result["fitted_state_unchanged"] is not True:
        raise ValueError("Saved model/prediction identity check failed")
    for name in ("controls_before", "controls_after"):
        if result[name].get("runtime_library_controls_verified") is not True:
            raise ValueError("Worker library controls are not verified")
    data = load_series(attempt.parents[2], plan, job["cell"])
    targets = targets_for(data, job)
    choices = expected_scores(config, job)
    if [record["candidate_id"] for record in result["records"]] != [c["candidate_id"] for c in choices]:
        raise ValueError("Unexpected or missing worker scoring candidates")
    if result["formal_timing_eligible"] is not False:
        raise ValueError("Search timing cannot claim formal comparative eligibility")
    components = scaler = zero_reference = None
    if job["stage"] == "hybrid":
        from forecasting.data import TrafficScaler
        from models.hybrid import combine_predictions
        directory = attempt.parents[2]
        picked = json.loads((directory / "selections/ridge.json").read_text())
        chosen = picked["selection"]["selected"]
        rid = f"{chosen['candidate_id']}--{job['cell']}--f{job['fold_index']+1}"
        dependency = load_success(directory, rid, checksum)
        if dependency is None:
            raise ValueError("Missing selected ridge dependency")
        path, _ = dependency
        expected_dependency = {"job_id": rid, "attempt": str(path.relative_to(directory)),
            "completion_sha256": sha256(path / "complete.json"),
            "ridge_selection_sha256": sha256(directory / "selections/ridge.json")}
        if result["dependency"] != expected_dependency or result["settings"]["alpha"] != chosen["parameters"]["alpha"]:
            raise ValueError("Residual fit does not reference the globally selected ridge")
        with np.load(attempt / "components.npz", allow_pickle=False) as saved:
            components = {name: saved[name] for name in saved.files}
        if set(components) != {"targets", "ridge_z", "correction_z"}:
            raise ValueError("Missing original hybrid component outputs")
        np.testing.assert_array_equal(components["targets"], targets)
        payload = json.loads((attempt / "model/model.json").read_text())
        ridge_payload = json.loads((path / "model/model.json").read_text())
        if payload["scaler"] != ridge_payload["scaler"]:
            raise ValueError("Residual and selected ridge scaling differ")
        scaler = TrafficScaler(**payload["scaler"])
        zero_reference = pd.read_csv(path / "predictions" / f"{chosen['candidate_id']}.csv", float_precision="round_trip")
        np.testing.assert_array_equal(zero_reference.target_index, targets)
        np.testing.assert_array_equal(scaler.inverse(components["ridge_z"]), zero_reference.prediction_raw)
        np.testing.assert_array_equal(np.maximum(0, scaler.inverse(components["ridge_z"])), zero_reference.prediction)
    for candidate, record in zip(choices, result["records"]):
        cid = candidate["candidate_id"]
        frame = pd.read_csv(attempt / "predictions" / f"{cid}.csv", float_precision="round_trip")
        np.testing.assert_array_equal(frame.target_index, targets)
        np.testing.assert_array_equal(frame.actual, data["traffic_observed"][targets])
        if components is not None:
            component_raw = scaler.inverse(combine_predictions(components["ridge_z"], components["correction_z"], candidate["parameters"]["weight"]))
            np.testing.assert_array_equal(component_raw, frame.prediction_raw)
            np.testing.assert_array_equal(np.maximum(0, component_raw), frame.prediction)
        np.testing.assert_array_equal(pd.to_datetime(frame.timestamp).to_numpy(), data["timestamps"][targets].to_numpy())
        if not frame.cell.eq(job["cell"]).all() or not frame.method.eq(job["method"]).all():
            raise ValueError("Forecast identity differs from the job")
        if job["seed"] is None:
            if not frame.seed.isna().all():
                raise ValueError("Unexpected deterministic forecast seed")
        elif not frame.seed.eq(job["seed"]).all():
            raise ValueError("Wrong LSTM forecast seed")
        evaluation = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
            job["fit_end"], job["score_start"], job["score_end"], targets,
            frame.prediction_raw.to_numpy(), frame.prediction.to_numpy())
        if evaluation != result["evaluations"][cid]:
            raise ValueError("Saved scores do not reproduce from forecasts")
        expected = {"candidate_id": cid, "cell": job["cell"], "fold_index": job["fold_index"],
                    "seed": job["seed"], "status": "passed", "scores": evaluation["scores"],
                    "mase_scale": evaluation["weekly_scale"]}
        if job["stage"] == "lstm":
            expected["selected_epochs"] = result["training"]["selected_epochs"]
        if record != expected:
            raise ValueError("Selection record differs from the verified predictions")
    return result


def collect_stage(directory, plan, config, checksum, stage):
    from forecasting.development_plan import build_jobs
    records, evidence = [], {}
    for job in build_jobs(config, stage):
        success = load_success(directory, job["id"], checksum)
        if success is None:
            raise ValueError(f"Stage {stage} is incomplete: {job['id']}")
        attempt, marker = success
        if marker["job"] != job:
            raise ValueError("Stored job differs from the declared plan")
        result = validate_result(attempt, job, plan, checksum, config)
        records.extend(result["records"])
        evidence[job["id"]] = {"attempt": str(attempt.relative_to(directory)),
                                "completion_sha256": sha256(attempt / "complete.json")}
    return records, evidence


def selection(directory, plan, config, checksum, stage):
    from forecasting.development_selection import select_stage
    records, evidence = collect_stage(directory, plan, config, checksum, stage)
    if stage == "hybrid":
        ridge = selection(directory, plan, config, checksum, "ridge")
        chosen = ridge["selection"]["selected"]["candidate_id"]
        ridge_records, _ = collect_stage(directory, plan, config, checksum, "ridge")
        records += [{**r, "candidate_id": "hybrid-zero"} for r in ridge_records if r["candidate_id"] == chosen]
        evidence["ridge_selection_sha256"] = sha256(directory / "selections/ridge.json")
    value = {"stage": stage, "plan_sha256": checksum, "evidence": evidence,
             "selection": select_stage(records, config, stage), "primary_comparator": None,
             "confirmation_or_final_evaluated": False}
    path = directory / "selections" / f"{stage}.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Saved selection differs from complete verified evidence")
    else:
        path.parent.mkdir(exist_ok=True)
        write_json_atomic(path, value)
    return value


def ridge_dependency(directory, plan, config, checksum, job):
    """Bind the correction to the globally chosen ridge, never a local winner."""
    from forecasting.development_plan import build_jobs
    from forecasting.development_selection import select_stage
    # Check all sealed score records without reparsing 135 forecast CSVs per fit.
    # The stage parent has already performed that full arithmetic validation.
    records, evidence = [], {}
    for ridge_job in build_jobs(config, "ridge"):
        success = load_success(directory, ridge_job["id"], checksum)
        if success is None:
            raise ValueError("The complete ridge stage must precede residual fitting")
        path, marker = success
        report = json.loads((path / "result.json").read_text())
        if marker["job"] != ridge_job or report["job"] != ridge_job or report["plan_sha256"] != checksum:
            raise ValueError("Ridge selection evidence has the wrong identity")
        records.extend(report["records"])
        evidence[ridge_job["id"]] = {"attempt": str(path.relative_to(directory)),
                                    "completion_sha256": sha256(path / "complete.json")}
    picked = {"stage": "ridge", "plan_sha256": checksum, "evidence": evidence,
              "selection": select_stage(records, config, "ridge"), "primary_comparator": None,
              "confirmation_or_final_evaluated": False}
    if json.loads((directory / "selections/ridge.json").read_text()) != picked:
        raise ValueError("Global ridge selection changed or lacks complete evidence")
    cid = picked["selection"]["selected"]["candidate_id"]
    dependency = next(j for j in build_jobs(config, "ridge") if j["candidate_id"] == cid
                      and j["cell"] == job["cell"] and j["fold_index"] == job["fold_index"])
    attempt, _ = load_success(directory, dependency["id"], checksum)
    return attempt, {"job_id": dependency["id"], "attempt": str(attempt.relative_to(directory)),
                     "completion_sha256": sha256(attempt / "complete.json"),
                     "ridge_selection_sha256": sha256(directory / "selections/ridge.json")}


def worker(directory, job_id, attempt_path):
    import numpy as np
    from forecasting.development_models import (fit_candidate, predict_candidate,
        predict_hybrid_components, finalize_hybrid_components)
    from forecasting.pilot_models import load_model, save_model, model_fingerprint
    from tools.cpu_controls import single_thread_cpu
    from tools.pilot_measurements import elapsed_call, memory_snapshot
    from tools.pilot_results import score_predictions, save_predictions
    plan, config, checksum = verify_plan(directory)
    job = next((j for j in plan["jobs"] if j["id"] == job_id), None)
    if job is None or attempt_path.parent != directory / "jobs" / job_id:
        raise ValueError("Worker must use a planned job and its assigned attempt")
    metadata = json.loads((attempt_path / "attempt.json").read_text())
    if (metadata.get("job") != job or metadata.get("plan_sha256") != checksum
            or metadata.get("schema_version") != 1
            or attempt_path.name != f"attempt-{metadata.get('attempt_number', 0):04d}"
            or metadata.get("attempt_number", 0) < 1
            or any((attempt_path / name).exists() for name in ("result.json", "complete.json", "failure.json"))):
        raise ValueError("Worker requires a fresh assigned attempt with matching provenance")
    data = load_series(directory, plan, job["cell"])
    targets = targets_for(data, job)
    dependency_path = dependency_info = ridge_bundle = None
    if job["stage"] == "hybrid":
        dependency_path, dependency_info = ridge_dependency(directory, plan, config, checksum, job)
    with single_thread_cpu() as controls:
        before = controls.verify()
        if dependency_path is not None:
            ridge_bundle = load_model(dependency_path / "model")
        fit_end = job["fit_end"]
        bundle, seconds = elapsed_call(lambda: fit_candidate(job,
            data["traffic"][:fit_end], data["observed"][:fit_end], data["timestamps"][:fit_end],
            config, ridge_bundle=ridge_bundle,
            progress=lambda text: print(text, flush=True)))
        fingerprint = model_fingerprint(bundle)
        (raw, final), prediction_seconds = elapsed_call(lambda: predict_candidate(bundle,
            data["traffic"], data["observed"], data["timestamps"], targets))
        save_model(bundle, attempt_path / "model")
        loaded = load_model(attempt_path / "model")
        if model_fingerprint(loaded) != fingerprint:
            raise AssertionError("Saved/reloaded fitted state differs from the original")
        reload_raw, reload_final = predict_candidate(loaded, data["traffic"], data["observed"], data["timestamps"], targets)
        np.testing.assert_array_equal(raw, reload_raw)
        np.testing.assert_array_equal(final, reload_final)
        variants = {}
        if job["stage"] == "hybrid":
            ridge_z, correction_z = predict_hybrid_components(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
            np.savez(attempt_path / "components.npz", targets=targets, ridge_z=ridge_z, correction_z=correction_z)
            zero = finalize_hybrid_components(bundle, ridge_z, correction_z, 0, config)
            reference = predict_candidate(ridge_bundle, data["traffic"], data["observed"], data["timestamps"], targets)
            for actual, expected in zip(zero, reference):
                np.testing.assert_array_equal(actual, expected)
            for candidate in expected_scores(config, job):
                variants[candidate["candidate_id"]] = finalize_hybrid_components(bundle,
                    ridge_z, correction_z, candidate["parameters"]["weight"], config)
            np.testing.assert_array_equal(variants[job["candidate_id"] + "-w1"][0], raw)
        else:
            variants[job["candidate_id"]] = (raw, final)
        records, evaluations = [], {}
        for cid, (prediction_raw, prediction) in variants.items():
            evaluation = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
                fit_end, job["score_start"], job["score_end"], targets, prediction_raw, prediction)
            if evaluation["mase_status"] != "available":
                raise ValueError("Unavailable MASE requires a declared cohort review, not pruning")
            save_predictions(attempt_path / "predictions" / f"{cid}.csv", job["cell"], job["method"],
                job["seed"], targets, data["timestamps"][targets], data["traffic_observed"][targets], prediction_raw, prediction)
            record = {"candidate_id": cid, "cell": job["cell"], "fold_index": job["fold_index"],
                      "seed": job["seed"], "status": "passed", "scores": evaluation["scores"],
                      "mase_scale": evaluation["weekly_scale"]}
            if job["stage"] == "lstm":
                record["selected_epochs"] = bundle.training_metadata["selected_epochs"]
            records.append(record)
            evaluations[cid] = evaluation
        if model_fingerprint(bundle) != fingerprint:
            raise AssertionError("Scoring changed the fitted model")
        after, memory = controls.verify(), memory_snapshot()
        result = {"status": "passed", "job": job, "plan_sha256": checksum,
                  "dependency": dependency_info, "settings": bundle.settings,
                  "training": bundle.training_metadata, "records": records, "evaluations": evaluations,
                  "fit_seconds": seconds, "initial_prediction_seconds": prediction_seconds,
                  "timing_scope": "search operation only; correction fit reuses ridge; not full hybrid timing",
                  "worker_lifetime_memory": memory, "model_state_sha256": fingerprint,
                  "reload_predictions_exact": True, "fitted_state_unchanged": True,
                  "controls_before": before, "controls_after": after, "formal_timing_eligible": False}
    # A cleanup failure must prevent publication of a passing worker result.
    write_json_atomic(attempt_path / "result.json", result)


def launch(directory, job, attempt, lock_fd):
    """Wait for the child even on interruption, so another run cannot overlap."""
    # Direct launch keeps the inherited lock in the numerical worker itself.
    arguments = [sys.executable, "-m", "tools.run_development_experiments",
                 "--worker", job["id"], "--attempt", str(attempt), "--output-dir", str(directory)]
    with (attempt / "worker.log").open("x") as stream:
        child = subprocess.Popen(arguments, cwd=ROOT, env={**os.environ, **CPU_ENV},
                                 stdout=stream, stderr=subprocess.STDOUT,
                                 pass_fds=(lock_fd,))
        try:
            return child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


@contextmanager
def stage_session(directory, stage, checksum):
    """Include failed/retried work in recorded parent-stage wall time."""
    sessions = directory / "sessions"
    sessions.mkdir(exist_ok=True)
    previous = sorted(sessions.iterdir())
    expected = [f"session-{number:04d}" for number in range(1, len(previous)+1)]
    if [path.name for path in previous] != expected or any(not path.is_dir() for path in previous):
        raise ValueError("Unexpected stage-session history")
    session = sessions / f"session-{len(previous)+1:04d}"
    session.mkdir()
    write_json_atomic(session / "started.json", {"stage": stage, "plan_sha256": checksum,
                      "started_at_utc": datetime.now(timezone.utc).isoformat()})
    started = time.perf_counter()
    outcome, error_type = "passed", None
    try:
        yield session
    except BaseException as error:
        outcome, error_type = "failed_or_interrupted", type(error).__name__
        raise
    finally:
        write_json_atomic(session / "finished.json", {"status": outcome, "error_type": error_type,
            "wall_seconds": time.perf_counter()-started,
            "scope": "parent stage wall time including all attempts, validation and selection; excludes imports, preparation and initial plan verification"})


def run_stage(directory, stage, *, resume=False, retry_failed=False, host_ready=False):
    from forecasting.development_plan import build_jobs
    from tools.check_cpu_controls import machine_information
    if not host_ready:
        raise ValueError("Confirm the declared host conditions with --host-ready before fitting")
    with run_lock(directory) as lock_fd:
        plan, config, checksum = verify_plan(directory)
        with stage_session(directory, stage, checksum):
            if stage == "hybrid":
                selection(directory, plan, config, checksum, "ridge")
            jobs = build_jobs(config, stage)
            for index, job in enumerate(jobs, 1):
                attempt, cached = begin_attempt(directory, job, checksum, resume=resume, retry_failed=retry_failed)
                if cached:
                    validate_result(attempt, job, plan, checksum, config)
                    print(f"Reused verified job {index}/{len(jobs)}: {job['id']}", flush=True)
                    continue
                started = time.perf_counter()
                outcome = "incomplete"
                try:
                    write_json_atomic(attempt / "host.json", {"host_conditions_confirmed_by_operator": True,
                                      "machine_before": machine_information()})
                    print(f"Development {stage} {index}/{len(jobs)}: {job['id']}", flush=True)
                    returncode = launch(directory, job, attempt, lock_fd)
                    if returncode != 0:
                        raise RuntimeError(f"Worker failed ({returncode}); inspect {attempt / 'worker.log'}")
                    validate_result(attempt, job, plan, checksum, config)
                    verify_current_identity(directory, plan, checksum)
                    outcome = "passed"
                except BaseException as error:
                    write_json_atomic(attempt / "failure.json", {"type": type(error).__name__, "message": str(error),
                                      "automatic_retry": False})
                    raise
                finally:
                    write_json_atomic(attempt / "elapsed.json", {"outcome": outcome,
                        "wall_seconds": time.perf_counter() - started,
                        "scope": "consumed attempt including worker startup, prediction, exports and validation"})
                complete_attempt(attempt, job, checksum)
            selected = selection(directory, plan, config, checksum, stage)
            verify_current_identity(directory, plan, checksum)
            return selected


def summarize(directory):
    from forecasting.development_plan import STAGES
    with run_lock(directory):
        plan, config, checksum = verify_plan(directory)
        selections = {stage: selection(directory, plan, config, checksum, stage) for stage in STAGES}
        attempts = []
        for path in sorted((directory / "jobs").glob("*/attempt-*")):
            timing = path / "elapsed.json"
            attempts.append({"attempt": str(path.relative_to(directory)),
                             "timing": json.loads(timing.read_text()) if timing.exists() else None})
        unknown = sum(x["timing"] is None for x in attempts)
        sessions = []
        for path in sorted((directory / "sessions").glob("session-*")):
            start = json.loads((path / "started.json").read_text())
            if start["plan_sha256"] != checksum or start["stage"] not in STAGES:
                raise ValueError("Stage-session identity differs from this run")
            finish = path / "finished.json"
            timing = json.loads(finish.read_text()) if finish.exists() else None
            if timing is not None and (not math.isfinite(timing["wall_seconds"]) or timing["wall_seconds"] < 0):
                raise ValueError("Invalid recorded stage duration")
            sessions.append({"session": path.name, "stage": start["stage"], "timing": timing})
        unknown_sessions = sum(s["timing"] is None for s in sessions)
        summary = {"status": "passed", "scope": "complete declared development search",
                   "plan_sha256": checksum, "selections": selections,
                   "attempts_including_failures": attempts,
                   "recorded_attempt_wall_seconds": sum(x["timing"]["wall_seconds"] for x in attempts if x["timing"]),
                   "attempts_with_unavailable_elapsed_time": unknown,
                   "stage_sessions_including_failures": sessions,
                   "recorded_stage_wall_seconds": sum(s["timing"]["wall_seconds"] for s in sessions if s["timing"]),
                   "sessions_with_unavailable_elapsed_time": unknown_sessions,
                   "recorded_stage_timing_complete": bool(sessions) and unknown == 0 and unknown_sessions == 0,
                   "timing_scope": "recorded stage commands; imports, preparation and initial plan checks excluded",
                   "formal_timing_eligible": False, "primary_comparator": None,
                   "holdout_forecasts_evaluated": False,
                   "next_decision": "Review development evidence before confirmation or any finite adaptation"}
        path = directory / "development-summary.json"
        if path.exists():
            if json.loads(path.read_text()) != summary:
                raise ValueError("Existing complete summary differs; preserve evidence")
        else:
            write_json_atomic(path, summary)
        return summary


def main():
    from forecasting.development_plan import STAGES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", nargs="?")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("prepare", *STAGES, "summarize"), default="prepare")
    parser.add_argument("--execute-development", action="store_true")
    parser.add_argument("--host-ready", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    startup_check()
    directory = Path(args.output_dir).resolve()
    if args.worker:
        if not args.attempt or args.data_directory or args.execute_development or args.host_ready or args.resume or args.retry_failed or args.stage != "prepare":
            parser.error("Internal worker accepts only its job, attempt and output directory")
        worker(directory, args.worker, Path(args.attempt).resolve())
        return
    if args.attempt or (args.retry_failed and not args.resume):
        parser.error("An explicit failed-job retry also requires --resume")
    if args.stage == "prepare":
        if not args.data_directory or args.execute_development or args.resume or args.retry_failed or args.host_ready:
            parser.error("Prepare takes only the prepared data directory and new output directory")
        result = prepare(args.data_directory, directory)
        print(json.dumps({"status": "prepared", "stage_counts": result["stage_counts"], "fitting_performed": False}, indent=2))
    elif args.stage == "summarize":
        if args.data_directory or args.execute_development or args.host_ready or args.resume or args.retry_failed:
            parser.error("Summarize takes only the output directory")
        result = summarize(directory)
        print(json.dumps({"status": result["status"], "scope": result["scope"], "primary_comparator": None}, indent=2))
    else:
        if args.data_directory or not args.execute_development:
            parser.error("Fitting stages require --execute-development and an existing prepared output directory")
        result = run_stage(directory, args.stage, resume=args.resume, retry_failed=args.retry_failed, host_ready=args.host_ready)
        print(json.dumps({"status": "passed", "stage": args.stage, "selected": result["selection"]["selected"]}, indent=2))


if __name__ == "__main__":
    main()
