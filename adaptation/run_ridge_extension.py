"""One bounded development extension; no confirmation/final data or fitting.

Prepare imports only the completed development run. Ridge and hybrid require
--execute-extension and --host-ready. Original evidence is read-only.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from adaptation.plan import (NEW_ALPHAS, PLAN_SHA256, SUMMARY_SHA256, build_jobs,
                             effective_config, scoring_candidates, select_ridge, select_hybrid)
from adaptation import historical
from tools import run_development_experiments as original
from tools.development_store import (begin_attempt, complete_attempt, load_success,
                                     run_lock, sha256, write_json_atomic)
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]
load_series = original.load_series
targets_for = original.targets_for
validate_training = original.validate_training


def read_json(path):
    sha256(path)  # Also reject symlinks and non-regular files.
    return json.loads(path.read_text())


def save_or_check(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing evidence differs; preserve {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, value)
    return value


def extension_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted((ROOT / "adaptation").rglob("*.py"))}


def source_identity(directory):
    if sha256(directory / "plan.json") != PLAN_SHA256:
        raise ValueError("Original development plan differs from the reviewed run")
    if sha256(directory / "development-summary.json") != SUMMARY_SHA256:
        raise ValueError("Original complete summary differs from the reviewed run")


def audit_original(directory):
    """Recompute all original scores/selections without creating locks or files."""
    from forecasting.development_plan import STAGES
    from forecasting.development_selection import select_stage
    from tools.development_store import _inventory, verify_inventory
    source_identity(directory)
    inventory = _inventory(directory, exclude=(".run.lock",))
    plan, config, checksum = historical.verify_original_plan(directory)
    summary = read_json(directory / "development-summary.json")
    all_records, selections = {}, {}
    for stage in STAGES:
        records, evidence = original.collect_stage(directory, plan, config, checksum, stage)
        all_records[stage] = records
        scored = list(records)
        if stage == "hybrid":
            cid = selections["ridge"]["selection"]["selected"]["candidate_id"]
            scored += [{**r, "candidate_id": "hybrid-zero"} for r in all_records["ridge"] if r["candidate_id"] == cid]
            evidence["ridge_selection_sha256"] = sha256(directory / "selections/ridge.json")
        selected = {"stage": stage, "plan_sha256": checksum, "evidence": evidence,
                    "selection": select_stage(scored, config, stage), "primary_comparator": None,
                    "confirmation_or_final_evaluated": False}
        if selected != read_json(directory / "selections" / f"{stage}.json"):
            raise ValueError("Original selection does not reproduce from its complete evidence")
        selections[stage] = selected
    if (summary["selections"] != selections or summary["plan_sha256"] != checksum
            or summary["status"] != "passed" or summary["holdout_forecasts_evaluated"] is not False):
        raise ValueError("Original summary differs from the reproduced selections")
    attempts = [{"attempt": str(p.relative_to(directory)),
                 "timing": read_json(p / "elapsed.json") if (p / "elapsed.json").exists() else None}
                for p in sorted((directory / "jobs").glob("*/attempt-*"))]
    sessions = []
    for p in sorted((directory / "sessions").glob("session-*")):
        start = read_json(p / "started.json")
        if start["plan_sha256"] != checksum or start["stage"] not in STAGES:
            raise ValueError("Original timing session identity differs")
        sessions.append({"session": p.name, "stage": start["stage"],
                         "timing": read_json(p / "finished.json") if (p / "finished.json").exists() else None})
    if attempts != summary["attempts_including_failures"] or sessions != summary["stage_sessions_including_failures"]:
        raise ValueError("Original timing records differ from the reviewed summary")
    # Pin every original artifact, including timing and unsuccessful attempts.
    verify_inventory(directory, inventory, exclude=(".run.lock",))
    return plan, config, all_records, inventory


def prepare(source, directory):
    from forecasting.development_plan import TUNING_CELLS
    from tools.run_development_pilot import code_hashes
    source, directory = source.resolve(), directory.resolve()
    if directory.exists() or source == directory or source in directory.parents or directory in source.parents:
        raise ValueError("Use a new output directory outside the original run")
    initial, config, records, inventory = audit_original(source)
    identity, machine = original.environment_identity()
    if identity != initial["environment_identity"]:
        raise ValueError("Preparation requires the original host, interpreter and packages")
    directory.mkdir(parents=True)
    (directory / "series").mkdir()
    for cell in TUNING_CELLS:
        relative = initial["series"][cell]["path"]
        shutil.copyfile(source / relative, directory / relative)
    shutil.copyfile(source / "plan.json", directory / "original-plan.json")
    shutil.copyfile(source / "development-summary.json", directory / "original-summary.json")
    write_json_atomic(directory / "original-records.json", records)
    write_json_atomic(directory / "original-inventory.json", inventory)
    plan = {"schema_version": 1, "scope": "single bounded ridge regularisation extension",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "original_directory": str(source), "original_plan_sha256": PLAN_SHA256,
            "original_summary_sha256": SUMMARY_SHA256,
            "original_records_sha256": sha256(directory / "original-records.json"),
            "original_inventory_sha256": sha256(directory / "original-inventory.json"),
            "original_code_sha256": initial["code_sha256"], "extension_code_sha256": extension_hashes(),
            "runtime_development_code_sha256": code_hashes(),
            "source_compatibility_sha256": historical.catalog_sha256(),
            "config_sha256": original.CONFIG_SHA256,
            "effective_config": effective_config(config),
            "series": initial["series"], "jobs": build_jobs(config, "ridge") + build_jobs(config, "hybrid"),
            "stage_counts": {"ridge": 54, "hybrid_if_new_alpha_wins": 162},
            "new_alphas": list(NEW_ALPHAS), "maximum_new_fits": 216,
            "stop_after_declared_extension": True, "holdout_forecasts_enabled": False,
            "formal_comparative_timing": False, "cpu_startup_profile": CPU_ENV,
            "environment_identity": identity, "machine_at_preparation": machine}
    write_json_atomic(directory / "plan.json", plan)
    write_json_atomic(directory / "plan-checksum.json", {"sha256": sha256(directory / "plan.json")})
    verify_plan(directory)
    verify_source_inventory(directory, plan)
    return plan


def verify_plan(directory):
    from forecasting.development_plan import TUNING_CELLS
    from tools.run_development_pilot import code_hashes
    config = original.read_config()
    plan = read_json(directory / "plan.json")
    checksum = read_json(directory / "plan-checksum.json")["sha256"]
    if sha256(directory / "plan.json") != checksum:
        raise ValueError("Extension plan changed")
    source = Path(plan["original_directory"])
    source_identity(source)
    if sha256(directory / "original-plan.json") != PLAN_SHA256 or sha256(directory / "original-summary.json") != SUMMARY_SHA256:
        raise ValueError("Copied original evidence changed")
    initial = read_json(directory / "original-plan.json")
    historical.verify_development_code(initial["code_sha256"])
    if (plan["schema_version"] != 1 or plan["original_plan_sha256"] != PLAN_SHA256
            or plan["original_summary_sha256"] != SUMMARY_SHA256
            or plan["config_sha256"] != original.CONFIG_SHA256
            or plan["original_code_sha256"] != initial["code_sha256"]
            or plan.get("runtime_development_code_sha256") != code_hashes()
            or plan.get("source_compatibility_sha256") != historical.catalog_sha256()
            or plan["extension_code_sha256"] != extension_hashes()
            or plan["effective_config"] != effective_config(config)
            or plan["series"] != initial["series"]
            or plan["jobs"] != build_jobs(config, "ridge") + build_jobs(config, "hybrid")
            or plan["new_alphas"] != list(NEW_ALPHAS) or plan["maximum_new_fits"] != 216
            or plan["stage_counts"] != {"ridge": 54, "hybrid_if_new_alpha_wins": 162}
            or plan["stop_after_declared_extension"] is not True
            or plan["holdout_forecasts_enabled"] is not False or plan["formal_comparative_timing"] is not False
            or plan["cpu_startup_profile"] != CPU_ENV):
        raise ValueError("Extension differs from its declared code, configuration or finite workload")
    identity, _ = original.environment_identity()
    if identity != plan["environment_identity"] or identity != initial["environment_identity"]:
        raise ValueError("Extension requires the same machine, interpreter and packages as the original run")
    for name in ("records", "inventory"):
        if sha256(directory / f"original-{name}.json") != plan[f"original_{name}_sha256"]:
            raise ValueError("Copied original records or inventory changed")
    if {p.name for p in (directory / "series").iterdir()} != {f"{c}.npz" for c in TUNING_CELLS}:
        raise ValueError("Unexpected development prefix files")
    for cell in TUNING_CELLS:
        data = load_series(directory, plan, cell)
        for job in build_jobs(config, "ridge")[:27]:
            if job["cell"] == cell and targets_for(data, job).tolist() != plan["series"][cell]["folds"][job["fold_index"]]["targets"]:
                raise ValueError("Development cohort changed")
    known = {j["id"] for j in plan["jobs"]}
    if (directory / "jobs").exists() and any(p.name not in known or p.is_symlink() for p in (directory / "jobs").iterdir()):
        raise ValueError("Unexpected extension job evidence")
    # Even resealed imported score records must reproduce the pinned summaries.
    from forecasting.development_selection import select_stage
    records = read_json(directory / "original-records.json")
    summary = read_json(directory / "original-summary.json")
    for stage in ("ridge", "hybrid"):
        scored = list(records[stage])
        if stage == "hybrid":
            cid = summary["selections"]["ridge"]["selection"]["selected"]["candidate_id"]
            scored += [{**r, "candidate_id": "hybrid-zero"} for r in records["ridge"] if r["candidate_id"] == cid]
        if select_stage(scored, config, stage) != summary["selections"][stage]["selection"]:
            raise ValueError("Imported scores differ from the original complete selection")
    return plan, config, checksum


def verify_source_inventory(directory, plan):
    from tools.development_store import verify_inventory
    verify_inventory(Path(plan["original_directory"]), read_json(directory / "original-inventory.json"), exclude=(".run.lock",))


def verify_current_identity(directory, plan, checksum):
    actual, _, current = verify_plan(directory)
    if actual != plan or current != checksum:
        raise ValueError("Extension identity changed while running")


def expected_scores(config, job):
    choices = scoring_candidates(config, job["stage"])
    if job["stage"] == "ridge":
        return [c for c in choices if c["candidate_id"] == job["candidate_id"]]
    return [c for c in choices if all(c["parameters"][k] == job["parameters"][k] for k in ("trees", "leaves"))]


def collect_stage(directory, plan, config, checksum, stage, *, validate=True):
    records, evidence = [], {}
    for job in build_jobs(config, stage):
        success = load_success(directory, job["id"], checksum)
        if success is None:
            raise ValueError(f"Incomplete extension {stage}: {job['id']}")
        attempt, marker = success
        if marker["job"] != job:
            raise ValueError("Stored extension job differs from plan")
        result = validate_result(attempt, job, plan, checksum, config) if validate else read_json(attempt / "result.json")
        if result["job"] != job or result["plan_sha256"] != checksum or result["status"] != "passed":
            raise ValueError("Extension result identity differs")
        records += result["records"]
        evidence[job["id"]] = {"attempt": str(attempt.relative_to(directory)), "completion_sha256": sha256(attempt / "complete.json")}
    return records, evidence


def checked_ridge_selection(directory, plan, config, checksum, *, validate=False, create=False):
    records, evidence = collect_stage(directory, plan, config, checksum, "ridge", validate=validate)
    initial = read_json(directory / "original-records.json")["ridge"]
    result = {"stage": "ridge", "plan_sha256": checksum, "evidence": evidence,
              "selection": select_ridge(initial, records, config), "original_plan_sha256": PLAN_SHA256}
    path = directory / "selections/ridge.json"
    if create:
        return save_or_check(path, result)
    if read_json(path) != result:
        raise ValueError("Global expanded ridge selection changed")
    return result


def ridge_dependency(directory, plan, config, checksum, job):
    picked = checked_ridge_selection(directory, plan, config, checksum)
    if not picked["selection"]["rebuild_corrections"]:
        raise ValueError("Original ridge winner forbids new corrections")
    cid = picked["selection"]["selected"]["candidate_id"]
    rid = f"{cid}--{job['cell']}--f{job['fold_index']+1}"
    attempt, _ = load_success(directory, rid, checksum)
    return attempt, {"job_id": rid, "attempt": str(attempt.relative_to(directory)),
                     "completion_sha256": sha256(attempt / "complete.json"),
                     "ridge_selection_sha256": sha256(directory / "selections/ridge.json")}


def selection(directory, plan, config, checksum, stage):
    ridge = checked_ridge_selection(directory, plan, config, checksum, validate=True, create=stage == "ridge")
    if stage == "ridge":
        return ridge
    original_records = read_json(directory / "original-records.json")
    ridge_records, _ = collect_stage(directory, plan, config, checksum, "ridge")
    if ridge["selection"]["rebuild_corrections"]:
        records, evidence = collect_stage(directory, plan, config, checksum, "hybrid")
    else:
        records, evidence = [], {}
        if any((directory / "jobs" / j["id"]).exists() for j in build_jobs(config, "hybrid")):
            raise ValueError("Original ridge winner forbids correction attempts")
    initial = read_json(directory / "original-summary.json")
    alpha = initial["selections"]["ridge"]["selection"]["selected"]["parameters"]["alpha"]
    value = {"stage": "hybrid", "plan_sha256": checksum, "evidence": evidence,
             "ridge_selection_sha256": sha256(directory / "selections/ridge.json"),
             "selection": select_hybrid(original_records["hybrid"], alpha, ridge["selection"],
                 original_records["ridge"] + ridge_records, records, config)}
    return save_or_check(directory / "selections/hybrid.json", value)


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
    if job["stage"] == "hybrid" and not checked_ridge_selection(directory, plan, config, checksum)["selection"]["rebuild_corrections"]:
        raise ValueError("Original ridge winner forbids new corrections")
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
            effective_config(config), ridge_bundle=ridge_bundle,
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
    arguments = [sys.executable, "-m", "adaptation.run_ridge_extension",
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
    from tools.check_cpu_controls import machine_information
    if not host_ready:
        raise ValueError("Confirm the declared host conditions with --host-ready before fitting")
    with run_lock(directory) as lock_fd:
        plan, config, checksum = verify_plan(directory)
        verify_source_inventory(directory, plan)
        if (directory / "extension-summary.json").exists():
            raise ValueError("Extension is already summarized; no further stage sessions are allowed")
        with stage_session(directory, stage, checksum):
            if stage == "hybrid":
                selection(directory, plan, config, checksum, "ridge")
            jobs = build_jobs(config, stage)
            if stage == "hybrid" and not checked_ridge_selection(directory, plan, config, checksum)["selection"]["rebuild_corrections"]:
                jobs = []
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
            verify_source_inventory(directory, plan)
            selected = selection(directory, plan, config, checksum, stage)
            verify_current_identity(directory, plan, checksum)
            verify_source_inventory(directory, plan)
            return selected


def cost_records(directory, checksum):
    """Record failures and unknown interrupted durations; never sum overlapping totals."""
    def timing(path):
        if not path.exists():
            return None
        value = read_json(path)
        seconds = value["wall_seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid recorded extension duration")
        return value
    attempts = [{"attempt": str(p.relative_to(directory)), "timing": timing(p / "elapsed.json")}
                for p in sorted((directory / "jobs").glob("*/attempt-*"))]
    sessions = []
    if (directory / "sessions").exists():
        paths = sorted((directory / "sessions").iterdir())
        if [p.name for p in paths] != [f"session-{i:04d}" for i in range(1, len(paths)+1)]:
            raise ValueError("Unexpected extension session history")
        for p in paths:
            start = read_json(p / "started.json")
            if start["plan_sha256"] != checksum or start["stage"] not in ("ridge", "hybrid"):
                raise ValueError("Extension session identity differs")
            sessions.append({"session": p.name, "stage": start["stage"], "timing": timing(p / "finished.json")})
    return {"attempts_including_failures": attempts, "stage_sessions_including_failures": sessions,
            "recorded_attempt_wall_seconds": math.fsum(x["timing"]["wall_seconds"] for x in attempts if x["timing"]),
            "recorded_stage_wall_seconds": math.fsum(x["timing"]["wall_seconds"] for x in sessions if x["timing"]),
            "attempts_with_unavailable_elapsed_time": sum(x["timing"] is None for x in attempts),
            "sessions_with_unavailable_elapsed_time": sum(x["timing"] is None for x in sessions),
            "scope": "additional search only; stage and attempt totals overlap and must not be added; correction fits exclude cached ridge; imports/preparation/initial verification excluded"}


def summarize(directory):
    with run_lock(directory):
        plan, config, checksum = verify_plan(directory)
        verify_source_inventory(directory, plan)
        ridge = selection(directory, plan, config, checksum, "ridge")
        hybrid = selection(directory, plan, config, checksum, "hybrid")
        initial = read_json(directory / "original-summary.json")
        value = {"status": "passed", "scope": "completed single bounded development extension",
                 "plan_sha256": checksum, "original_plan_sha256": PLAN_SHA256,
                 "original_summary_sha256": SUMMARY_SHA256,
                 "selections": {"ridge": ridge, "hybrid": hybrid},
                 "initial_selections": initial["selections"],
                 "standalone_ridge": ridge["selection"]["selected"],
                 "unchanged_direct_lightgbm": initial["selections"]["direct"]["selection"]["selected"],
                 "new_successful_fit_count": 216 if ridge["selection"]["rebuild_corrections"] else 54,
                 "additional_tuning_cost": cost_records(directory, checksum),
                 "initial_recorded_stage_wall_seconds": initial["recorded_stage_wall_seconds"],
                 "formal_timing_eligible": False, "primary_comparator": None,
                 "holdout_forecasts_evaluated": False, "automatic_further_expansion": False,
                 "next_decision": "Review all extension outcomes and additional tuning cost; no automatic expansion or confirmation"}
        verify_current_identity(directory, plan, checksum)
        verify_source_inventory(directory, plan)
        return save_or_check(directory / "extension-summary.json", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("prepare", "ridge", "hybrid", "summarize"), default="prepare")
    parser.add_argument("--execute-extension", action="store_true")
    parser.add_argument("--host-ready", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    original.startup_check()
    # resolve() also prevents source/output aliasing through parent symlinks.
    directory = Path(args.output_dir).resolve()
    if args.worker:
        if not args.attempt or args.original_dir or args.execute_extension or args.host_ready or args.resume or args.retry_failed or args.stage != "prepare":
            parser.error("Internal worker accepts only job, attempt and output directory")
        worker(directory, args.worker, Path(args.attempt).resolve())
        return
    if args.attempt or (args.retry_failed and not args.resume):
        parser.error("Retry needs --resume and --retry-failed")
    if args.stage == "prepare":
        if not args.original_dir or args.execute_extension or args.host_ready or args.resume or args.retry_failed:
            parser.error("Prepare takes only --original-dir and a new --output-dir")
        result = prepare(Path(args.original_dir).resolve(), directory)
        print(json.dumps({"status": "prepared", "stage_counts": result["stage_counts"], "fitting_performed": False}, indent=2))
    elif args.stage == "summarize":
        if args.original_dir or args.execute_extension or args.host_ready or args.resume or args.retry_failed:
            parser.error("Summarize takes only the prepared output directory")
        result = summarize(directory)
        print(json.dumps({"status": result["status"], "new_successful_fit_count": result["new_successful_fit_count"], "primary_comparator": None}, indent=2))
    else:
        if args.original_dir or not args.execute_extension or not args.host_ready:
            parser.error("Fitting requires --execute-extension and --host-ready")
        result = run_stage(directory, args.stage, resume=args.resume, retry_failed=args.retry_failed, host_ready=args.host_ready)
        print(json.dumps({"status": "passed", "stage": args.stage, "selected": result["selection"]["selected"]}, indent=2))


if __name__ == "__main__":
    main()
