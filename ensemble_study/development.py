"""Complete only the 108 missing standalone L1 development fits.

Preparation audits and pins historical results and copies development prefixes.
Each numerical worker fits one new full-target tree and checks native reload.
The parent validates closed worker outputs before sealing an immutable attempt.
Historical fits are never rerun; weight replay belongs to the evidence module.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ensemble_study import common
from forecasting.development_plan import FOLDS, TUNING_CELLS
from tools.development_store import (begin_attempt, complete_attempt, load_success,
    run_lock, sha256, verify_inventory, write_json_atomic, _inventory, _read_json)
from tools.run_cpu import CPU_ENV
from tools.run_development_experiments import (read_config, load_series, targets_for,
                                              stage_session, startup_check)

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "missing_l1_development_completion"
MAXIMUM_FITS = 108


def build_jobs():
    """Four capacities x nine unchanged cells x three chronological folds."""
    return [{"id": f"l1-t{trees}-l{leaves}--{cell}--f{fold+1}",
             "cell": cell, "fold_index": fold,
             "condition": {"id": "lightgbm_l1", "kind": "direct", "objective": "l1",
                           "trees": trees, "leaves": leaves, "ridge_weight": 0.},
             "fit_end": start, "score_start": start, "score_end": end, "seed": None}
            for trees, leaves in common.MISSING for cell in TUNING_CELLS
            for fold, (start, end) in enumerate(FOLDS)]


def _operator(operator, host_ready):
    if operator not in ("automated", "user") or type(host_ready) is not bool:
        raise ValueError("Record the actual operator and a boolean readiness assertion")
    if operator == "user" and not host_ready:
        raise ValueError("User-operated execution requires --host-ready")
    if operator == "automated" and host_ready:
        raise ValueError("Automated execution must use measured readiness, not a user assertion")


def _source_anchors(source):
    return {name: digest for name, digest in source["files_sha256"].items()
            if name in ("plan.json", "plan-checksum.json", "development-summary.json", "l1-summary.json")}


def verify_sources(plan, *, full=False):
    """Full inventories at stage boundaries; immutable anchor checks per worker."""
    if set(plan["sources"]) != {"original", "l1"}:
        raise ValueError("Both historical source directories are required")
    for source in plan["sources"].values():
        if set(source) != {"directory", "files_sha256"}:
            raise ValueError("Historical source snapshot schema differs")
        directory = Path(source["directory"])
        if not directory.is_absolute() or "plan.json" not in source["files_sha256"]:
            raise ValueError("Source snapshots require absolute directories and their plans")
        if full:
            verify_inventory(directory, source["files_sha256"], exclude=(".run.lock",))
        else:
            for name, digest in _source_anchors(source).items():
                if sha256(directory / name) != digest:
                    raise ValueError("Historical source anchor changed")


def _check_cohort(directory, plan, job):
    data = load_series(directory, plan, job["cell"])
    targets = targets_for(data, job)
    expected = plan["series"][job["cell"]]["folds"][job["fold_index"]]["targets"]
    if not len(targets) or targets.tolist() != expected:
        raise ValueError("Development targets differ from the original common cohort")
    return data, targets


def verify_plan(directory, *, check_native=True, full_sources=False,
                operator="automated", host_ready=False):
    directory = Path(directory).absolute()
    _operator(operator, host_ready)
    plan = _read_json(directory / "plan.json")
    checksum = sha256(directory / "plan.json")
    if _read_json(directory / "plan-checksum.json") != {"sha256": checksum}:
        raise ValueError("The immutable L1 completion plan changed")
    protocol = common.protocol()
    config = read_config()
    if (plan["schema_version"] != 1 or plan["scope"] != SCOPE
            or plan["study_id"] != common.STUDY_ID
            or plan["protocol_sha256"] != sha256(common.PROTOCOL)
            or plan["config_sha256"] != sha256(ROOT / "run_config.yaml")
            or plan["code_sha256"] != common.code_hashes("development")
            or plan["jobs"] != build_jobs() or len(plan["jobs"]) != MAXIMUM_FITS
            or protocol["maximum_new_development_successful_fits"] != MAXIMUM_FITS
            or plan["cpu_startup_profile"] != CPU_ENV
            or plan["holdout_forecasts_enabled"] is not False
            or plan["formal_comparative_timing"] is not False
            or set(plan["series"]) != set(TUNING_CELLS)):
        raise ValueError("L1 completion code, protocol, workload or chronology drift")
    source_audit = plan["source_audit"]
    if (source_audit.get("status") != "passed"
            or source_audit.get("environment_identity") != plan["environment_identity"]
            or source_audit.get("reused_counts") != {"l2": 162, "l1": 54, "ridge": 27}
            or source_audit.get("sources") != plan["sources"]):
        raise ValueError("Original component reuse audit differs from the sealed plan")
    receipt = plan["publication"]
    if sha256(Path(receipt["path"])) != receipt["sha256"]:
        raise ValueError("Published-byte verification receipt changed")
    common.verify_publication(Path(receipt["path"]), scope="development", commit=receipt["commit"])
    verify_sources(plan, full=full_sources)
    original = _read_json(Path(plan["sources"]["original"]["directory"]) / "plan.json")
    if original["series"] != plan["series"] or original["environment_identity"] != plan["environment_identity"]:
        raise ValueError("Copied data descriptions or host differ from the original development plan")
    expected = {f"{cell}.npz" for cell in TUNING_CELLS}
    if {p.name for p in (directory / "series").iterdir()} != expected:
        raise ValueError("Missing or additional development series")
    for cell, entry in plan["series"].items():
        if entry["path"] != f"series/{cell}.npz" or sha256(directory / entry["path"]) != entry["sha256"]:
            raise ValueError("Copied development data changed")
    if (directory / "jobs").exists():
        known = {job["id"] for job in plan["jobs"]}
        if any(path.name not in known for path in (directory / "jobs").iterdir()):
            raise ValueError("Unexpected job outside the 108-fit completion")
    for key in ("readiness_before", "readiness_after"):
        common.verify_saved_readiness(plan[key], plan["environment_identity"])
    if check_native:
        common.native_readiness(plan["environment_identity"], operator=operator, host_ready=host_ready)
    return plan, config, checksum


def prepare(original_directory, l1_directory, directory, publication,
            *, operator="automated", host_ready=False):
    """Audit reuse and publish a new immutable plan; no model is fitted."""
    _operator(operator, host_ready)
    common.protocol()
    original_directory, l1_directory, directory, publication = [Path(p).absolute()
        for p in (original_directory, l1_directory, directory, publication)]
    if directory.exists() or any(source == directory or source in directory.parents or directory in source.parents
                                 for source in (original_directory, l1_directory)):
        raise ValueError("Use a new output directory outside both preserved source runs")
    receipt = common.verify_publication(publication, scope="development")
    initial = _read_json(original_directory / "plan.json")
    readiness_before = common.native_readiness(initial["environment_identity"], operator, host_ready)
    from ensemble_study.evidence import audit_sources
    from tools.cpu_controls import single_thread_cpu
    with single_thread_cpu() as controls:
        controls.verify()
        audit = audit_sources(original_directory, l1_directory)
        controls.verify()
    expected_sources = {"original": {"directory": str(original_directory),
                                      "files_sha256": _inventory(original_directory, exclude=(".run.lock",))},
                        "l1": {"directory": str(l1_directory),
                               "files_sha256": _inventory(l1_directory, exclude=(".run.lock",))}}
    if (audit.get("status") != "passed" or audit.get("sources") != expected_sources
            or audit.get("environment_identity") != initial["environment_identity"]
            or audit.get("reused_counts") != {"l2": 162, "l1": 54, "ridge": 27}):
        raise ValueError("Historical audit does not establish the precise reused component budget")
    read_config()
    jobs = build_jobs()
    if len(jobs) != MAXIMUM_FITS or set(initial["series"]) != set(TUNING_CELLS):
        raise ValueError("The completion requires the fixed nine-cell, three-fold panel")
    for cell, entry in initial["series"].items():
        if entry["path"] != f"series/{cell}.npz":
            raise ValueError("Unsafe source development series path")
        for job in (j for j in jobs[:27] if j["cell"] == cell):
            _check_cohort(original_directory, initial, job)
    readiness_after = common.native_readiness(initial["environment_identity"], operator, host_ready)
    verify_sources({"sources": expected_sources}, full=True)
    directory.mkdir(parents=True)
    (directory / "series").mkdir()
    for entry in initial["series"].values():
        shutil.copyfile(original_directory / entry["path"], directory / entry["path"])
    plan = {"schema_version": 1, "study_id": common.STUDY_ID, "scope": SCOPE,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256(common.PROTOCOL), "config_sha256": sha256(ROOT / "run_config.yaml"),
            "code_sha256": common.code_hashes("development"), "cpu_startup_profile": CPU_ENV,
            "publication": {"path": str(publication), "sha256": sha256(publication), "commit": receipt["commit"]},
            "sources": expected_sources, "source_audit": audit, "series": deepcopy(initial["series"]),
            "environment_identity": initial["environment_identity"], "jobs": jobs,
            "readiness_before": readiness_before, "readiness_after": readiness_after,
            "holdout_forecasts_enabled": False, "formal_comparative_timing": False,
            "timing_scope": "Additional development search; not formal selected-model deployment cost",
            "maximum_new_successful_fits": MAXIMUM_FITS, "new_ridge_fits": 0,
            "new_weight_replay_fits": 0}
    write_json_atomic(directory / "plan.json", plan)
    write_json_atomic(directory / "plan-checksum.json", {"sha256": sha256(directory / "plan.json")})
    verify_plan(directory, check_native=False)
    return plan


def worker(directory, job_id, attempt, *, operator="automated", host_ready=False):
    """One fresh numerical worker: fit, original-unit forecast and exact reload."""
    import numpy as np
    from ensemble_study.model import fit_selected, predict_with_components, save_model, load_model, fingerprint
    from tools.cpu_controls import single_thread_cpu
    from tools.pilot_measurements import memory_snapshot
    from tools.pilot_results import score_predictions, save_predictions
    directory, attempt = Path(directory).absolute(), Path(attempt).absolute()
    plan, config, checksum = verify_plan(directory, check_native=False)
    job = next((job for job in plan["jobs"] if job["id"] == job_id), None)
    if job is None or attempt.parent != directory / "jobs" / job_id:
        raise ValueError("Unknown assigned development job or attempt")
    metadata = _read_json(attempt / "attempt.json")
    actor = _read_json(attempt / "operator.json")
    if actor != {"operator": operator, "host_ready_assertion": host_ready}:
        raise ValueError("Worker operator differs from the current invocation's assigned operator")
    _operator(actor["operator"], actor["host_ready_assertion"])
    if (metadata["job"] != job or metadata["plan_sha256"] != checksum
            or attempt.name != f"attempt-{metadata['attempt_number']:04d}"
            or any((attempt / name).exists() for name in ("complete.json", "failure.json", "result.json", "model"))):
        raise ValueError("Development worker requires its fresh assigned attempt")
    data, targets = _check_cohort(directory, plan, job)
    readiness_before = common.native_readiness(plan["environment_identity"], operator, host_ready)
    with single_thread_cpu() as controls:
        controls_before = controls.verify()
        bundle, stats = fit_selected(job["condition"], data["traffic"], data["observed"],
                                    data["timestamps"], job["fit_end"], config)
        fit_memory = memory_snapshot()
        state = fingerprint(bundle)
        output = predict_with_components(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
        if output["ridge_raw"] is not None or not np.array_equal(output["raw"], output["direct_raw"]):
            raise AssertionError("A new completion fit must be standalone full-target LightGBM")
        save_model(bundle, attempt / "model")
        restored = load_model(attempt / "model")
        reloaded = predict_with_components(restored, data["traffic"], data["observed"], data["timestamps"], targets)
        for key in ("raw", "final", "direct_raw"):
            np.testing.assert_array_equal(output[key], reloaded[key])
        if fingerprint(bundle) != state or fingerprint(restored) != state:
            raise AssertionError("Prediction or native reload changed model state")
        csv = save_predictions(attempt / "predictions.csv", job["cell"], job["condition"]["id"],
            None, targets, data["timestamps"][targets], data["traffic_observed"][targets], output["raw"], output["final"])
        score = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
            job["fit_end"], job["score_start"], job["score_end"], targets, output["raw"], output["final"])
        controls_after = controls.verify()
    readiness_after = common.native_readiness(plan["environment_identity"], operator, host_ready)
    verify_plan(directory, check_native=False)
    result = {"status": "passed", "job": job, "plan_sha256": checksum, "condition": job["condition"],
        "fit_seconds": stats["fit_seconds"], "training": stats, "development_fit_memory": fit_memory,
        "model_state_sha256": state, "model_files_sha256": _inventory(attempt / "model"),
        "predictions_sha256": csv["sha256"], "evaluation": score,
        "controls_before": controls_before, "controls_after": controls_after,
        "readiness_before": readiness_before, "readiness_after": readiness_after,
        "reload_predictions_exact": True, "fitted_state_unchanged": True,
        "component_prediction_identity": "direct_raw equals prediction_raw; no ridge component",
        "formal_comparative_timing": False}
    write_json_atomic(attempt / "result.json", result)
    return result


def read_predictions(attempt, job, plan):
    import numpy as np
    import pandas as pd
    attempt = Path(attempt)
    frame = pd.read_csv(attempt / "predictions.csv", float_precision="round_trip", keep_default_na=False,
                        parse_dates=["timestamp"], dtype={"seed": str})
    if set(frame) != {"cell", "method", "seed", "target_index", "timestamp", "actual", "prediction_raw", "prediction"}:
        raise ValueError("Development forecast CSV schema differs")
    data, targets = _check_cohort(attempt.parents[2], plan, job)
    np.testing.assert_array_equal(frame.target_index, targets)
    np.testing.assert_array_equal(frame.timestamp.to_numpy(), data["timestamps"][targets].to_numpy())
    np.testing.assert_array_equal(frame.actual, data["traffic_observed"][targets])
    np.testing.assert_array_equal(frame.prediction, np.maximum(0., frame.prediction_raw))
    if (not np.isfinite(frame.prediction_raw).all() or not frame.cell.eq(job["cell"]).all()
            or not frame.method.eq(job["condition"]["id"]).all() or not frame.seed.eq("").all()):
        raise ValueError("Development forecast identity or numerical values differ")
    return frame


def validate_result(attempt, job, plan, checksum, *, reload_model=False):
    """Recompute saved forecast arithmetic and optionally verify exact native reload."""
    import numpy as np
    from tools.pilot_results import score_predictions
    attempt = Path(attempt)
    result = _read_json(attempt / "result.json")
    actor = _read_json(attempt / "operator.json")
    if set(actor) != {"operator", "host_ready_assertion"}:
        raise ValueError("Unexpected attempt-operator metadata")
    _operator(actor["operator"], actor["host_ready_assertion"])
    if (result["status"] != "passed" or result["job"] != job or result["plan_sha256"] != checksum
            or result["condition"] != job["condition"] or result["formal_comparative_timing"] is not False
            or result["reload_predictions_exact"] is not True or result["fitted_state_unchanged"] is not True):
        raise ValueError("New development result differs from its frozen job")
    training = result["training"]
    if (training["fit_interval"] != [0, job["fit_end"]] or training["condition"] != job["condition"]
            or training["cached_component_reused"] is not False
            or training["target_design"] != "independent full traffic target"
            or training["component_fit_counts"] != {"direct_lightgbm": 1}
            or training["tree_parameters"].get("objective") != "regression_l1"):
        raise ValueError("New fit is not a fresh standalone L1 full-target model")
    seconds = result["fit_seconds"]
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0 or training["fit_seconds"] != seconds):
        raise ValueError("Development fitting duration must be finite and positive")
    for key in ("controls_before", "controls_after"):
        if result[key].get("runtime_library_controls_verified") is not True:
            raise ValueError("Worker numerical controls were not verified")
    for key in ("readiness_before", "readiness_after"):
        common.verify_saved_readiness(result[key], plan["environment_identity"])
        if (result[key]["operator"] != actor["operator"]
                or result[key]["host_ready_assertion"] != actor["host_ready_assertion"]):
            raise ValueError("Worker readiness does not describe the assigned execution operator")
    if (_inventory(attempt / "model") != result["model_files_sha256"]
            or sha256(attempt / "predictions.csv") != result["predictions_sha256"]):
        raise ValueError("Saved model or predictions changed")
    data, targets = _check_cohort(attempt.parents[2], plan, job)
    from preprocessing.windows import eligible_targets
    expected_training_count = int(eligible_targets(data["traffic"][:job["fit_end"]], data["observed"][:job["fit_end"]]).sum())
    if training["training_target_count"] != expected_training_count:
        raise ValueError("Supervised training rows differ from the common fitting mask")
    frame = read_predictions(attempt, job, plan)
    score = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
        job["fit_end"], job["score_start"], job["score_end"], targets,
        frame.prediction_raw.to_numpy(), frame.prediction.to_numpy())
    if score != result["evaluation"]:
        raise ValueError("Saved development metrics do not reproduce")
    if reload_model:
        from ensemble_study.model import load_model, fingerprint, predict_with_components
        from tools.cpu_controls import single_thread_cpu
        with single_thread_cpu() as controls:
            controls.verify()
            restored = load_model(attempt / "model")
            if (restored.condition != job["condition"] or restored.settings["fit_end"] != job["fit_end"]
                    or fingerprint(restored) != result["model_state_sha256"]):
                raise ValueError("Reloaded new model differs from its condition or fitted state")
            output = predict_with_components(restored, data["traffic"], data["observed"], data["timestamps"], targets)
            np.testing.assert_array_equal(output["raw"], frame.prediction_raw)
            np.testing.assert_array_equal(output["final"], frame.prediction)
            np.testing.assert_array_equal(output["direct_raw"], output["raw"])
            if output["ridge_raw"] is not None:
                raise ValueError("A standalone completion model must not contain ridge")
            controls.verify()
    return {"job": job, "attempt": attempt, "result": result, "frame": frame}


def completed_records(directory, *, reload_models=True):
    """Expose the exact complete 108-fit cohort for the evidence/selection audit."""
    directory = Path(directory).absolute()
    plan, _, checksum = verify_plan(directory, check_native=False, full_sources=True)
    records = []
    for job in plan["jobs"]:
        saved = load_success(directory, job["id"], checksum)
        if saved is None:
            raise ValueError("All 108 new successful development fits are required before selection")
        attempt, _ = saved
        records.append(validate_result(attempt, job, plan, checksum, reload_model=reload_models))
    return records


def launch(directory, job, attempt, lock_fd, operator, host_ready):
    command = [sys.executable, "-m", "ensemble_study.development", "--output-dir", str(directory),
               "--worker", job["id"], "--attempt", str(attempt), "--operator", operator]
    if host_ready:
        command.append("--host-ready")
    with (attempt / "worker.log").open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, **CPU_ENV},
                                 stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
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


def run(directory, *, resume=False, retry_failed=False, operator="automated", host_ready=False):
    """Sequential immutable attempts; never train an already-successful job."""
    directory = Path(directory).absolute()
    _operator(operator, host_ready)
    with run_lock(directory) as lock_fd:
        plan, _, checksum = verify_plan(directory, full_sources=True, operator=operator, host_ready=host_ready)
        with stage_session(directory, "missing_l1_completion", checksum):
            for index, job in enumerate(plan["jobs"], 1):
                attempt, cached = begin_attempt(directory, job, checksum, resume=resume, retry_failed=retry_failed)
                if cached:
                    validate_result(attempt, job, plan, checksum, reload_model=True)
                    print(f"Reused completion {index}/{len(plan['jobs'])}: {job['id']}", flush=True)
                    continue
                started, outcome = time.perf_counter(), "failed_or_interrupted"
                try:
                    write_json_atomic(attempt / "operator.json", {"operator": operator,
                        "host_ready_assertion": host_ready})
                    print(f"New L1 completion {index}/{len(plan['jobs'])}: {job['id']}", flush=True)
                    if launch(directory, job, attempt, lock_fd, operator, host_ready) != 0:
                        raise RuntimeError(f"Worker failed; inspect {attempt / 'worker.log'}")
                    validate_result(attempt, job, plan, checksum, reload_model=True)
                    verify_plan(directory, check_native=False)
                    outcome = "passed"
                except BaseException as error:
                    write_json_atomic(attempt / "failure.json", {"type": type(error).__name__,
                        "message": str(error), "automatic_retry": False})
                    raise
                finally:
                    write_json_atomic(attempt / "elapsed.json", {"outcome": outcome,
                        "wall_seconds": time.perf_counter()-started,
                        "scope": "Consumed attempt including worker startup, readiness, fitting, exports and audit"})
                complete_attempt(attempt, job, checksum)
        verify_plan(directory, full_sources=True, operator=operator, host_ready=host_ready)
        rows = completed_records(directory, reload_models=False)
        attempts = [path for job in plan["jobs"] for path in (directory / "jobs" / job["id"]).iterdir()]
        return {"status": "passed", "scope": SCOPE, "new_successful_fit_count": len(rows),
                "attempt_count": len(attempts), "failed_or_incomplete_attempt_count": len(attempts)-len(rows),
                "reused_historical_fit_counts": plan["source_audit"]["reused_counts"],
                "holdout_forecasts_evaluated": False, "formal_comparative_timing": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), default="prepare")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-dir", type=Path)
    parser.add_argument("--l1-dir", type=Path)
    parser.add_argument("--publication", type=Path)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--operator", choices=("automated", "user"), required=True)
    parser.add_argument("--host-ready", action="store_true")
    parser.add_argument("--execute-development", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    startup_check()
    _operator(args.operator, args.host_ready)
    inputs = any(x is not None for x in (args.original_dir, args.l1_dir, args.publication))
    if args.worker:
        if (args.stage != "prepare" or not args.attempt or inputs or args.summary_dir
                or args.execute_development or args.resume or args.retry_failed):
            parser.error("Internal workers accept only assigned attempt, job, operator and output")
        worker(args.output_dir, args.worker, args.attempt, operator=args.operator, host_ready=args.host_ready)
        return
    if args.attempt or (args.retry_failed and not args.resume):
        parser.error("Invalid attempt or retry arguments")
    if args.stage == "prepare":
        if (not all((args.original_dir, args.l1_dir, args.publication)) or args.summary_dir
                or args.execute_development or args.resume or args.retry_failed):
            parser.error("Prepare requires original results, historical L1 results and publication receipt")
        plan = prepare(args.original_dir, args.l1_dir, args.output_dir, args.publication,
                       operator=args.operator, host_ready=args.host_ready)
        result = {"status": "prepared", "new_successful_fit_budget": len(plan["jobs"]), "fitting_performed": False}
    elif args.stage == "run":
        if inputs or args.summary_dir or not args.execute_development:
            parser.error("Run requires its existing plan and --execute-development")
        result = run(args.output_dir, resume=args.resume, retry_failed=args.retry_failed,
                     operator=args.operator, host_ready=args.host_ready)
    else:
        if inputs or not args.summary_dir or args.execute_development or args.resume or args.retry_failed:
            parser.error("Summarize requires existing completion results and a separate --summary-dir")
        with run_lock(args.output_dir):
            plan, _, _ = verify_plan(args.output_dir, full_sources=True,
                                    operator=args.operator, host_ready=args.host_ready)
            from ensemble_study.evidence import summarize_evidence, save_evidence
            result = summarize_evidence(Path(plan["sources"]["original"]["directory"]),
                Path(plan["sources"]["l1"]["directory"]), args.output_dir, protocol_path=common.PROTOCOL)
            save_evidence(args.summary_dir, result)
        result = {"status": "passed", "summary_directory": str(args.summary_dir.absolute())}
    import json
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
