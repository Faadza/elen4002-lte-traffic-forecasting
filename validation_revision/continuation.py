"""Append-only continuation under both the original plan and a published amendment.

Original model, fit, scoring and data modules are imported without modification.
Only LSTM prediction validation uses the separately published worker. No runtime
monkey-patching or hash-check bypass is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.development_store import (sha256, _read_json, _inventory, _attempts,
    load_success, begin_attempt, complete_attempt, run_lock, write_json_atomic)
from tools.run_cpu import CPU_ENV
from tools.run_development_experiments import stage_session
from ensemble_study.evaluation_plan import ROOT, verify_plan
from ensemble_study.evaluation_run import validate_result, summarize, execution_identity
from docs.commands.confirmation_recovery import (readiness_failure, wait_for_quiet,
    enforce_budget, RecoveryInterrupted, interrupt)
from validation_revision.measurement import RULE, TOLERANCE, verify_audit

AMENDMENT = ROOT/"docs/commands/confirmation-validation-v2.json"
RECEIPT = ROOT.parent/"context/reconciliation-review/confirmation-validation-publication-2026-10-09.json"


def inventory_digest(path):
    return hashlib.sha256(json.dumps(_inventory(path), sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def amendment_identity(digest):
    return {"id": RULE, "plan_sha256": digest,
            "publication_receipt_sha256": sha256(RECEIPT)}


def verify_amendment(directory):
    """Verify the additional plan; never change the original plan or receipt."""
    amendment, published = _read_json(AMENDMENT), _read_json(RECEIPT)
    if published.get("verified") is not True:
        raise ValueError("Numerical amendment publication is not verified")
    for name, digest in published["payload_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Amendment publication differs: {name}")
    required = {str(AMENDMENT.relative_to(ROOT)), "docs/commands/resume_ensemble_confirmation_v2_mac.sh"}
    required.update(str(p.relative_to(ROOT)) for p in (ROOT/"validation_revision").glob("*.py"))
    if not required <= published["payload_sha256"].keys():
        raise ValueError("Unpublished amendment source")
    if (Path(directory) != ROOT/amendment["output_directory"]
            or sha256(directory/"plan.json") != amendment["base_plan_sha256"]
            or amendment["rule"] != RULE
            or amendment["rtol"] != TOLERANCE or amendment["atol"] != TOLERANCE
            or sys.executable != amendment["python_executable"]):
        raise ValueError("Amendment plan, rule, directory or interpreter differs")
    base = _read_json(directory/"plan.json")
    for name, digest in base["code_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Original numerical source changed: {name}")
    for entry in amendment["diagnostic_evidence"]:
        if sha256(ROOT.parent/entry["path"]) != entry["sha256"]:
            raise ValueError("Independent diagnostic evidence changed")
    for name, digest in amendment["additional_dependencies_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Amendment helper changed: {name}")
    if "docs/commands/confirmation_recovery.py" not in amendment["additional_dependencies_sha256"]:
        raise ValueError("Recovery helper must be pinned explicitly")
    return amendment, sha256(AMENDMENT)


def validate_amended_result(attempt, job, plan, checksum, amendment, digest, *, read_forecasts=True):
    result = validate_result(attempt, job, plan, checksum, read_forecasts=read_forecasts)
    relative = str(attempt.relative_to(ROOT/amendment["output_directory"]))
    if relative in amendment["historical_attempts"]:
        if inventory_digest(attempt) != amendment["historical_attempts"][relative]:
            raise ValueError("Historical attempt changed")
        if job["method"] == "lstm" and job["id"] not in amendment["audited_completed_lstm_jobs"]:
            raise ValueError("Historical LSTM lacks the independent amended-rule audit")
    else:
        expected = amendment_identity(digest)
        if _read_json(attempt/"validation-amendment.json") != expected:
            raise ValueError("New attempt lacks the published amendment identity")
        if job["method"] == "lstm":
            predicted = result["predict"]
            if predicted.get("validation_amendment") != expected:
                raise ValueError("LSTM prediction used an unamended worker")
            if plan["common_origins"]:
                audit = predicted["timings"]["numerical_validation"]
                verify_audit(audit, len(plan["common_origins"]))
                import numpy as np
                scaler = _read_json(attempt/"model/single/model.json")["scaler"]
                if (audit["mean"] != scaler["mean"] or audit["sd"] != scaler["sd"]
                        or scaler["fit_end"] != job["fit_end"]):
                    raise ValueError("Numerical audit uses a different fitting scaler")
                with np.load(attempt/"reference.npz", allow_pickle=False) as reference:
                    positions = np.searchsorted(reference["targets"], plan["common_origins"])
                    np.testing.assert_array_equal(reference["targets"][positions], plan["common_origins"])
                    np.testing.assert_array_equal(audit["batch_z"],
                        (reference["raw"][positions]-scaler["mean"])/scaler["sd"])
            if job["id"] == amendment["diagnosed_failure_job"]:
                if result["fit"]["model_state_sha256"] != amendment["diagnosed_model_state_sha256"]:
                    raise ValueError("Retried LSTM model differs from the diagnosed same-seed fit")
    return result


def snapshot(directory, plan, amendment, digest):
    """Audit every success and failure without selecting partial model rankings."""
    state = {"complete": [], "failures": [], "counts": {}, "next_job": None}
    present = set()
    for job in plan["jobs"]:
        saved = load_success(directory, job["id"], amendment["base_plan_sha256"])
        attempts = _attempts(directory, job["id"])
        state["counts"][job["id"]] = len(attempts)
        for attempt in attempts:
            relative = str(attempt.relative_to(directory))
            present.add(relative)
            if _read_json(attempt/"attempt.json")["job"] != job:
                raise ValueError("Attempt settings differ")
            if relative in amendment["historical_attempts"]:
                if inventory_digest(attempt) != amendment["historical_attempts"][relative]:
                    raise ValueError("Historical evidence changed")
            elif _read_json(attempt/"validation-amendment.json") != amendment_identity(digest):
                raise ValueError("Unexpected attempt outside this continuation")
            if (attempt/"failure.json").exists():
                diagnosed = relative == amendment["diagnosed_failure_attempt"]
                if not (diagnosed and relative in amendment["historical_attempts"]) and not readiness_failure(attempt):
                    raise ValueError(f"Unrecognized failure; stop for investigation: {attempt}")
                state["failures"].append(relative)
            elif not (attempt/"complete.json").exists():
                raise ValueError(f"Incomplete/active attempt requires review: {attempt}")
        if saved:
            if state["next_job"] is not None:
                raise ValueError("Successful jobs are not a sequential prefix")
            validate_amended_result(saved[0], job, plan, amendment["base_plan_sha256"],
                amendment, digest, read_forecasts=False)
            state["complete"].append(job["id"])
        elif state["next_job"] is None:
            state["next_job"] = job["id"]
    if not amendment["historical_attempts"].keys() <= present:
        raise ValueError("Historical attempts are missing")
    if (directory/"jobs").exists() and {p.name for p in (directory/"jobs").iterdir()} - {j["id"] for j in plan["jobs"]}:
        raise ValueError("Unknown job directory")
    return state


def launch(directory, job, attempt, phase, lock_fd):
    module = "validation_revision.worker" if phase == "predict" and job["method"] == "lstm" else "ensemble_study.evaluation_run"
    arguments = [sys.executable, "-B", "-m", module, "--output-dir", str(directory),
                 "--worker", job["id"], "--attempt", str(attempt)]
    if module == "ensemble_study.evaluation_run":
        arguments.extend(["--phase", phase])
    with (attempt/f"{phase}.log").open("x") as stream:
        child = None
        try:
            child = subprocess.Popen(arguments, cwd=ROOT, env={**os.environ, **CPU_ENV},
                stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
            return child.wait()
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise


def run(directory):
    import psutil
    execution = execution_identity("user", True)
    with run_lock(directory) as lock_fd:
        amendment, digest = verify_amendment(directory)
        plan, config, checksum = verify_plan(directory)
        state = snapshot(directory, plan, amendment, digest)
        with stage_session(directory, "confirmation", checksum) as session:
            write_json_atomic(session/"operator.json", execution)
            write_json_atomic(session/"validation-amendment.json", amendment_identity(digest))
            with (session/"recovery-events.jsonl").open("x") as journal:
                def emit(event, **data):
                    journal.write(json.dumps({"event": event, "at_utc": datetime.now(timezone.utc).isoformat(),
                        **data}, sort_keys=True, allow_nan=False)+"\n")
                    journal.flush()
                for index, job in enumerate(plan["jobs"], 1):
                    if job["id"] in state["complete"]:
                        continue
                    while True:
                        state["next_job"] = job["id"]
                        enforce_budget(state, amendment["recovery_limits"])
                        wait_for_quiet(amendment["recovery_limits"], emit, sample=psutil.cpu_percent)
                        verify_amendment(directory)
                        verify_plan(directory)
                        attempt, cached = begin_attempt(directory, job, checksum, resume=True, retry_failed=True)
                        if cached:
                            raise ValueError("Unexpected concurrent completion")
                        write_json_atomic(attempt/"operator.json", execution)
                        write_json_atomic(attempt/"validation-amendment.json", amendment_identity(digest))
                        state["counts"][job["id"]] += 1
                        started, outcome = time.perf_counter(), "failed_or_interrupted"
                        error_seen = None
                        try:
                            print(f"confirmation {index}/{len(plan['jobs'])}: {job['id']}", flush=True)
                            for phase in ("fit", "predict"):
                                code = launch(directory, job, attempt, phase, lock_fd)
                                if code != 0:
                                    raise RuntimeError(f"{phase} worker failed ({code}); inspect {attempt/(phase+'.log')}")
                            validate_amended_result(attempt, job, plan, checksum, amendment, digest)
                            verify_amendment(directory)
                            verify_plan(directory)
                            write_json_atomic(attempt/"result.json", {"status": "passed", "job": job,
                                "plan_sha256": checksum, "validation_amendment": amendment_identity(digest),
                                "fit_sha256": sha256(attempt/"fit.json"), "predict_sha256": sha256(attempt/"predict.json")})
                            outcome = "passed"
                        except BaseException as error:
                            write_json_atomic(attempt/"failure.json", {"type": type(error).__name__,
                                "message": str(error), "automatic_retry": False})
                            error_seen = error
                        finally:
                            write_json_atomic(attempt/"elapsed.json", {"outcome": outcome,
                                "wall_seconds": time.perf_counter()-started, "scope": "complete consumed attempt"})
                        if error_seen is None:
                            complete_attempt(attempt, job, checksum)
                            state["complete"].append(job["id"])
                            emit("completed", job=job["id"], completed=len(state["complete"]))
                            break
                        state["failures"].append(str(attempt.relative_to(directory)))
                        if not isinstance(error_seen, RuntimeError) or not readiness_failure(attempt):
                            raise error_seen
                        enforce_budget(state, amendment["recovery_limits"])
                        emit("readiness_retry", attempt=str(attempt.relative_to(directory)))
                        print("CPU readiness rejected this attempt; retained it. Cooling down and retrying the same job.", flush=True)
                        time.sleep(amendment["recovery_limits"]["cooldown_seconds"])
                verified = snapshot(directory, plan, amendment, digest)
                if len(verified["complete"]) != len(plan["jobs"]):
                    raise ValueError("Incomplete fixed panel")
        result = summarize(directory)
        record = {"status": "passed", "base_plan_sha256": checksum,
            "validation_amendment": amendment_identity(digest),
            "base_summary_sha256": sha256(directory/"summary.json"),
            "successful_jobs": len(plan["jobs"]), "failed_attempts": len(verified["failures"]),
            "historical_successes_preserved": len(amendment["historical_completed_jobs"]),
            "final_requires_same_validation_rule": RULE}
        path = directory/"validation-summary.json"
        if path.exists():
            if _read_json(path) != record:
                raise ValueError("Existing amendment summary differs")
        else:
            write_json_atomic(path, record)
        print(json.dumps({"status": "passed", "period": "confirmation", "summary": str(path)}, indent=2))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    directory = ROOT/_read_json(AMENDMENT)["output_directory"]
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    if args.check:
        with run_lock(directory):
            amendment, digest = verify_amendment(directory)
            plan, _, _ = verify_plan(directory, check_environment=False)
            state = snapshot(directory, plan, amendment, digest)
            enforce_budget(state, amendment["recovery_limits"])
        print(json.dumps({"status": "passed", "fitting_performed": False,
            "completed": len(state["complete"]), "failed_attempts": len(state["failures"]),
            "remaining": len(plan["jobs"])-len(state["complete"])}))
    else:
        from tools.run_development_experiments import startup_check
        startup_check()
        run(directory)


if __name__ == "__main__":
    main()
