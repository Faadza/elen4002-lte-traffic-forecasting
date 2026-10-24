"""Final evaluation under the original scientific plan and published validation amendment.

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

CONFIG = ROOT/"docs/commands/final-evaluation-v1.json"
RECEIPT = ROOT.parent/"context/reconciliation-review/final-evaluation-publication-2026-10-09.json"


def inventory_digest(path):
    return hashlib.sha256(json.dumps(_inventory(path), sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def amendment_identity(digest):
    return {"id": RULE, "plan_sha256": digest,
            "publication_receipt_sha256": sha256(RECEIPT)}


def verify_publication():
    config, receipt = _read_json(CONFIG), _read_json(RECEIPT)
    if receipt.get("verified") is not True:
        raise ValueError("Final publication has not been verified")
    for name, digest in receipt["payload_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Final publication differs: {name}")
    required = {str(CONFIG.relative_to(ROOT)), "docs/commands/run_ensemble_final_mac.sh"}
    required.update(str(p.relative_to(ROOT)) for p in (ROOT/"final_stage").glob("*.py"))
    if not required <= receipt["payload_sha256"].keys():
        raise ValueError("Unpublished final source")
    for name, digest in config["effective_dependencies_sha256"].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f"Frozen dependency differs: {name}")
    from validation_revision.continuation import verify_amendment as verify_confirmation_amendment
    confirmation = ROOT/config["confirmation_directory"]
    verify_confirmation_amendment(confirmation)
    for name, digest in config["confirmation_sha256"].items():
        if sha256(confirmation/name) != digest:
            raise ValueError(f"Confirmation evidence changed: {name}")
    verify_confirmation_audits(config, receipt)
    summary = _read_json(confirmation/"summary.json")
    amended = _read_json(confirmation/"validation-summary.json")
    if (summary["status"] != "passed" or summary["period"] != "confirmation"
            or amended["status"] != "passed" or amended["successful_jobs"] != 741
            or amended["base_summary_sha256"] != config["confirmation_sha256"]["summary.json"]
            or amended["final_requires_same_validation_rule"] != RULE
            or summary["primary_model"] != config["primary_model"]
            or summary["primary_comparator"] != config["primary_comparator"]
            or config["rule"] != RULE or config["rtol"] != TOLERANCE or config["atol"] != TOLERANCE
            or sys.executable != config["python_executable"]):
        raise ValueError("Unverified confirmation freeze, interpreter or validation rule")
    return config


def verify_confirmation_audits(config, receipt):
    required_audits = {"docs/l1-l2-confirmation-results/arithmetic-audit.json",
                       "docs/l1-l2-confirmation-results/integrity-audit.json"}
    if set(config["independent_audits_sha256"]) != required_audits:
        raise ValueError("Both independent confirmation audits are required")
    for name, digest in config["independent_audits_sha256"].items():
        path = ROOT/name
        if (receipt["payload_sha256"].get(name) != digest
                or sha256(path) != digest or _read_json(path).get("status") != "passed"):
            raise ValueError("Published confirmation audit did not pass")


def validation_plan(directory, base, config):
    if (base["period"] != "final" or base["primary_model"] != config["primary_model"]
            or base["primary_comparator"] != config["primary_comparator"]
            or base["selection_sha256"] != config["selection_sha256"]
            or len(base["jobs"]) != 1539 or sum(j["predict"] for j in base["jobs"]) != 741
            or base["confirmation_freeze"]["sha256"] != config["confirmation_sha256"]["summary.json"]):
        raise ValueError("Final scientific plan differs from the complete frozen design")
    return {"schema_version": 1, "rule": RULE, "rtol": TOLERANCE, "atol": TOLERANCE,
        "output_directory": str(directory.relative_to(ROOT)),
        "base_plan_sha256": sha256(directory/"plan.json"),
        "cost_plan_sha256": sha256(directory/"cost-plan.json"),
        "configuration_sha256": sha256(CONFIG), "publication_receipt_sha256": sha256(RECEIPT),
        "confirmation_sha256": config["confirmation_sha256"],
        "historical_attempts": {}, "historical_completed_jobs": [],
        "audited_completed_lstm_jobs": [], "diagnosed_failure_job": None,
        "diagnosed_failure_attempt": None, "recovery_limits": config["recovery_limits"],
        "independent_final_fits": True, "settings_retuned": False}


def verify_amendment(directory):
    config = verify_publication()
    directory = Path(directory)
    if directory != ROOT/config["output_directory"]:
        raise ValueError("Unexpected final output directory")
    base = _read_json(directory/"plan.json")
    expected = validation_plan(directory, base, config)
    path = directory/"validation-plan.json"
    if _read_json(path) != expected:
        raise ValueError("Final validation plan changed")
    return expected, sha256(path)


def prepare_final(*, operator, host_ready):
    from ensemble_study.evaluation_plan import prepare
    execution_identity(operator, host_ready)
    config = verify_publication()
    directory = ROOT/config["output_directory"]
    confirmation = ROOT/config["confirmation_directory"]
    # These checks precede final-data export and any final fitting.
    from validation_revision.continuation import snapshot as confirmation_snapshot
    from validation_revision.continuation import verify_amendment as confirmation_amendment
    amendment, digest = confirmation_amendment(confirmation)
    confirmed, _, _ = verify_plan(confirmation)
    with run_lock(confirmation):
        state = confirmation_snapshot(confirmation, confirmed, amendment, digest)
        if len(state["complete"]) != 741:
            raise ValueError("Confirmation is incomplete")
    if not directory.exists():
        prepare(config["data_directory"], config["original_directory"], directory, "final",
            config["base_evaluation_commit"], config["selection_file"], config["selection_sha256"],
            config["base_evaluation_receipt"], confirmation, operator=operator, host_ready=host_ready)
    with run_lock(directory):
        base, _, _ = verify_plan(directory)
        path = directory/"validation-plan.json"
        expected = validation_plan(directory, base, config)
        if not path.exists():
            if (directory/"jobs").exists() or (directory/"sessions").exists():
                raise ValueError("Final work predates this validation plan")
            write_json_atomic(path, expected)
        verify_amendment(directory)
    print(json.dumps({"status": "prepared", "period": "final", "jobs": 1539,
        "prediction_jobs": 741, "fitting_performed": False}, indent=2))


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
    module = "final_stage.worker" if phase == "predict" and job["method"] == "lstm" else "ensemble_study.evaluation_run"
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


def run(directory, *, operator, host_ready):
    import psutil
    execution = execution_identity(operator, host_ready)
    if not (directory/"plan.json").is_file() or not (directory/"validation-plan.json").is_file():
        raise ValueError("Prepare and verify both final plans before execution")
    with run_lock(directory) as lock_fd:
        amendment, digest = verify_amendment(directory)
        plan, config, checksum = verify_plan(directory)
        state = snapshot(directory, plan, amendment, digest)
        with stage_session(directory, "final", checksum) as session:
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
                            print(f"final {index}/{len(plan['jobs'])}: {job['id']}", flush=True)
                            for phase in (("fit", "predict") if job["predict"] else ("fit",)):
                                code = launch(directory, job, attempt, phase, lock_fd)
                                if code != 0:
                                    raise RuntimeError(f"{phase} worker failed ({code}); inspect {attempt/(phase+'.log')}")
                            result = validate_amended_result(attempt, job, plan, checksum, amendment, digest)
                            if job["repeat"] > 1 and job["method"] != "lstm":
                                first_id = job["id"].rsplit("--r", 1)[0]+"--r1"
                                first, _ = load_success(directory, first_id, checksum)
                                if _read_json(first/"fit.json")["model_state_sha256"] != result["fit"]["model_state_sha256"]:
                                    raise ValueError("Repeated deterministic fit changed fitted state")
                            verify_amendment(directory)
                            verify_plan(directory)
                            write_json_atomic(attempt/"result.json", {"status": "passed", "job": job,
                                "plan_sha256": checksum, "validation_amendment": amendment_identity(digest),
                                "fit_sha256": sha256(attempt/"fit.json"), "predict_sha256": sha256(attempt/"predict.json") if job["predict"] else None})
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
            "all_final_lstm_use_same_validation_rule": RULE}
        path = directory/"validation-summary.json"
        if path.exists():
            if _read_json(path) != record:
                raise ValueError("Existing amendment summary differs")
        else:
            write_json_atomic(path, record)
        print(json.dumps({"status": "passed", "period": "final", "summary": str(path)}, indent=2))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--operator", choices=("automated", "user"))
    parser.add_argument("--host-ready", action="store_true")
    args = parser.parse_args()
    if args.check and (args.operator or args.host_ready):
        parser.error("Read-only check does not assert an execution operator")
    if not args.check and not args.operator:
        parser.error("Preparation and execution require the actual operator")
    directory = ROOT/_read_json(CONFIG)["output_directory"]
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    if args.check:
        config = verify_publication()
        if not directory.exists():
            print(json.dumps({"status": "passed", "scope": "published_final_preparation_preflight",
                "fitting_performed": False, "final_directory_created": False,
                "jobs_after_preparation": 1539, "prediction_jobs": 741}))
            return
        if not (directory/"plan.json").is_file() or not (directory/"validation-plan.json").is_file():
            raise ValueError("Final preparation is incomplete; run --prepare to verify and seal it before execution")
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
        if args.prepare:
            prepare_final(operator=args.operator, host_ready=args.host_ready)
        else:
            run(directory, operator=args.operator, host_ready=args.host_ready)


if __name__ == "__main__":
    main()
