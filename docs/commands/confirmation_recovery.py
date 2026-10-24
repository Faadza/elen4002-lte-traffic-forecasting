"""Bounded user-operated recovery around the unchanged confirmation runner.

This supervisor never fits, scores, edits an attempt, or accepts a failed
readiness check. Every retry is a fresh attempt made by the original runner.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.development_store import (_read_json, _inventory, _attempts,
    load_success, run_lock, sha256)

CONFIG = ROOT / "docs/commands/confirmation-recovery-v1.json"
RECEIPT = ROOT.parent / "context/reconciliation-review/confirmation-recovery-publication-2026-10-09.json"
BACKGROUND_ERROR = "ValueError: Background workload exceeds the declared readiness check or is unavailable"


class RecoveryInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Recovery interrupted by signal {signum}; no automatic restart")


def interrupt(signum, frame):
    raise RecoveryInterrupted(signum)


def verify_inputs(config, publication):
    """Read-only pins, independent of restricted native host queries."""
    if publication.get("verified") is not True:
        raise ValueError("Recovery publication has not been verified")
    for name, digest in publication["payload_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Recovery publication differs: {name}")
    for required in (str(CONFIG.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT)),
                     "docs/commands/resume_ensemble_confirmation_mac.sh"):
        if required not in publication["payload_sha256"]:
            raise ValueError("Incomplete recovery publication")
    directory = ROOT / config["output_directory"]
    if sha256(directory / "plan.json") != config["plan_sha256"]:
        raise ValueError("Frozen confirmation plan changed")
    plan = _read_json(directory / "plan.json")
    if _read_json(directory / "plan-checksum.json") != {"sha256": config["plan_sha256"]}:
        raise ValueError("Plan checksum record differs")
    if (plan["period"] != "confirmation" or plan["publication_commit"] != config["evaluation_commit"]
            or plan["selection_sha256"] != config["selection_sha256"]
            or sha256(plan["selection_file"]) != config["selection_sha256"]):
        raise ValueError("Frozen selection or period changed")
    path = Path(plan["publication_record"]["path"])
    if (sha256(path) != config["evaluation_receipt_sha256"]
            or sha256(path) != plan["publication_record"]["sha256"]):
        raise ValueError("Existing evaluation receipt changed")
    original = _read_json(path)
    if original["code_sha256"] != plan["code_sha256"]:
        raise ValueError("Runtime manifest differs")
    for name, digest in plan["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen runtime changed: {name}")
    directories = {Path(n).parts[0] for n in plan["code_sha256"]}
    current = {str(p.relative_to(ROOT)) for d in directories for p in (ROOT/d).glob("*.py")}
    expected = {n for n in plan["code_sha256"] if n.endswith(".py")}
    if current != expected or sha256(ROOT/"ensemble_study/protocol.json") != original["protocol_sha256"]:
        raise ValueError("Runtime membership or protocol changed")
    if (sha256(ROOT/config["handoff_path"]) != config["handoff_sha256"]
            or sys.executable != config["python_executable"]):
        raise ValueError("Use the published handoff and original interpreter")
    return directory, plan


def readiness_failure(attempt):
    """Classify only the precise recorded worker exception; never stdout alone."""
    failure = _read_json(attempt / "failure.json")
    if failure.get("type") != "RuntimeError" or failure.get("automatic_retry") is not False:
        return False
    for phase in ("fit", "predict"):
        log = attempt / f"{phase}.log"
        expected = f"{phase} worker failed (1); inspect {log}"
        if failure.get("message") != expected or not log.is_file():
            continue
        text = log.read_text()
        return (text.rstrip().endswith(BACKGROUND_ERROR)
                and 'ensemble_study/common.py' in text
                and "in native_readiness" in text and "in verify_saved_readiness" in text
                and ("readiness_before = native_readiness(" in text
                     or "readiness_after = native_readiness(" in text))
    return False


def snapshot(directory, plan, digest):
    """Validate complete inventories without computing partial model rankings."""
    jobs = {j["id"]: j for j in plan["jobs"]}
    jobdir = directory / "jobs"
    if jobdir.exists() and any(p.name not in jobs for p in jobdir.iterdir()):
        raise ValueError("Unexpected job directory")
    state = {"complete": [], "failures": [], "attempts": {}, "counts": {}, "next_job": None}
    for job_id, job in jobs.items():
        success = load_success(directory, job_id, digest)
        attempts = _attempts(directory, job_id)
        state["counts"][job_id] = len(attempts)
        for attempt in attempts:
            if _read_json(attempt/"attempt.json")["job"] != job:
                raise ValueError("Attempt settings differ from frozen job")
            relative = str(attempt.relative_to(directory))
            if (attempt/"failure.json").exists():
                if not readiness_failure(attempt):
                    raise ValueError(f"Unrecognized failure; manual review required: {attempt}")
                state["failures"].append(relative)
            elif not (attempt/"complete.json").exists():
                raise ValueError(f"Interrupted or active attempt; manual review required: {attempt}")
            state["attempts"][relative] = _inventory(attempt)
        if success:
            if state["next_job"] is not None:
                raise ValueError("Completed jobs do not form the runner's sequential prefix")
            state["complete"].append(job_id)
        elif state["next_job"] is None:
            state["next_job"] = job_id
    return state


def verify_transition(before, after, returncode):
    for name, hashes in before["attempts"].items():
        if after["attempts"].get(name) != hashes:
            raise ValueError("An existing attempt changed during recovery")
    if after["complete"][:len(before["complete"])] != before["complete"]:
        raise ValueError("Previously successful jobs changed")
    new_failures = set(after["failures"]) - set(before["failures"])
    new_attempts = set(after["attempts"]) - set(before["attempts"])
    new_successes = len(after["complete"]) - len(before["complete"])
    if len(new_attempts) != new_successes + len(new_failures):
        raise ValueError("Unexpected attempt creation; stop for review")
    if returncode == 0:
        if after["next_job"] is not None or new_failures:
            raise ValueError("Runner returned success without completing the fixed panel")
        return None
    if returncode != 1 or len(new_failures) != 1:
        raise ValueError("Stop on interruption, unknown error, or missing/multiple new failures")
    failure = next(iter(new_failures))
    if Path(failure).parent.name != after["next_job"]:
        raise ValueError("New failure does not match the next incomplete job")
    return failure


def session_snapshot(directory, digest):
    sessions = directory/"sessions"
    result = {}
    if not sessions.exists():
        return result
    paths = sorted(sessions.iterdir())
    if [p.name for p in paths] != [f"session-{i:04d}" for i in range(1, len(paths)+1)]:
        raise ValueError("Unexpected runner session history")
    for path in paths:
        started = _read_json(path/"started.json")
        finished = _read_json(path/"finished.json")
        if (started.get("plan_sha256") != digest or started.get("stage") != "confirmation"
                or _read_json(path/"operator.json") != {"operator": "user", "host_ready_assertion": True}
                or finished.get("status") not in ("passed", "failed_or_interrupted")):
            raise ValueError("Runner session metadata differs")
        result[path.name] = {"files": _inventory(path), "status": finished["status"]}
    return result


def verify_sessions(before, after, returncode):
    if (any(after.get(name) != files for name, files in before.items())
            or len(set(after)-set(before)) != 1):
        raise ValueError("Expected exactly one new runner session; concurrent/unrecorded run detected")
    name = next(iter(set(after)-set(before)))
    expected_status = "passed" if returncode == 0 else "failed_or_interrupted"
    if after[name]["status"] != expected_status:
        raise ValueError("Runner session outcome disagrees with process exit; stop for review")
    return name


def verify_parent_failure(log, attempt):
    message = _read_json(attempt/"failure.json")["message"]
    if not log.read_text().rstrip().endswith(f"RuntimeError: {message}"):
        raise ValueError("This invocation did not report the new worker failure; stop for review")


def enforce_budget(state, config):
    job = state["next_job"]
    if job is not None and state["counts"][job] >= config["maximum_attempts_per_job"]:
        raise RuntimeError(f"Retry limit reached for {job}; preserve evidence and investigate workload")
    if job is not None and len(state["failures"]) >= config["maximum_failed_attempts_total"]:
        raise RuntimeError("Confirmation recovery failure limit reached; stop for review")


def wait_for_quiet(config, emit, *, sample, sleep=time.sleep, clock=time.monotonic):
    """Later diagnostic readings never replace a worker's failed readiness."""
    deadline = clock() + config["maximum_quiet_wait_seconds"]
    while clock() < deadline:
        values = [sample(interval=1) for _ in range(2)]
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not 0 <= v <= 100 for v in values):
            raise ValueError("CPU load could not be measured; stop rather than assume readiness")
        passed = all(v <= config["maximum_cpu_percent"] for v in values)
        emit("cpu_wait", samples_percent=values, quiet=passed)
        print(f"CPU readings: {values[0]:.1f}%, {values[1]:.1f}% — "
              + ("quiet; ready to resume." if passed else "waiting for a quiet Mac."), flush=True)
        if passed:
            return
        sleep(min(config["poll_seconds"], max(0, deadline-clock())))
    raise RuntimeError("CPU stayed busy for the maximum quiet wait; no further attempt launched")


def launch(command, directory, emit):
    """Keep the original runner visible; stop its full process group on Ctrl-C."""
    with (directory/"runner.log").open("x") as log:
        child = None
        try:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True)
            emit("runner_started", pid=child.pid, command=command)
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return child.wait()
        except BaseException:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise


def recover(config, directory, plan, session, emit):
    import psutil
    invocation = 0
    while True:
        with run_lock(directory):
            verify_inputs(config, _read_json(RECEIPT))
            before = snapshot(directory, plan, config["plan_sha256"])
            before_sessions = session_snapshot(directory, config["plan_sha256"])
        enforce_budget(before, config)
        print(f"Verified {len(before['complete'])}/{len(plan['jobs'])} completed jobs; "
              f"{len(before['failures'])} recorded readiness failures.", flush=True)
        wait_for_quiet(config, emit, sample=psutil.cpu_percent)
        # Recheck after waiting: another manual invocation must not be mistaken
        # for work performed by this supervisor.
        with run_lock(directory):
            verify_inputs(config, _read_json(RECEIPT))
            if (snapshot(directory, plan, config["plan_sha256"]) != before
                    or session_snapshot(directory, config["plan_sha256"]) != before_sessions):
                raise ValueError("Run state changed while waiting; stop for review")
        invocation += 1
        logdir = session/f"invocation-{invocation:04d}"
        logdir.mkdir()
        code = launch(["/bin/bash", str(ROOT/config["handoff_path"]), "--retry-failed"], logdir, emit)
        with run_lock(directory):
            verify_inputs(config, _read_json(RECEIPT))
            after = snapshot(directory, plan, config["plan_sha256"])
            after_sessions = session_snapshot(directory, config["plan_sha256"])
        runner_session = verify_sessions(before_sessions, after_sessions, code)
        failure = verify_transition(before, after, code)
        if failure is not None:
            verify_parent_failure(logdir/"runner.log", directory/failure)
        emit("runner_finished", returncode=code, completed=len(after["complete"]),
             failed_attempt_count=len(after["failures"]), newly_failed_attempt=failure,
             runner_session=runner_session)
        if failure is None:
            summary = _read_json(directory/"summary.json")
            if (summary.get("status") != "passed" or summary.get("plan_sha256") != config["plan_sha256"]
                    or summary.get("selection_sha256") != config["selection_sha256"]):
                raise ValueError("Runner did not produce the frozen confirmation summary")
            emit("passed", summary_sha256=sha256(directory/"summary.json"))
            print("Confirmation passed. Its summary is ready for independent review.", flush=True)
            return
        enforce_budget(after, config)
        emit("external_retry_scheduled", attempt=failure, cooldown_seconds=config["cooldown_seconds"])
        print(f"Readiness rejection recorded at {failure}. Waiting {config['cooldown_seconds']} seconds "
              "before checking CPU and resuming; successful jobs will be reused.", flush=True)
        time.sleep(config["cooldown_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Verify pins and stored attempts only; never execute")
    mode.add_argument("--run", action="store_true", help="User-operated bounded recovery; original native checks apply")
    args = parser.parse_args()
    config = _read_json(CONFIG)
    directory, plan = verify_inputs(config, _read_json(RECEIPT))
    if args.check:
        with run_lock(directory):
            state = snapshot(directory, plan, config["plan_sha256"])
        enforce_budget(state, config)
        print(json.dumps({"status": "passed", "fitting_performed": False,
            "completed": len(state["complete"]), "failed_attempts": len(state["failures"]),
            "remaining": len(plan["jobs"])-len(state["complete"])}))
        return
    logs = directory.parent/"confirmation-recovery"
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupt)
    with run_lock(logs):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        session = logs/f"{stamp}-{os.getpid()}"
        session.mkdir()
        with (session/"events.jsonl").open("x") as stream:
            def emit(event, **data):
                stream.write(json.dumps({"event": event, "at_utc": datetime.now(timezone.utc).isoformat(),
                    **data}, sort_keys=True, allow_nan=False)+"\n")
                stream.flush()
            emit("started", operator="user", plan_sha256=config["plan_sha256"],
                 recovery_publication_sha256=sha256(RECEIPT), limits=config,
                 worker_failed_samples_available=False)
            try:
                recover(config, directory, plan, session, emit)
            except BaseException as error:
                emit("stopped", type=type(error).__name__, message=str(error))
                raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Recovery interrupted; all attempts retained. No automatic restart.", file=sys.stderr)
        sys.exit(130)
    except RecoveryInterrupted as error:
        print(str(error), file=sys.stderr)
        sys.exit(128+error.signum)
    except Exception as error:
        print(f"Recovery stopped: {error}", file=sys.stderr)
        sys.exit(1)
