"""Invented calendars exercise preparation, worker audit and immutable resume.

Historical audit/host probes are explicitly replaced by test fixtures. No LTE
files or real native-readiness evidence are read or produced by these tests.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ensemble_study import common
from ensemble_study import development as completion
from tools.development_store import (_inventory, _read_json, begin_attempt,
    complete_attempt, load_success, sha256, write_json_atomic)


IDENTITY = {"cpu_model": "invented test CPU"}


def readiness(operator="automated", host_ready=False):
    return {"status": "passed", "operator": operator, "host_ready_assertion": host_ready,
        "checked_at_utc": "2020-01-01T00:00:00+00:00", "identity": IDENTITY,
        "machine": {"system": "Darwin", "cpu_model": {"returncode": 0, "stdout": IDENTITY["cpu_model"]},
            "power_source": {"returncode": 0, "stdout": "AC Power"},
            "power_profiles": {"returncode": 0, "stdout": "lowpowermode 0"}},
        "background": {"method": "two 1-second totalCPU samples", "samples_percent": [0., 0.],
            "maximum_allowed_percent": 20.0, "passed": True}}


@pytest.fixture
def environment(tmp_path, monkeypatch):
    original, old_l1 = tmp_path / "original", tmp_path / "historical-l1"
    original.mkdir()
    old_l1.mkdir()
    (original / "series").mkdir()
    n = np.arange(6132, dtype=np.float64)
    traffic = 35 + .001*n + 3*np.sin(n*.17) + 2*np.cos(n*.037)
    observed = np.ones(len(n), dtype=bool)
    observed[190] = False
    actual = traffic.copy()
    actual[190] = np.nan
    traffic[190] = traffic[189]
    clock = pd.date_range("2020-01-06", periods=len(n), freq="h")
    series = {}
    for cell in completion.TUNING_CELLS:
        path = original / "series" / f"{cell}.npz"
        np.savez(path, traffic=traffic, traffic_observed=actual, observed=observed, timestamps=clock.to_numpy())
        series[cell] = {"path": f"series/{cell}.npz", "sha256": sha256(path), "folds": [
            {"targets": list(range(start, end)), "counts": {"score": end-start}}
            for start, end in completion.FOLDS]}
    write_json_atomic(original / "plan.json", {"environment_identity": IDENTITY, "series": series})
    write_json_atomic(original / "development-summary.json", {"status": "invented fixture only"})
    write_json_atomic(old_l1 / "plan.json", {"environment_identity": IDENTITY})
    write_json_atomic(old_l1 / "l1-summary.json", {"status": "invented fixture only"})
    def audited(source, historical):
        assert source == original and historical == old_l1
        return {"status": "passed", "environment_identity": IDENTITY,
                "reused_counts": {"l2": 162, "l1": 54, "ridge": 27},
                "sources": {"original": {"directory": str(original), "files_sha256": _inventory(original)},
                            "l1": {"directory": str(old_l1), "files_sha256": _inventory(old_l1)}}}
    monkeypatch.setitem(sys.modules, "ensemble_study.evidence", SimpleNamespace(audit_sources=audited))
    actors = []
    def native(expected, operator="automated", host_ready=False):
        assert expected == IDENTITY
        actors.append((operator, host_ready))
        return readiness(operator, host_ready)
    monkeypatch.setattr(common, "native_readiness", native)
    publication = tmp_path / "publication.json"
    write_json_atomic(publication, {"verified": True, "commit": "a"*40, "selection_sha256": None,
        "protocol_sha256": sha256(common.PROTOCOL), "code_sha256": common.code_hashes("development")})
    return SimpleNamespace(original=original, historical=old_l1, output=tmp_path / "completion",
                           publication=publication, actors=actors)


def prepare(env):
    return completion.prepare(env.original, env.historical, env.output, env.publication)


def reseal(directory, plan):
    write_json_atomic(directory / "plan.json", plan, replace=True)
    write_json_atomic(directory / "plan-checksum.json", {"sha256": sha256(directory / "plan.json")}, replace=True)


@pytest.fixture
def fitted_attempt(environment):
    env = environment
    plan = prepare(env)
    checksum = sha256(env.output / "plan.json")
    job = plan["jobs"][0]
    attempt, cached = begin_attempt(env.output, job, checksum)
    assert cached is False
    write_json_atomic(attempt / "operator.json", {"operator": "automated", "host_ready_assertion": False})
    # A real numerical fit/reload of invented data; host fixture is not a claim
    # of native experiment execution or formal cost evidence.
    completion.worker(env.output, job["id"], attempt)
    return env, plan, checksum, job, attempt


def test_exact_four_missing_capacities_nine_cells_three_folds():
    jobs = completion.build_jobs()
    assert len(jobs) == len({j["id"] for j in jobs}) == 108
    assert {(j["condition"]["trees"], j["condition"]["leaves"]) for j in jobs} == set(common.MISSING)
    assert {(j["condition"]["trees"], j["condition"]["leaves"]) for j in jobs}.isdisjoint({(128,7), (32,31)})
    assert all(j["condition"]["objective"] == "l1" and j["condition"]["ridge_weight"] == 0 for j in jobs)
    assert all(j["fit_end"] == j["score_start"] and j["score_end"]-j["score_start"] == 672 for j in jobs)
    assert max(j["score_end"] for j in jobs) == 6132


def test_prepare_pins_full_sources_and_copies_data_without_fitting(environment):
    from ensemble_study import model
    with patch.object(model, "fit_selected", side_effect=AssertionError("Preparation cannot fit")):
        plan = prepare(environment)
    assert plan["maximum_new_successful_fits"] == 108
    assert plan["new_ridge_fits"] == plan["new_weight_replay_fits"] == 0
    assert not (environment.output / "jobs").exists()
    for entry in plan["series"].values():
        assert (environment.output / entry["path"]).read_bytes() == (environment.original / entry["path"]).read_bytes()
    completion.verify_plan(environment.output, check_native=False, full_sources=True)
    assert environment.actors == [("automated", False), ("automated", False)]
    with pytest.raises(ValueError, match="new output"):
        prepare(environment)


def test_prepare_rejects_unverified_publication_before_source_audit(environment):
    value = _read_json(environment.publication)
    value["verified"] = False
    write_json_atomic(environment.publication, value, replace=True)
    with pytest.raises(ValueError, match="Published-byte"):
        prepare(environment)
    assert not environment.output.exists()
    assert not environment.actors


def test_readiness_failure_prevents_creating_any_output(environment, monkeypatch):
    def blocked(*args, **kwargs):
        raise ValueError("Native CPU query unavailable")
    monkeypatch.setattr(common, "native_readiness", blocked)
    with pytest.raises(ValueError, match="CPU query unavailable"):
        prepare(environment)
    assert not environment.output.exists()


@pytest.mark.parametrize("change", ["job", "series", "source", "receipt", "protocol", "operator"])
def test_prepare_evidence_and_job_mutations_fail(environment, change):
    plan = prepare(environment)
    if change == "job":
        plan["jobs"][0]["condition"]["objective"] = "l2"
        reseal(environment.output, plan)
    elif change == "series":
        (environment.output / next(iter(plan["series"].values()))["path"]).write_bytes(b"changed")
    elif change == "source":
        (environment.original / "development-summary.json").write_text("{}")
    elif change == "receipt":
        (environment.publication).write_text("{}")
    elif change == "protocol":
        plan["protocol_sha256"] = "0"*64
        reseal(environment.output, plan)
    else:
        plan["readiness_before"]["operator"] = "pretend"
        reseal(environment.output, plan)
    with pytest.raises(ValueError):
        completion.verify_plan(environment.output, check_native=False)


def test_worker_real_invented_fit_reload_and_parent_validation(fitted_attempt):
    env, plan, checksum, job, attempt = fitted_attempt
    record = completion.validate_result(attempt, job, plan, checksum, reload_model=True)
    assert record["result"]["training"]["component_fit_counts"] == {"direct_lightgbm": 1}
    assert record["result"]["condition"]["objective"] == "l1"
    assert record["result"]["reload_predictions_exact"] is True
    assert len(record["frame"]) == 672
    assert record["frame"].seed.eq("").all()
    (attempt / "worker.log").write_text("invented worker finished\n")
    complete_attempt(attempt, job, checksum)
    assert load_success(env.output, job["id"], checksum)[0] == attempt
    with pytest.raises(ValueError, match="All 108"):
        completion.completed_records(env.output, reload_models=False)


@pytest.mark.parametrize("field", ["scores", "readiness", "operator", "fit_count", "objective"])
def test_saved_result_drift_rejected(fitted_attempt, field):
    _, plan, checksum, job, attempt = fitted_attempt
    result = _read_json(attempt / "result.json")
    if field == "scores": result["evaluation"]["scores"]["MAE"] += 1
    elif field == "readiness": result["controls_after"]["runtime_library_controls_verified"] = False
    elif field == "operator":
        result["readiness_after"]["operator"] = "user"
        result["readiness_after"]["host_ready_assertion"] = True
    elif field == "fit_count": result["training"]["training_target_count"] -= 1
    else: result["training"]["tree_parameters"]["objective"] = "regression"
    write_json_atomic(attempt / "result.json", result, replace=True)
    with pytest.raises(ValueError):
        completion.validate_result(attempt, job, plan, checksum)


def test_worker_rejects_operator_mismatch_before_fitting(environment):
    plan = prepare(environment)
    checksum = sha256(environment.output / "plan.json")
    job = plan["jobs"][0]
    attempt, _ = begin_attempt(environment.output, job, checksum)
    write_json_atomic(attempt / "operator.json", {"operator": "user", "host_ready_assertion": True})
    with pytest.raises(ValueError, match="operator differs"):
        completion.worker(environment.output, job["id"], attempt, operator="automated")
    assert not (attempt / "model").exists()


def test_sealed_success_cannot_change_or_be_refitted(fitted_attempt):
    env, plan, checksum, job, attempt = fitted_attempt
    complete_attempt(attempt, job, checksum)
    same, cached = begin_attempt(env.output, job, checksum, resume=True)
    assert same == attempt and cached is True
    with pytest.raises(ValueError, match="fresh assigned"):
        completion.worker(env.output, job["id"], attempt)
    with pytest.raises(ValueError, match="explicit resume"):
        begin_attempt(env.output, job, checksum)


def test_runner_current_actor_can_change_and_resume_keeps_original_operator(environment, monkeypatch):
    env = environment
    # Scope one-job control flow while leaving the production scheduler's 108
    # jobs verified separately. No condition or model settings are changed.
    first = completion.build_jobs()[0]
    monkeypatch.setattr(completion, "build_jobs", lambda: [copy.deepcopy(first)])
    monkeypatch.setattr(completion, "MAXIMUM_FITS", 1)
    original_protocol = common.protocol
    monkeypatch.setattr(common, "protocol", lambda: {**original_protocol(), "maximum_new_development_successful_fits": 1})
    prepare(env)  # The preparation operator is automated.
    calls = []
    def inline_launch(directory, job, attempt, fd, operator, host_ready):
        calls.append((operator, host_ready))
        completion.worker(directory, job["id"], attempt, operator=operator, host_ready=host_ready)
        (attempt / "worker.log").write_text("invented child closed log\n")
        return 0
    monkeypatch.setattr(completion, "launch", inline_launch)
    completed = completion.run(env.output, operator="user", host_ready=True)
    assert completed["new_successful_fit_count"] == 1
    assert calls == [("user", True)]
    attempt = env.output / "jobs" / first["id"] / "attempt-0001"
    original_files = _inventory(attempt)
    assert _read_json(attempt / "operator.json") == {"operator": "user", "host_ready_assertion": True}
    assert _read_json(env.output / "plan.json")["readiness_before"]["operator"] == "automated"
    again = completion.run(env.output, resume=True, operator="automated")
    assert again["new_successful_fit_count"] == 1
    assert calls == [("user", True)]
    assert _inventory(attempt) == original_files


def test_failure_needs_explicit_retry_and_preserves_attempt(environment, monkeypatch):
    env = environment
    first = completion.build_jobs()[0]
    monkeypatch.setattr(completion, "build_jobs", lambda: [copy.deepcopy(first)])
    monkeypatch.setattr(completion, "MAXIMUM_FITS", 1)
    original_protocol = common.protocol
    monkeypatch.setattr(common, "protocol", lambda: {**original_protocol(), "maximum_new_development_successful_fits": 1})
    prepare(env)
    def fail(directory, job, attempt, fd, operator, host_ready):
        (attempt / "worker.log").write_text("invented failure\n")
        return 1
    monkeypatch.setattr(completion, "launch", fail)
    with pytest.raises(RuntimeError, match="Worker failed"):
        completion.run(env.output)
    attempt = env.output / "jobs" / first["id"] / "attempt-0001"
    before = _inventory(attempt)
    with pytest.raises(ValueError, match="retry_failed"):
        completion.run(env.output, resume=True)
    def succeed(directory, job, attempt, fd, operator, host_ready):
        completion.worker(directory, job["id"], attempt, operator=operator, host_ready=host_ready)
        (attempt / "worker.log").write_text("invented recovered worker\n")
        return 0
    monkeypatch.setattr(completion, "launch", succeed)
    result = completion.run(env.output, resume=True, retry_failed=True)
    assert result["attempt_count"] == 2 and result["failed_or_incomplete_attempt_count"] == 1
    assert _inventory(attempt) == before


@pytest.mark.parametrize("operator,ready", [("user", False), ("automated", True), ("pretend", False)])
def test_operator_assertion_cannot_be_fabricated(operator, ready):
    with pytest.raises(ValueError):
        completion._operator(operator, ready)


def test_launcher_uses_sequential_fresh_worker_and_inherited_lock(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    child = SimpleNamespace(wait=lambda: 0)
    with patch.object(completion.subprocess, "Popen", return_value=child) as process:
        assert completion.launch(tmp_path, {"id": "one"}, attempt, 123, "user", True) == 0
    arguments, options = process.call_args.args[0], process.call_args.kwargs
    assert arguments[:3] == [sys.executable, "-m", "ensemble_study.development"]
    assert "--host-ready" in arguments and arguments[arguments.index("--operator")+1] == "user"
    assert options["pass_fds"] == (123,)
    assert all(options["env"][key] == value for key, value in completion.CPU_ENV.items())
    assert options["stdout"].closed
