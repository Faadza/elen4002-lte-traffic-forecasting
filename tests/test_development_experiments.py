"""Invented-data runner boundaries, saved arithmetic and resumable execution.

No supplied traffic is loaded or fitted by these checks. Numerical worker
checks use the same runtime CPU-control context as actual experiment workers.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from forecasting import development_plan
from tools import run_development_experiments as runner
from tools.development_store import begin_attempt, complete_attempt, load_success, sha256


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())


def invented_series(length=6132):
    t = np.arange(length, dtype=np.float64)
    values = 20 + 2*np.sin(2*np.pi*t/24) + t/1000
    flags = np.ones(length, dtype=bool)
    # A genuine gap invalidates this label but preserves its causal input.
    flags[4200] = False
    observed = values.copy()
    observed[~flags] = np.nan
    values[4200] = values[4199]
    return {"traffic": values, "traffic_observed": observed, "observed": flags,
            "timestamps": pd.date_range("2020-01-06", periods=length, freq="h")}


def save_series(directory, cell, data):
    path = directory / "series" / f"{cell}.npz"
    path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(path, **{k: v.to_numpy() if isinstance(v, pd.DatetimeIndex) else v
                    for k, v in data.items()})
    return {"path": f"series/{cell}.npz", "sha256": sha256(path)}


def put_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def reseal_plan(directory, plan):
    put_json(directory / "plan.json", plan)
    put_json(directory / "plan-checksum.json", {"sha256": sha256(directory / "plan.json")})


@pytest.fixture
def prepared(tmp_path, monkeypatch, config):
    from tools import check_prepared_data
    data = invented_series(8760)
    hourly = pd.concat([pd.DataFrame({"cell": cell, "ts": data["timestamps"],
        "traffic": data["traffic"], "traffic_observed": data["traffic_observed"],
        "observed": data["observed"]}) for cell in development_plan.TUNING_CELLS], ignore_index=True)
    report = {"tuning_cells": list(development_plan.TUNING_CELLS),
              "source_sha256": config["data"]["raw_sha256"]}
    monkeypatch.setattr(check_prepared_data, "load_validated_data", lambda _: (hourly, report))
    monkeypatch.setattr(runner, "environment_identity", lambda: ({"machine": "invented"}, {"recorded": True}))
    directory = tmp_path / "experiment"
    plan = runner.prepare(tmp_path / "invented-input", directory)
    return directory, plan, hourly


def test_prepare_physically_excludes_holdout_and_accepts_unequal_cohort(prepared, config, monkeypatch):
    directory, plan, hourly = prepared
    assert len(plan["jobs"]) == 864
    assert plan["stage_counts"] == {"baselines": 81, "ridge": 135, "direct": 162, "hybrid": 162, "lstm": 324}
    assert plan["holdout_forecasts_enabled"] is False
    assert not (directory / "jobs").exists()
    assert not (directory / "selections").exists()
    for cell in development_plan.TUNING_CELLS:
        data = runner.load_series(directory, plan, cell)
        assert all(len(v) == 6132 for v in data.values())
        assert data["timestamps"][-1] == pd.Timestamp("2020-09-17 11:00:00")
        first = plan["series"][cell]["folds"][0]
        assert first["counts"]["score"] == 671
        assert 4200 not in first["targets"]
        assert plan["series"][cell]["folds"][1]["counts"]["score"] == 672
    checked, _, _ = runner.verify_plan(directory)
    assert checked == plan


def test_prepare_does_not_overwrite_an_existing_directory(prepared):
    directory, plan, _ = prepared
    before = (directory / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        runner.prepare("unused", directory)
    assert (directory / "plan.json").read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("config_sha256", "a"*64), ("source_sha256", "a"*64),
    ("prepared_checklist_sha256", "a"*64), ("code_sha256", {}),
    ("jobs", []), ("stage_counts", {}), ("cpu_startup_profile", {}),
    ("holdout_forecasts_enabled", True), ("schema_version", 2),
])
def test_resume_rejects_incompatible_plan_even_with_recomputed_checksum(prepared, field, value):
    directory, plan, _ = prepared
    altered = copy.deepcopy(plan)
    altered[field] = value
    reseal_plan(directory, altered)
    with pytest.raises(ValueError, match="fixed code, data or development workload"):
        runner.verify_plan(directory)


def test_plan_checksum_rejects_unrecorded_changes(prepared):
    directory, plan, _ = prepared
    plan["created_at_utc"] = "altered"
    put_json(directory / "plan.json", plan)
    with pytest.raises(ValueError, match="Run plan changed"):
        runner.verify_plan(directory)


def test_resume_requires_same_machine_interpreter_and_packages(prepared, monkeypatch):
    directory, _, _ = prepared
    monkeypatch.setattr(runner, "environment_identity", lambda: ({"machine": "different"}, {}))
    with pytest.raises(ValueError, match="same machine"):
        runner.verify_plan(directory)


def test_plan_rechecks_common_targets_before_fitting(prepared):
    directory, plan, _ = prepared
    plan["series"][development_plan.TUNING_CELLS[0]]["folds"][0]["targets"].append(4200)
    reseal_plan(directory, plan)
    with pytest.raises(ValueError, match="target cohort"):
        runner.verify_plan(directory)


@pytest.mark.parametrize("change", ["extra_series", "unexpected_job", "path", "series_bytes"])
def test_resume_rejects_untracked_or_changed_evidence(prepared, change):
    directory, plan, _ = prepared
    cell = development_plan.TUNING_CELLS[0]
    if change == "extra_series":
        (directory / "series/extra.npz").write_bytes(b"extra")
    elif change == "unexpected_job":
        (directory / "jobs/unplanned").mkdir(parents=True)
    elif change == "path":
        plan["series"][cell]["path"] = "series/../outside.npz"
        reseal_plan(directory, plan)
    else:
        (directory / plan["series"][cell]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        runner.verify_plan(directory)


@pytest.mark.parametrize("change", ["holdout_length", "extra_array", "bad_mask", "genuine_mismatch", "irregular_clock"])
def test_worker_series_contract_cannot_be_bypassed_by_rehashing(tmp_path, change):
    data = invented_series(6133 if change == "holdout_length" else 6132)
    if change == "extra_array":
        data["extra"] = np.ones(6132)
    elif change == "bad_mask":
        data["observed"] = data["observed"].astype(int)
        data["observed"][0] = 2
    elif change == "genuine_mismatch":
        data["traffic_observed"][0] = np.nan
    elif change == "irregular_clock":
        data["timestamps"] = data["timestamps"].to_numpy()
        data["timestamps"][1] = data["timestamps"][0]
    entry = save_series(tmp_path, "invented-cell", data)
    with pytest.raises((ValueError, AssertionError)):
        runner.load_series(tmp_path, {"series": {"invented-cell": entry}}, "invented-cell")


def test_configuration_file_hash_is_checked_before_parsing(tmp_path, monkeypatch):
    (tmp_path / "run_config.yaml").write_text("not: the approved configuration\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="changing the configuration"):
        runner.read_config()


@pytest.fixture
def small_worker(tmp_path, config, monkeypatch):
    data = invented_series()
    plan = {"series": {"invented-cell": save_series(tmp_path, "invented-cell", data)}, "jobs": []}
    checksum = "e"*64
    monkeypatch.setattr(runner, "verify_plan", lambda _: (plan, config, checksum))
    def make_job(stage):
        job = copy.deepcopy(development_plan.build_jobs(config, stage)[0])
        job.update(cell="invented-cell", fit_end=280, score_start=280, score_end=310,
                   id=f"{job['candidate_id']}--invented-cell--f1")
        plan["jobs"].append(job)
        attempt, _ = begin_attempt(tmp_path, job, checksum)
        return job, attempt
    return tmp_path, plan, config, checksum, data, make_job


def test_real_workers_roundtrip_prefix_boundary_and_manual_errors(small_worker, monkeypatch):
    from forecasting import development_models
    directory, plan, config, checksum, data, make_job = small_worker
    original = development_models.fit_candidate
    captured = []
    def fit_spy(job, values, flags, clock, config, **kwargs):
        assert len(values) == len(flags) == len(clock) == job["fit_end"] == 280
        assert clock[-1] == data["timestamps"][279]
        captured.append(job["stage"])
        return original(job, values, flags, clock, config, **kwargs)
    monkeypatch.setattr(development_models, "fit_candidate", fit_spy)
    ridge_attempt = None
    dependency_info = None
    for stage in ("ridge", "direct", "hybrid"):
        job, attempt = make_job(stage)
        if stage == "hybrid":
            monkeypatch.setattr(runner, "ridge_dependency", lambda *args: (ridge_attempt, dependency_info))
            monkeypatch.setattr(development_models, "fit_ridge", lambda *args: pytest.fail("Hybrid refitted cached ridge"))
        runner.worker(directory, job["id"], attempt)
        result = runner.validate_result(attempt, job, plan, checksum, config)
        assert result["reload_predictions_exact"] and result["fitted_state_unchanged"]
        assert result["formal_timing_eligible"] is False
        assert result["controls_before"]["runtime_library_controls_verified"] is True
        assert result["controls_after"]["runtime_library_controls_verified"] is True
        assert not (attempt / "complete.json").exists()
        complete_attempt(attempt, job, checksum)
        if stage == "ridge":
            ridge_attempt = attempt
            (directory / "selections").mkdir()
            put_json(directory / "selections/ridge.json", {"selection": {"selected": {
                "candidate_id": job["candidate_id"], "parameters": job["parameters"]}}})
            dependency_info = {"job_id": job["id"], "attempt": str(attempt.relative_to(directory)),
                "completion_sha256": sha256(attempt / "complete.json"),
                "ridge_selection_sha256": sha256(directory / "selections/ridge.json")}
        else:
            assert result["settings"]["requested_trees"] == 32
        pair_mask = data["observed"][168:280] & data["observed"][:112]
        scale = np.abs(data["traffic_observed"][168:280] - data["traffic_observed"][:112])[pair_mask].mean()
        for record in result["records"]:
            frame = pd.read_csv(attempt / "predictions" / f"{record['candidate_id']}.csv", float_precision="round_trip")
            residual = frame.actual.to_numpy() - frame.prediction.to_numpy()
            assert record["scores"]["MAE"] == pytest.approx(np.abs(residual).mean())
            assert record["scores"]["RMSE"] == pytest.approx(np.sqrt(np.mean(residual**2)))
            assert record["scores"]["MASE"] == pytest.approx(np.abs(residual).mean()/scale)
            assert record["mase_scale"] == pytest.approx(scale)
            np.testing.assert_array_equal(frame.prediction, np.maximum(frame.prediction_raw, 0))
        if stage == "hybrid":
            assert len(result["records"]) == 4
            assert result["training"]["component_fit_counts"] == {"ridge": 0, "correction": 1}
            assert result["dependency"] == dependency_info
            assert (attempt / "components.npz").is_file()
    assert captured == ["ridge", "direct", "hybrid"]


@pytest.mark.parametrize("change", ["prediction", "actual", "target", "score", "record", "control", "reload", "formal"])
def test_saved_result_recalculation_rejects_changed_evidence(small_worker, change):
    directory, plan, config, checksum, _, make_job = small_worker
    job, attempt = make_job("baselines")
    runner.worker(directory, job["id"], attempt)
    result = json.loads((attempt / "result.json").read_text())
    cid = job["candidate_id"]
    csv = attempt / "predictions" / f"{cid}.csv"
    if change in ("prediction", "actual", "target"):
        frame = pd.read_csv(csv, float_precision="round_trip")
        field = {"prediction": "prediction", "actual": "actual", "target": "target_index"}[change]
        frame.loc[0, field] += 1
        frame.to_csv(csv, index=False)
    elif change == "score":
        result["evaluations"][cid]["scores"]["MAE"] += 1
    elif change == "record":
        result["records"][0]["scores"]["MAE"] += 1
    elif change == "control":
        result["controls_after"]["runtime_library_controls_verified"] = False
    elif change == "reload":
        result["reload_predictions_exact"] = False
    else:
        result["formal_timing_eligible"] = True
    put_json(attempt / "result.json", result)
    with pytest.raises((ValueError, AssertionError)):
        runner.validate_result(attempt, job, plan, checksum, config)


def test_worker_control_cleanup_failure_cannot_publish_pass(small_worker, monkeypatch):
    from tools import cpu_controls
    directory, _, _, _, _, make_job = small_worker
    job, attempt = make_job("baselines")
    real_context = cpu_controls.single_thread_cpu
    @contextmanager
    def failing_cleanup():
        with real_context() as controls:
            yield controls
        raise RuntimeError("invented cleanup failure")
    monkeypatch.setattr(cpu_controls, "single_thread_cpu", failing_cleanup)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        runner.worker(directory, job["id"], attempt)
    assert not (attempt / "result.json").exists()
    assert not (attempt / "complete.json").exists()


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    from tools import check_cpu_controls
    job = {"id": "invented-stage-job", "stage": "ridge"}
    plan, config, checksum = {}, {}, "d"*64
    monkeypatch.setattr(runner, "verify_current_identity", lambda *args: None)
    monkeypatch.setattr(runner, "verify_plan", lambda _: (plan, config, checksum))
    monkeypatch.setattr(development_plan, "build_jobs", lambda *args: [job])
    monkeypatch.setattr(check_cpu_controls, "machine_information", lambda: {"fixture": True})
    monkeypatch.setattr(runner, "selection", lambda *args: {"selection": {"selected": None}})
    validations = []
    def validate(attempt, *args):
        assert (attempt / "worker-output.txt").read_text() == "complete worker output"
        validations.append(attempt)
    monkeypatch.setattr(runner, "validate_result", validate)
    return tmp_path, job, checksum, validations


def test_stage_requires_host_confirmation_before_creating_attempt(lifecycle):
    directory, _, _, _ = lifecycle
    with pytest.raises(ValueError, match="host-ready"):
        runner.run_stage(directory, "ridge")
    assert not (directory / "jobs").exists()


def test_failed_worker_retained_explicit_retry_and_strict_success_reuse(lifecycle, monkeypatch):
    directory, job, checksum, validations = lifecycle
    launches = []
    def launch(_, __, attempt, lock):
        launches.append(attempt)
        (attempt / "worker.log").write_text("first failure" if len(launches) == 1 else "success")
        (attempt / "worker-output.txt").write_text("complete worker output")
        return 7 if len(launches) == 1 else 0
    monkeypatch.setattr(runner, "launch", launch)
    with pytest.raises(RuntimeError, match="Worker failed"):
        runner.run_stage(directory, "ridge", host_ready=True)
    first = launches[0]
    assert (first / "failure.json").is_file()
    assert json.loads((first / "elapsed.json").read_text())["outcome"] == "incomplete"
    assert not (first / "complete.json").exists()
    assert not validations
    old_files = {p.name: p.read_bytes() for p in first.iterdir()}
    with pytest.raises(ValueError, match="retry_failed"):
        runner.run_stage(directory, "ridge", resume=True, host_ready=True)
    runner.run_stage(directory, "ridge", resume=True, retry_failed=True, host_ready=True)
    assert len(launches) == 2 and launches[1].name == "attempt-0002"
    assert {p.name: p.read_bytes() for p in first.iterdir()} == old_files
    assert load_success(directory, job["id"], checksum)[0] == launches[1]
    assert json.loads((launches[1] / "elapsed.json").read_text())["outcome"] == "passed"
    runner.run_stage(directory, "ridge", resume=True, host_ready=True)
    assert len(launches) == 2
    assert validations == [launches[1], launches[1]]
    (launches[1] / "worker-output.txt").write_text("corrupt")
    with pytest.raises(ValueError, match="inventory differs"):
        runner.run_stage(directory, "ridge", resume=True, host_ready=True)
    assert len(launches) == 2


def test_validation_failure_or_interruption_never_seals_attempt(lifecycle, monkeypatch):
    directory, _, _, _ = lifecycle
    def interrupted(*args):
        raise KeyboardInterrupt("invented interruption")
    monkeypatch.setattr(runner, "launch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run_stage(directory, "ridge", host_ready=True)
    first = next((directory / "jobs").glob("*/attempt-*"))
    assert not (first / "complete.json").exists()
    assert json.loads((first / "failure.json").read_text())["type"] == "KeyboardInterrupt"
    assert (first / "elapsed.json").is_file()


def test_worker_exit_success_still_requires_saved_result_validation(lifecycle, monkeypatch):
    directory, _, _, _ = lifecycle
    monkeypatch.setattr(runner, "launch", lambda *args: 0)
    monkeypatch.setattr(runner, "validate_result", lambda *args: (_ for _ in ()).throw(ValueError("bad stored metrics")))
    with pytest.raises(ValueError, match="bad stored metrics"):
        runner.run_stage(directory, "ridge", host_ready=True)
    attempt = next((directory / "jobs").glob("*/attempt-*"))
    assert (attempt / "failure.json").exists()
    assert not (attempt / "complete.json").exists()


def test_launch_terminates_and_reaps_interrupted_child(tmp_path, monkeypatch):
    events = []
    class Child:
        count = 0
        def wait(self, timeout=None):
            self.count += 1
            events.append(("wait", timeout))
            if self.count == 1:
                raise KeyboardInterrupt
            if self.count == 2:
                raise subprocess.TimeoutExpired("invented", timeout)
            return -9
        def terminate(self):
            events.append(("terminate", None))
        def kill(self):
            events.append(("kill", None))
    def popen(arguments, **kwargs):
        assert kwargs["pass_fds"] == (123,)
        assert arguments[:3] == [sys.executable, "-m", "tools.run_development_experiments"]
        assert all(kwargs["env"][k] == v for k, v in runner.CPU_ENV.items())
        return Child()
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(KeyboardInterrupt):
        runner.launch(tmp_path, {"id": "fixture"}, tmp_path, 123)
    assert events == [("wait", None), ("terminate", None), ("wait", 5), ("kill", None), ("wait", None)]


@pytest.mark.parametrize("arguments", [
    ["--stage", "ridge"], ["--stage", "lstm"], ["--stage", "confirmation"],
    ["--stage", "final"], ["--stage", "all"],
    ["data", "--execute-development"], ["data", "--stage", "ridge", "--execute-development"],
    ["--stage", "ridge", "--execute-development", "--retry-failed"],
    ["data", "--host-ready"], ["--stage", "summarize", "--execute-development"],
])
def test_cli_never_implicitly_runs_fits(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["runner", "--output-dir", str(tmp_path), *arguments])
    monkeypatch.setattr(runner, "prepare", lambda *args: pytest.fail("Unexpected preparation"))
    monkeypatch.setattr(runner, "run_stage", lambda *args, **kwargs: pytest.fail("Unexpected fitting"))
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_cli_default_only_prepares_and_explicit_stage_runs_once(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runner, "prepare", lambda *args: calls.append(("prepare", args)) or {"stage_counts": {}})
    monkeypatch.setattr(runner, "run_stage", lambda *args, **kwargs: calls.append(("run", args, kwargs)) or {"selection": {"selected": None}})
    monkeypatch.setattr(sys, "argv", ["runner", "invented-data", "--output-dir", str(tmp_path)])
    runner.main()
    assert len(calls) == 1 and calls[0][0] == "prepare"
    assert json.loads(capsys.readouterr().out)["fitting_performed"] is False
    monkeypatch.setattr(sys, "argv", ["runner", "--output-dir", str(tmp_path), "--stage", "direct", "--execute-development", "--host-ready"])
    runner.main()
    assert len(calls) == 2 and calls[1][1][1] == "direct"
    assert calls[1][2] == {"resume": False, "retry_failed": False, "host_ready": True}


def test_startup_variables_checked_before_execution(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(RuntimeError, match="run_cpu.py"):
        runner.startup_check()


def test_launch_actual_child_inherits_run_lock_and_cpu_environment(tmp_path, monkeypatch):
    """A wrapper grandchild would lose the lock; probe the actual launched PID."""
    from tools.development_store import run_lock
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    (tools_dir / "__init__.py").write_text("")
    (tools_dir / "run_development_experiments.py").write_text('''
import json, os
fd = int(os.environ["INVENTED_LOCK_FD"])
print(json.dumps({"lock_inode": os.fstat(fd).st_ino,
                  "omp": os.environ["OMP_NUM_THREADS"],
                  "hash_seed": os.environ["PYTHONHASHSEED"]}))
''')
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "17")
    directory = tmp_path / "run"
    directory.mkdir()
    attempt = directory / "attempt"
    attempt.mkdir()
    with run_lock(directory) as lock:
        monkeypatch.setenv("INVENTED_LOCK_FD", str(lock))
        assert runner.launch(directory, {"id": "fixture"}, attempt, lock) == 0
    probe = json.loads((attempt / "worker.log").read_text())
    assert probe["lock_inode"] == (directory / ".run.lock").stat().st_ino
    assert probe["omp"] == "1" and probe["hash_seed"] == runner.CPU_ENV["PYTHONHASHSEED"]


@pytest.mark.parametrize("change", ["job", "plan", "number", "completed", "failed", "existing_result"])
def test_worker_rejects_nonfresh_or_mismatched_attempt_before_fit(small_worker, monkeypatch, change):
    from forecasting import development_models
    directory, _, _, _, _, make_job = small_worker
    job, attempt = make_job("ridge")
    metadata = json.loads((attempt / "attempt.json").read_text())
    if change == "job":
        metadata["job"]["parameters"]["alpha"] = 1
    elif change == "plan":
        metadata["plan_sha256"] = "c"*64
    elif change == "number":
        metadata["attempt_number"] = 2
    else:
        name = {"completed": "complete.json", "failed": "failure.json", "existing_result": "result.json"}[change]
        put_json(attempt / name, {})
    put_json(attempt / "attempt.json", metadata)
    monkeypatch.setattr(development_models, "fit_candidate", lambda *args, **kwargs: pytest.fail("Invalid attempt fitted"))
    with pytest.raises(ValueError, match="fresh assigned attempt"):
        runner.worker(directory, job["id"], attempt)


@pytest.fixture
def hybrid_evidence(small_worker, monkeypatch):
    """Two real small fits, one explicit selected-ridge dependency; no search."""
    directory, plan, config, checksum, data, make_job = small_worker
    ridge_job, ridge_attempt = make_job("ridge")
    runner.worker(directory, ridge_job["id"], ridge_attempt)
    runner.validate_result(ridge_attempt, ridge_job, plan, checksum, config)
    complete_attempt(ridge_attempt, ridge_job, checksum)
    (directory / "selections").mkdir()
    selection_path = directory / "selections/ridge.json"
    picked = {"selection": {"selected": {"candidate_id": ridge_job["candidate_id"],
              "parameters": ridge_job["parameters"]}}}
    put_json(selection_path, picked)
    dependency = {"job_id": ridge_job["id"], "attempt": str(ridge_attempt.relative_to(directory)),
        "completion_sha256": sha256(ridge_attempt / "complete.json"),
        "ridge_selection_sha256": sha256(selection_path)}
    monkeypatch.setattr(runner, "ridge_dependency", lambda *args: (ridge_attempt, dependency))
    job, attempt = make_job("hybrid")
    runner.worker(directory, job["id"], attempt)
    runner.validate_result(attempt, job, plan, checksum, config)
    return directory, plan, config, checksum, job, attempt, selection_path


@pytest.mark.parametrize("change", ["correction", "ridge", "component_targets", "scaler", "alpha", "dependency", "global_selection"])
def test_parent_rechecks_hybrid_addition_and_selected_ridge(hybrid_evidence, change):
    directory, plan, config, checksum, job, attempt, selection_path = hybrid_evidence
    if change in ("correction", "ridge", "component_targets"):
        with np.load(attempt / "components.npz", allow_pickle=False) as saved:
            components = {k: saved[k] for k in saved.files}
        name = {"correction": "correction_z", "ridge": "ridge_z", "component_targets": "targets"}[change]
        components[name][0] += 1
        np.savez(attempt / "components.npz", **components)
    elif change == "scaler":
        path = attempt / "model/model.json"
        model = json.loads(path.read_text())
        model["scaler"]["mean"] += 1
        put_json(path, model)
    elif change == "global_selection":
        picked = json.loads(selection_path.read_text())
        picked["selection"]["selected"]["candidate_id"] = "ridge-a1"
        picked["selection"]["selected"]["parameters"]["alpha"] = 1
        put_json(selection_path, picked)
    else:
        result = json.loads((attempt / "result.json").read_text())
        if change == "alpha":
            result["settings"]["alpha"] = 1
        else:
            result["dependency"]["completion_sha256"] = "a"*64
        put_json(attempt / "result.json", result)
    with pytest.raises((ValueError, AssertionError)):
        runner.validate_result(attempt, job, plan, checksum, config)


@pytest.fixture
def short_lstm_evidence(small_worker):
    directory, plan, config, checksum, _, make_job = small_worker
    config["models"]["lstm"].update(maximum_epochs=1, patience_epochs=1, inner_stopping_tail_hours=24)
    job, attempt = make_job("lstm")
    runner.worker(directory, job["id"], attempt)
    result = runner.validate_result(attempt, job, plan, checksum, config)
    return result, job, config


def test_lstm_worker_retains_real_effective_budget_and_contained_phases(short_lstm_evidence):
    result, job, config = short_lstm_evidence
    training = result["training"]
    assert training["effective_budgets"] == {"maximum_epochs": 1, "patience_epochs": 1, "inner_stopping_tail_hours": 24}
    assert training["selected_epochs"] == training["selection_epochs_run"] == 1
    assert training["stop_reason"] == "maximum_epochs"
    assert training["maximum_epochs_reached"] is True
    assert len(training["stopping_history"]) == len(training["refit_history"]) == 1
    phases = training["phase_timings"]
    assert 0 < phases["selection_seconds"] + phases["refit_seconds"] <= result["fit_seconds"]
    assert training["component_fit_counts"] == {"inner_selection": 1, "full_prefix_refit": 1}
    assert result["records"][0]["selected_epochs"] == 1
    assert result["formal_timing_eligible"] is False


@pytest.mark.parametrize("change", ["fit_end", "lookback", "job", "fit_interval", "counts", "fit_time", "width", "seed", "batch", "selected", "history", "refit", "budget", "phase_zero", "phase_overrun"])
def test_saved_lstm_effective_settings_and_phase_durations_are_checked(short_lstm_evidence, change):
    result, job, config = short_lstm_evidence
    altered = copy.deepcopy(result)
    settings, training = altered["settings"], altered["training"]
    if change == "fit_end":
        settings["fit_end"] += 1
    elif change == "lookback":
        settings["lookback"] = 24
    elif change == "job":
        settings["development_job"]["cell"] = "other-cell"
    elif change == "fit_interval":
        training["fit_interval"][1] += 1
    elif change == "counts":
        training["component_fit_counts"]["inner_selection"] = 2
    elif change == "fit_time":
        altered["fit_seconds"] = float("nan")
    elif change == "width":
        settings["hidden_units"] = 64
    elif change == "seed":
        settings["seed"] += 1
    elif change == "batch":
        settings["batch_size"] += 1
    elif change == "selected":
        training["selected_epochs"] = 2
    elif change == "history":
        training["stopping_history"] = []
    elif change == "refit":
        training["refit_history"] = []
    elif change == "budget":
        training["effective_budgets"]["maximum_epochs"] += 1
    elif change == "phase_zero":
        training["phase_timings"]["refit_seconds"] = 0
    else:
        training["phase_timings"]["selection_seconds"] = altered["fit_seconds"] + 1
    with pytest.raises(ValueError):
        runner.validate_training(altered, job, config)


def test_loaded_state_identity_checked_even_when_predictions_would_match(small_worker, monkeypatch):
    from forecasting import pilot_models
    directory, _, _, _, _, make_job = small_worker
    job, attempt = make_job("baselines")
    original = pilot_models.load_model
    def altered_load(path):
        model = original(path)
        model.settings["unexpected_setting"] = "different persisted state"
        return model
    monkeypatch.setattr(pilot_models, "load_model", altered_load)
    with pytest.raises(AssertionError, match="fitted state differs"):
        runner.worker(directory, job["id"], attempt)
    assert not (attempt / "result.json").exists()


def test_parent_detects_identity_drift_before_sealing_last_job(lifecycle, monkeypatch):
    directory, _, _, validations = lifecycle
    events = []
    def launch(_, __, attempt, lock):
        events.append("worker exit")
        (attempt / "worker-output.txt").write_text("complete worker output")
        return 0
    def drift(*args):
        events.append("identity check")
        assert validations
        raise ValueError("Code, plan or environment changed while the stage was running")
    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(runner, "verify_current_identity", drift)
    with pytest.raises(ValueError, match="changed while"):
        runner.run_stage(directory, "ridge", host_ready=True)
    assert events == ["worker exit", "identity check"]
    attempt = validations[0]
    assert not (attempt / "complete.json").exists()
    assert (attempt / "failure.json").exists()


@pytest.mark.parametrize("change", ["plan", "code", "environment"])
def test_current_identity_rechecks_recorded_provenance(prepared, monkeypatch, change):
    from tools import run_development_pilot
    directory, plan, _ = prepared
    checksum = sha256(directory / "plan.json")
    runner.verify_current_identity(directory, plan, checksum)
    if change == "plan":
        (directory / "plan.json").write_text("changed plan")
    elif change == "code":
        monkeypatch.setattr(run_development_pilot, "code_hashes", lambda: {"changed": "file"})
    else:
        monkeypatch.setattr(runner, "environment_identity", lambda: ({"new": "machine"}, {}))
    with pytest.raises(ValueError, match="changed while"):
        runner.verify_current_identity(directory, plan, checksum)


def test_stage_session_preserves_failed_and_retried_parent_walltime(tmp_path, monkeypatch):
    times = iter([10., 13., 20., 25.])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(times))
    with pytest.raises(KeyboardInterrupt):
        with runner.stage_session(tmp_path, "ridge", "a"*64):
            raise KeyboardInterrupt
    first = tmp_path / "sessions/session-0001"
    old = {p.name: p.read_bytes() for p in first.iterdir()}
    with runner.stage_session(tmp_path, "ridge", "a"*64):
        pass
    assert {p.name: p.read_bytes() for p in first.iterdir()} == old
    failed = json.loads((first / "finished.json").read_text())
    passed = json.loads((tmp_path / "sessions/session-0002/finished.json").read_text())
    assert failed["wall_seconds"] == 3 and failed["status"] == "failed_or_interrupted"
    assert failed["error_type"] == "KeyboardInterrupt"
    assert passed["wall_seconds"] == 5 and passed["status"] == "passed"


@pytest.mark.parametrize("missing", ["none", "attempt", "session", "all_sessions"])
def test_summary_retains_unknown_cost_and_does_not_double_count(lifecycle, missing):
    directory, job, checksum, _ = lifecycle
    attempt = directory / "jobs" / job["id"] / "attempt-0001"
    attempt.mkdir(parents=True)
    if missing != "attempt":
        put_json(attempt / "elapsed.json", {"outcome": "passed", "wall_seconds": 2.0})
    if missing != "all_sessions":
        for index, value in enumerate((3.0, 7.0), 1):
            session = directory / "sessions" / f"session-{index:04d}"
            session.mkdir(parents=True)
            put_json(session / "started.json", {"stage": "ridge", "plan_sha256": checksum})
            if missing != "session" or index == 1:
                put_json(session / "finished.json", {"wall_seconds": value, "status": "failed_or_interrupted" if index == 1 else "passed"})
    summary = runner.summarize(directory)
    assert summary["attempts_with_unavailable_elapsed_time"] == (missing == "attempt")
    assert summary["sessions_with_unavailable_elapsed_time"] == (missing == "session")
    assert summary["recorded_stage_timing_complete"] is (missing == "none")
    assert summary["recorded_attempt_wall_seconds"] == (0 if missing == "attempt" else 2.0)
    assert summary["recorded_stage_wall_seconds"] == (0 if missing == "all_sessions" else 3 if missing == "session" else 10)
    assert summary["formal_timing_eligible"] is False
    assert summary["primary_comparator"] is None
    assert summary["holdout_forecasts_evaluated"] is False
    assert runner.summarize(directory) == summary
