"""Check pilot boundaries with invented arrays and stub jobs, never model fits."""
from contextlib import contextmanager
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools import run_development_pilot as runner


def install_module(monkeypatch, name, **functions):
    module = ModuleType(name)
    module.__dict__.update(functions)
    monkeypatch.setitem(sys.modules, name, module)


def test_fixed_jobs_keep_all_models_repeats_and_seeds():
    jobs = runner.pilot_jobs()
    assert len(jobs) == 15
    assert len({job["id"] for job in jobs}) == 15
    assert tuple(dict.fromkeys(job["method"] for job in jobs)) == runner.METHODS
    for method in runner.METHODS:
        subset = [job for job in jobs if job["method"] == method]
        assert [job["repeat"] for job in subset] == ([1] if method in runner.BASELINES else [1, 2, 3])
        assert [job["seed"] for job in subset] == ([2026, 2027, 2028] if method == "lstm"
                                                   else [None] * len(subset))


def test_existing_directory_is_preserved_before_controls(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    previous = output / "previous.json"
    previous.write_text("retain this evidence")
    monkeypatch.setattr(runner, "single_thread_cpu", lambda: pytest.fail("must not configure"))
    with pytest.raises(ValueError, match="new output directory"):
        runner.run_pilot("unused", output)
    assert previous.read_text() == "retain this evidence"
    assert not (output / "incomplete.json").exists()


def test_missing_startup_settings_stop_before_output_creation(monkeypatch, tmp_path):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    output = tmp_path / "missing-settings"
    monkeypatch.setattr(runner, "single_thread_cpu", lambda: pytest.fail("must not configure"))
    with pytest.raises(RuntimeError, match="Launch with tools/run_cpu"):
        runner.run_pilot("unused", output)
    assert not output.exists()


@pytest.fixture
def stub_driver(monkeypatch):
    for key, value in runner.CPU_ENV.items():
        monkeypatch.setenv(key, value)
    events = []
    plan = {"jobs": [{"id": "first"}, {"id": "second"}]}

    class Controls:
        def verify(self):
            events.append("verify_context")

    @contextmanager
    def context():
        events.append("enter_context")
        yield Controls()
        events.append("leave_context")

    def prepare(data_directory, directory, host_ready):
        events.append("prepare")
        (directory / "prepared.txt").write_text("invented fixture")
        return plan

    monkeypatch.setattr(runner, "single_thread_cpu", context)
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "verify_inputs", lambda *args: events.append("verify_inputs"))
    monkeypatch.setattr(runner, "summarize", lambda *args: events.append("summarize") or {"status": "passed"})
    return events, plan


def test_driver_runs_fresh_phases_sequentially_then_saves(monkeypatch, tmp_path, stub_driver):
    events, plan = stub_driver
    monkeypatch.setattr(runner, "launch", lambda directory, job, phase:
                        events.append((job["id"], phase)))
    output = tmp_path / "pilot"
    result = runner.run_pilot("unused", output)
    assert events == ["enter_context", "prepare", "verify_context", "leave_context",
                      ("first", "fit"), ("first", "predict"),
                      ("second", "fit"), ("second", "predict"), "verify_inputs", "summarize"]
    assert json.loads((output / "checks.json").read_text()) == result
    assert result["output_sha256"] == {"prepared.txt": runner.sha256(output / "prepared.txt")}
    assert not (output / "incomplete.json").exists()


def test_failed_job_retains_partial_evidence_and_stops(monkeypatch, tmp_path, stub_driver):
    events, plan = stub_driver

    def launch(directory, job, phase):
        events.append((job["id"], phase))
        if phase == "predict":
            (directory / "failed-job.log").write_text("diagnostic retained")
            raise RuntimeError("invented worker failure")

    monkeypatch.setattr(runner, "launch", launch)
    output = tmp_path / "pilot"
    with pytest.raises(RuntimeError, match="invented worker failure"):
        runner.run_pilot("unused", output)
    incomplete = json.loads((output / "incomplete.json").read_text())
    assert incomplete["status"] == "incomplete"
    assert incomplete["automatic_retry"] is False
    assert incomplete["formal_timing_eligible"] is False
    assert (output / "failed-job.log").read_text() == "diagnostic retained"
    assert not (output / "checks.json").exists()
    assert ("second", "fit") not in events and "summarize" not in events


def test_launch_uses_explicit_interpreter_launcher_and_retains_failed_log(monkeypatch, tmp_path, capsys):
    captured = {}

    def run(arguments, **kwargs):
        captured.update(arguments=arguments, **kwargs)
        kwargs["stdout"].write("invented worker diagnostic\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="retained log"):
        runner.launch(tmp_path, {"id": "ridge-1"}, "fit")
    assert captured["arguments"] == [sys.executable, str(runner.ROOT / "tools/run_cpu.py"),
        "-m", "tools.run_development_pilot", "--worker", "fit", "--job", "ridge-1",
        "--output-dir", str(tmp_path)]
    assert captured["cwd"] == runner.ROOT
    assert captured["stderr"] == runner.subprocess.STDOUT
    assert (tmp_path / "ridge-1-fit.log").read_text() == "invented worker diagnostic\n"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_worker_saves_pass_only_after_cleanup(monkeypatch, tmp_path, cleanup_fails):
    events = []
    plan = {"jobs": [{"id": "ridge-1"}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    destination = tmp_path / "fit.json"

    @contextmanager
    def context():
        yield object()
        events.append("cleanup")
        if cleanup_fails:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(runner, "verify_inputs", lambda *args: events.append("verify_inputs"))
    monkeypatch.setattr(runner, "read_config", lambda: {})
    monkeypatch.setattr(runner, "single_thread_cpu", context)
    monkeypatch.setattr(runner, "fit_worker", lambda *args: (destination, {"status": "passed"}))
    original_write = runner.write_json

    def write(path, result):
        assert events[-2:] == ["cleanup", "verify_inputs"]
        original_write(path, result)

    monkeypatch.setattr(runner, "write_json", write)
    if cleanup_fails:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            runner.worker(tmp_path, "ridge-1", "fit")
        assert not destination.exists()
    else:
        runner.worker(tmp_path, "ridge-1", "fit")
        assert json.loads(destination.read_text()) == {"status": "passed"}


@pytest.mark.parametrize("method", ["persistence", "ridge", "ridge_lightgbm_hybrid", "lstm"])
def test_fit_boundary_excludes_export_and_reference_prediction(monkeypatch, tmp_path, method):
    events = []
    seed = 2026 if method == "lstm" else None
    job = {"id": f"{method}-1", "method": method, "repeat": 1, "seed": seed}
    data = {"traffic": np.arange(4788, dtype=float), "observed": np.ones(4788, dtype=bool),
            "timestamps": pd.date_range("2020-01-01", periods=4788, freq="h"),
            "targets": np.array([4116, 4117])}
    bundle = SimpleNamespace(training_metadata={"scope": "stub"}, settings={"scope": "stub"})

    def fit_model(actual_method, traffic, observed, timestamps, fit_end, config, *, seed):
        events.append("fit")
        assert actual_method == method and fit_end == 4116
        assert len(traffic) == len(observed) == len(timestamps) == 4116
        assert traffic[-1] == 4115
        assert seed == job["seed"]
        return bundle

    def save_model(actual_bundle, directory):
        events.append("save")
        assert actual_bundle is bundle
        directory.mkdir()
        (directory / "model.json").write_text("invented state")

    def predict_model(*args):
        events.append("predict")
        return np.array([1., 2.]), np.array([1., 2.])

    def elapsed_call(function):
        events.append("timer_start")
        result = function()
        events.append("timer_stop")
        return result, 0.25

    class Controls:
        def verify(self):
            events.append("controls")
            return {"runtime_library_controls_verified": True}

    install_module(monkeypatch, "forecasting.pilot_models", fit_model=fit_model,
                   save_model=save_model, predict_model=predict_model,
                   model_fingerprint=lambda bundle: "unchanged-state")
    install_module(monkeypatch, "tools.pilot_measurements", elapsed_call=elapsed_call,
                   memory_snapshot=lambda: events.append("memory") or {"peak_rss_bytes": 100})
    monkeypatch.setattr(runner, "load_series", lambda directory: data)
    monkeypatch.setattr(runner, "verify_series", lambda data, plan: None)
    path, result = runner.fit_worker(tmp_path, {}, job, {}, Controls())
    if method in runner.BASELINES:
        assert events == ["controls", "fit", "controls", "save", "predict"]
        assert result["fit_seconds"] is None and result["fit_memory"] is None
        assert result["fit_kind"] == "not_applicable"
    else:
        assert events == ["controls", "timer_start", "fit", "timer_stop", "memory", "controls", "save", "predict"]
        assert result["fit_seconds"] == 0.25
        assert result["fit_memory"] == {"peak_rss_bytes": 100}
        assert result["fit_kind"] == ("development_epoch_selection_and_refit" if method == "lstm" else "complete_fit")
    assert result["formal_timing_eligible"] is False
    assert not path.exists()  # The outer worker writes only after context cleanup.
    assert (path.parent / "reference_predictions.npz").exists()


def test_current_config_loads_with_runtime_checksum():
    checksums = json.loads((runner.ROOT / "docs/data-bundle-checksums.json").read_text())
    assert runner.CONFIG_SHA256 == checksums["runtime_config_sha256"]
    config = runner.read_config()
    assert config["models"]["ridge"]["alphas"] == [0.1, 1, 10, 100, 1000]


def test_changed_scientific_config_rejected_before_parsing(monkeypatch):
    monkeypatch.setattr(runner, "sha256", lambda path: "unexpected-hash")
    with pytest.raises(ValueError, match="changing run_config"):
        runner.read_config()


@pytest.fixture
def invented_series():
    from tools.pilot_measurements import select_origins
    traffic = np.arange(4788, dtype=np.float64)
    targets = np.arange(4116, 4788)
    origins = select_origins(targets)
    data = {"traffic": traffic, "traffic_observed": traffic.copy(),
            "observed": np.ones(4788, dtype=bool),
            "timestamps": pd.date_range("2020-01-01", periods=4788, freq="h"),
            "targets": targets, "origins": origins}
    plan = {"targets": targets.tolist(), "origins": origins.tolist()}
    return data, plan


def test_cohort_matches_physical_arrays_and_fixed_origin_selection(invented_series):
    data, plan = invented_series
    runner.verify_series(data, plan)
    assert len(data["targets"]) == 672 and len(data["origins"]) == 100


@pytest.mark.parametrize("mutation", ["plan_targets", "targets_both", "origins_both", "missing_target", "longer_series"])
def test_changed_cohort_or_origins_are_rejected(invented_series, mutation):
    data, plan = invented_series
    if mutation == "plan_targets":
        plan["targets"] = plan["targets"][1:]
    elif mutation == "targets_both":
        data["targets"] = data["targets"][1:]
        plan["targets"] = data["targets"].tolist()
    elif mutation == "origins_both":
        data["origins"] = data["origins"].copy()
        data["origins"][1] += 1
        plan["origins"] = data["origins"].tolist()
    elif mutation == "missing_target":
        data["observed"][4300] = False
    else:
        data["traffic"] = np.r_[data["traffic"], 4788.]
        data["observed"] = np.r_[data["observed"], True]
    with pytest.raises((AssertionError, ValueError)):
        runner.verify_series(data, plan)


@pytest.mark.parametrize("mutation", ["code", "config", "input_bytes", "jobs", "interval"])
def test_worker_plan_rejects_drift(monkeypatch, tmp_path, mutation):
    (tmp_path / "series.npz").write_bytes(b"invented input bytes")
    hashes = {"tools/runner.py": "reviewed"}
    monkeypatch.setattr(runner, "code_hashes", lambda: hashes.copy())
    plan = {"code_sha256": hashes.copy(), "config_sha256": runner.CONFIG_SHA256,
            "series_sha256": runner.sha256(tmp_path / "series.npz"), "jobs": runner.pilot_jobs(),
            "cell": "Cell_003791", "fit_interval": [0, 4116], "score_interval": [4116, 4788]}
    runner.verify_inputs(tmp_path, plan)
    if mutation == "code":
        plan["code_sha256"]["tools/runner.py"] = "changed"
    elif mutation == "config":
        plan["config_sha256"] = "changed"
    elif mutation == "input_bytes":
        (tmp_path / "series.npz").write_bytes(b"changed input bytes")
    elif mutation == "jobs":
        plan["jobs"] = plan["jobs"][:-1]
    else:
        plan["score_interval"][1] += 1
    with pytest.raises(ValueError):
        runner.verify_inputs(tmp_path, plan)


@pytest.fixture
def saved_pilot(monkeypatch, tmp_path, invented_series):
    """Fabricate saved evidence; no model implementation is called."""
    from tools import check_cpu_controls
    from tools.pilot_results import score_predictions, save_predictions
    data, plan = invented_series
    plan.update(jobs=runner.pilot_jobs(), cell="Cell_003791", scope="invented pilot fixture",
                fit_interval=[0, 4116], score_interval=[4116, 4788])
    monkeypatch.setattr(runner, "load_series", lambda directory: data)
    monkeypatch.setattr(check_cpu_controls, "machine_information", lambda: {"scope": "invented fixture"})
    actual = data["traffic_observed"][data["targets"]]
    predictions = actual + 0.5
    evaluation = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
        4116, 4116, 4788, data["targets"], predictions, predictions)
    for job in plan["jobs"]:
        output = tmp_path / "jobs" / job["id"]
        (output / "model").mkdir(parents=True)
        (output / "model" / "state.txt").write_text("invented model placeholder")
        (output / "training.json").write_text("{}")
        np.savez(output / "reference_predictions.npz", raw=predictions, final=predictions)
        save_predictions(output / "predictions.csv", plan["cell"], job["method"], job["seed"],
                         data["targets"], data["timestamps"][data["targets"]], actual, predictions, predictions)
        fit = {"status": "passed", "job": job,
               "fit_seconds": None if job["method"] in runner.BASELINES else 1.,
               "fit_kind": "invented", "fit_memory": None, "artifact_bytes": 26,
               "artifact_sha256": runner.file_hashes(output / "model"),
               "training_sha256": runner.sha256(output / "training.json"),
               "reference_sha256": runner.sha256(output / "reference_predictions.npz")}
        prediction = {"status": "passed", "job": job, "evaluation": evaluation,
                      "predictions_sha256": runner.sha256(output / "predictions.csv")}
        (output / "fit.json").write_text(json.dumps(fit))
        (output / "predict.json").write_text(json.dumps(prediction))
    return tmp_path, plan


def test_summary_recomputes_saved_metrics_and_retains_declared_order(saved_pilot):
    directory, plan = saved_pilot
    result = runner.summarize(directory, plan)
    assert result["status"] == "passed"
    assert result["common_target_count"] == 672 and result["latency_origin_count"] == 100
    assert [row["method"] for row in result["methods_in_declared_order"]] == list(runner.METHODS)
    assert result["formal_timing_eligible"] is False and result["model_ranking_performed"] is False
    assert result["all_reload_predictions_exact"] is True
    assert all(row["scores"]["MAE"] == 0.5 for row in result["methods_in_declared_order"])


@pytest.mark.parametrize("mutation", ["incomplete", "metrics", "model", "training", "reference", "actual", "seed"])
def test_summary_rejects_incomplete_or_inconsistent_saved_evidence(saved_pilot, mutation):
    directory, plan = saved_pilot
    output = directory / "jobs" / plan["jobs"][0]["id"]
    prediction = json.loads((output / "predict.json").read_text())
    if mutation == "incomplete":
        prediction["status"] = "incomplete"
    elif mutation == "metrics":
        prediction["evaluation"]["scores"]["MAE"] = 99.
    elif mutation == "model":
        (output / "model" / "state.txt").write_text("changed model")
    elif mutation == "training":
        (output / "training.json").write_text('{"changed":true}')
    elif mutation == "reference":
        (output / "reference_predictions.npz").write_bytes(b"changed predictions")
    else:
        frame = pd.read_csv(output / "predictions.csv")
        if mutation == "actual":
            frame.loc[0, "actual"] += 1.
        else:
            frame.loc[0, "seed"] = 2026
        frame.to_csv(output / "predictions.csv", index=False, float_format="%.17g")
        # Refresh the outer hash: the content-level checks must still reject it.
        prediction["predictions_sha256"] = runner.sha256(output / "predictions.csv")
    (output / "predict.json").write_text(json.dumps(prediction))
    with pytest.raises((AssertionError, ValueError)):
        runner.summarize(directory, plan)
