"""Check pilot boundaries with invented arrays and stub jobs, never model fits."""
from contextlib import contextmanager
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools import run_development_capacity as runner


def install_module(monkeypatch, name, **functions):
    module = ModuleType(name)
    module.__dict__.update(functions)
    monkeypatch.setitem(sys.modules, name, module)


def test_fixed_jobs_keep_both_dropout_configurations_and_every_seed():
    jobs = runner.capacity_jobs()
    assert len(jobs) == len({job["id"] for job in jobs}) == 15
    assert tuple(dict.fromkeys(job["group"] for job in jobs)) == runner.GROUPS
    assert tuple(dict.fromkeys(job["method"] for job in jobs)) == runner.METHODS
    for group in runner.GROUPS:
        subset = [job for job in jobs if job["group"] == group]
        assert [job["repeat"] for job in subset] == [1, 2, 3]
        assert [job["seed"] for job in subset] == ([2026, 2027, 2028] if group.startswith("lstm")
                                                   else [None] * 3)
    assert [j["dropout"] for j in jobs if j["method"] == "lstm"] == [0., 0., 0., .2, .2, .2]


def test_existing_directory_is_preserved_before_controls(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    previous = output / "previous.json"
    previous.write_text("retain this evidence")
    monkeypatch.setattr(runner, "single_thread_cpu", lambda: pytest.fail("must not configure"))
    with pytest.raises(ValueError, match="new output directory"):
        runner.run_capacity("unused", output)
    assert previous.read_text() == "retain this evidence"
    assert not (output / "incomplete.json").exists()


def test_missing_startup_settings_stop_before_output_creation(monkeypatch, tmp_path):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    output = tmp_path / "missing-settings"
    monkeypatch.setattr(runner, "single_thread_cpu", lambda: pytest.fail("must not configure"))
    with pytest.raises(RuntimeError, match="Launch with tools/run_cpu"):
        runner.run_capacity("unused", output)
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
    result = runner.run_capacity("unused", output)
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
        runner.run_capacity("unused", output)
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
        "-m", "tools.run_development_capacity", "--worker", "fit", "--job", "ridge-1",
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
    traffic = np.arange(6132, dtype=np.float64)
    targets = np.arange(5460, 6132)
    origins = select_origins(targets)
    data = {"traffic": traffic, "traffic_observed": traffic.copy(),
            "observed": np.ones(6132, dtype=bool),
            "timestamps": pd.date_range("2020-01-01", periods=6132, freq="h"),
            "targets": targets, "origins": origins}
    data["observed"][168:179] = False
    data["traffic_observed"][168:179] = np.nan
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
        data["observed"][5600] = False
    else:
        data["traffic"] = np.r_[data["traffic"], 6132.]
        data["observed"] = np.r_[data["observed"], True]
    with pytest.raises((AssertionError, ValueError)):
        runner.verify_series(data, plan)


@pytest.mark.parametrize("mutation", ["code", "config", "input_bytes", "jobs", "interval"])
def test_worker_plan_rejects_drift(monkeypatch, tmp_path, mutation):
    (tmp_path / "series.npz").write_bytes(b"invented input bytes")
    hashes = {"tools/runner.py": "reviewed"}
    monkeypatch.setattr(runner, "code_hashes", lambda: hashes.copy())
    plan = {"code_sha256": hashes.copy(), "config_sha256": runner.CONFIG_SHA256,
            "series_sha256": runner.sha256(tmp_path / "series.npz"), "jobs": runner.capacity_jobs(),
            "cell": "Cell_003791", "fit_interval": [0, 5460], "score_interval": [5460, 6132]}
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
    plan.update(jobs=runner.capacity_jobs(), cell="Cell_003791", scope="invented pilot fixture",
                fit_interval=[0, 5460], score_interval=[5460, 6132])
    monkeypatch.setattr(runner, "load_series", lambda directory: data)
    monkeypatch.setattr(check_cpu_controls, "machine_information", lambda: {"scope": "invented fixture"})
    actual = data["traffic_observed"][data["targets"]]
    predictions = actual + 0.5
    evaluation = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
        5460, 5460, 6132, data["targets"], predictions, predictions)
    for job in plan["jobs"]:
        output = tmp_path / "jobs" / job["id"]
        (output / "model").mkdir(parents=True)
        (output / "model" / "state.txt").write_text("invented model placeholder")
        settings, training = invented_fit_evidence(job)
        (output / "training.json").write_text(json.dumps(training))
        np.savez(output / "reference_predictions.npz", raw=predictions, final=predictions)
        save_predictions(output / "predictions.csv", plan["cell"], job["method"], job["seed"],
                         data["targets"], data["timestamps"][data["targets"]], actual, predictions, predictions)
        fit = {"status": "passed", "job": job,
               "fit_seconds": 1., "settings": settings,
               "controls_before": {"runtime_library_controls_verified": True},
               "controls_after": {"runtime_library_controls_verified": True},
               "fit_kind": "invented", "fit_memory": None, "artifact_bytes": 26,
               "artifact_sha256": runner.file_hashes(output / "model"),
               "training_sha256": runner.sha256(output / "training.json"),
               "reference_sha256": runner.sha256(output / "reference_predictions.npz")}
        prediction = {"status": "passed", "job": job, "evaluation": evaluation,
                      "controls_before": {"runtime_library_controls_verified": True},
                      "controls_after": {"runtime_library_controls_verified": True},
                      "reload_predictions_exact": True, "fitted_state_unchanged": True,
                      "predictions_sha256": runner.sha256(output / "predictions.csv")}
        (output / "fit.json").write_text(json.dumps(fit))
        (output / "predict.json").write_text(json.dumps(prediction))
    return tmp_path, plan


def test_summary_recomputes_saved_metrics_and_retains_declared_order(saved_pilot):
    directory, plan = saved_pilot
    result = runner.summarize(directory, plan)
    assert result["status"] == "passed"
    assert result["common_target_count"] == 672 and result["latency_origin_count"] == 100
    assert [row["group"] for row in result["configurations_in_declared_order"]] == list(runner.GROUPS)
    assert result["component_fitting_phases"] == 24
    assert len(result["lstm_training_diagnostics"]) == 6
    assert result["formal_timing_eligible"] is False and result["model_ranking_performed"] is False
    assert result["all_reload_predictions_exact"] is True
    assert all(row["scores"]["MAE"] == 0.5 for row in result["configurations_in_declared_order"])


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


def invented_fit_evidence(job):
    settings = {"fit_end": 5460, "lookback": 168}
    method = job["method"]
    counts = {"ridge": {"ridge": 1}, "direct_lightgbm": {"direct_lightgbm": 1},
              "ridge_lightgbm_hybrid": {"ridge": 1, "correction": 1},
              "lstm": {"inner_selection": 1, "full_prefix_refit": 1}}
    training = {"component_fit_counts": counts[method]}
    if method in ("ridge", "ridge_lightgbm_hybrid"):
        settings["alpha"] = 0.1
    if method in ("direct_lightgbm", "ridge_lightgbm_hybrid"):
        settings.update(requested_trees=512, leaf_cap=31)
        training["actual_trees"] = 512
    if method == "ridge_lightgbm_hybrid":
        settings["weight"] = 1.0
    if method == "lstm":
        settings.update(hidden_units=32, dropout=job["dropout"], seed=job["seed"],
                        batch_size=32, selected_epochs=91)
        training.update(effective_budgets={"maximum_epochs": 100, "patience_epochs": 10,
                                           "inner_stopping_tail_hours": 336},
                        selected_epochs=91, selection_epochs_run=100,
                        stopping_history=[{}]*100, refit_history=[{}]*91,
                        stop_reason="maximum_epochs", maximum_epochs_reached=True,
                        phase_timings={"selection_seconds": .4, "refit_seconds": .3})
    return settings, training


def test_cap_hit_is_recorded_without_changing_budget_or_failing():
    job = runner.capacity_jobs()[-1]
    settings, training = invented_fit_evidence(job)
    runner.validate_fit_evidence(job, settings, training, 1.)
    assert training["selected_epochs"] == 91 and training["stop_reason"] == "maximum_epochs"


@pytest.mark.parametrize("change", ["architecture", "dropout", "budget", "phases", "phase_time", "epoch_count", "cap_flag"])
def test_fit_evidence_rejects_changed_budget_or_invalid_accounting(change):
    job = runner.capacity_jobs()[-1]
    settings, training = invented_fit_evidence(job)
    if change == "architecture": settings["hidden_units"] = 16
    elif change == "dropout": settings["dropout"] = 0.
    elif change == "budget": training["effective_budgets"]["maximum_epochs"] = 200
    elif change == "phases": training["component_fit_counts"] = {"full_prefix_refit": 1}
    elif change == "phase_time": training["phase_timings"]["refit_seconds"] = 4.
    elif change == "epoch_count": training["selected_epochs"] = 101
    else: training["maximum_epochs_reached"] = False
    with pytest.raises(ValueError):
        runner.validate_fit_evidence(job, settings, training, 1.)


def test_summary_keeps_lstm_architectures_separate_and_embeds_fit_controls(saved_pilot):
    directory, plan = saved_pilot
    result = runner.summarize(directory, plan)
    groups = result["configurations_in_declared_order"]
    assert [g["dropout"] for g in groups[-2:]] == [0., .2]
    assert all(len(g["fit_seconds"]["observations"]) == 3 for g in groups)
    assert all(row["fit"]["controls_before"]["runtime_library_controls_verified"] for row in result["jobs"])
    assert sum(len(row["training"].get("refit_history", [])) for row in result["jobs"]) == 6*91


def test_summary_rejects_unverified_fitting_controls(saved_pilot):
    directory, plan = saved_pilot
    path = directory / "jobs" / plan["jobs"][0]["id"] / "fit.json"
    report = json.loads(path.read_text())
    report["controls_after"]["runtime_library_controls_verified"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="runtime controls"):
        runner.summarize(directory, plan)


@pytest.mark.parametrize("group", runner.GROUPS)
def test_fit_timer_contains_prefix_fit_but_excludes_export_and_predictions(monkeypatch, tmp_path, invented_series, group):
    from forecasting import capacity_models, pilot_models
    from tools import pilot_measurements
    events = []
    job = next(j for j in runner.capacity_jobs() if j["group"] == group)
    settings, training = invented_fit_evidence(job)
    bundle = SimpleNamespace(settings=settings, training_metadata=training)
    data, plan = invented_series

    def fit(method, values, observed, clock, fit_end, config, **kwargs):
        events.append("fit")
        assert len(values) == len(observed) == len(clock) == fit_end == 5460
        assert values[-1] == 5459.
        assert kwargs == {"seed": job["seed"], "dropout": job["dropout"]}
        return bundle

    def elapsed(function):
        events.append("timer_start")
        result = function()
        events.append("timer_stop")
        return result, 1.

    def save(model, directory):
        events.append("save")
        directory.mkdir()
        (directory / "model.json").write_text("invented artifact")

    def predict(*args):
        events.append("predict")
        return np.ones(672), np.ones(672)

    class Controls:
        def verify(self):
            events.append("controls")
            return {"runtime_library_controls_verified": True}

    monkeypatch.setattr(runner, "load_series", lambda directory: data)
    monkeypatch.setattr(capacity_models, "fit_model", fit)
    monkeypatch.setattr(pilot_models, "save_model", save)
    monkeypatch.setattr(pilot_models, "predict_model", predict)
    monkeypatch.setattr(pilot_models, "model_fingerprint", lambda model: "unchanged")
    monkeypatch.setattr(pilot_measurements, "elapsed_call", elapsed)
    monkeypatch.setattr(pilot_measurements, "memory_snapshot", lambda: events.append("memory") or {"peak_rss_bytes": 100})
    path, result = runner.fit_worker(tmp_path, plan, job, {}, Controls())
    assert events == ["controls", "timer_start", "fit", "timer_stop", "memory", "controls", "save", "predict"]
    assert result["fit_seconds"] == 1.
    assert not path.exists()  # The outer worker saves success after context cleanup.
