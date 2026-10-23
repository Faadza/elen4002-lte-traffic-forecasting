"""Invented-input checks of chronological gates, workloads and worker handoff."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from final_evaluation import plan, run
from tools.development_store import begin_attempt, sha256, write_json_atomic


def test_frozen_workload_and_no_loss_or_weight_search():
    cells = [f"Cell_{n:06d}" for n in range(57)]
    confirmation, final = (plan.jobs_for(p, cells) for p in plan.PERIODS)
    assert len(confirmation) == 570 and len(final) == 1026
    assert sum(j["predict"] for j in final) == sum(j["predict"] for j in confirmation) == 570
    assert sum(j["method"] not in plan.BASELINES for j in confirmation) == 399
    assert sum(j["method"] not in plan.BASELINES for j in final) == 855
    for period, jobs in (("confirmation", confirmation), ("final", final)):
        assert len({j["id"] for j in jobs}) == len(jobs)
        assert {j["method"] for j in jobs} == set(plan.METHODS)
        assert {j["seed"] for j in jobs if j["method"] == "lstm"} == set(plan.SEEDS)
        assert all((j["fit_end"], j["score_end"]) == plan.PERIODS[period] for j in jobs)
    assert plan.FROZEN["tree_objective"] == "regression"
    assert plan.FROZEN["blend_ridge_weight"] == .2
    assert plan.FROZEN["residual_weight"] == 1


@pytest.mark.parametrize("period,cells", [("test", ["a"]), ("final", []), ("final", ["a", "a"])])
def test_invalid_panel_or_period(period, cells):
    with pytest.raises(ValueError):
        plan.jobs_for(period, cells)


def test_manifest_covers_every_runtime_package():
    hashes = plan.code_hashes()
    for name in ("final_evaluation/run.py", "final_evaluation/plan.py", "final_evaluation/analysis.py",
                 "final_evaluation/measurement.py", "final_evaluation/holdout_models.py",
                 "blend_revision/model.py", "models/ridge.py", "forecasting/lstm_training.py"):
        assert hashes[name] == sha256(plan.ROOT / name)


def test_host_power_requirements():
    machine = {"system": "Darwin", "power_source": {"returncode": 0, "stdout": "Now drawing from 'AC Power'"},
               "power_profiles": {"returncode": 0, "stdout": "lowpowermode 0\nlowpowermode 0"}}
    plan.host_check(machine)
    for path, value in (("system", "Linux"), ("power_source", {"returncode": 0, "stdout": "Battery Power"}),
                        ("power_profiles", {"returncode": 0, "stdout": "lowpowermode 1"}),
                        ("power_profiles", {"returncode": 1, "stdout": ""})):
        bad = deepcopy(machine); bad[path] = value
        with pytest.raises(ValueError): plan.host_check(bad)


def test_final_cannot_be_used_as_its_own_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "development_evidence", lambda p: {"environment_identity": {}})
    monkeypatch.setattr(plan, "environment_identity", lambda: ({}, {}))
    monkeypatch.setattr(plan, "read_config", lambda: {})
    monkeypatch.setattr(run, "summarize", lambda *a, **k: {"period": "final", "status": "passed",
        "primary_model": plan.BLEND, "frozen": plan.FROZEN, "primary_comparator": "ridge"})
    with pytest.raises(ValueError, match="sealed confirmation"):
        plan.prepare(tmp_path / "data", tmp_path / "original", tmp_path / "new", "final", "a"*40,
                     confirmation=tmp_path / "old-final")
    assert not (tmp_path / "new").exists()


def test_final_requires_confirmation_before_data_load(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "development_evidence", lambda p: {"environment_identity": {}})
    monkeypatch.setattr(plan, "environment_identity", lambda: ({}, {}))
    monkeypatch.setattr(plan, "read_config", lambda: {})
    with pytest.raises(ValueError, match="requires sealed confirmation"):
        plan.prepare(tmp_path / "data", tmp_path / "original", tmp_path / "new", "final", "a"*40)
    assert not (tmp_path / "new").exists()


def test_series_rejects_future_rows_and_changed_bytes(tmp_path):
    period = "confirmation"; start, end = plan.PERIODS[period]
    (tmp_path / "series").mkdir()
    path = tmp_path / "series/Cell_test.npz"
    values = 100 + np.arange(end+1, dtype=float) % 24
    clock = pd.date_range("2018-01-01", periods=end+1, freq="h").to_numpy()
    np.savez(path, traffic=values, traffic_observed=values, observed=np.ones(end+1, dtype=bool), timestamps=clock)
    p = {"period": period, "series": {"Cell_test": {"path": "series/Cell_test.npz",
         "sha256": sha256(path), "targets": list(range(start, end))}}}
    with pytest.raises(ValueError, match="period boundary"):
        plan.load_series(tmp_path, p, "Cell_test")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="series changed"):
        plan.load_series(tmp_path, p, "Cell_test")


@pytest.mark.parametrize("method", ["persistence", plan.BLEND, plan.RESIDUAL, "lstm"])
def test_complete_worker_serialization_and_metric_audit(tmp_path, monkeypatch, method):
    # These 432 invented hours are not the real evaluation panel. Mock only
    # plan/host gates, retaining CPU controls, serialization and score checking.
    start, end = 400, 432
    values = 100 + 7*np.sin(np.arange(end)/7) + np.arange(end)/20
    observed = np.ones(end, dtype=bool)
    clock = pd.date_range("2018-01-01", periods=end, freq="h")
    data = {"traffic": values, "traffic_observed": values.copy(), "observed": observed,
            "timestamps": clock}
    targets = np.arange(start, end)
    job = {"id": f"{method}--Cell_test--r1", "method": method, "cell": "Cell_test",
           "repeat": 1, "seed": 2026 if method == "lstm" else None,
           "fit_end": start, "score_start": start, "score_end": end,
           "predict": True, "period": "confirmation"}
    p = {"jobs": [job], "common_origins": [start, end-1],
         "series": {"Cell_test": {"targets": targets.tolist()}}}
    config = plan.read_config(); digest = "a"*64
    monkeypatch.setattr(run, "verify_plan", lambda p: (deepcopy(globals_plan), config, digest))
    globals_plan = p
    monkeypatch.setattr(run, "load_series", lambda *a: (data, targets))
    monkeypatch.setattr(run, "host_check", lambda m: None)
    import tools.check_cpu_controls
    monkeypatch.setattr(tools.check_cpu_controls, "machine_information", lambda: {"invented_test": True})
    attempt, cached = begin_attempt(tmp_path, job, digest)
    run.worker(tmp_path, job["id"], attempt, "fit")
    run.worker(tmp_path, job["id"], attempt, "predict")
    report = run.validate_result(attempt, job, p, digest)
    if method == "persistence":
        assert report["fit"]["fit_seconds"] is None
    else:
        assert report["fit"]["fit_seconds"] > 0
        assert report["fit"]["training"]["cached_component_reused"] is False
    assert report["predict"]["reload_predictions_exact"]
    frame = run.read_predictions(attempt, job, p)
    if method == "persistence":
        np.testing.assert_array_equal(frame.prediction, values[targets-1])
    with pytest.raises(ValueError, match="already exists"):
        run.worker(tmp_path, job["id"], attempt, "fit")
    prediction = json.loads((attempt / "predict.json").read_text())
    prediction["evaluation"]["scores"]["MAE"] += .01
    (attempt / "predict.json").write_text(json.dumps(prediction))
    with pytest.raises(ValueError, match="arithmetic"):
        run.validate_result(attempt, job, p, digest)


def test_run_requires_host_conditions_before_any_fit(tmp_path):
    with pytest.raises(ValueError, match="host-ready"):
        run.run(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_development_source_requires_unchanged_unlisted_file(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "ROOT", tmp_path)
    source = tmp_path / "models" / "ridge.py"
    source.parent.mkdir()
    source.write_text("original numerical code\n")
    archived = {"models/ridge.py": sha256(source)}
    assert plan.verify_development_code(archived) == {}
    source.write_text("changed numerical code\n")
    with pytest.raises(ValueError, match="Unreviewed development source change"):
        plan.verify_development_code(archived)
    source.unlink()
    with pytest.raises(ValueError, match="Expected a regular file"):
        plan.verify_development_code(archived)


@pytest.mark.parametrize("name", sorted(plan.REVIEWED_SOURCE_REVISIONS))
def test_development_source_accepts_only_exact_reviewed_pair(monkeypatch, name):
    archived, current = plan.REVIEWED_SOURCE_REVISIONS[name]
    monkeypatch.setattr(plan, "sha256", lambda path: archived)
    assert plan.verify_development_code({name: archived}) == {}
    monkeypatch.setattr(plan, "sha256", lambda path: current)
    assert plan.verify_development_code({name: archived}) == {
        name: {"original_sha256": archived, "current_sha256": current}}
    with pytest.raises(ValueError, match="Unreviewed development source change"):
        plan.verify_development_code({name: "0" * 64})
    with pytest.raises(ValueError, match="Unreviewed development source change"):
        plan.verify_development_code({"models/unreviewed.py": archived})
    monkeypatch.setattr(plan, "sha256", lambda path: "0" * 64)
    with pytest.raises(ValueError, match="Unreviewed development source change"):
        plan.verify_development_code({name: archived})


def test_development_pins_recheck_archive_and_recorded_revisions(tmp_path, monkeypatch):
    archived_plan = tmp_path / "plan.json"
    archived_plan.write_text(json.dumps({"code_sha256": {"tools/example.py": "old"}}))
    evidence = {"directory": str(tmp_path), "pins": {"plan.json": sha256(archived_plan)}}
    applied = {"tools/example.py": {"original_sha256": "old", "current_sha256": "reviewed"}}
    monkeypatch.setattr(plan, "verify_development_code", lambda hashes: deepcopy(applied))
    with pytest.raises(ValueError, match="Recorded development source revisions differ"):
        plan.verify_development_pins(evidence)
    evidence["source_revisions"] = deepcopy(applied)
    plan.verify_development_pins(evidence)
    evidence["source_revisions"]["tools/example.py"]["current_sha256"] = "different"
    with pytest.raises(ValueError, match="Recorded development source revisions differ"):
        plan.verify_development_pins(evidence)
    evidence["source_revisions"] = deepcopy(applied)
    archived_plan.write_text(json.dumps({"code_sha256": {}}))
    with pytest.raises(ValueError, match="Original development evidence changed"):
        plan.verify_development_pins(evidence)
