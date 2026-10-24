"""Invented evidence tests; no native fitting or real holdout access."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ensemble_study import common, evidence
from ensemble_study.selection import CAPACITIES, OBJECTIVES, WEIGHTS, make_condition
from forecasting.data import fit_scaler
from forecasting.development_plan import FOLDS, TUNING_CELLS
from tools.development_store import _inventory, _read_json, sha256


def lstm_records():
    jobs, results = [], {}
    for cell in TUNING_CELLS:
        for fold in range(3):
            for seed in common.FIXED["seeds"]:
                job = {"id": f"lstm-h32-d0p2--{cell}--f{fold+1}--s{seed}",
                       "stage": "lstm", "candidate_id": "lstm-h32-d0p2",
                       "cell": cell, "fold_index": fold, "seed": seed}
                jobs.append(job)
                results[job["id"]] = {"job": job, "training": {"selected_epochs": 23}}
    return jobs, results


def test_fixed_epochs_rederived_from_all81_evidence_records():
    jobs, results = lstm_records()
    result = evidence.derive_fixed_epochs(jobs, results)
    assert result["selected_result_count"] == 81
    assert result["fixed_epochs"] == 23
    assert result["median_selected_epochs"] == 23


@pytest.mark.parametrize("bad", ["missing", "duplicate", "wrong_seed", "wrong_epoch", "bool_epoch"])
def test_fixed_epochs_fail_on_incomplete_or_changed_evidence(bad):
    jobs, results = lstm_records()
    if bad == "missing":
        results.pop(jobs[0]["id"])
    elif bad == "duplicate":
        jobs.append(copy.deepcopy(jobs[0]))
    elif bad == "wrong_seed":
        jobs[0]["seed"] = 99
    elif bad == "wrong_epoch":
        for value in results.values():
            value["training"]["selected_epochs"] = 24
    else:
        results[jobs[0]["id"]]["training"]["selected_epochs"] = True
    with pytest.raises(ValueError):
        evidence.derive_fixed_epochs(jobs, results)


def beta_zero_fixture(monkeypatch):
    import l1_revision.model as model
    traffic = np.arange(1, 225, dtype=float)
    flags = np.ones(224, dtype=bool)
    data = {"traffic": traffic, "observed": flags,
            "timestamps": pd.date_range("2026-01-01", periods=224, freq="h")}
    scaler = fit_scaler(traffic[:200], flags[:200], 200)
    bundle = SimpleNamespace(beta=0, ridge_bundle=None, scaler=scaler,
        settings={"training_target_count": 32}, model=SimpleNamespace(num_feature=lambda: 340))
    monkeypatch.setattr(model, "load_model", lambda _: bundle)
    monkeypatch.setattr(model, "model_fingerprint", lambda _: "0" * 64)
    job = {"parameters": {"beta": 0}, "fit_end": 200}
    result = {"dependency": None, "training": {"component_fit_counts": {"direct_lightgbm_l1": 1}}}
    return bundle, job, result, data


def test_beta_zero_reconstructs_fitting_targets_without_any_fit(monkeypatch, tmp_path):
    bundle, job, result, data = beta_zero_fixture(monkeypatch)
    value = evidence._check_beta_zero(tmp_path, job, result, data)
    assert value["fitting_target_count"] == 32
    assert len(value["reconstructed_full_target_labels_sha256"]) == 64
    assert value["model_state_sha256"] == "0" * 64


@pytest.mark.parametrize("bad", ["beta", "dependency", "ridge_bundle", "target_count", "feature_count"])
def test_beta_zero_refuses_partial_models_or_wrong_training_prefix(monkeypatch, tmp_path, bad):
    bundle, job, result, data = beta_zero_fixture(monkeypatch)
    if bad == "beta":
        job["parameters"]["beta"] = .25
    elif bad == "dependency":
        result["dependency"] = {"ridge": "old"}
    elif bad == "ridge_bundle":
        bundle.ridge_bundle = object()
    elif bad == "target_count":
        bundle.settings["training_target_count"] = 31
    else:
        bundle.model.num_feature = lambda: 1
    with pytest.raises(ValueError):
        evidence._check_beta_zero(tmp_path, job, result, data)


def test_forecast_reader_checks_alignment_identity_and_raw_final_bound(tmp_path):
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h")
    data = {"traffic_observed": np.array([1., 2., 3.]), "timestamps": timestamps}
    job = {"cell": "A", "method": "lightgbm_l1", "seed": None}
    frame = pd.DataFrame({"cell": ["A", "A"], "method": ["lightgbm_l1"] * 2,
        "seed": ["", ""], "target_index": [1, 2], "timestamp": timestamps[1:],
        "actual": [2., 3.], "prediction_raw": [-1., 4.], "prediction": [0., 4.]})
    path = tmp_path / "predictions.csv"
    frame.to_csv(path, index=False)
    assert len(evidence._frame(path, job, data, np.array([1, 2]))) == 2
    frame.loc[0, "prediction"] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        evidence._frame(path, job, data, np.array([1, 2]))


def test_condition_diagnostics_preserve_lstm_seed_rmse_and_all_folds():
    cells = ["A"]
    values = np.arange(6132, dtype=float) % 13 + 20
    data = {"traffic": values, "traffic_observed": values,
            "observed": np.ones(6132, dtype=bool),
            "timestamps": pd.date_range("2026-01-01", periods=6132, freq="h")}
    history = {"data": {"A": data}, "base_raw": {"lstm": {}}}
    cohorts = {}
    for fold, (start, end) in enumerate(FOLDS):
        targets = np.arange(start, end)
        actual = values[targets]
        history["base_raw"]["lstm"][("A", fold)] = {
            "2026": actual - 2, "2027": actual + 2, "2028": actual.copy()}
        cohorts[("l2", 32, 7, "A", fold)] = {
            "targets": targets, "actual": actual, "weekly_scale": 2,
            "ridge_raw": actual - 2, "direct_raw": actual + 2}
    result = evidence._condition_diagnostics(make_condition("lstm"), history, cohorts, cells)
    assert result["scores"]["MAE"] == pytest.approx(4 / 3)
    assert result["scores"]["RMSE"] == pytest.approx(4 / 3)
    assert len(result["folds"]) == 3
    assert result["folds"][0]["summary"]["cells"]["A"]["seed_count"] == 3
    blend = make_condition("blend", objective="l2", trees=32, leaves=7, ridge_weight=.5)
    assert evidence._condition_diagnostics(blend, history, cohorts, cells)["scores"]["MAE"] == 0


def evidence_payload(tmp_path):
    sources = {}
    for name in ("original", "l1", "completion"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "data.json").write_text(json.dumps({"source": name}))
        sources[name] = {"directory": str(directory), "files_sha256": _inventory(directory)}
    selection = {"status": "passed", "primary_candidate": "ridge",
        "protocol_sha256": sha256(common.PROTOCOL), "evidence": {
            "historical_sources": {"sources": {k: sources[k] for k in ("original", "l1")}},
            "new_completion": sources["completion"],
            "replay_code_sha256": common.code_hashes("development")}}
    tables = {}
    for objective in OBJECTIVES:
        for weight in WEIGHTS:
            tables[f"weight-predictions/{objective}-r{format(weight, '.12g').replace('.', 'p')}.csv"] = pd.DataFrame({
                "objective": [objective], "ridge_weight": [weight], "cell": ["A"], "fold_index": [0],
                "target_index": [4116], "timestamp": pd.to_datetime(["2026-01-01"]),
                "actual": [10.], "ridge_raw": [8.], "lightgbm_raw": [12.],
                "prediction_raw": [weight*8+(1-weight)*12], "prediction": [weight*8+(1-weight)*12]})
    return {"selection": selection, "development_summary": {"selection": selection}, "prediction_tables": tables}


def test_save_evidence_writes_hash_manifest_and_refuses_overwrite(tmp_path):
    payload = evidence_payload(tmp_path)
    output = tmp_path / "summary"
    result = evidence.save_evidence(output, payload)
    assert result["status"] == "passed"
    marker = _read_json(output / "complete.json")
    assert marker["files_sha256"] == _inventory(output, exclude=("complete.json",))
    assert result["selection_sha256"] == sha256(output / "selection.json")
    with pytest.raises(ValueError, match="never overwrite"):
        evidence.save_evidence(output, payload)


def test_save_refuses_source_subdirectory_and_changed_evidence(tmp_path):
    payload = evidence_payload(tmp_path)
    with pytest.raises(ValueError, match="separate"):
        evidence.save_evidence(tmp_path / "original" / "summary", payload)
    (tmp_path / "l1" / "data.json").write_text("changed")
    with pytest.raises(ValueError, match="inventory differs"):
        evidence.save_evidence(tmp_path / "summary", payload)
    assert not (tmp_path / "summary").exists()


def test_audit_sources_does_not_return_raw_arrays(monkeypatch):
    report = {"status": "passed", "reused_counts": {"l2": 162, "l1": 54, "ridge": 27}}
    monkeypatch.setattr(evidence, "_load_historical", lambda *args: {"audit": report, "raw": np.ones(2)})
    assert evidence.audit_sources("original", "historical") == report
    json.dumps(evidence.audit_sources("original", "historical"), allow_nan=False)


def test_completion_costs_keep_failure_attempts_and_do_not_add_nested_totals(tmp_path):
    for index, successful in ((1, False), (2, True)):
        directory = tmp_path / "jobs" / "one-job" / f"attempt-{index:04d}"
        directory.mkdir(parents=True)
        (directory / "attempt.json").write_text(json.dumps({"plan_sha256": "abc"}))
        (directory / ("complete.json" if successful else "failure.json")).write_text("{}")
        (directory / "elapsed.json").write_text(json.dumps({"wall_seconds": index + 2}))
    session = tmp_path / "sessions" / "session-0001"
    session.mkdir(parents=True)
    (session / "started.json").write_text(json.dumps({"plan_sha256": "abc", "stage": "missing_l1_completion"}))
    (session / "finished.json").write_text(json.dumps({"wall_seconds": 10}))
    result = evidence.completion_cost_records(tmp_path, "abc")
    assert result["successful_attempt_count"] == 1
    assert result["failed_or_interrupted_attempt_count"] == 1
    assert result["recorded_attempt_wall_seconds"] == 7
    assert result["recorded_stage_wall_seconds"] == 10


def test_prediction_tables_save_both_components_and_every_declared_weight():
    history = {"data": {"A": {"timestamps": pd.date_range("2026-01-01", periods=6132, freq="h")}}}
    cohorts = {}
    for objective in OBJECTIVES:
        for fold, (start, end) in enumerate(FOLDS):
            cohorts[(objective, 32, 7, "A", fold)] = {
                "targets": np.array([start]), "actual": np.array([1.]),
                "ridge_raw": np.array([-10.]), "direct_raw": np.array([10.]), "weekly_scale": 2}
    tables = evidence._prediction_tables(cohorts, {o: {"trees": 32, "leaves": 7} for o in OBJECTIVES}, history, ["A"])
    assert len(tables) == 16
    frame = tables["weight-predictions/l2-r0p5.csv"]
    assert len(frame) == 3
    assert frame.ridge_raw.eq(-10).all()
    assert frame.lightgbm_raw.eq(10).all()
    assert frame.prediction_raw.eq(0).all()
    assert frame.prediction.eq(0).all()
