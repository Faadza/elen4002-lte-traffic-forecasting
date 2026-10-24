"""Synthetic checks of frozen dynamic panels, chronological gates and workers."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from ensemble_study import evaluation_plan as plan, evaluation_run as run
from ensemble_study import common
from ensemble_study.selection import make_condition
from final_evaluation.analysis import select_primary_comparator, paired_calendar_intervals
from tools.development_store import begin_attempt, sha256


def selection(weights=None):
    components = {"l2": {"trees": 128, "leaves": 7}, "l1": {"trees": 32, "leaves": 31}}
    weights = weights or {"l2": .2, "l1": .3}
    return {"schema_version": 1, "status": "passed", "study_id": plan.STUDY_ID,
            "selected_components": components, "selected_weights": weights,
            "conditions": plan.expected_conditions(components, weights),
            "primary_candidate": plan.condition_for("l1", weights["l1"], components["l1"])["id"],
            "fixed": deepcopy(plan.FIXED), "protocol_sha256": sha256(common.PROTOCOL),
            "evidence": {"synthetic": True}}


@pytest.mark.parametrize("weights,count", [({"l2": .5, "l1": .5}, 9),
    ({"l2": .2, "l1": .5}, 10), ({"l2": .2, "l1": .3}, 11), ({"l2": 1, "l1": 1}, 9),
    ({"l2": 0, "l1": 0}, 9)])
def test_dynamic_panel_deduplication_and_complete_cost_counts(weights, count):
    chosen = plan.validate_selection(selection(weights))
    assert len(chosen["conditions"]) == count
    cells = [f"Cell_{n:06d}" for n in range(57)]
    confirm = plan.jobs_for("confirmation", cells, chosen["conditions"])
    final = plan.jobs_for("final", cells, chosen["conditions"])
    assert len(confirm) == 57*(count+2)
    assert len(final) == 57*(3*count-6)
    assert sum(j["predict"] for j in final) == len(confirm)
    assert len({j["id"] for j in final}) == len(final)
    assert {j["seed"] for j in final if j["method"] == "lstm"} == set(plan.SEEDS)
    assert sum(c["id"] == "ridge" for c in chosen["conditions"]) == 1
    assert tuple(c["id"] for c in chosen["conditions"][:7]) == plan.STANDALONES
    assert not any("hybrid" in c["id"] for c in chosen["conditions"])


@pytest.mark.parametrize("change", [
    lambda s: s["conditions"].pop(),
    lambda s: s["conditions"].append(deepcopy(s["conditions"][0])),
    lambda s: s.update(primary_candidate="lstm"),
    lambda s: s["selected_weights"].update(l1=.4),
    lambda s: s["selected_components"]["l2"].update(trees=256),
    lambda s: s["fixed"].update(lstm_epochs=24),
    lambda s: s.update(study_id="old_fixed_80_20"),
    lambda s: s["conditions"][0].update(added=True),
    lambda s: s.update(evidence={}),
])
def test_selection_rejects_drift_and_old_scope(change):
    item = selection(); change(item)
    with pytest.raises(ValueError): plan.validate_selection(item)


def test_published_selection_and_source_receipt_must_match(tmp_path):
    selected = selection()
    source = tmp_path / "selection.json"; source.write_text(json.dumps(selected))
    digest = sha256(source)
    assert plan.read_selection(source, digest) == selected
    receipt = {"commit": "a"*40, "verified": True, "selection_sha256": digest,
               "protocol_sha256": selected["protocol_sha256"], "code_sha256": plan.code_hashes()}
    path = tmp_path / "publication.json"; path.write_text(json.dumps(receipt))
    plan.verify_publication(path, selected, digest, "a"*40)
    receipt["code_sha256"]["ensemble_study/evaluation_run.py"] = "b"*64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Published-byte"):
        plan.verify_publication(path, selected, digest, "a"*40)
    source.write_text(json.dumps({**selected, "extra": 1}))
    with pytest.raises(ValueError, match="selection bytes"):
        plan.read_selection(source, digest)


def test_final_requires_matching_sealed_confirmation_before_data_load(tmp_path, monkeypatch):
    chosen = selection()
    monkeypatch.setattr(plan, "read_selection", lambda *a: chosen)
    monkeypatch.setattr(plan, "verify_publication", lambda *a: {})
    monkeypatch.setattr(plan, "development_evidence", lambda *a: {"environment_identity": {}})
    monkeypatch.setattr(common, "native_readiness", lambda *a, **k: {})
    import tools.check_prepared_data
    def forbidden(_): raise AssertionError("data was read before final gate")
    monkeypatch.setattr(tools.check_prepared_data, "load_validated_data", forbidden)
    args = (tmp_path/"data", tmp_path/"original", tmp_path/"new", "final", "a"*40,
            tmp_path/"selection", "b"*64, tmp_path/"receipt")
    with pytest.raises(ValueError, match="requires sealed confirmation"):
        plan.prepare(*args)
    for changes in ({"period": "final"}, {"selection_sha256": "c"*64}, {"status": "failed"},
                    {"primary_model": "ridge"}, {"primary_comparator": "ensemble_l1_r0p5"}):
        previous = {"period": "confirmation", "status": "passed", "study_id": plan.STUDY_ID,
                    "selection_sha256": "b"*64, "primary_model": chosen["primary_candidate"],
                    "primary_comparator": "lightgbm_l1", **changes}
        monkeypatch.setattr(run, "summarize", lambda *a, _p=previous, **k: _p)
        with pytest.raises(ValueError, match="sealed confirmation"):
            plan.prepare(*args, confirmation=tmp_path/"bad")
    assert not (tmp_path/"new").exists()


def test_component_arithmetic_rejects_clip_before_blend():
    condition = make_condition("blend", objective="l1", trees=32, leaves=7, ridge_weight=.2)
    component = {"ridge_raw": np.array([-100., 20.]), "direct_raw": np.array([10., -2.]),
                 "raw": np.array([-12., 2.4]), "final": np.array([0., 2.4])}
    # Construct using exact floating-point operation order expected by the model.
    component["raw"] = .2*component["ridge_raw"] + .8*component["direct_raw"]
    component["final"] = np.maximum(0., component["raw"])
    run.audit_components(condition, component)
    component["raw"] = .2*np.maximum(0., component["ridge_raw"]) + .8*np.maximum(0., component["direct_raw"])
    component["final"] = component["raw"].copy()
    with pytest.raises(AssertionError): run.audit_components(condition, component)


def test_separate_immutable_cost_projection_binds_same_final_jobs(tmp_path):
    chosen = selection()
    cells = ["A", "B"]
    frozen = {"period": "final", "selection": chosen, "selection_sha256": "b"*64,
              "cells": cells, "jobs": plan.jobs_for("final", cells, chosen["conditions"]),
              "common_origins": [7450, 7500]}
    projection = plan.cost_plan_projection(frozen, "a"*64)
    assert projection["accuracy_plan_sha256"] == "a"*64
    assert len(projection["jobs"]) == len(frozen["jobs"])
    assert sum(j["measure_prediction"] for j in projection["jobs"]) == 2*(len(chosen["conditions"])+2)
    assert sum(j["artifact_representative"] for j in projection["jobs"]) == 2*len(chosen["conditions"])
    assert all(j["seed"] == 2026 for j in projection["jobs"]
               if j["method"] == "lstm" and j["artifact_representative"])
    path = tmp_path/"cost-plan.json"; path.write_text(json.dumps(projection))
    plan.verify_cost_plan(tmp_path, frozen, "a"*64)
    with pytest.raises(ValueError, match="cost plan differs"):
        plan.verify_cost_plan(tmp_path, frozen, "c"*64)
    projection["jobs"][-1]["measure_prediction"] = True
    path.write_text(json.dumps(projection))
    with pytest.raises(ValueError, match="cost plan differs"):
        plan.verify_cost_plan(tmp_path, frozen, "a"*64)
    with pytest.raises(ValueError, match="Confirmation"):
        plan.verify_cost_plan(tmp_path, {"period": "confirmation"}, "a"*64)


@pytest.mark.parametrize("condition", [make_condition("persistence"),
    make_condition("direct", objective="l1", trees=32, leaves=7),
    make_condition("blend", objective="l1", trees=32, leaves=7, ridge_weight=.5),
    make_condition("blend", objective="l2", trees=32, leaves=7, ridge_weight=.2)])
def test_invented_worker_complete_reload_component_and_metric_audit(tmp_path, monkeypatch, condition):
    start, end = 400, 412
    values = 100 + 7*np.sin(np.arange(end)/7) + np.arange(end)/20
    observed = np.ones(end, dtype=bool)
    clock = pd.date_range("2018-01-01", periods=end, freq="h")
    data = {"traffic": values, "traffic_observed": values.copy(), "observed": observed, "timestamps": clock}
    targets = np.arange(start, end)
    method = condition["id"]
    job = {"id": f"{method}--Cell_test--r1", "method": method, "condition": condition,
        "cell": "Cell_test", "repeat": 1, "seed": None, "fit_end": start,
        "score_start": start, "score_end": end, "predict": True, "period": "confirmation"}
    p = {"jobs": [job], "common_origins": [start, end-1], "series": {"Cell_test": {"targets": targets.tolist()}},
         "development_evidence": {"environment_identity": {}}, "operator": "user", "host_ready_assertion": True}
    config, digest = plan.read_config(), "a"*64
    monkeypatch.setattr(run, "verify_plan", lambda _: (deepcopy(p), config, digest))
    monkeypatch.setattr(run, "load_series", lambda *a: (data, targets))
    checks = []
    def readiness(*a, **k):
        checks.append(k)
        return {"synthetic": True, "operator": k["operator"]}
    monkeypatch.setattr(common, "native_readiness", readiness)
    monkeypatch.setattr(common, "verify_saved_readiness", lambda *a: None)
    attempt, cached = begin_attempt(tmp_path, job, digest)
    (attempt/"operator.json").write_text(json.dumps(run.execution_identity("automated", False)))
    run.worker(tmp_path, job["id"], attempt, "fit")
    run.worker(tmp_path, job["id"], attempt, "predict")
    report = run.validate_result(attempt, job, p, digest)
    assert len(checks) == 4 and all(c["operator"] == "automated" for c in checks)
    assert report["fit"]["execution_operator"]["operator"] == "automated"
    assert report["fit"]["training"]["condition"] == condition
    assert report["fit"]["training"]["cached_component_reused"] is False
    assert report["predict"]["reload_predictions_exact"]
    if condition["kind"] == "blend":
        components = run.read_components(attempt, condition)
        assert components["ridge_raw"] is not None and components["direct_raw"] is not None
    with pytest.raises(ValueError, match="already exists"):
        run.worker(tmp_path, job["id"], attempt, "fit")
    result = json.loads((attempt/"predict.json").read_text())
    result["evaluation"]["scores"]["MAE"] += 1
    (attempt/"predict.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="arithmetic"):
        run.validate_result(attempt, job, p, digest)


def test_each_invocation_requires_truthful_actor_independent_of_preparation(tmp_path):
    assert run.execution_identity("automated", False) == {"operator": "automated", "host_ready_assertion": False}
    assert run.execution_identity("user", True)["operator"] == "user"
    for actor, ready in ((None, False), ("user", False), ("human", True), ("automated", 1)):
        with pytest.raises(ValueError, match="actual --operator"):
            run.execution_identity(actor, ready)
    with pytest.raises(ValueError, match="actual --operator"):
        run.run(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_seventh_standalone_can_win_and_endpoint_self_interval_is_zero():
    cells = ["A"]
    summaries = {m: {"status": "complete", "expected_cells": cells,
                    "contributors": {"MASE": cells}, "scores": {"MASE": 1.}}
                 for m in plan.STANDALONES}
    summaries["lightgbm_l1"]["scores"]["MASE"] = .9
    assert select_primary_comparator(summaries, candidate_order=plan.STANDALONES)["method"] == "lightgbm_l1"
    actual = np.arange(50., 90.)
    panel = {"A": {"actual": actual, "eligible": np.ones(40, bool), "weekly_scale": 2.,
                    "hybrid_by_seed": {"None": actual+1}, "comparator_by_seed": {"None": actual+1}}}
    intervals = paired_calendar_intervals(panel, cells, replicates=20, block_lengths=(7, 14))
    assert all(x["interval"] == [0., 0.] for x in intervals["blocks"].values())


@pytest.mark.parametrize("period", ["confirmation", "final"])
def test_complete_saved_forecast_summary_uses_dynamic_panel_and_primary(tmp_path, monkeypatch, period):
    # This is invented saved-forecast arithmetic, not a real evaluation or fit.
    from tests.test_ensemble_measurement import records, CONDITIONS
    chosen = selection()
    assert chosen["conditions"] == CONDITIONS
    cells = ["A", "B"]
    reports = records(period, tuple(cells))
    jobs = plan.jobs_for(period, cells, CONDITIONS)
    by_id = {r["job"]["id"]: r for r in reports}
    start, end = plan.PERIODS[period]
    values = 100 + np.arange(end, dtype=float)/100 + np.sin(np.arange(end)/8)
    clock = pd.date_range("2018-01-01", periods=end, freq="h")
    observed = np.ones(end, dtype=bool)
    targets = np.arange(start, end)
    data = {"traffic": values, "traffic_observed": values, "observed": observed, "timestamps": clock}
    p = {"period": period, "selection": chosen, "selection_sha256": "b"*64,
         "primary_model": chosen["primary_candidate"], "cells": cells, "jobs": jobs,
         "primary_comparator": "lightgbm_l1" if period == "final" else None,
         "scope": "invented test", "design_limits": [], "series": {c: {"targets": targets.tolist()} for c in cells}}
    monkeypatch.setattr(run, "verify_plan", lambda _: (p, {}, "a"*64))
    monkeypatch.setattr(run, "load_series", lambda *a: (data, targets))
    paths = {}
    for job in jobs:
        path = tmp_path/"jobs"/job["id"]/"attempt-0001"
        paths[job["id"]] = path
    monkeypatch.setattr(run, "load_success", lambda _, job_id, checksum: (paths[job_id], {}))
    monkeypatch.setattr(run, "validate_result", lambda attempt, job, *a: by_id[job["id"]])
    monkeypatch.setattr(run, "sha256", lambda _: "c"*64)
    monkeypatch.setattr(run, "_read_json", lambda _: {"model_state_sha256": "d"*64})
    for row in reports:
        row["fit"]["model_state_sha256"] = "d"*64
    def component(condition):
        actual = values[targets]
        ridge, direct = actual+10, actual+(1 if condition["objective"] == "l1" else 2)
        kind = condition["kind"]
        if kind == "blend": raw = condition["ridge_weight"]*ridge + (1-condition["ridge_weight"])*direct
        elif kind == "ridge": raw = ridge
        elif kind == "direct": raw = direct
        else: raw = actual+3
        return {"raw": raw, "final": np.maximum(0., raw),
                "ridge_raw": ridge if kind in ("ridge", "blend") else None,
                "direct_raw": direct if kind in ("direct", "blend") else None}
    monkeypatch.setattr(run, "read_components", lambda _, condition: component(condition))
    def predictions(attempt, job, _):
        comp = component(job["condition"])
        return pd.DataFrame({"target_index": targets, "prediction_raw": comp["raw"]})
    monkeypatch.setattr(run, "read_predictions", predictions)
    summary = run._summarize(tmp_path, seal=False)
    assert set(summary["methods"]) == {c["id"] for c in CONDITIONS}
    assert summary["primary_comparator"] == "lightgbm_l1"
    assert summary["primary_model"] == "ensemble_l1_r0p3"
    assert summary["designated_primary_comparison"]["metrics"]["MASE"]["absolute_difference"] > 0
    assert summary["saved_ensembles_match_independent_components_exactly"]
    assert (summary["uncertainty"] is not None) == (period == "final")
