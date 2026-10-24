"""Invented-data scientific, provenance and failure-path checks; no LTE fits."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from adaptation import plan as extension
from adaptation import run_ridge_extension as runner
from forecasting import development_plan
from forecasting.development_selection import select_stage
from tools.development_store import begin_attempt, complete_attempt, load_success, sha256, _inventory
from tests.test_development_experiments import invented_series, save_series, put_json


@pytest.fixture
def config():
    return runner.original.read_config()


def records(choices, scores=None):
    scores = scores or {}
    return [{"candidate_id": c["candidate_id"], "cell": cell, "fold_index": fold,
             "seed": None, "status": "passed", "mase_scale": 2.0,
             "scores": {"MASE": scores.get(c["candidate_id"], 1.0),
                        "MAE": 2*scores.get(c["candidate_id"], 1.0),
                        "RMSE": 3*scores.get(c["candidate_id"], 1.0)}}
            for c in choices for cell in development_plan.TUNING_CELLS for fold in range(3)]


def ridge_rows(config, old=0.9, new=0.8):
    initial = records(development_plan.candidates(config, "ridge"), {"ridge-a1000": old})
    added = records(extension.fit_candidates(config, "ridge"), {"ridge-a10000": new, "ridge-a100000": new+0.1})
    return initial, added


def test_budget_chronology_and_overlay_are_exact_and_nonmutating(config):
    before = copy.deepcopy(config)
    effective = extension.effective_config(config)
    assert effective["models"]["ridge"]["alphas"] == [0.1, 1, 10, 100, 1000, 10000, 100000]
    effective["models"]["ridge"]["alphas"] = before["models"]["ridge"]["alphas"]
    assert effective == config == before
    for stage, n in (("ridge", 54), ("hybrid", 162)):
        jobs = extension.build_jobs(config, stage)
        assert len(jobs) == len({j["id"] for j in jobs}) == n
        assert {j["cell"] for j in jobs} == set(development_plan.TUNING_CELLS)
        assert {(j["fit_end"], j["score_start"], j["score_end"]) for j in jobs} == {(a,a,b) for a,b in development_plan.FOLDS}
        assert all(j["seed"] is None and j["score_end"] <= 6132 for j in jobs)
    assert len(extension.scoring_candidates(config, "hybrid")) == 24
    assert {c["parameters"]["weight"] for c in extension.scoring_candidates(config, "hybrid")} == {.25,.5,.75,1}
    with pytest.raises(ValueError):
        extension.fit_candidates(config, "confirmation")


@pytest.mark.parametrize("old,new,rebuild,alpha", [(0.8,0.9,False,1000),(0.9,0.8,True,10000),(.8,.8,True,10000)])
def test_global_ridge_selection_and_conditional_branch(config, old, new, rebuild, alpha):
    initial, added = ridge_rows(config, old, new)
    result = extension.select_ridge(initial, added, config)
    assert len(result["candidates"]) == 7
    assert result["selected"]["parameters"]["alpha"] == alpha
    assert result["rebuild_corrections"] is rebuild
    assert result["primary_comparator"] is None


def test_selection_matches_original_aggregation_bit_for_bit(config):
    choices = development_plan.candidates(config, "ridge")
    rows = records(choices)
    for i, r in enumerate(rows):
        r["scores"] = {"MASE": .17 + (i % 13) / 71, "MAE": 2*(.17+(i%13)/71), "RMSE": .7 + (i%17)/31}
    assert extension.aggregate(rows, config, choices) == select_stage(rows, config, "ridge")["candidates"]


def test_ridge_ties_use_full_precision_not_rounded_scores(config):
    initial, added = ridge_rows(config, 1.0, 1.0)
    for row in added:
        score = 1.0 if row["candidate_id"] == "ridge-a10000" else np.nextafter(1.0, 2.0)
        row["scores"].update(MASE=score, MAE=2*score)
    assert extension.select_ridge(initial, added, config)["selected"]["parameters"]["alpha"] == 10000
    for row in added:
        row["scores"].update(MASE=1.0, MAE=2.0)
    assert extension.select_ridge(initial, added, config)["selected"]["parameters"]["alpha"] == 100000


@pytest.mark.parametrize("damage", ["missing","duplicate","nan","failed","cell","fold","seed","scale","metric"])
def test_no_partial_or_invalid_ridge_cohorts_enter_selection(config, damage):
    initial, added = ridge_rows(config)
    if damage == "missing": added.pop()
    elif damage == "duplicate": added.append(copy.deepcopy(added[0]))
    elif damage == "nan": added[0]["scores"]["MASE"] = float("nan")
    elif damage == "failed": added[0]["status"] = "failed"
    elif damage == "cell": added[0]["cell"] = "unknown"
    elif damage == "fold": added[0]["fold_index"] = True
    elif damage == "seed": added[0]["seed"] = 2026
    elif damage == "metric": added[0]["scores"]["MAE"] += 1
    else:
        added[0]["mase_scale"] = 3
        added[0]["scores"]["MAE"] = 3*added[0]["scores"]["MASE"]
    with pytest.raises(ValueError): extension.select_ridge(initial, added, config)


@pytest.mark.parametrize("zero,old,new,origin,weight", [
    (.8,.8,.8,"expanded_ridge",0), (.9,.8,.8,"initial",.25),
    (.9,.85,.8,"extension",.25), (.9,.8,.85,"initial",.25)])
def test_hybrid_ties_fallback_alpha_and_zero_identity(config, zero, old, new, origin, weight):
    initial, added = ridge_rows(config, 1.0, zero)
    ridge = extension.select_ridge(initial, added, config)
    old_rows = records(development_plan.score_candidates(config,"hybrid")[1:])
    new_rows = records(extension.scoring_candidates(config,"hybrid"))
    for collection, score in ((old_rows,old),(new_rows,new)):
        for r in collection: r["scores"].update(MASE=score, MAE=2*score)
    result = extension.select_hybrid(old_rows,1000,ridge,initial+added,new_rows,config)
    chosen = result["selected"]
    assert chosen["origin"] == origin and chosen["parameters"]["weight"] == weight
    assert chosen["ridge_alpha"] == (1000 if origin == "initial" else 10000)
    assert result["hybrid_success_claimed"] is False
    assert result["ridge_preferred"] is (weight == 0)
    assert len(result["original_positive_candidates"]) == 24
    if weight:
        assert chosen["parameters"]["trees"] == 32 and chosen["parameters"]["leaves"] == 7


def test_skipped_corrections_need_no_new_records_and_preserve_original(config):
    initial, added = ridge_rows(config, .8,.9)
    ridge = extension.select_ridge(initial,added,config)
    old = records(development_plan.score_candidates(config,"hybrid")[1:])
    result = extension.select_hybrid(old,1000,ridge,initial+added,[],config)
    assert result["correction_stage_skipped"] and result["skip_reason"]
    assert len(result["candidates"]) == 2
    with pytest.raises(ValueError, match="forbids"):
        extension.select_hybrid(old,1000,ridge,initial+added,records(extension.scoring_candidates(config,"hybrid")),config)


def test_hybrid_rejects_cross_round_denominator_change(config):
    initial,added = ridge_rows(config)
    old = records(development_plan.score_candidates(config,"hybrid")[1:])
    new = records(extension.scoring_candidates(config,"hybrid"))
    for r in new:
        r["mase_scale"] = 3.0
        r["scores"]["MAE"] = 3*r["scores"]["MASE"]
    with pytest.raises(ValueError,match="share"):
        extension.select_hybrid(old,1000,extension.select_ridge(initial,added,config),initial+added,new,config)


@pytest.fixture
def prepared(tmp_path, monkeypatch, config):
    source = tmp_path / "original"
    source.mkdir()
    series = {}
    data = invented_series()
    for cell in development_plan.TUNING_CELLS:
        series[cell] = save_series(source, cell, data)
        series[cell]["folds"] = [{"targets": runner.targets_for(data,j).tolist()} for j in development_plan.build_jobs(config,"ridge")[:27] if j["cell"] == cell]
    from tools.run_development_pilot import code_hashes
    archived_code = code_hashes()
    for name, pair in runner.historical.catalog()["development_revisions"].items():
        archived_code[name] = pair["original_sha256"]
    original_plan = {"series":series,"code_sha256":archived_code,"environment_identity":{"invented":True}}
    put_json(source / "plan.json", original_plan)
    initial, _ = ridge_rows(config)
    hybrid = records(development_plan.score_candidates(config,"hybrid")[1:])
    all_records = {"ridge":initial,"hybrid":hybrid}
    ridge_selection = select_stage(initial,config,"ridge")
    hybrid_zero = [{**r,"candidate_id":"hybrid-zero"} for r in initial if r["candidate_id"] == ridge_selection["selected"]["candidate_id"]]
    summary = {"selections":{"ridge":{"selection":ridge_selection},"hybrid":{"selection":select_stage(hybrid+hybrid_zero,config,"hybrid")}}}
    put_json(source / "development-summary.json", summary)
    monkeypatch.setattr(runner,"PLAN_SHA256",sha256(source / "plan.json"))
    monkeypatch.setattr(runner,"SUMMARY_SHA256",sha256(source / "development-summary.json"))
    monkeypatch.setattr(runner.original,"environment_identity",lambda:({"invented":True},{"invented":True}))
    monkeypatch.setattr(runner,"audit_original",lambda _: (original_plan,config,all_records,_inventory(source)))
    directory = tmp_path / "extension"
    before = _inventory(source)
    runner.prepare(source,directory)
    return source,directory,before


def test_preparation_reads_only_original_prefixes_and_preserves_source(prepared):
    from tools.run_development_pilot import code_hashes
    source,directory,before = prepared
    plan,_,_ = runner.verify_plan(directory)
    assert _inventory(source) == before
    assert len(plan["jobs"]) == 216
    assert not (directory / "jobs").exists()
    assert not (directory / "selections").exists()
    assert plan["runtime_development_code_sha256"] == code_hashes()
    assert plan["original_code_sha256"] != plan["runtime_development_code_sha256"]
    assert plan["source_compatibility_sha256"] == runner.historical.catalog_sha256()
    for cell in development_plan.TUNING_CELLS:
        assert all(len(a) == 6132 for a in runner.load_series(directory,plan,cell).values())


@pytest.mark.parametrize("change", ["effective_config","new_alphas","jobs","maximum_new_fits","series","extension_code_sha256","holdout_forecasts_enabled",
                                  "runtime_development_code_sha256", "source_compatibility_sha256"])
def test_resealed_changed_plan_fails_closed(prepared,change):
    _,directory,_ = prepared
    p=runner.read_json(directory / "plan.json")
    p[change] = True if change == "holdout_forecasts_enabled" else []
    put_json(directory / "plan.json",p)
    put_json(directory / "plan-checksum.json",{"sha256":sha256(directory / "plan.json")})
    with pytest.raises(ValueError): runner.verify_plan(directory)


@pytest.mark.parametrize("change", ["source_artifact","source_summary","prefix","records","extra_job","environment"])
def test_changed_original_or_extension_evidence_is_rejected(prepared, monkeypatch, change):
    source,directory,_=prepared
    p,_,_=runner.verify_plan(directory)
    if change == "source_artifact":
        (source / "extra.txt").write_text("unexpected")
        with pytest.raises(ValueError,match="inventory"):
            runner.verify_source_inventory(directory,p)
        return
    if change == "source_summary": (source / "development-summary.json").write_text("changed")
    elif change == "prefix": (directory / "series" / f"{development_plan.TUNING_CELLS[0]}.npz").write_bytes(b"changed")
    elif change == "records": (directory / "original-records.json").write_text("changed")
    elif change == "extra_job": (directory / "jobs/unplanned").mkdir(parents=True)
    else: monkeypatch.setattr(runner.original,"environment_identity",lambda:({"changed":True},{}))
    with pytest.raises(ValueError): runner.verify_plan(directory)


@pytest.mark.parametrize("relation", ["same","child","parent","existing"])
def test_prepare_never_overwrites_or_nests_original(prepared,relation):
    source,directory,before=prepared
    output={"same":source,"child":source/"new","parent":source.parent,"existing":directory}[relation]
    with pytest.raises(ValueError,match="new output"):
        runner.prepare(source,output)
    assert _inventory(source)==before


def test_prepare_rejects_overlap_through_symlink_alias(prepared):
    source,_,before=prepared
    alias=source.parent / "alias"
    alias.symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError,match="new output"):
        runner.prepare(source,alias / "nested")
    assert _inventory(source)==before


@pytest.mark.parametrize("change",["extra_file","session_timing"])
def test_audit_rejects_source_mutation_during_readonly_verification(tmp_path,config,monkeypatch,change):
    initial,_=ridge_rows(config)
    checksum="a"*64
    selection={"stage":"ridge","plan_sha256":checksum,"evidence":{},
               "selection":select_stage(initial,config,"ridge"),"primary_comparator":None,
               "confirmation_or_final_evaluated":False}
    (tmp_path / "selections").mkdir()
    put_json(tmp_path / "selections/ridge.json",selection)
    session=tmp_path / "sessions/session-0001"
    session.mkdir(parents=True)
    put_json(session / "started.json",{"stage":"ridge","plan_sha256":checksum})
    put_json(session / "finished.json",{"status":"passed","wall_seconds":2.0})
    put_json(tmp_path / "development-summary.json",{"status":"passed","plan_sha256":checksum,
             "selections":{"ridge":selection},"holdout_forecasts_evaluated":False,
             "attempts_including_failures":[],"stage_sessions_including_failures":[{
                 "session":"session-0001","stage":"ridge","timing":{"status":"passed","wall_seconds":2.0}}]})
    monkeypatch.setattr(runner,"source_identity",lambda _:None)
    monkeypatch.setattr(runner.historical,"verify_original_plan",lambda _:({},config,checksum))
    monkeypatch.setattr(development_plan,"STAGES",("ridge",))
    def collect(*args):
        if change=="extra_file": (tmp_path / "changed-during-audit.txt").write_text("new evidence")
        else: put_json(session / "finished.json",{"status":"passed","wall_seconds":3.0})
        return initial,{}
    monkeypatch.setattr(runner.original,"collect_stage",collect)
    with pytest.raises(ValueError,match="inventory|timing"):
        runner.audit_original(tmp_path)


@pytest.fixture
def small_worker(tmp_path,config,monkeypatch):
    data=invented_series()
    cell=development_plan.TUNING_CELLS[0]
    plan={"series":{cell:save_series(tmp_path,cell,data)},"jobs":[]}
    checksum="e"*64
    monkeypatch.setattr(runner,"verify_plan",lambda _:(plan,config,checksum))
    def make(stage):
        j=copy.deepcopy(extension.build_jobs(config,stage)[0])
        j.update(fit_end=280,score_start=280,score_end=310)
        plan["jobs"].append(j)
        attempt,_=begin_attempt(tmp_path,j,checksum)
        return j,attempt
    return tmp_path,plan,config,checksum,data,make


def test_actual_invented_workers_preserve_alpha_residual_identity_and_metrics(small_worker,monkeypatch):
    from forecasting import development_models
    directory,plan,config,checksum,data,make=small_worker
    original_fit=development_models.fit_candidate
    def fit(job,values,flags,clock,effective,**kwargs):
        assert len(values)==len(flags)==len(clock)==280
        assert effective == extension.effective_config(config)
        return original_fit(job,values,flags,clock,effective,**kwargs)
    monkeypatch.setattr(development_models,"fit_candidate",fit)
    ridge_job,ridge_attempt=make("ridge")
    runner.worker(directory,ridge_job["id"],ridge_attempt)
    ridge_result=runner.validate_result(ridge_attempt,ridge_job,plan,checksum,config)
    complete_attempt(ridge_attempt,ridge_job,checksum)
    assert ridge_result["settings"]["alpha"]==10000
    (directory / "selections").mkdir()
    picked={"selection":{"selected":{"candidate_id":ridge_job["candidate_id"],"parameters":{"alpha":10000}},"rebuild_corrections":True}}
    put_json(directory / "selections/ridge.json",picked)
    monkeypatch.setattr(runner,"checked_ridge_selection",lambda *a,**kw:picked)
    dependency={"job_id":ridge_job["id"],"attempt":str(ridge_attempt.relative_to(directory)),
                "completion_sha256":sha256(ridge_attempt / "complete.json"),
                "ridge_selection_sha256":sha256(directory / "selections/ridge.json")}
    monkeypatch.setattr(runner,"ridge_dependency",lambda *a:(ridge_attempt,dependency))
    monkeypatch.setattr(development_models,"fit_ridge",lambda *a:pytest.fail("Residual stage refitted ridge"))
    job,attempt=make("hybrid")
    runner.worker(directory,job["id"],attempt)
    result=runner.validate_result(attempt,job,plan,checksum,config)
    assert result["training"]["component_fit_counts"]=={"ridge":0,"correction":1}
    assert result["settings"]["alpha"]==10000 and len(result["records"])==4
    assert result["reload_predictions_exact"] and result["fitted_state_unchanged"]
    assert result["controls_before"]["runtime_library_controls_verified"]
    assert result["controls_after"]["runtime_library_controls_verified"]
    assert result["formal_timing_eligible"] is False
    for record in result["records"]:
        frame=pd.read_csv(attempt / "predictions" / f"{record['candidate_id']}.csv",float_precision="round_trip")
        residual=frame.actual.to_numpy()-frame.prediction.to_numpy()
        assert record["scores"]["MAE"]==pytest.approx(np.abs(residual).mean())
        assert record["scores"]["RMSE"]==pytest.approx(np.sqrt(np.mean(residual**2)))
        assert record["scores"]["MASE"]==pytest.approx(np.abs(residual).mean()/record["mase_scale"])
        np.testing.assert_array_equal(frame.prediction,np.maximum(frame.prediction_raw,0))
    with np.load(attempt / "components.npz",allow_pickle=False) as saved:
        components={k:saved[k] for k in saved.files}
    components["ridge_z"][0]+=1
    np.savez(attempt / "components.npz",**components)
    with pytest.raises(AssertionError): runner.validate_result(attempt,job,plan,checksum,config)


def test_worker_never_runs_correction_for_original_ridge_winner(small_worker,monkeypatch):
    from forecasting import development_models
    directory,_,_,_,_,make=small_worker
    job,attempt=make("hybrid")
    monkeypatch.setattr(runner,"checked_ridge_selection",lambda *a:{"selection":{"rebuild_corrections":False}})
    monkeypatch.setattr(development_models,"fit_candidate",lambda *a,**kw:pytest.fail("Unexpected correction fit"))
    with pytest.raises(ValueError,match="forbids"):
        runner.worker(directory,job["id"],attempt)


@pytest.fixture
def lifecycle(tmp_path,monkeypatch):
    job={"id":"invented-job","stage":"ridge"}
    checksum="d"*64
    monkeypatch.setattr(runner,"verify_plan",lambda _:({}, {}, checksum))
    monkeypatch.setattr(runner,"verify_current_identity",lambda *a:None)
    monkeypatch.setattr(runner,"verify_source_inventory",lambda *a:None)
    monkeypatch.setattr(runner,"build_jobs",lambda *a:[job])
    monkeypatch.setattr(runner,"selection",lambda *a:{"selection":{"selected":None}})
    monkeypatch.setattr(runner,"validate_result",lambda *a:None)
    return tmp_path,job,checksum


def test_fail_retry_and_success_resume_preserve_all_attempt_bytes(lifecycle,monkeypatch):
    directory,job,checksum=lifecycle
    launches=[]
    def launch(_,__,attempt,lock):
        launches.append(attempt)
        (attempt / "worker.log").write_text("invented")
        return 1 if len(launches)==1 else 0
    monkeypatch.setattr(runner,"launch",launch)
    with pytest.raises(RuntimeError): runner.run_stage(directory,"ridge",host_ready=True)
    before=_inventory(launches[0])
    with pytest.raises(ValueError): runner.run_stage(directory,"ridge",host_ready=True,resume=True)
    runner.run_stage(directory,"ridge",host_ready=True,resume=True,retry_failed=True)
    runner.run_stage(directory,"ridge",host_ready=True,resume=True)
    assert len(launches)==2 and _inventory(launches[0])==before
    assert load_success(directory,job["id"],checksum)[0]==launches[1]
    costs=runner.cost_records(directory,checksum)
    assert len(costs["attempts_including_failures"])==2
    assert costs["recorded_stage_wall_seconds"]>0
    (launches[1] / "worker.log").write_text("changed")
    with pytest.raises(ValueError,match="inventory"):
        runner.run_stage(directory,"ridge",host_ready=True,resume=True)


def test_stage_skips_all_correction_workers_for_old_winner(lifecycle,monkeypatch):
    directory,_,_=lifecycle
    monkeypatch.setattr(runner,"checked_ridge_selection",lambda *a:{"selection":{"rebuild_corrections":False}})
    monkeypatch.setattr(runner,"launch",lambda *a:pytest.fail("No correction worker permitted"))
    runner.run_stage(directory,"hybrid",host_ready=True)
    assert not (directory / "jobs").exists()


def test_identity_failure_does_not_seal_attempt(lifecycle,monkeypatch):
    directory,_,_=lifecycle
    monkeypatch.setattr(runner,"launch",lambda *a:0)
    def fail(*args): raise ValueError("Code changed")
    monkeypatch.setattr(runner,"verify_current_identity",fail)
    with pytest.raises(ValueError,match="Code changed"):
        runner.run_stage(directory,"ridge",host_ready=True)
    attempt=next((directory / "jobs").glob("*/attempt-*"))
    assert (attempt / "failure.json").exists() and not (attempt / "complete.json").exists()


def test_stage_rechecks_source_after_selection(lifecycle,monkeypatch):
    directory,_,_=lifecycle
    monkeypatch.setattr(runner,"build_jobs",lambda *a:[])
    state={"selected":False}
    def selection(*a):
        state["selected"]=True
        return {"selection":{"selected":None}}
    def inventory(*a):
        if state["selected"]: raise ValueError("Source changed during selection")
    monkeypatch.setattr(runner,"selection",selection)
    monkeypatch.setattr(runner,"verify_source_inventory",inventory)
    with pytest.raises(ValueError,match="Source changed"):
        runner.run_stage(directory,"ridge",host_ready=True)
    finish=runner.read_json(directory / "sessions/session-0001/finished.json")
    assert finish["status"]=="failed_or_interrupted"


def test_summarized_extension_forbids_new_stage_sessions(lifecycle):
    directory,_,_=lifecycle
    put_json(directory / "extension-summary.json",{"status":"passed"})
    with pytest.raises(ValueError,match="already summarized"):
        runner.run_stage(directory,"ridge",host_ready=True)
    assert not (directory / "sessions").exists()


def test_costs_keep_unknown_interrupted_duration_and_reject_invalid_values(tmp_path):
    attempt=tmp_path / "jobs/invented/attempt-0001"
    attempt.mkdir(parents=True)
    value=runner.cost_records(tmp_path,"a"*64)
    assert value["attempts_with_unavailable_elapsed_time"]==1
    assert value["recorded_attempt_wall_seconds"]==0
    put_json(attempt / "elapsed.json",{"wall_seconds":-1})
    with pytest.raises(ValueError,match="duration"):
        runner.cost_records(tmp_path,"a"*64)


@pytest.mark.parametrize("args", [
    [],["--stage","ridge"],["--stage","ridge","--execute-extension"],
    ["--stage","confirmation"],["--stage","final"],["--stage","all"],
    ["--stage","summarize","--execute-extension"],
    ["--original-dir","x","--host-ready"],
    ["--stage","ridge","--execute-extension","--host-ready","--retry-failed"]])
def test_cli_never_implicitly_fits(tmp_path,monkeypatch,args):
    monkeypatch.setattr(sys,"argv",["runner","--output-dir",str(tmp_path),*args])
    monkeypatch.setattr(runner,"run_stage",lambda *a,**kw:pytest.fail("Unexpected stage"))
    monkeypatch.setattr(runner,"prepare",lambda *a:pytest.fail("Unexpected prepare"))
    with pytest.raises(SystemExit) as error: runner.main()
    assert error.value.code==2
