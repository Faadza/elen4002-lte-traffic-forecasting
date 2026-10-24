"""Prepare, execute or summarize the frozen 216-fit development-only revision.

Uses only the completed original run and its nine saved development prefixes.
Every successful attempt is immutable. Failed attempts require explicit retry.
"""
import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from adaptation.plan import PLAN_SHA256, SUMMARY_SHA256, aggregate
from adaptation import historical as archive
from adaptation.run_ridge_extension import (audit_original, read_json, save_or_check,
                                           source_identity, stage_session)
from forecasting.development_plan import TUNING_CELLS
from l1_revision.plan import COUNTS, STAGE, build_jobs, scientific_specification, select
from tools import run_development_experiments as original
from tools.development_store import (begin_attempt, complete_attempt, load_success,
                                     run_lock, sha256, verify_inventory, write_json_atomic)
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]


def package_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p)
            for package in ("l1_revision", "adaptation", "blend_revision")
            for p in sorted((ROOT / package).rglob("*")) if p.suffix in (".py", ".json")}


def historical_blend(source, initial, config):
    """Reproduce the fixed, already selected 20/80 blend without fitting."""
    import numpy as np
    import pandas as pd
    from tools.pilot_results import score_predictions
    rows = []
    for job in build_jobs(config)[:27]:
        data = original.load_series(source, initial, job["cell"])
        targets = original.targets_for(data, job)
        frames = []
        for cid in ("ridge-a1000", "direct-t128-l7"):
            rid = f"{cid}--{job['cell']}--f{job['fold_index']+1}"
            success = load_success(source, rid, PLAN_SHA256)
            if success is None:
                raise ValueError("Missing historical blend component")
            path, _ = success
            frame = pd.read_csv(path / "predictions" / f"{cid}.csv", float_precision="round_trip")
            np.testing.assert_array_equal(frame.target_index, targets)
            np.testing.assert_array_equal(frame.actual, data["traffic_observed"][targets])
            frames.append(frame)
        raw = 0.2 * frames[0].prediction_raw.to_numpy() + 0.8 * frames[1].prediction_raw.to_numpy()
        evaluation = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
            job["fit_end"], job["score_start"], job["score_end"], targets, raw, np.maximum(0, raw))
        rows.append({"candidate_id": "historical-blend-r0p2", "cell": job["cell"],
                     "fold_index": job["fold_index"], "seed": None, "status": "passed",
                     "scores": evaluation["scores"], "mase_scale": evaluation["weekly_scale"]})
    choice = {"candidate_id": "historical-blend-r0p2", "method": "ridge_direct_blend",
              "parameters": {"ridge_alpha": 1000, "ridge_weight": 0.2, "direct_trees": 128, "direct_leaves": 7}}
    return {"records": rows, "summary": aggregate(rows, config, [choice])[0],
            "interpretation": "historical development-selected reference; no new tuning or fits"}


def prepare(source, directory):
    from tools.run_development_pilot import code_hashes
    source, directory = source.resolve(), directory.resolve()
    if directory.exists() or source == directory or source in directory.parents or directory in source.parents:
        raise ValueError("Use a new output directory outside the original run")
    initial, config, records, inventory = audit_original(source)
    selected = read_json(source / "development-summary.json")["selections"]
    if (selected["ridge"]["selection"]["selected"]["candidate_id"] != "ridge-a1000"
            or selected["direct"]["selection"]["selected"]["candidate_id"] != "direct-t128-l7"):
        raise ValueError("Original selected components differ from this revision")
    historical = historical_blend(source, initial, config)
    identity, machine = original.environment_identity()
    if identity != initial["environment_identity"]:
        raise ValueError("Preparation requires the original host, interpreter and packages")
    directory.mkdir(parents=True)
    (directory / "series").mkdir()
    for cell in TUNING_CELLS:
        relative = initial["series"][cell]["path"]
        shutil.copyfile(source / relative, directory / relative)
    for old, new in (("plan.json", "original-plan.json"), ("development-summary.json", "original-summary.json")):
        shutil.copyfile(source / old, directory / new)
    for name, value in (("original-records", records), ("original-inventory", inventory), ("historical-blend", historical)):
        write_json_atomic(directory / f"{name}.json", value)
    plan = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_specification": scientific_specification(config), "original_directory": str(source),
            "original_plan_sha256": PLAN_SHA256, "original_summary_sha256": SUMMARY_SHA256,
            "original_code_sha256": initial["code_sha256"], "revision_and_dependency_code_sha256": package_hashes(),
            "runtime_development_code_sha256": code_hashes(),
            "source_compatibility_sha256": archive.catalog_sha256(),
            "config_sha256": original.CONFIG_SHA256, "jobs": build_jobs(config), "series": initial["series"],
            "imported_sha256": {name: sha256(directory / name) for name in
                ("original-records.json", "original-inventory.json", "historical-blend.json")},
            "environment_identity": identity, "machine_at_preparation": machine, "cpu_startup_profile": CPU_ENV}
    write_json_atomic(directory / "plan.json", plan)
    write_json_atomic(directory / "plan-checksum.json", {"sha256": sha256(directory / "plan.json")})
    verify_plan(directory)
    verify_source_inventory(directory, plan)
    return plan


def verify_source_inventory(directory, plan):
    verify_inventory(Path(plan["original_directory"]), read_json(directory / "original-inventory.json"), exclude=(".run.lock",))


def verify_plan(directory):
    """Validate a new/current run, including its exact runtime and original host."""
    return _verify_plan(directory, archived=False)


def verify_archived_plan(directory):
    """Read the one pinned historical L1 run; never authorise its execution."""
    return _verify_plan(directory, archived=True)


def _verify_plan(directory, *, archived):
    from tools.run_development_pilot import code_hashes
    config = original.read_config()
    plan = read_json(directory / "plan.json")
    checksum = read_json(directory / "plan-checksum.json")["sha256"]
    if sha256(directory / "plan.json") != checksum:
        raise ValueError("Revision plan changed")
    source_identity(Path(plan["original_directory"]))
    if sha256(directory / "original-plan.json") != PLAN_SHA256 or sha256(directory / "original-summary.json") != SUMMARY_SHA256:
        raise ValueError("Copied original evidence changed")
    initial = read_json(directory / "original-plan.json")
    archive.verify_development_code(initial["code_sha256"])
    if archived:
        if checksum != archive.L1_PLAN_SHA256:
            raise ValueError("L1 archive is not the reviewed historical plan")
        archive.verify_archived_config(plan["config_sha256"])
        archive.verify_l1_package(plan["revision_and_dependency_code_sha256"], package_hashes())
    elif (plan["config_sha256"] != original.CONFIG_SHA256
            or plan.get("runtime_development_code_sha256") != code_hashes()
            or plan.get("source_compatibility_sha256") != archive.catalog_sha256()
            or plan["revision_and_dependency_code_sha256"] != package_hashes()):
        raise ValueError("Revision differs from its current code or configuration")
    if (plan["schema_version"] != 1 or plan["original_plan_sha256"] != PLAN_SHA256
            or plan["original_summary_sha256"] != SUMMARY_SHA256
            or plan["original_code_sha256"] != initial["code_sha256"]
            or plan["scientific_specification"] != scientific_specification(config)
            or plan["jobs"] != build_jobs(config) or plan["series"] != initial["series"]
            or plan["cpu_startup_profile"] != CPU_ENV):
        raise ValueError("Revision differs from its fixed source, scientific plan or workload")
    if plan["environment_identity"] != initial["environment_identity"]:
        raise ValueError("Revision requires the original machine, interpreter and package versions")
    if not archived:
        identity, _ = original.environment_identity()
        if identity != plan["environment_identity"]:
            raise ValueError("Revision requires the original machine, interpreter and package versions")
    expected_imports = {"original-records.json", "original-inventory.json", "historical-blend.json"}
    if set(plan["imported_sha256"]) != expected_imports:
        raise ValueError("Incomplete imported evidence")
    for name, digest in plan["imported_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError("Imported evidence changed")
    if {p.name for p in (directory / "series").iterdir()} != {f"{cell}.npz" for cell in TUNING_CELLS}:
        raise ValueError("Unexpected development prefix files")
    for cell in TUNING_CELLS:
        data = original.load_series(directory, plan, cell)
        for job in build_jobs(config)[:27]:
            if job["cell"] == cell and original.targets_for(data, job).tolist() != plan["series"][cell]["folds"][job["fold_index"]]["targets"]:
                raise ValueError("Development target cohort changed")
    known = {job["id"] for job in plan["jobs"]}
    if (directory / "jobs").exists() and any(p.name not in known or p.is_symlink() for p in (directory / "jobs").iterdir()):
        raise ValueError("Unexpected revision job evidence")
    return plan, config, checksum


def verify_current_identity(directory, plan, checksum):
    current, _, actual = verify_plan(directory)
    if current != plan or actual != checksum:
        raise ValueError("Revision identity changed during execution")


def ridge_dependency(plan, job):
    if job["parameters"]["beta"] == 0:
        return None, None
    source = Path(plan["original_directory"])
    rid = f"ridge-a1000--{job['cell']}--f{job['fold_index']+1}"
    success = load_success(source, rid, PLAN_SHA256)
    if success is None:
        raise ValueError("Missing immutable original ridge dependency")
    path, marker = success
    expected_job = next(j for j in read_json(source / "plan.json")["jobs"] if j["id"] == rid)
    if marker["job"] != expected_job:
        raise ValueError("Original ridge dependency job differs")
    return path, {"job_id": rid, "attempt": str(path.relative_to(source)),
                  "original_plan_sha256": PLAN_SHA256, "completion_sha256": sha256(path / "complete.json"),
                  "ridge_selection_sha256": sha256(source / "selections/ridge.json")}


def validate_result(attempt, job, plan, checksum, config):
    import numpy as np
    import pandas as pd
    from l1_revision.model import load_model, model_fingerprint, predict_candidate, predict_components
    from tools.cpu_controls import single_thread_cpu
    from tools.pilot_results import score_predictions
    result = read_json(attempt / "result.json")
    if result["job"] != job or result["plan_sha256"] != checksum or result["status"] != "passed":
        raise ValueError("Worker result identity differs")
    if (result["reload_predictions_exact"] is not True or result["fitted_state_unchanged"] is not True
            or result["formal_timing_eligible"] is not False):
        raise ValueError("Worker state or timing claims are invalid")
    for name in ("controls_before", "controls_after"):
        if result[name].get("runtime_library_controls_verified") is not True:
            raise ValueError("Worker CPU library controls are unverified")
    for name in ("fit_seconds", "initial_prediction_seconds"):
        if isinstance(result[name], bool) or not isinstance(result[name], (int, float)) or not math.isfinite(result[name]) or result[name] <= 0:
            raise ValueError("Worker measured durations must be positive and finite")
    _, dependency = ridge_dependency(plan, job)
    if result["dependency"] != dependency:
        raise ValueError("Wrong ridge dependency; beta=0 must not use ridge")
    data = original.load_series(attempt.parents[2], plan, job["cell"])
    targets = original.targets_for(data, job)
    frame = pd.read_csv(attempt / "predictions.csv", float_precision="round_trip")
    np.testing.assert_array_equal(frame.target_index, targets)
    np.testing.assert_array_equal(frame.actual, data["traffic_observed"][targets])
    np.testing.assert_array_equal(pd.to_datetime(frame.timestamp).to_numpy(), data["timestamps"][targets].to_numpy())
    if not frame.cell.eq(job["cell"]).all() or not frame.method.eq(job["method"]).all() or not frame.seed.isna().all():
        raise ValueError("Forecast identity differs from its planned cohort")
    with single_thread_cpu() as controls:
        controls.verify()
        bundle = load_model(attempt / "model")
        fingerprint = model_fingerprint(bundle)
        if (bundle.settings != result["settings"]
                or fingerprint != result["model_state_sha256"]):
            raise ValueError("Stored model differs from its worker result")
        raw, final = predict_candidate(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
        np.testing.assert_array_equal(raw, frame.prediction_raw)
        np.testing.assert_array_equal(final, frame.prediction)
        ridge_z, tree_z = predict_components(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
        with np.load(attempt / "components.npz", allow_pickle=False) as saved:
            if set(saved.files) != {"targets", "ridge_z", "tree_z"}:
                raise ValueError("Incomplete standardized component evidence")
            np.testing.assert_array_equal(saved["targets"], targets)
            np.testing.assert_array_equal(saved["ridge_z"], ridge_z)
            np.testing.assert_array_equal(saved["tree_z"], tree_z)
        np.testing.assert_array_equal(bundle.scaler.inverse(bundle.beta * ridge_z + tree_z), raw)
        if dependency is not None:
            from forecasting.pilot_models import load_model as load_ridge, model_fingerprint as ridge_fingerprint
            source = Path(plan["original_directory"]) / dependency["attempt"] / "model"
            if ridge_fingerprint(load_ridge(source)) != ridge_fingerprint(bundle.ridge_bundle):
                raise ValueError("Serialized ridge state differs from its original dependency")
        if model_fingerprint(bundle) != fingerprint:
            raise ValueError("Reload verification mutated fitted state")
        controls.verify()
    # The fitted bundle is verified against the fixed job by the model loader;
    # this explicit comparison also prevents valid models from another cohort.
    settings, training = result["settings"], result["training"]
    if (settings["fit_end"] != job["fit_end"] or bundle.beta != job["parameters"]["beta"]
            or settings["requested_trees"] != job["parameters"]["trees"] or settings["leaf_cap"] != job["parameters"]["leaves"]
            or settings["job"] != job or settings["lookback"] != 168
            or training["fit_interval"] != [0, job["fit_end"]]
            or training["training_target_count"] != settings["training_target_count"]
            or training["actual_trees"] != bundle.model.current_iteration()
            or training["component_fit_counts"] != {"direct_lightgbm_l1" if bundle.beta == 0 else "correction_l1": 1}
            or training.get("ridge_reused", False) != (bundle.beta > 0)):
        raise ValueError("Fitted settings differ from the fixed candidate/cohort")
    from models.lightgbm import build_parameters
    expected_parameters = build_parameters(config["models"]["trees"], job["parameters"]["trees"], job["parameters"]["leaves"])
    expected_parameters["objective"] = "regression_l1"
    if settings["tree_parameters"] != expected_parameters:
        raise ValueError("L1 objective or common fixed tree parameters changed")
    evaluation = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
        job["fit_end"], job["score_start"], job["score_end"], targets, raw, final)
    if evaluation != result["evaluation"] or evaluation["mase_status"] != "available":
        raise ValueError("Saved scores differ from independently scored complete forecasts")
    record = {"candidate_id": job["candidate_id"], "cell": job["cell"], "fold_index": job["fold_index"],
              "seed": None, "status": "passed", "scores": evaluation["scores"], "mase_scale": evaluation["weekly_scale"]}
    if result["record"] != record:
        raise ValueError("Selection record differs from verified forecasts")
    return result


def worker(directory, job_id, attempt):
    import numpy as np
    from forecasting.pilot_models import load_model as load_ridge
    from l1_revision.model import fit_candidate, predict_candidate, predict_components, save_model, load_model, model_fingerprint
    from tools.cpu_controls import single_thread_cpu
    from tools.pilot_measurements import elapsed_call, memory_snapshot
    from tools.pilot_results import score_predictions, save_predictions
    plan, config, checksum = verify_plan(directory)
    job = next((j for j in plan["jobs"] if j["id"] == job_id), None)
    if job is None or attempt.parent != directory / "jobs" / job_id:
        raise ValueError("Worker requires a planned job and assigned attempt")
    metadata = read_json(attempt / "attempt.json")
    number = metadata.get("attempt_number", 0)
    if (metadata.get("job") != job or metadata.get("plan_sha256") != checksum or metadata.get("schema_version") != 1
            or number < 1 or attempt.name != f"attempt-{number:04d}"
            or any((attempt / name).exists() for name in ("result.json", "complete.json", "failure.json"))):
        raise ValueError("Worker requires a fresh attempt with matching provenance")
    data = original.load_series(directory, plan, job["cell"])
    targets = original.targets_for(data, job)
    dependency_path, dependency = ridge_dependency(plan, job)
    with single_thread_cpu() as controls:
        before = controls.verify()
        ridge = None if dependency_path is None else load_ridge(dependency_path / "model")
        end = job["fit_end"]
        bundle, seconds = elapsed_call(lambda: fit_candidate(job, data["traffic"][:end],
            data["observed"][:end], data["timestamps"][:end], config, ridge_bundle=ridge))
        fingerprint = model_fingerprint(bundle)
        (raw, final), prediction_seconds = elapsed_call(lambda: predict_candidate(bundle,
            data["traffic"], data["observed"], data["timestamps"], targets))
        save_model(bundle, attempt / "model")
        loaded = load_model(attempt / "model")
        if model_fingerprint(loaded) != fingerprint:
            raise AssertionError("Reloaded state differs")
        again = predict_candidate(loaded, data["traffic"], data["observed"], data["timestamps"], targets)
        np.testing.assert_array_equal(raw, again[0])
        np.testing.assert_array_equal(final, again[1])
        ridge_z, tree_z = predict_components(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
        np.savez(attempt / "components.npz", targets=targets, ridge_z=ridge_z, tree_z=tree_z)
        evaluation = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
            end, job["score_start"], job["score_end"], targets, raw, final)
        if evaluation["mase_status"] != "available":
            raise ValueError("Unavailable MASE needs cohort review, never silent pruning")
        save_predictions(attempt / "predictions.csv", job["cell"], job["method"], None, targets,
                         data["timestamps"][targets], data["traffic_observed"][targets], raw, final)
        if model_fingerprint(bundle) != fingerprint:
            raise AssertionError("Scoring changed fitted state")
        result = {"status": "passed", "job": job, "plan_sha256": checksum, "dependency": dependency,
                  "settings": bundle.settings, "training": bundle.training_metadata,
                  "record": {"candidate_id": job["candidate_id"], "cell": job["cell"], "fold_index": job["fold_index"],
                             "seed": None, "status": "passed", "scores": evaluation["scores"], "mase_scale": evaluation["weekly_scale"]},
                  "evaluation": evaluation, "fit_seconds": seconds, "initial_prediction_seconds": prediction_seconds,
                  "timing_scope": "additional tree search only; positive beta reuses ridge; not full hybrid deployment cost",
                  "worker_lifetime_memory": memory_snapshot(), "model_state_sha256": fingerprint,
                  "reload_predictions_exact": True, "fitted_state_unchanged": True,
                  "controls_before": before, "controls_after": controls.verify(), "formal_timing_eligible": False}
    write_json_atomic(attempt / "result.json", result)


def launch(directory, job, attempt, lock_fd):
    arguments = [sys.executable, "-m", "l1_revision.run_experiment", "--worker", job["id"],
                 "--attempt", str(attempt), "--output-dir", str(directory)]
    with (attempt / "worker.log").open("x") as stream:
        child = subprocess.Popen(arguments, cwd=ROOT, env={**os.environ, **CPU_ENV},
                                 stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))
        try:
            return child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def collect(directory, plan, config, checksum):
    records, evidence = [], {}
    for job in plan["jobs"]:
        success = load_success(directory, job["id"], checksum)
        if success is None:
            raise ValueError(f"Incomplete revision: {job['id']}")
        path, marker = success
        if marker["job"] != job:
            raise ValueError("Stored job differs from the frozen plan")
        records.append(validate_result(path, job, plan, checksum, config)["record"])
        evidence[job["id"]] = {"attempt": str(path.relative_to(directory)), "completion_sha256": sha256(path / "complete.json")}
    return records, evidence


def run(directory, *, resume=False, retry_failed=False, host_ready=False, operator="user"):
    from tools.check_cpu_controls import machine_information
    if not host_ready or operator not in ("user", "automated"):
        raise ValueError("Execution needs explicit host readiness and operator identity")
    with run_lock(directory) as lock_fd:
        plan, config, checksum = verify_plan(directory)
        verify_source_inventory(directory, plan)
        if (directory / "l1-summary.json").exists():
            raise ValueError("Revision is already summarized; no further sessions allowed")
        with stage_session(directory, STAGE, checksum):
            for index, job in enumerate(plan["jobs"], 1):
                attempt, cached = begin_attempt(directory, job, checksum, resume=resume, retry_failed=retry_failed)
                if cached:
                    validate_result(attempt, job, plan, checksum, config)
                    print(f"Reused verified job {index}/216: {job['id']}", flush=True)
                    continue
                start, outcome = time.perf_counter(), "incomplete"
                try:
                    write_json_atomic(attempt / "host.json", {"operator": operator,
                        "host_conditions_confirmed_by_operator": True,
                        "host_conditions_confirmed_by_user": operator == "user",
                        "readiness_basis": "operator live AC power, low power and background load checks",
                        "machine_before": machine_information()})
                    print(f"Development L1 {index}/216: {job['id']}", flush=True)
                    if launch(directory, job, attempt, lock_fd) != 0:
                        raise RuntimeError(f"Worker failed; inspect {attempt / 'worker.log'}")
                    validate_result(attempt, job, plan, checksum, config)
                    verify_current_identity(directory, plan, checksum)
                    outcome = "passed"
                except BaseException as error:
                    write_json_atomic(attempt / "failure.json", {"type": type(error).__name__, "message": str(error), "automatic_retry": False})
                    raise
                finally:
                    write_json_atomic(attempt / "elapsed.json", {"outcome": outcome, "wall_seconds": time.perf_counter() - start,
                        "scope": "consumed attempt including startup, prediction, exports and validation"})
                complete_attempt(attempt, job, checksum)
            records, evidence = collect(directory, plan, config, checksum)
            result = {"plan_sha256": checksum, "selection": select(records, config), "evidence": evidence}
            verify_current_identity(directory, plan, checksum)
            verify_source_inventory(directory, plan)
            return save_or_check(directory / "selection.json", result)


def cost_records(directory, checksum):
    def timing(path):
        if not path.exists():
            return None
        value = read_json(path)
        seconds = value["wall_seconds"]
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid recorded revision duration")
        return value
    attempts = [{"attempt": str(p.relative_to(directory)), "timing": timing(p / "elapsed.json")}
                for p in sorted((directory / "jobs").glob("*/attempt-*"))]
    sessions = []
    paths = sorted((directory / "sessions").iterdir()) if (directory / "sessions").exists() else []
    if [p.name for p in paths] != [f"session-{i:04d}" for i in range(1, len(paths)+1)]:
        raise ValueError("Unexpected stage-session history")
    for p in paths:
        start = read_json(p / "started.json")
        if start["plan_sha256"] != checksum or start["stage"] != STAGE:
            raise ValueError("Revision timing session differs")
        sessions.append({"session": p.name, "stage": STAGE, "timing": timing(p / "finished.json")})
    return {"attempts_including_failures": attempts, "stage_sessions_including_failures": sessions,
            "recorded_attempt_wall_seconds": math.fsum(x["timing"]["wall_seconds"] for x in attempts if x["timing"]),
            "recorded_stage_wall_seconds": math.fsum(x["timing"]["wall_seconds"] for x in sessions if x["timing"]),
            "attempts_with_unavailable_elapsed_time": sum(x["timing"] is None for x in attempts),
            "sessions_with_unavailable_elapsed_time": sum(x["timing"] is None for x in sessions),
            "scope": "additional search; overlapping attempt/stage totals must not be added; cached ridge fit cost excluded; not formal complete-model comparison"}


def development_comparisons(selection, references):
    """Describe adaptation evidence without declaring held-out hybrid success."""
    positive = selection["best_positive_beta_l1"]
    comparators = {"best_standalone_l1": selection["best_standalone_l1"], **references}
    comparisons = {}
    for name, comparator in comparators.items():
        score, baseline = positive["scores"]["MASE"], comparator["scores"]["MASE"]
        comparisons[name] = {"positive_beta_candidate_id": positive["candidate_id"],
            "comparator_candidate_id": comparator["candidate_id"], "MASE_difference": score - baseline,
            "relative_MASE_change_percent": 100 * (score / baseline - 1) if baseline else None,
            "positive_beta_lower_on_development": score < baseline}
    direct_options = [selection["best_standalone_l1"], references["original_l2_direct"]]
    best_direct = min(direct_options, key=lambda row: (row["scores"]["MASE"],
        row["candidate_id"] != references["original_l2_direct"]["candidate_id"]))
    return {"best_positive_against_each_retained_reference": comparisons,
            "best_retained_standalone_development_candidate": best_direct,
            "best_standalone_rule": "lowest development MASE across retained L2 and L1 selections; exact tie retains original L2",
            "future_confirmation_integration": "separate frozen protocol integration required; primary comparator remains pending confirmation",
            "hybrid_success_claimed": False}


def summarize(directory):
    with run_lock(directory):
        plan, config, checksum = verify_plan(directory)
        verify_source_inventory(directory, plan)
        records, evidence = collect(directory, plan, config, checksum)
        selection = {"plan_sha256": checksum, "selection": select(records, config), "evidence": evidence}
        save_or_check(directory / "selection.json", selection)
        original_summary = read_json(directory / "original-summary.json")
        references = {"original_l2_direct": original_summary["selections"]["direct"]["selection"]["selected"],
                      "original_residual_hybrid": original_summary["selections"]["hybrid"]["selection"]["selected"],
                      "fixed_20_80_blend": read_json(directory / "historical-blend.json")["summary"]}
        value = {"status": "passed", "scope": scientific_specification(config)["scope"], "plan_sha256": checksum,
                 "selection": selection, "new_successful_tree_fit_count": len(records), "stage_counts": COUNTS,
                 "historical_references": references,
                 "development_comparisons": development_comparisons(selection["selection"], references),
                 "additional_tuning_cost": cost_records(directory, checksum), "primary_comparator": None,
                 "formal_timing_eligible": False, "holdout_forecasts_evaluated": False,
                 "automatic_further_expansion": False,
                 "interpretation": "development adaptation only; compare positive beta against new matched L1 standalone, then freeze before confirmation"}
        verify_current_identity(directory, plan, checksum)
        verify_source_inventory(directory, plan)
        return save_or_check(directory / "l1-summary.json", value)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("prepare", "fit", "summarize"), default="prepare")
    parser.add_argument("--execute-revision", action="store_true")
    parser.add_argument("--host-ready", action="store_true")
    parser.add_argument("--operator", choices=("user", "automated"), default="user")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", help=argparse.SUPPRESS)
    args = parser.parse_args()
    original.startup_check()
    directory = Path(args.output_dir).resolve()
    if args.worker:
        if not args.attempt or args.original_dir or args.execute_revision or args.host_ready or args.resume or args.retry_failed or args.stage != "prepare":
            parser.error("Internal worker takes only output directory, job and assigned attempt")
        worker(directory, args.worker, Path(args.attempt).resolve())
        return
    if args.attempt or (args.retry_failed and not args.resume):
        parser.error("Retry requires --resume and --retry-failed")
    if args.stage == "prepare":
        if not args.original_dir or args.execute_revision or args.host_ready or args.resume or args.retry_failed:
            parser.error("Prepare takes only original directory and a new output directory")
        result = prepare(Path(args.original_dir), directory)
        print(json.dumps({"status": "prepared", "stage_counts": COUNTS, "fitting_performed": False}, indent=2))
    elif args.stage == "summarize":
        if args.original_dir or args.execute_revision or args.host_ready or args.resume or args.retry_failed:
            parser.error("Summarize takes only the prepared output directory")
        result = summarize(directory)
        print(json.dumps({"status": result["status"], "new_successful_tree_fit_count": result["new_successful_tree_fit_count"],
                          "selection": result["selection"]["selection"]}, indent=2))
    else:
        if args.original_dir or not args.execute_revision or not args.host_ready:
            parser.error("Fitting requires --execute-revision and --host-ready")
        result = run(directory, resume=args.resume, retry_failed=args.retry_failed, host_ready=args.host_ready, operator=args.operator)
        print(json.dumps({"status": "passed", "selection": result["selection"]}, indent=2))


if __name__ == "__main__":
    main()
