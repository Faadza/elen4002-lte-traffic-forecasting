"""Run frozen confirmation/final fits in sequential fresh CPU workers.

Preparation never trains. Final preparation requires complete sealed confirmation.
Successful attempts are immutable; failed attempts need an explicit retry.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.development_store import (begin_attempt, complete_attempt, load_success,
    run_lock, sha256, write_json_atomic, _read_json, _inventory)
from tools.run_development_experiments import startup_check, stage_session
from tools.run_cpu import CPU_ENV
from ensemble_study.evaluation_plan import (ROOT, BASELINES, STANDALONES, PERIODS,
    SEEDS, STUDY_ID, load_series, verify_plan, prepare)


def audit_components(condition, component):
    """Independently check full-target raw components and one final lower bound."""
    import numpy as np
    expected = {"raw", "final", "ridge_raw", "direct_raw"}
    if set(component) != expected:
        raise ValueError("Unexpected component forecast inventory")
    raw, final = np.asarray(component["raw"]), np.asarray(component["final"])
    if raw.ndim != 1 or not len(raw) or final.shape != raw.shape or not np.isfinite(raw).all():
        raise ValueError("Nonfinite or misaligned raw forecast")
    np.testing.assert_array_equal(final, np.maximum(0., raw))
    required = {"ridge": {"ridge_raw"}, "direct": {"direct_raw"},
                "blend": {"ridge_raw", "direct_raw"}, "baseline": set(), "lstm": set()}[condition["kind"]]
    for name in ("ridge_raw", "direct_raw"):
        values = component[name]
        if name not in required:
            if values is not None:
                raise ValueError("Unused component must be absent")
        elif values is None or np.shape(values) != raw.shape or not np.isfinite(values).all():
            raise ValueError("Missing or invalid full-target component forecast")
    if condition["kind"] == "blend":
        r = condition["ridge_weight"]
        expected_raw = r * np.asarray(component["ridge_raw"]) + (1-r) * np.asarray(component["direct_raw"])
        np.testing.assert_array_equal(raw, expected_raw)
    elif required:
        np.testing.assert_array_equal(raw, component[next(iter(required))])


def read_components(attempt, condition):
    import numpy as np
    with np.load(attempt / "reference.npz", allow_pickle=False) as reference:
        required = {"targets", "raw", "final"}
        if condition["kind"] in ("ridge", "blend"):
            required.add("ridge_raw")
        if condition["kind"] in ("direct", "blend"):
            required.add("direct_raw")
        if set(reference.files) != required:
            raise ValueError("Saved component forecast inventory differs")
        components = {name: reference[name].copy() if name in reference.files else None
                      for name in ("raw", "final", "ridge_raw", "direct_raw")}
    audit_components(condition, components)
    return components


def execution_identity(operator, host_ready):
    """Describe this invocation, independently of whoever prepared its plan."""
    if (operator not in ("automated", "user") or not isinstance(host_ready, bool)
            or (operator == "user" and host_ready is not True)):
        raise ValueError("Run requires its actual --operator; a user invocation also requires --host-ready")
    return {"operator": operator, "host_ready_assertion": host_ready}


def read_execution_identity(attempt):
    value = _read_json(attempt / "operator.json")
    if set(value) != {"operator", "host_ready_assertion"}:
        raise ValueError("Invalid immutable attempt operator record")
    if value != execution_identity(value["operator"], value["host_ready_assertion"]):
        raise ValueError("Attempt execution operator differs")
    return value


def worker(directory, job_id, attempt, phase):
    """One fresh process for fit or prediction, with independent CPU readback."""
    import numpy as np
    from tools.cpu_controls import single_thread_cpu
    from ensemble_study.common import native_readiness
    from tools.pilot_measurements import memory_snapshot
    plan, config, checksum = verify_plan(directory)
    job = next((j for j in plan["jobs"] if j["id"] == job_id), None)
    if job is None or attempt.parent != directory / "jobs" / job_id:
        raise ValueError("Unknown job or attempt directory")
    metadata = _read_json(attempt / "attempt.json")
    if (metadata["job"] != job or metadata["plan_sha256"] != checksum
            or attempt.name != f"attempt-{metadata['attempt_number']:04d}"
            or any((attempt / n).exists() for n in ("complete.json", "failure.json", "result.json"))):
        raise ValueError("Worker requires its fresh assigned attempt")
    data, targets = load_series(directory, plan, job["cell"])
    identity = plan["development_evidence"]["environment_identity"]
    execution = read_execution_identity(attempt)
    readiness_before = native_readiness(identity, operator=execution["operator"],
                                        host_ready=execution["host_ready_assertion"])
    with single_thread_cpu() as controls:
        from ensemble_study.model import (fit_selected, predict_selected, predict_with_components,
            save_model, load_model, fingerprint)
        from ensemble_study.measurement import measure_predictions_allow_empty
        before = controls.verify()
        if phase == "fit":
            if (attempt / "fit.json").exists():
                raise ValueError("Fit payload already exists")
            end = job["fit_end"]
            bundle, stats = fit_selected(job["condition"], data["traffic"][:end],
                data["observed"][:end], data["timestamps"][:end], end, config,
                seed=job["seed"])
            memory = memory_snapshot() if job["method"] not in BASELINES else None
            if job["method"] in BASELINES:
                stats["fit_seconds"] = None
            state_digest = fingerprint(bundle)
            save_model(bundle, attempt / "model")
            hashes = _inventory(attempt / "model")
            reference_sha = None
            if job["predict"]:
                component = predict_with_components(bundle, data["traffic"], data["observed"],
                                                    data["timestamps"], targets)
                audit_components(job["condition"], component)
                np.savez(attempt / "reference.npz", targets=targets,
                         **{k: v for k, v in component.items() if v is not None})
                reference_sha = sha256(attempt / "reference.npz")
                if fingerprint(bundle) != state_digest:
                    raise AssertionError("Reference prediction mutated fitted state")
            result = {"status": "passed", "job": job, "plan_sha256": checksum,
                "fit_seconds": stats["fit_seconds"], "training": stats,
                "fit_memory": memory, "model_state_sha256": state_digest,
                "model_files_sha256": hashes, "artifact_bytes": sum(
                    (attempt / "model" / n).stat().st_size for n in hashes),
                "reference_sha256": reference_sha}
        elif phase == "predict":
            if not job["predict"] or (attempt / "predict.json").exists():
                raise ValueError("Unexpected or repeated prediction phase")
            fitted = _read_json(attempt / "fit.json")
            if (fitted["job"] != job or fitted["plan_sha256"] != checksum
                    or _inventory(attempt / "model") != fitted["model_files_sha256"]
                    or sha256(attempt / "reference.npz") != fitted["reference_sha256"]):
                raise ValueError("Fitting/model/reference evidence differs")
            bundle = load_model(attempt / "model")
            if fingerprint(bundle) != fitted["model_state_sha256"]:
                raise ValueError("Reloaded model state differs")
            scaler = getattr(bundle, "scaler", None)
            sd = scaler.sd if scaler is not None else 1.
            rtol, atol = (1e-5, 1e-5 * sd) if job["method"] == "lstm" else (1e-10, 1e-9 * max(1., sd))
            def predict(indices):
                return predict_selected(bundle, data["traffic"], data["observed"], data["timestamps"], indices)
            measured = measure_predictions_allow_empty(predict, targets,
                np.asarray(plan["common_origins"], dtype=np.int64), rtol=rtol, atol=atol)
            memory = memory_snapshot()
            with np.load(attempt / "reference.npz", allow_pickle=False) as reference:
                np.testing.assert_array_equal(reference["targets"], targets)
                np.testing.assert_array_equal(reference["raw"], measured["raw"])
                np.testing.assert_array_equal(reference["final"], measured["final"])
            component = predict_with_components(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
            audit_components(job["condition"], component)
            with np.load(attempt / "reference.npz", allow_pickle=False) as reference:
                for name, values in component.items():
                    if values is not None:
                        np.testing.assert_array_equal(reference[name], values)
            if fingerprint(bundle) != fitted["model_state_sha256"]:
                raise AssertionError("Prediction mutated fitted state")
            from tools.pilot_results import save_predictions, score_predictions
            csv = save_predictions(attempt / "predictions.csv", job["cell"], job["method"],
                job["seed"], targets, data["timestamps"][targets], data["traffic_observed"][targets],
                measured["raw"], measured["final"])
            scored = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
                job["fit_end"], job["score_start"], job["score_end"], targets,
                measured["raw"], measured["final"])
            result = {"status": "passed", "job": job, "plan_sha256": checksum,
                "timings": measured["timings"], "prediction_memory": memory,
                "predictions_sha256": csv["sha256"], "evaluation": scored,
                "reload_predictions_exact": True, "fitted_state_unchanged": True}
        else:
            raise ValueError("Unknown internal worker phase")
        result.update(controls_before=before, controls_after=controls.verify())
    readiness_after = native_readiness(identity, operator=execution["operator"],
                                       host_ready=execution["host_ready_assertion"])
    if read_execution_identity(attempt) != execution:
        raise ValueError("Execution identity changed during the worker")
    result.update(readiness_before=readiness_before, readiness_after=readiness_after,
                  execution_operator=execution)
    verify_plan(directory)
    write_json_atomic(attempt / f"{phase}.json", result)


def validate_result(attempt, job, plan, checksum, *, read_forecasts=True):
    """Recompute saved scores; never accept a partial or mismatched cohort."""
    fit = _read_json(attempt / "fit.json")
    predict = _read_json(attempt / "predict.json") if job["predict"] else None
    execution = read_execution_identity(attempt)
    for result in (fit, predict):
        if result is None:
            continue
        if (result["status"] != "passed" or result["job"] != job or result["plan_sha256"] != checksum
                or result["execution_operator"] != execution):
            raise ValueError("Worker result differs from assigned job/plan")
        for key in ("controls_before", "controls_after"):
            if result[key].get("runtime_library_controls_verified") is not True:
                raise ValueError("Unverified worker CPU controls")
        from ensemble_study.common import verify_saved_readiness
        for name in ("readiness_before", "readiness_after"):
            verify_saved_readiness(result[name], plan["development_evidence"]["environment_identity"])
            if result[name]["operator"] != execution["operator"]:
                raise ValueError("Worker readiness operator differs from its execution attempt")
    if (fit["training"]["fit_interval"] != [0, job["fit_end"]]
            or fit["training"]["condition"] != job["condition"]
            or fit["training"]["cached_component_reused"]):
        raise ValueError("Complete fitting prefix or fresh-component accounting differs")
    counts = {"baseline": {}, "ridge": {"ridge": 1}, "direct": {"direct_lightgbm": 1},
              "blend": {"ridge": 1, "direct_lightgbm": 1}, "lstm": {"full_prefix_fit": 1}}
    if fit["training"]["component_fit_counts"] != counts[job["condition"]["kind"]]:
        raise ValueError("Fitting must include all fresh independently trained components")
    seconds = fit["fit_seconds"]
    if job["method"] in BASELINES:
        if seconds is not None or fit["fit_memory"] is not None:
            raise ValueError("Naive rules have no learned fitting cost")
    elif isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Complete fitting time must be positive and finite")
    if _inventory(attempt / "model") != fit["model_files_sha256"]:
        raise ValueError("Model files changed after fitting")
    if fit["artifact_bytes"] != sum((attempt / "model" / n).stat().st_size for n in fit["model_files_sha256"]):
        raise ValueError("Deployment artifact size differs")
    if job["predict"]:
        if (sha256(attempt / "reference.npz") != fit["reference_sha256"]
                or sha256(attempt / "predictions.csv") != predict["predictions_sha256"]
                or predict["timings"]["origins"] != plan["common_origins"]
                or predict["timings"]["target_count"] != len(plan["series"][job["cell"]]["targets"])
                or predict["reload_predictions_exact"] is not True
                or predict["fitted_state_unchanged"] is not True):
            raise ValueError("Forecast or common timing-workload evidence differs")
        if read_forecasts:
            frame = read_predictions(attempt, job, plan)
            from tools.pilot_results import score_predictions
            data, targets = load_series(attempt.parents[2], plan, job["cell"])
            scored = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
                job["fit_end"], job["score_start"], job["score_end"], targets,
                frame.prediction_raw.to_numpy(), frame.prediction.to_numpy())
            if scored != predict["evaluation"]:
                raise ValueError("Saved forecast arithmetic does not reproduce")
    elif (attempt / "predict.json").exists() or (attempt / "predictions.csv").exists():
        raise ValueError("Timing-only repeat cannot add another scoring candidate")
    return {"job": job, "fit": fit, "predict": predict}


def read_predictions(attempt, job, plan):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(attempt / "predictions.csv", float_precision="round_trip",
                        keep_default_na=False, parse_dates=["timestamp"], dtype={"seed": str})
    data, targets = load_series(attempt.parents[2], plan, job["cell"])
    np.testing.assert_array_equal(frame.target_index, targets)
    np.testing.assert_array_equal(frame.actual, data["traffic_observed"][targets])
    np.testing.assert_array_equal(frame.timestamp.to_numpy(), data["timestamps"][targets].to_numpy())
    np.testing.assert_array_equal(frame.prediction, np.maximum(0., frame.prediction_raw))
    if (not np.isfinite(frame.prediction_raw).all() or not frame.cell.eq(job["cell"]).all()
            or not frame.method.eq(job["method"]).all()
            or not frame.seed.eq("" if job["seed"] is None else str(job["seed"])).all()):
        raise ValueError("Forecast identity or numeric values differ")
    with np.load(attempt / "reference.npz", allow_pickle=False) as reference:
        np.testing.assert_array_equal(reference["targets"], targets)
        np.testing.assert_array_equal(reference["raw"], frame.prediction_raw)
        np.testing.assert_array_equal(reference["final"], frame.prediction)
    read_components(attempt, job["condition"])
    return frame


def launch(directory, job, attempt, phase, lock_fd):
    arguments = [sys.executable, "-m", "ensemble_study.evaluation_run", "--output-dir", str(directory),
                 "--worker", job["id"], "--attempt", str(attempt), "--phase", phase]
    with (attempt / f"{phase}.log").open("x") as stream:
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


def run(directory, *, operator=None, host_ready=False, resume=False, retry_failed=False):
    execution = execution_identity(operator, host_ready)
    with run_lock(directory) as lock_fd:
        plan, config, checksum = verify_plan(directory)
        with stage_session(directory, plan["period"], checksum) as session:
            write_json_atomic(session / "operator.json", execution)
            for index, job in enumerate(plan["jobs"], 1):
                attempt, cached = begin_attempt(directory, job, checksum, resume=resume, retry_failed=retry_failed)
                if cached:
                    validate_result(attempt, job, plan, checksum)
                    print(f"Reused {index}/{len(plan['jobs'])}: {job['id']}", flush=True)
                    continue
                write_json_atomic(attempt / "operator.json", execution)
                started, outcome = time.perf_counter(), "failed_or_interrupted"
                try:
                    print(f"{plan['period']} {index}/{len(plan['jobs'])}: {job['id']}", flush=True)
                    for phase in (("fit", "predict") if job["predict"] else ("fit",)):
                        code = launch(directory, job, attempt, phase, lock_fd)
                        if code != 0:
                            raise RuntimeError(f"{phase} worker failed ({code}); inspect {attempt / (phase+'.log')}")
                    result = validate_result(attempt, job, plan, checksum)
                    if job["repeat"] > 1 and job["method"] != "lstm":
                        first = {**job, "repeat": 1, "predict": True, "id": job["id"].rsplit("--r", 1)[0]+"--r1"}
                        previous, _ = load_success(directory, first["id"], checksum)
                        if _read_json(previous / "fit.json")["model_state_sha256"] != result["fit"]["model_state_sha256"]:
                            raise ValueError("Repeated deterministic fit changed model state")
                    verify_plan(directory)
                    write_json_atomic(attempt / "result.json", {"status": "passed", "job": job,
                        "plan_sha256": checksum, "fit_sha256": sha256(attempt / "fit.json"),
                        "predict_sha256": sha256(attempt / "predict.json") if job["predict"] else None})
                    outcome = "passed"
                except BaseException as error:
                    write_json_atomic(attempt / "failure.json", {"type": type(error).__name__,
                        "message": str(error), "automatic_retry": False})
                    raise
                finally:
                    write_json_atomic(attempt / "elapsed.json", {"outcome": outcome,
                        "wall_seconds": time.perf_counter()-started, "scope": "complete consumed attempt"})
                complete_attempt(attempt, job, checksum)
        return summarize(directory)


def summarize(directory, *, seal=True):
    from tools.cpu_controls import single_thread_cpu
    with single_thread_cpu():
        return _summarize(directory, seal=seal)


def _summarize(directory, *, seal=True):
    """Audit all fixed-panel predictions before choosing or publishing a comparator."""
    import numpy as np
    from evaluation.metrics import seasonal_scale
    from final_evaluation.analysis import (score_cell, aggregate_method, compare_methods,
        select_primary_comparator, paired_calendar_intervals)
    from ensemble_study.measurement import aggregate_costs
    plan, config, checksum = verify_plan(directory)
    conditions = plan["selection"]["conditions"]
    method_ids = [c["id"] for c in conditions]
    primary_model = plan["primary_model"]
    reports, evidence, raw = [], {}, {m: {c: {} for c in plan["cells"]} for m in method_ids}
    component_forecasts = {}
    start, end = PERIODS[plan["period"]]
    for job in plan["jobs"]:
        saved = load_success(directory, job["id"], checksum)
        if saved is None:
            raise ValueError("Complete every planned job before summary or comparator selection")
        attempt, marker = saved
        result = validate_result(attempt, job, plan, checksum)
        reports.append(result)
        evidence[job["id"]] = {"attempt": str(attempt.relative_to(directory)),
                               "completion_sha256": sha256(attempt / "complete.json")}
        if job["predict"]:
            frame = read_predictions(attempt, job, plan)
            full = np.full(end-start, np.nan)
            full[frame.target_index.to_numpy()-start] = frame.prediction_raw
            raw[job["method"]][job["cell"]][str(job["seed"])] = full
            component_forecasts[(job["method"], job["cell"], job["seed"])] = read_components(attempt, job["condition"])
        elif job["repeat"] > 1:
            first_id = job["id"].rsplit("--r", 1)[0]+"--r1"
            first, _ = load_success(directory, first_id, checksum)
            if _read_json(first / "fit.json")["model_state_sha256"] != result["fit"]["model_state_sha256"]:
                raise ValueError("Deterministic repetition differs from fixed first deployment")
    cell_data = {}
    methods = {m: {} for m in method_ids}
    for cell in plan["cells"]:
        data, targets = load_series(directory, plan, cell)
        eligible = np.zeros(end-start, dtype=bool)
        eligible[targets-start] = True
        scale = seasonal_scale(data["traffic_observed"][:start], data["observed"][:start], start, lag=168)
        observed = data["observed"]
        genuine_history = np.asarray([observed[t-168:t].all() for t in range(start, end)])
        actual = data["traffic_observed"][start:end]
        for condition in conditions:
            if condition["kind"] != "blend":
                continue
            component = component_forecasts[(condition["id"], cell, None)]
            np.testing.assert_array_equal(component["ridge_raw"], raw["ridge"][cell]["None"][eligible])
            direct = f"lightgbm_{condition['objective']}"
            np.testing.assert_array_equal(component["direct_raw"], raw[direct][cell]["None"][eligible])
        cell_data[cell] = {"actual": actual, "eligible": eligible, "weekly_scale": scale}
        for method in method_ids:
            expected_seeds = [str(x) for x in SEEDS] if method == "lstm" else ["None"]
            methods[method][cell] = score_cell(data["timestamps"][start:end], actual, observed[start:end],
                eligible, raw[method][cell], scale, genuine_history, expected_seeds=expected_seeds)
    summaries = {m: aggregate_method(methods[m], plan["cells"]) for m in method_ids}
    costs = aggregate_costs(reports, plan["cells"], plan["period"], conditions)
    comparator_selection = select_primary_comparator(summaries,
        {m: costs[m]["comparison_latency_seconds"] for m in method_ids},
        candidate_order=STANDALONES) if plan["period"] == "confirmation" else None
    primary = comparator_selection["method"] if comparator_selection else plan["primary_comparator"]
    comparisons = {m: compare_methods(summaries[primary_model], summaries[m]) for m in method_ids}
    for method, comparison in comparisons.items():
        for metric, values in comparison.items():
            first, other = summaries[primary_model]["scores"][metric], summaries[method]["scores"][metric]
            difference = None if first is None or other is None else first-other
            values["primary_minus_comparator"] = values.pop("hybrid_minus_comparator")
            values.update(primary_macro=first, comparator_macro=other,
                          absolute_difference=difference,
                          percentage_difference_vs_comparator=None if difference is None or other == 0 else 100*difference/other,
                          percentage_denominator="named comparator macro metric",
                          percentage_status="unavailable_missing_or_zero_comparator" if difference is None or other == 0 else "available")
    interval = None
    if plan["period"] == "final":
        panel = {c: {**cell_data[c], "hybrid_by_seed": raw[primary_model][c],
                     "comparator_by_seed": raw[primary][c]} for c in plan["cells"]}
        interval = paired_calendar_intervals(panel, plan["cells"])
        interval["statistic"] = "primary_condition_minus_frozen_comparator_macro_MASE"
    result = {"status": "passed", "study_id": STUDY_ID, "period": plan["period"], "plan_sha256": checksum,
        "scope": plan["scope"], "primary_model": primary_model,
        "selection_sha256": plan["selection_sha256"], "conditions": conditions,
        "primary_comparator": primary, "comparator_selection": comparator_selection,
        "cell_count": len(plan["cells"]), "methods": summaries,
        "primary_minus_comparators": comparisons, "costs": costs, "uncertainty": interval,
        "evidence": evidence, "formal_complete_cost_period": "final",
        "cost_plan_sha256": sha256(directory / "cost-plan.json") if plan["period"] == "final" else None,
        "limitations": plan["design_limits"], "final_settings_retuned": False,
        "primary_is_interior_ensemble": next(c for c in conditions if c["id"] == primary_model)["kind"] == "blend",
        "primary_identical_to_comparator": primary_model == primary,
        "designated_primary_comparison": {"candidate": primary_model, "comparator": primary,
                                           "metrics": comparisons[primary]},
        "saved_ensembles_match_independent_components_exactly": True}
    verify_plan(directory)
    path = directory / "summary.json"
    if path.exists():
        if _read_json(path) != result:
            raise ValueError("Existing summary changed; preserve the earlier evidence")
    elif seal:
        write_json_atomic(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), default="prepare")
    parser.add_argument("--period", choices=tuple(PERIODS))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--original-dir", type=Path)
    parser.add_argument("--publication-commit")
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--publication-record", type=Path)
    parser.add_argument("--operator", choices=("automated", "user"))
    parser.add_argument("--confirmation-dir", type=Path)
    parser.add_argument("--execute-evaluation", action="store_true")
    parser.add_argument("--host-ready", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=("fit", "predict"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    startup_check()
    directory = args.output_dir.absolute()
    preparation_args = any(x is not None for x in (args.period, args.data_dir, args.original_dir,
        args.publication_commit, args.confirmation_dir, args.selection_file,
        args.selection_sha256, args.publication_record))
    if args.worker:
        if (not args.attempt or not args.phase or args.stage != "prepare" or preparation_args
                or args.operator or args.execute_evaluation or args.host_ready or args.resume or args.retry_failed):
            parser.error("Internal worker accepts only assigned job, attempt, phase and output")
        worker(directory, args.worker, args.attempt.absolute(), args.phase)
        return
    if args.attempt or args.phase or (args.retry_failed and not args.resume):
        parser.error("Invalid worker/retry options")
    if args.stage == "prepare":
        if (not args.data_dir or not args.original_dir or not args.period or not args.publication_commit
                or not args.selection_file or not args.selection_sha256 or not args.publication_record
                or not args.operator or args.execute_evaluation or args.resume or args.retry_failed):
            parser.error("Prepare requires data, original results, period, published selection/receipt and truthful operator")
        result = prepare(args.data_dir, args.original_dir, directory, args.period,
            args.publication_commit, args.selection_file, args.selection_sha256,
            args.publication_record, args.confirmation_dir,
            operator=args.operator, host_ready=args.host_ready)
        print(json.dumps({"status": "prepared", "period": result["period"], "jobs": len(result["jobs"]),
            "prediction_jobs": sum(j["predict"] for j in result["jobs"]), "fitting_performed": False}, indent=2))
    elif args.stage == "run":
        if preparation_args or not args.operator or not args.execute_evaluation:
            parser.error("Run requires an existing plan, actual --operator and --execute-evaluation")
        result = run(directory, operator=args.operator, host_ready=args.host_ready,
                     resume=args.resume, retry_failed=args.retry_failed)
        print(json.dumps({"status": "passed", "period": result["period"],
                          "primary_comparator": result["primary_comparator"]}, indent=2))
    else:
        if preparation_args or args.operator or args.execute_evaluation or args.host_ready or args.resume or args.retry_failed:
            parser.error("Summarize accepts only the existing output directory")
        with run_lock(directory):
            result = summarize(directory)
        print(json.dumps({"status": "passed", "period": result["period"],
                          "primary_comparator": result["primary_comparator"]}, indent=2))


if __name__ == "__main__":
    main()
