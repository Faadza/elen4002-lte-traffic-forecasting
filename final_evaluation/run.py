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
from final_evaluation.plan import (ROOT, BASELINES, BLEND, RESIDUAL, METHODS, PERIODS,
    SEEDS, load_series, verify_plan, prepare, host_check)


def worker(directory, job_id, attempt, phase):
    """One fresh process for fit or prediction, with independent CPU readback."""
    import numpy as np
    from tools.cpu_controls import single_thread_cpu
    from tools.check_cpu_controls import machine_information
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
    machine_before = machine_information()
    host_check(machine_before)
    with single_thread_cpu() as controls:
        from final_evaluation.holdout_models import (fit_selected, predict_selected,
            save_selected, load_selected, selected_fingerprint)
        from final_evaluation.measurement import measure_predictions_allow_empty
        before = controls.verify()
        if phase == "fit":
            if (attempt / "fit.json").exists():
                raise ValueError("Fit payload already exists")
            end = job["fit_end"]
            bundle, stats = fit_selected(job["method"], data["traffic"][:end],
                data["observed"][:end], data["timestamps"][:end], end, config,
                seed=job["seed"], progress=None)
            memory = memory_snapshot() if job["method"] not in BASELINES else None
            if job["method"] in BASELINES:
                stats["fit_seconds"] = None
            fingerprint = selected_fingerprint(bundle)
            save_selected(bundle, attempt / "model")
            hashes = _inventory(attempt / "model")
            reference_sha = None
            if job["predict"]:
                raw, final = predict_selected(bundle, data["traffic"], data["observed"],
                                               data["timestamps"], targets)
                np.savez(attempt / "reference.npz", targets=targets, raw=raw, final=final)
                reference_sha = sha256(attempt / "reference.npz")
                if selected_fingerprint(bundle) != fingerprint:
                    raise AssertionError("Reference prediction mutated fitted state")
            result = {"status": "passed", "job": job, "plan_sha256": checksum,
                "fit_seconds": stats["fit_seconds"], "training": stats,
                "fit_memory": memory, "model_state_sha256": fingerprint,
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
            bundle = load_selected(attempt / "model")
            if selected_fingerprint(bundle) != fitted["model_state_sha256"]:
                raise ValueError("Reloaded model state differs")
            scaler = getattr(bundle, "scaler", None)
            if job["method"] == BLEND:
                scaler = bundle.ridge.scaler
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
            if selected_fingerprint(bundle) != fitted["model_state_sha256"]:
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
    machine_after = machine_information()
    host_check(machine_after)
    result.update(machine_before=machine_before, machine_after=machine_after)
    verify_plan(directory)
    write_json_atomic(attempt / f"{phase}.json", result)


def validate_result(attempt, job, plan, checksum, *, read_forecasts=True):
    """Recompute saved scores; never accept a partial or mismatched cohort."""
    fit = _read_json(attempt / "fit.json")
    predict = _read_json(attempt / "predict.json") if job["predict"] else None
    for result in (fit, predict):
        if result is None:
            continue
        if result["status"] != "passed" or result["job"] != job or result["plan_sha256"] != checksum:
            raise ValueError("Worker result differs from assigned job/plan")
        for key in ("controls_before", "controls_after"):
            if result[key].get("runtime_library_controls_verified") is not True:
                raise ValueError("Unverified worker CPU controls")
        host_check(result["machine_before"])
        host_check(result["machine_after"])
    if fit["training"]["fit_interval"] != [0, job["fit_end"]] or fit["training"]["cached_component_reused"]:
        raise ValueError("Complete fitting prefix or fresh-component accounting differs")
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
    return frame


def launch(directory, job, attempt, phase, lock_fd):
    arguments = [sys.executable, "-m", "final_evaluation.run", "--output-dir", str(directory),
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


def run(directory, *, resume=False, retry_failed=False, host_ready=False):
    if not host_ready:
        raise ValueError("Confirm AC power, Low Power Mode off and closed heavy applications with --host-ready")
    with run_lock(directory) as lock_fd:
        plan, config, checksum = verify_plan(directory)
        with stage_session(directory, plan["period"], checksum):
            for index, job in enumerate(plan["jobs"], 1):
                attempt, cached = begin_attempt(directory, job, checksum, resume=resume, retry_failed=retry_failed)
                if cached:
                    validate_result(attempt, job, plan, checksum)
                    print(f"Reused {index}/{len(plan['jobs'])}: {job['id']}", flush=True)
                    continue
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
    from final_evaluation.measurement import aggregate_costs
    plan, config, checksum = verify_plan(directory)
    reports, evidence, raw = [], {}, {m: {c: {} for c in plan["cells"]} for m in METHODS}
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
        elif job["repeat"] > 1:
            first_id = job["id"].rsplit("--r", 1)[0]+"--r1"
            first, _ = load_success(directory, first_id, checksum)
            if _read_json(first / "fit.json")["model_state_sha256"] != result["fit"]["model_state_sha256"]:
                raise ValueError("Deterministic repetition differs from fixed first deployment")
    cell_data = {}
    methods = {m: {} for m in METHODS}
    for cell in plan["cells"]:
        data, targets = load_series(directory, plan, cell)
        eligible = np.zeros(end-start, dtype=bool)
        eligible[targets-start] = True
        scale = seasonal_scale(data["traffic_observed"][:start], data["observed"][:start], start, lag=168)
        observed = data["observed"]
        genuine_history = np.asarray([observed[t-168:t].all() for t in range(start, end)])
        actual = data["traffic_observed"][start:end]
        from blend_revision.model import blend_raw_predictions
        expected_blend, _ = blend_raw_predictions(raw["ridge"][cell]["None"][eligible],
                                                   raw["direct_lightgbm"][cell]["None"][eligible], .2)
        np.testing.assert_array_equal(raw[BLEND][cell]["None"][eligible], expected_blend)
        cell_data[cell] = {"actual": actual, "eligible": eligible, "weekly_scale": scale}
        for method in METHODS:
            expected_seeds = [str(x) for x in SEEDS] if method == "lstm" else ["None"]
            methods[method][cell] = score_cell(data["timestamps"][start:end], actual, observed[start:end],
                eligible, raw[method][cell], scale, genuine_history, expected_seeds=expected_seeds)
    summaries = {m: aggregate_method(methods[m], plan["cells"]) for m in METHODS}
    costs = aggregate_costs(reports, plan["cells"], plan["period"])
    comparator_selection = select_primary_comparator(summaries,
        {m: costs[m]["comparison_latency_seconds"] for m in METHODS}) if plan["period"] == "confirmation" else None
    primary = comparator_selection["method"] if comparator_selection else plan["primary_comparator"]
    comparisons = {m: compare_methods(summaries[BLEND], summaries[m]) for m in METHODS if m != BLEND}
    interval = None
    if plan["period"] == "final":
        panel = {c: {**cell_data[c], "hybrid_by_seed": raw[BLEND][c],
                     "comparator_by_seed": raw[primary][c]} for c in plan["cells"]}
        interval = paired_calendar_intervals(panel, plan["cells"])
    result = {"status": "passed", "period": plan["period"], "plan_sha256": checksum,
        "scope": plan["scope"], "primary_model": BLEND, "secondary_hybrid": RESIDUAL,
        "primary_comparator": primary, "comparator_selection": comparator_selection,
        "frozen": plan["frozen"], "cell_count": len(plan["cells"]), "methods": summaries,
        "blend_minus_comparators": comparisons, "costs": costs, "uncertainty": interval,
        "evidence": evidence, "formal_complete_cost_period": "final",
        "limitations": plan["design_limits"], "final_settings_retuned": False,
        "saved_blend_matches_independent_components_exactly": True}
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
                                                  args.publication_commit, args.confirmation_dir))
    if args.worker:
        if (not args.attempt or not args.phase or args.stage != "prepare" or preparation_args
                or args.execute_evaluation or args.host_ready or args.resume or args.retry_failed):
            parser.error("Internal worker accepts only assigned job, attempt, phase and output")
        worker(directory, args.worker, args.attempt.absolute(), args.phase)
        return
    if args.attempt or args.phase or (args.retry_failed and not args.resume):
        parser.error("Invalid worker/retry options")
    if args.stage == "prepare":
        if (not args.data_dir or not args.original_dir or not args.period or not args.publication_commit
                or args.execute_evaluation or args.host_ready or args.resume or args.retry_failed):
            parser.error("Prepare requires data, original results, period and published commit only")
        result = prepare(args.data_dir, args.original_dir, directory, args.period,
                         args.publication_commit, args.confirmation_dir)
        print(json.dumps({"status": "prepared", "period": result["period"], "jobs": len(result["jobs"]),
            "prediction_jobs": sum(j["predict"] for j in result["jobs"]), "fitting_performed": False}, indent=2))
    elif args.stage == "run":
        if preparation_args or not args.execute_evaluation:
            parser.error("Run requires an existing plan and --execute-evaluation")
        result = run(directory, resume=args.resume, retry_failed=args.retry_failed, host_ready=args.host_ready)
        print(json.dumps({"status": "passed", "period": result["period"],
                          "primary_comparator": result["primary_comparator"]}, indent=2))
    else:
        if preparation_args or args.execute_evaluation or args.host_ready or args.resume or args.retry_failed:
            parser.error("Summarize accepts only the existing output directory")
        with run_lock(directory):
            result = summarize(directory)
        print(json.dumps({"status": "passed", "period": result["period"],
                          "primary_comparator": result["primary_comparator"]}, indent=2))


if __name__ == "__main__":
    main()
