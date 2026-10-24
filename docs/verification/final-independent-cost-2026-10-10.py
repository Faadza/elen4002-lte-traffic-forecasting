#!/usr/bin/env python3
"""Read-only independent reconstruction of the completed final cost experiment.

Uses only Python's standard library; imports no research execution/aggregation
module and performs no model fit, reload, or forecast. The only write is a NEW
audit JSON. Invoke with --repo and --output; existing output is never replaced.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


BASELINES = ("persistence", "daily_seasonal_naive", "weekly_seasonal_naive")
SEEDS = (2026, 2027, 2028)
P95_LABEL = "linear empirical quantile; descriptive, not a confidence interval"
INPUT_HASHES = {}
CHECKS = Counter()
CPU_SAMPLES = []
WORKER_INTERVALS = []


def require(value, label):
    CHECKS[label] += 1
    if not value:
        raise AssertionError(label)


def digest(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    value = h.hexdigest()
    if path in INPUT_HASHES:
        require(INPUT_HASHES[path] == value, "input unchanged when read again")
    INPUT_HASHES[path] = value
    return value


def read(path):
    digest(path)
    return json.loads(Path(path).read_text())


def positive(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def linear_quantile(values, probability):
    values = sorted(values)
    position = (len(values)-1)*probability
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def times(values):
    if not values:
        return None
    require(all(positive(x) for x in values), "finite positive timer observations")
    return {"count": len(values), "median_seconds": statistics.median(values),
            "minimum_seconds": min(values), "maximum_seconds": max(values),
            "p95_seconds": linear_quantile(values, .95), "p95_method": P95_LABEL}


def byte_summary(values):
    return {"observations_bytes": values, "median_bytes": statistics.median(values),
            "minimum_bytes": min(values), "maximum_bytes": max(values)} if values else None


def equal(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label+" keys")
        for key in expected:
            equal(actual[key], expected[key], label+"/"+str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label+" length")
        for left, right in zip(actual, expected):
            equal(left, right, label+" item")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(actual, expected, rel_tol=1e-14, abs_tol=1e-15), label)
    else:
        require(actual == expected, label)


def rss(value):
    require(value["system"] == "Darwin" and value["native_peak_unit"] == "bytes",
            "native macOS RSS units are bytes")
    require(value["source"] == "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
            "RSS source")
    require(value["incremental_model_memory"] is False
            and value["requires_separate_fresh_fit_and_prediction_workers"] is True,
            "RSS labeled total fresh-worker process memory")
    native = value["native_peak_value"]
    require(isinstance(native, int) and native > 0
            and value["process_peak_rss_bytes"] == native, "RSS native conversion")
    return native


def worker_controls(report, identity, operator):
    require(report["status"] == "passed" and report["execution_operator"] == operator,
            "worker status and truthful execution operator")
    before, after = report["controls_before"], report["controls_after"]
    require(before["process_id"] == after["process_id"]
            and before["caller_thread_id"] == after["caller_thread_id"],
            "worker process and numerical thread stable")
    for controls in (before, after):
        require(controls["system"] == "Darwin" and controls["requested_threads"] == 1,
                "native worker requested one CPU thread")
        require(all(controls[k] is True for k in ("runtime_library_controls_verified",
                "startup_controls_verified", "detected_pool_limits_verified")),
                "startup and supported runtime controls verified")
        require(controls["apple_blas"]["current_mode"] == 1
                and controls["apple_blas"]["caller_thread_setting_verified"] is True,
                "Apple Accelerate caller-thread mode verified")
        require(controls["native_pools"] and all(p["num_threads"] == 1
                for p in controls["native_pools"]), "all detected native pools limited to one")
        require(controls["torch"]["intraop_threads"] == 1
                and controls["torch"]["interop_threads"] == 1
                and controls["torch"]["deterministic_algorithms"] is True,
                "PyTorch thread and determinism controls")
    checks = (report["readiness_before"], report["readiness_after"])
    for ready in checks:
        require(ready["identity"] == identity and ready["status"] == "passed",
                "recorded native identity and readiness")
        require(ready["operator"] == operator["operator"]
                and ready["host_ready_assertion"] == operator["host_ready_assertion"],
                "readiness operator matches actual invocation")
        require(ready["exclusive_cores_or_fixed_frequency_established"] is False,
                "no unsupported exclusive-core or frequency claim")
        background = ready["background"]
        samples = background["samples_percent"]
        require(background["method"] == "two 1-second totalCPU samples"
                and background["maximum_allowed_percent"] == 20
                and background["passed"] is True and len(samples) == 2
                and all(0 <= x <= 20 for x in samples), "two acceptable CPU readiness samples")
        CPU_SAMPLES.extend(samples)
        machine = ready["machine"]
        require(machine["power_source"]["returncode"] == 0
                and "Now drawing from 'AC Power'" in machine["power_source"]["stdout"],
                "recorded AC power")
        modes = [line.split()[-1] for line in machine["power_profiles"]["stdout"].splitlines()
                 if "lowpowermode" in line]
        require(machine["power_profiles"]["returncode"] == 0 and modes
                and set(modes) == {"0"}, "recorded low-power mode off")
    start, stop = [datetime.fromisoformat(x["checked_at_utc"]) for x in checks]
    require(stop >= start, "worker readiness chronological order")
    WORKER_INTERVALS.append((start, stop, before["process_id"]))
    return before["process_id"]


def audit(repo):
    directory = repo/"results/local_runs/l1-l2-independent-ensemble/final"
    plan = read(directory/"plan.json")
    cost_plan = read(directory/"cost-plan.json")
    saved = read(directory/"summary.json")
    validation_plan = read(directory/"validation-plan.json")
    validation = read(directory/"validation-summary.json")
    plan_sha = digest(directory/"plan.json")
    require(saved["status"] == validation["status"] == "passed", "sealed final status")
    require(saved["plan_sha256"] == cost_plan["accuracy_plan_sha256"]
            == validation_plan["base_plan_sha256"] == plan_sha, "scientific plan hash binding")
    require(saved["cost_plan_sha256"] == validation_plan["cost_plan_sha256"]
            == digest(directory/"cost-plan.json"), "cost plan hash binding")
    require(validation["base_summary_sha256"] == digest(directory/"summary.json"),
            "final summary amendment hash binding")
    require(plan["period"] == cost_plan["period"] == saved["period"] == "final",
            "formal final period")
    conditions, cells = plan["selection"]["conditions"], plan["cells"]
    require(conditions == cost_plan["conditions"] == saved["conditions"], "frozen condition membership")
    require(len(conditions) == 11 and len(cells) == len(set(cells)) == 57, "11 conditions and 57 cells")
    require(cells == cost_plan["cells"], "cost and scientific cell panels identical")
    require(plan["primary_model"] == "ensemble_l1_r0p1"
            and plan["primary_comparator"] == "lightgbm_l1", "frozen primary identities")
    origins = plan["common_origins"]
    require(len(origins) == 100 and origins == sorted(set(origins))
            and cost_plan["common_origins"] == origins, "100 fixed common latency origins")

    # Derive the entire job set from the frozen conditions rather than calling
    # jobs_for or trusting the supplied cardinality/summary totals.
    expected = []
    for condition in conditions:
        method = condition["id"]
        reps = (1,) if method in BASELINES else (1, 2, 3)
        for cell in cells:
            for repeat in reps:
                expected.append({"cell": cell, "condition": condition, "fit_end": 7446,
                    "id": f"{method}--{cell}--r{repeat}", "method": method, "period": "final",
                    "predict": method == "lstm" or repeat == 1, "repeat": repeat,
                    "score_end": 8760, "score_start": 7446,
                    "seed": SEEDS[repeat-1] if method == "lstm" else None})
    require(plan["jobs"] == expected and len(expected) == 1539, "exact independently reconstructed job plan")
    expected_cost = [{"accuracy_candidate": j["predict"], "artifact_representative": j["repeat"] == 1,
        "cell": j["cell"], "id": j["id"], "learned_fit": j["method"] not in BASELINES,
        "measure_prediction": j["predict"], "method": j["method"], "repeat": j["repeat"],
        "seed": j["seed"]} for j in expected]
    require(cost_plan["jobs"] == expected_cost, "cost repetitions and deployment representatives")
    require({p.name for p in (directory/"jobs").iterdir()} == {j["id"] for j in expected},
            "no extra or missing final job directories")

    # Verify the frozen source files that implement the reviewed timer boundaries.
    config = read(repo/"docs/commands/final-evaluation-v1.json")
    require(digest(repo/"docs/commands/final-evaluation-v1.json")
            == validation_plan["configuration_sha256"], "final code configuration pin")
    source_pins = dict(plan["code_sha256"])
    source_pins.update(config["effective_dependencies_sha256"])
    receipt = read(repo.parent/"context/reconciliation-review/final-evaluation-publication-2026-10-09.json")
    require(digest(repo.parent/"context/reconciliation-review/final-evaluation-publication-2026-10-09.json")
            == validation_plan["publication_receipt_sha256"], "published final adapter receipt")
    source_pins.update(receipt["payload_sha256"])
    for name, checksum in source_pins.items():
        require(digest(repo/name) == checksum, "reviewed execution/publication source hash")

    identity = plan["development_evidence"]["environment_identity"]
    rows, failed, targets_by_cell = [], [], {}
    artifact_files_checked = artifact_total_bytes = 0
    complete_hashes = {}
    for job in expected:
        attempts = sorted((directory/"jobs"/job["id"]).glob("attempt-*"))
        accepted = [p for p in attempts if (p/"complete.json").exists()]
        require(len(accepted) == 1, "exactly one accepted attempt per planned job")
        for number, attempt in enumerate(attempts, 1):
            metadata = read(attempt/"attempt.json")
            require(metadata["job"] == job and metadata["plan_sha256"] == plan_sha
                    and metadata["attempt_number"] == number, "attempt assignment and numbering")
            operator = read(attempt/"operator.json")
            require(operator == {"host_ready_assertion": True, "operator": "user"},
                    "actual overnight execution operator")
            elapsed = read(attempt/"elapsed.json")
            require(positive(elapsed["wall_seconds"]), "positive complete-attempt wall time")
            if attempt not in accepted:
                failure = read(attempt/"failure.json")
                digest(attempt/"fit.log")
                log = (attempt/"fit.log").read_text()
                require("Background workload exceeds the declared readiness check or is unavailable" in log,
                        "failed attempts are declared CPU readiness rejections")
                require(not (attempt/"fit.json").exists() and not (attempt/"predict.json").exists()
                        and not (attempt/"result.json").exists(), "failed attempt excluded from accepted cost samples")
                before = "readiness_before = native_readiness" in log
                after = "readiness_after = native_readiness" in log
                require(before != after, "failed readiness stage identifiable")
                require(before or job["method"] in BASELINES, "no failed learned fit after model work")
                failed.append({"attempt": str(attempt.relative_to(directory)), "job": job["id"],
                    "method": job["method"], "wall_seconds": elapsed["wall_seconds"],
                    "failure_phase": "before fit" if before else "after baseline setup/reference export",
                    "learned_fitting_performed": False,
                    "fit_seconds": None, "failure_record_automatic_retry": failure["automatic_retry"],
                    "retained_model_files": sorted(str(p.relative_to(attempt/"model"))
                        for p in (attempt/"model").rglob("*") if p.is_file())})
                continue
            require(not (attempt/"failure.json").exists() and elapsed["outcome"] == "passed",
                    "accepted attempt not a failure")
            complete = read(attempt/"complete.json")
            require(complete["status"] == "passed" and complete["job"] == job
                    and complete["plan_sha256"] == plan_sha, "accepted completion identity")
            complete_hashes[str(attempt.relative_to(directory))] = digest(attempt/"complete.json")
            require(saved["evidence"][job["id"]] == {
                "attempt": str(attempt.relative_to(directory)),
                "completion_sha256": digest(attempt/"complete.json")}, "summary uses this exact accepted attempt")
            cost_names = ["fit.json", "result.json", "elapsed.json", "attempt.json", "operator.json"]
            if job["predict"]:
                cost_names.append("predict.json")
            for name in cost_names:
                require(digest(attempt/name) == complete["files_sha256"][name],
                        "cost evidence hash matches accepted completion")
            fit, result = read(attempt/"fit.json"), read(attempt/"result.json")
            require(fit["job"] == job and fit["plan_sha256"] == plan_sha
                    and result["job"] == job and result["fit_sha256"] == digest(attempt/"fit.json"),
                    "fit/result identity and hash")
            fit_pid = worker_controls(fit, identity, operator)
            training = fit["training"]
            expected_counts = {"baseline": {}, "ridge": {"ridge": 1},
                "direct": {"direct_lightgbm": 1}, "blend": {"ridge": 1, "direct_lightgbm": 1},
                "lstm": {"full_prefix_fit": 1}}[job["condition"]["kind"]]
            require(training["component_fit_counts"] == expected_counts
                    and training["cached_component_reused"] is False
                    and training["fit_interval"] == [0, 7446]
                    and training["condition"] == job["condition"], "fresh complete-model prefix fitting")
            require(training["fit_seconds"] == fit["fit_seconds"]
                    and training["fit_clock"] == "perf_counter_ns", "fit timer source and units")
            if job["method"] == "lstm":
                require(training["seed"] == job["seed"] and training["epochs"] == 23
                        and training["hidden_units"] == 32 and training["dropout"] == .2
                        and training["early_stopping"] is False, "fixed LSTM model and three prescribed seeds")
            elif job["condition"]["kind"] in ("direct", "blend"):
                require(training["tree_parameters"]["objective"] == (
                    "regression_l1" if job["condition"]["objective"] == "l1" else "regression"),
                    "tree objective identity")
            fit_rss = None
            if job["method"] in BASELINES:
                require(fit["fit_seconds"] is None and fit["fit_memory"] is None,
                        "baseline learned fitting not applicable")
            else:
                require(positive(fit["fit_seconds"]), "learned fit is measured positive seconds")
                fit_rss = rss(fit["fit_memory"])
            manifest = read(attempt/"model/ensemble.json")
            kind = job["condition"]["kind"]
            components = {"blend": ["ridge", "direct"], "ridge": ["ridge"],
                "direct": ["direct"], "lstm": ["single"], "baseline": ["single"]}[kind]
            require(manifest["condition"] == job["condition"] and manifest["components"] == components,
                    "deployment artifact contains every required component")
            actual_files = {str(p.relative_to(attempt/"model")): p for p in (attempt/"model").rglob("*") if p.is_file()}
            require(set(actual_files) == set(fit["model_files_sha256"])
                    == {"ensemble.json", *manifest["files_sha256"]}, "complete model-only artifact file inventory")
            actual_size = 0
            for name, path in actual_files.items():
                checksum = digest(path)
                require(checksum == fit["model_files_sha256"][name]
                        == complete["files_sha256"]["model/"+name], "model artifact bytes bound to accepted fit")
                if name != "ensemble.json":
                    require(checksum == manifest["files_sha256"][name], "nested deployment component hash")
                actual_size += path.stat().st_size
                artifact_files_checked += 1
            require(actual_size == fit["artifact_bytes"], "artifact size sums all actual model/config/scaler bytes")
            artifact_total_bytes += actual_size
            prediction, predict_rss, latency, batch = None, None, [], []
            if job["predict"]:
                prediction = read(attempt/"predict.json")
                require(prediction["job"] == job and prediction["plan_sha256"] == plan_sha
                        and result["predict_sha256"] == digest(attempt/"predict.json"), "prediction/result identity and hash")
                predict_pid = worker_controls(prediction, identity, operator)
                require(predict_pid != fit_pid, "fit and prediction ran in separate fresh processes")
                predict_rss = rss(prediction["prediction_memory"])
                timing = prediction["timings"]
                require(timing["warmup_passes"] == 1 and timing["timed_passes"] == 3,
                        "one warmup and three timed prediction passes")
                require(timing["origins"] == origins and timing["origin_count"] == 100,
                        "every prediction uses the same 100 origins")
                count = timing["target_count"]
                require(isinstance(count, int) and 0 < count <= 1314, "valid final batch size")
                require(targets_by_cell.setdefault(job["cell"], count) == count,
                        "batch target count identical across all methods/seeds")
                batch = timing["batch_seconds"]
                singles = timing["single_origin_seconds"]
                require(len(batch) == 3 and len(singles) == 3
                        and all(len(pass_) == 100 for pass_ in singles), "complete batch and origin timing workload")
                latency = [x for pass_ in singles for x in pass_]
                equal(timing["batch_summary"], times(batch), "independent worker batch summary")
                equal(timing["single_origin_summary"], times(latency), "independent worker latency summary")
                require(timing["clock"]["function"] == "time.perf_counter_ns"
                        and timing["clock"]["unit"] == "seconds"
                        and timing["clock"]["implementation"] == "mach_absolute_time()"
                        and timing["clock"]["monotonic"] is True,
                        "monotonic native elapsed seconds clock")
                require(timing["single_vs_batch_checked"] is True
                        and timing["identical_workload_repeats_equal"] is True
                        and timing["timer_excludes_output_checks_and_metric_scoring"] is True,
                        "prediction checks excluded from declared timers")
                require(prediction["reload_predictions_exact"] is True
                        and prediction["fitted_state_unchanged"] is True,
                        "recorded deployment reload and state checks")
            else:
                require(not (attempt/"predict.json").exists() and result["predict_sha256"] is None,
                        "deterministic extra fit repetitions add no inference or accuracy samples")
            rows.append({"job": job, "fit": fit, "predict": prediction,
                "fit_rss": fit_rss, "predict_rss": predict_rss, "latency": latency, "batch": batch,
                "artifact_bytes": actual_size, "wall_seconds": elapsed["wall_seconds"]})

    require(len(failed) == validation["failed_attempts"] == 4, "four retained failed attempts")
    require(len(rows) == validation["successful_jobs"] == 1539, "1539 accepted attempts")
    require(sum(r["predict"] is not None for r in rows) == 741, "741 accepted prediction workers")
    for method in [c["id"] for c in conditions if c["kind"] != "lstm"]:
        for cell in cells:
            fits = [r for r in rows if r["job"]["method"] == method and r["job"]["cell"] == cell]
            require(len({r["fit"]["model_state_sha256"] for r in fits}) == 1,
                    "deterministic complete state identical across all fit repetitions")
    intervals = sorted(WORKER_INTERVALS)
    require(all(left[1] < right[0] for left, right in zip(intervals, intervals[1:])),
            "accepted worker readiness intervals are sequential and nonoverlapping")

    costs = {}
    for condition in conditions:
        method = condition["id"]
        selected = [r for r in rows if r["job"]["method"] == method]
        predictions = [r for r in selected if r["predict"] is not None]
        fits = [r["fit"]["fit_seconds"] for r in selected if r["fit"]["fit_seconds"] is not None]
        panel = [math.fsum(r["fit"]["fit_seconds"] for r in selected if r["job"]["repeat"] == repeat)
                 for repeat in (1, 2, 3)] if fits else []
        latency = [x for r in selected for x in r["latency"]]
        batches = [x for r in predictions for x in r["batch"]]
        per_forecast = [seconds/r["predict"]["timings"]["target_count"]
                        for r in predictions for seconds in r["batch"]]
        batch_panel = [{"seed": seed, "pass": pass_+1,
            "seconds": math.fsum(r["batch"][pass_] for r in predictions if r["job"]["seed"] == seed)}
            for seed in (SEEDS if method == "lstm" else (None,)) for pass_ in range(3)]
        artifact_bytes = sum(r["artifact_bytes"] for r in selected if r["job"]["repeat"] == 1)
        rebuilt = {"condition": condition, "cell_count": len(cells),
            "fit_observation_count": len(fits), "fit_seconds_all_observations": fits,
            "fit_seconds_per_cell_observation_summary": times(fits),
            "panel_fit_seconds": {"observations": panel, "labels": ([{"seed": seed} for seed in SEEDS]
                if method == "lstm" else [{"repeat": rep} for rep in (1, 2, 3)]) if fits else [],
                "summary": times(panel), "scope": "sum of measured complete fits across cells for each repeat or seed"},
            "fit_process_peak_rss": byte_summary([r["fit_rss"] for r in selected if r["fit_rss"] is not None]),
            "prediction_process_peak_rss": byte_summary([r["predict_rss"] for r in predictions]),
            "deployed_artifact_bytes": artifact_bytes, "single_origin_latency_seconds": latency,
            "single_origin_latency_summary": times(latency), "comparison_latency_seconds": statistics.median(latency),
            "batch_target_counts": targets_by_cell, "common_latency_origin_count": 100,
            "common_latency_origins": origins}
        for name, value in rebuilt.items():
            equal(saved["costs"][method][name], value, "independent final cost summary/"+name)
        per_cell = []
        for cell in cells:
            cell_rows = [r for r in selected if r["job"]["cell"] == cell]
            cell_fits = [r["fit"]["fit_seconds"] for r in cell_rows if r["fit"]["fit_seconds"] is not None]
            per_cell.append({"cell": cell,
                "fit_seconds": {"observations": cell_fits, "summary": times(cell_fits)},
                "fit_observation_labels": [{"seed": r["job"]["seed"]} if method == "lstm" else
                    {"repeat": r["job"]["repeat"]} for r in cell_rows] if fits else [],
                "fit_process_peak_rss": byte_summary([r["fit_rss"] for r in cell_rows if r["fit_rss"] is not None]),
                "deployed_artifact_bytes": cell_rows[0]["artifact_bytes"],
                "prediction": [{"repeat": r["job"]["repeat"], "seed": r["job"]["seed"],
                    "timings": r["predict"]["timings"], "process_peak_rss_bytes": r["predict_rss"]}
                    for r in cell_rows if r["predict"] is not None]})
        equal(saved["costs"][method]["per_cell"], per_cell, "independent per-cell cost summary")
        costs[method] = {"condition": condition, "fit_observation_count": len(fits),
            "fit_seconds_per_cell_observation_summary": times(fits),
            "panel_fit_seconds": rebuilt["panel_fit_seconds"],
            "single_origin_latency_summary": times(latency),
            "batch_seconds_per_cell_observation_summary": times(batches),
            "batch_seconds_per_forecast_observation_summary": times(per_forecast),
            "panel_batch_observations": batch_panel,
            "panel_batch_seconds_summary": times([x["seconds"] for x in batch_panel]),
            "panel_batch_seconds_per_forecast_summary": times([x["seconds"]/sum(targets_by_cell.values()) for x in batch_panel]),
            "fit_process_peak_rss": rebuilt["fit_process_peak_rss"],
            "prediction_process_peak_rss": rebuilt["prediction_process_peak_rss"],
            "deployed_artifact_bytes": artifact_bytes,
            "deployed_artifact_cell_summary_bytes": byte_summary([r["artifact_bytes"] for r in selected if r["job"]["repeat"] == 1]),
            "saved_size_representative": "one model per cell, seed 2026" if method == "lstm" else "one complete model per cell, repeat 1",
            "accepted_attempt_wall_seconds_sum": math.fsum(r["wall_seconds"] for r in selected),
            "accepted_measured_fit_seconds_sum": math.fsum(fits),
            "accepted_measured_batch_seconds_sum": math.fsum(batches),
            "accepted_measured_origin_seconds_sum": math.fsum(latency),
            "accuracy_scores_copied_from_summary_not_reaudited_here": saved["methods"][method]["scores"]}

    primary = costs[plan["primary_model"]]
    ratios = {}
    for method, compared in costs.items():
        if method == plan["primary_model"]:
            continue
        pairs = {
            "median_complete_fit_per_cell": (primary["fit_seconds_per_cell_observation_summary"], compared["fit_seconds_per_cell_observation_summary"]),
            "median_complete_fit_57_cell_panel": (primary["panel_fit_seconds"]["summary"], compared["panel_fit_seconds"]["summary"]),
            "median_single_origin_latency": (primary["single_origin_latency_summary"], compared["single_origin_latency_summary"]),
            "p95_single_origin_latency": (primary["single_origin_latency_summary"], compared["single_origin_latency_summary"]),
            "median_57_cell_batch_seconds": (primary["panel_batch_seconds_summary"], compared["panel_batch_seconds_summary"])}
        ratio = {name: left["p95_seconds" if name.startswith("p95") else "median_seconds"]/
                 right["p95_seconds" if name.startswith("p95") else "median_seconds"]
                 if left and right else None for name, (left, right) in pairs.items()}
        for phase in ("fit", "prediction"):
            left, right = primary[phase+"_process_peak_rss"], compared[phase+"_process_peak_rss"]
            ratio["median_"+phase+"_process_rss"] = left["median_bytes"]/right["median_bytes"] if left and right else None
        ratio["complete_deployed_artifact_bytes"] = primary["deployed_artifact_bytes"]/compared["deployed_artifact_bytes"]
        ratio["accuracy_relative_change_percent_from_saved_summary"] = {metric:
            100*(primary["accuracy_scores_copied_from_summary_not_reaudited_here"][metric]/
                 compared["accuracy_scores_copied_from_summary_not_reaudited_here"][metric]-1)
            for metric in ("MASE", "MAE", "RMSE")}
        ratios[method] = ratio

    sessions = []
    retry_events = []
    quiet_events = []
    for session in sorted((directory/"sessions").iterdir()):
        started, finished = read(session/"started.json"), read(session/"finished.json")
        require(started["plan_sha256"] == plan_sha and finished["status"] == "passed", "successful final stage session")
        digest(session/"recovery-events.jsonl")
        events = [json.loads(line) for line in (session/"recovery-events.jsonl").read_text().splitlines()]
        retry_events.extend(e for e in events if e["event"] == "readiness_retry")
        quiet_events.extend(e for e in events if e["event"] == "cpu_wait")
        sessions.append({"directory": str(session.relative_to(directory)), "started": started, "finished": finished})
    require({e["attempt"] for e in retry_events} == {f["attempt"] for f in failed},
            "session journal proves all four automatic readiness retries despite failure flag false")
    accepted_wall = math.fsum(r["wall_seconds"] for r in rows)
    failed_wall = math.fsum(f["wall_seconds"] for f in failed)
    session_wall = math.fsum(s["finished"]["wall_seconds"] for s in sessions)
    source_inventory = {str(path.relative_to(repo.parent)): checksum for path, checksum in sorted(INPUT_HASHES.items())}
    inventory_bytes = json.dumps(source_inventory, sort_keys=True, separators=(",", ":")).encode()
    complete_bytes = json.dumps(complete_hashes, sort_keys=True, separators=(",", ":")).encode()
    return {"status": "passed", "audit_date": "2026-10-10", "scope": "independent final computational-cost arithmetic and timer-boundary review",
        "independence": "stdlib arithmetic and direct accepted-attempt JSON/file reads; no production aggregation helpers imported; no fit, reload, forecast or source edits",
        "run_directory": str(directory), "primary_model": plan["primary_model"], "primary_comparator": plan["primary_comparator"],
        "plan_sha256": plan_sha, "cost_plan_sha256": digest(directory/"cost-plan.json"),
        "summary_sha256": digest(directory/"summary.json"), "validation_summary_sha256": digest(directory/"validation-summary.json"),
        "audit_script_sha256": digest(Path(__file__)),
        "accepted_completion_inventory_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "cost_input_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "cost_input_inventory_hash_format": "SHA256 of UTF-8 json.dumps(relative-to-project-path:SHA256 mapping,sort_keys=True,separators=(',',':')); audit script hashed separately",
        "cost_input_inventory": source_inventory,
        "checked_model_artifact_file_count": artifact_files_checked,
        "checked_all_repetition_model_artifact_bytes": artifact_total_bytes,
        "source_code_and_publication_pins": source_pins,
        "checked_assertion_count": sum(CHECKS.values()),
        "schedule": {"conditions": len(conditions), "cells": len(cells), "accepted_attempts": len(rows),
            "learned_complete_model_fit_observations": sum(r["fit"]["fit_seconds"] is not None for r in rows),
            "baseline_setups": sum(r["job"]["method"] in BASELINES for r in rows),
            "prediction_workers": sum(r["predict"] is not None for r in rows),
            "total_successful_workers": len(WORKER_INTERVALS), "failed_attempts": len(failed),
            "accepted_readiness_snapshots": 2*len(WORKER_INTERVALS),
            "origin_samples_per_non_lstm_condition": 57*100*3,
            "origin_samples_for_three_separate_lstm_seeds": 57*100*3*3,
            "batch_targets_per_cell": targets_by_cell, "targets_per_panel": sum(targets_by_cell.values()),
            "common_origins": origins},
        "runtime_evidence": {"native_identity": identity, "operator": "user",
            "cpu_readiness_sample_count": len(CPU_SAMPLES), "maximum_recorded_readiness_cpu_percent": max(CPU_SAMPLES),
            "unique_accepted_worker_process_ids": len({x[2] for x in WORKER_INTERVALS}),
            "recorded_intervals_nonoverlapping": True, "all_fit_predict_pairs_separate_processes": True,
            "checked_startup_and_runtime_controls_before_and_after": True,
            "AC_power_and_low_power_mode_off_before_and_after": True,
            "RSS_units": "native Darwin bytes; report MiB by division by 2**20"},
        "timer_boundary_review": {
            "fit": "perf_counter_ns brackets _fit_complete: prefix checking/copying, observed-prefix scaler, causal features, every fresh component fit; fixed23-epoch LSTM complete fit. Configuration argument validation and checked_model are outside timer. Imports, disk loading, saving, reference prediction, metrics and runtime/readiness verification are excluded.",
            "ensemble_fit": "one fresh ridge and one fresh full-target LightGBM; both inside one complete-model timer, shared scaler/features built once. No cached components or residual fits.",
            "prediction": "loaded-model callback includes model checks, causal features/scaling, every deployed component, inverse transform, weighted combination and one final clipping. One warmup plus three timed passes; reload/export/score/output-equality checks excluded.",
            "lstm_amendment": "unchanged deployed callback and timing workload. Independent float64 numerical check follows all timed work; prediction RSS sampled immediately before this extra reference audit.",
            "RSS": "ru_maxrss process lifetime high-water mark sampled after work, before export/additional numerical auditing. Includes imported dependencies, loaded arrays/model, warmups and output checks; never incremental model memory.",
            "artifact": "all actual nested model/config/scaler files counted and hashed; complete independent ensemble has both components. Repeat1 per cell or LSTMseed2026 only; excludes extra repetition/seed copies, traffic and predictions.",
            "batch": "three eligible-batch passes per prediction worker. Panel totals sum one pass across57cells; for LSTM nine panel observations are three passes for each separately deployed seed. Per-forecast batch time is amortized throughput, not one-origin latency.",
            "aggregate": "all prescribed observations retained; median, min/max, descriptive linearly interpolated p95. Fit panel totals sum cells for each repeat/seed separately; LSTM seed variation is not same-seed repeatability."},
        "costs": costs, "primary_divided_by_comparator_ratios": ratios,
        "workflow_cost": {"sessions": sessions, "session_wall_seconds_sum": session_wall,
            "accepted_attempt_wall_seconds_sum": accepted_wall, "failed_attempt_wall_seconds_sum": failed_wall,
            "all_attempt_wall_seconds_sum": accepted_wall+failed_wall,
            "parent_outside_attempts_seconds": session_wall-accepted_wall-failed_wall,
            "accepted_measured_fit_seconds_sum": math.fsum(c["accepted_measured_fit_seconds_sum"] for c in costs.values()),
            "accepted_measured_batch_seconds_sum": math.fsum(c["accepted_measured_batch_seconds_sum"] for c in costs.values()),
            "accepted_measured_origin_seconds_sum": math.fsum(c["accepted_measured_origin_seconds_sum"] for c in costs.values()),
            "failed_attempts": failed, "readiness_retry_journal_events": retry_events,
            "recovery_cooldown_seconds_prescribed": len(retry_events)*validation_plan["recovery_limits"]["cooldown_seconds"],
            "quiet_check_event_count": len(quiet_events), "nonquiet_check_event_count": sum(e["quiet"] is False for e in quiet_events),
            "accounting_note": "Session includes attempts, parent verification and readiness waiting/cooldown; fit/inference timers are nested and must not be added to attempt/session totals. Final summary/bootstrap occurs after stage_session closes, so session time excludes final summarization and is not the full terminal-command duration. Failure automatic_retry:false is a legacy record field; session recovery-events explicitly proves all4 actual retries. No rejected learned model fit was performed; one rejected persistence setup exported a model/reference outside accepted samples."},
        "limitations": [
            "Formal here means the predeclared final complete-system measurement protocol on this host; one run does not certify arbitrary deployment performance.",
            "Elapsed wall-clock times include scheduling delays; the process runtime APIs and two readiness samples do not establish exclusive cores, fixed frequency, constant temperature or continuous background-load absence.",
            "Library-control snapshots retain formal_timing_eligible:false because that helper alone is not a timing certificate; the full worker/workload/native-readiness evidence supplies the study timing context.",
            "Condition execution is sequential in fixed order, not randomized; temporal machine conditions can contribute to small timing differences. Do not attribute a tiny measured blend speed advantage to a causal implementation benefit.",
            "Process RSS includes a large dependency/runtime baseline and is not incremental model memory; do not sum worker peaks to claim simultaneous panel memory.",
            "The batch amortized seconds/forecast and individual-origin latency answer different workloads and must not be interchanged.",
            "LSTM cost pools three required separately deployed seeds; it is not a three-model prediction ensemble. Seed2026 is the declared saved-size representative, not a selected best seed.",
            "Three fitting repeats and three inference passes support descriptive observed medians/ranges, not stable population confidence intervals for compute.",
            "Historical search and108-fit development completion costs are separate; they are not added to final selected-model cost.",
            "This audit independently checks cost arithmetic and bindings, not accuracy/uncertainty arithmetic or every forecast; copied accuracy scores require the separate arithmetic audit.",
            "No measured energy, network packet latency, operational congestion reduction, or multi-cell simultaneous-serving memory conclusion follows from these measurements."],
        "result": "Every reconstructed saved per-cell and panel fitting, latency, RSS and deployment-size aggregate agrees with accepted observations within1e-14 relative/1e-15 absolute floating-point comparison tolerance; all11 conditions and prescribed repetitions/seeds retained."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Independent audit output already exists; refusing overwrite")
    result = audit(args.repo.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "assertions": result["checked_assertion_count"], "schedule": {k: v for k, v in result["schedule"].items()
        if k not in ("batch_targets_per_cell", "common_origins")}}, indent=2))


if __name__ == "__main__":
    main()
