"""Fixed-workload measurement and complete-panel computational summaries.

These helpers never fit models or select forecasting settings. The caller uses
fresh, sequential numerical workers and supplies the already frozen workload.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real

import numpy as np

from ensemble_study.evaluation_plan import BASELINES, SEEDS, jobs_for
from tools.pilot_measurements import (
    _exact_repeat, _indices, _predictions, clock_information, elapsed_call,
    measure_predictions, timing_summary,
)


def measure_predictions_allow_empty(predict, targets, origins, *, rtol, atol):
    """Use the checked pilot routine, or record unavailable common latency.

No common origins does not justify dropping cells or substituting a different
latency workload. Batch forecasts and their three timed passes remain usable.
"""
    targets = _indices(targets, "Targets")
    origins = _indices(origins, "Origins", allow_empty=True)
    if len(origins):
        result = measure_predictions(predict, targets, origins, rtol=rtol, atol=atol)
        result["timings"]["single_origin_status"] = "available"
        return result
    for name, value in (("rtol", rtol), ("atol", atol)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    clock = clock_information()
    reference = _predictions(predict(targets), len(targets))
    seconds = []
    for _ in range(3):
        observed, duration = elapsed_call(lambda: predict(targets))
        _exact_repeat(_predictions(observed, len(targets)), reference)
        seconds.append(duration)
    return {"raw": reference[0], "final": reference[1], "timings": {
        "clock": clock, "warmup_passes": 1, "timed_passes": 3,
        "target_count": len(targets), "origins": [], "origin_count": 0,
        "batch_seconds": seconds, "batch_summary": timing_summary(seconds),
        "single_origin_seconds": [], "single_origin_summary": None,
        "single_origin_status": "unavailable_no_origins_common_to_all_cells",
        "single_origin_sample_order": "pass then ascending origin",
        "single_vs_batch_tolerance": {"rtol": float(rtol), "atol": float(atol),
                                      "units": "original dataset traffic units"},
        "identical_workload_repeats_equal": True,
        "single_vs_batch_checked": False,
        "timer_excludes_output_checks_and_metric_scoring": True,
    }}


def _number(value, name, *, positive=False, integer=False):
    valid_type = Integral if integer else Real
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, valid_type)
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")
    return int(value) if integer else float(value)


def _rss(report):
    if not isinstance(report, dict) or report.get("incremental_model_memory") is not False:
        raise ValueError("Memory must explicitly describe total worker process RSS")
    return _number(report.get("process_peak_rss_bytes"), "Process peak RSS", integer=True)


def _byte_summary(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {"observations_bytes": list(samples), "median_bytes": float(np.median(values)),
            "minimum_bytes": min(samples), "maximum_bytes": max(samples)} if samples else None


def _timings(report):
    if not isinstance(report, dict):
        raise ValueError("Missing prediction timing report")
    timings = report.get("timings")
    if not isinstance(timings, dict) or timings.get("warmup_passes") != 1 or timings.get("timed_passes") != 3:
        raise ValueError("Prediction must use one warmup and three timed passes")
    count = _number(timings.get("target_count"), "Batch target count", positive=True, integer=True)
    origins = _indices(timings.get("origins"), "Common origins", allow_empty=True)
    if len(origins) > 100 or timings.get("origin_count") != len(origins):
        raise ValueError("Invalid common-origin workload")
    batch = timings.get("batch_seconds")
    if not isinstance(batch, list) or len(batch) != 3:
        raise ValueError("Exactly three batch measurements are required")
    batch = [_number(x, "Batch duration") for x in batch]
    if timings.get("batch_summary") != timing_summary(batch):
        raise ValueError("Batch timing summary differs from its samples")
    singles = timings.get("single_origin_seconds")
    if len(origins):
        if (not isinstance(singles, list) or len(singles) != 3
                or any(not isinstance(row, list) or len(row) != len(origins) for row in singles)):
            raise ValueError("Each common origin needs three latency observations")
        flat = [_number(x, "Single-origin duration") for row in singles for x in row]
        if timings.get("single_origin_summary") != timing_summary(flat):
            raise ValueError("Latency summary differs from its samples")
        if timings.get("single_vs_batch_checked") is not True:
            raise ValueError("Single-origin forecasts must match the batch kernel within tolerance")
    else:
        if (singles != [] or timings.get("single_origin_summary") is not None
                or timings.get("single_vs_batch_checked") is not False):
            raise ValueError("An empty common workload must have unavailable latency")
        flat = []
    if timings.get("identical_workload_repeats_equal") is not True:
        raise ValueError("Repeated identical predictions must be checked")
    return count, origins.tolist(), flat, _rss(report.get("prediction_memory"))


def aggregate_costs(jobresults, cells, period, conditions):
    """Require every planned fit/seed; summarize one shared inference workload.

Input is a list (or ID-keyed mapping) of ``{job, fit, predict}`` records. Fit
reports contain fit_seconds, fit_memory and artifact_bytes. Predict is None for
final deterministic repeats two/three and otherwise contains timings plus
prediction_memory. Confirmation fit observations remain descriptive only;
formal complete-model cost is measured on final. LSTM seeds are separately
deployed models, never a forecast ensemble.
    """
    cells = list(cells)
    expected = {job["id"]: job for job in jobs_for(period, cells, conditions)}
    if isinstance(jobresults, Mapping):
        if set(jobresults) != set(expected):
            raise ValueError("Cost records require the exact complete planned job set")
        records = list(jobresults.values())
    else:
        records = list(jobresults)
    actual = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("job"), dict):
            raise ValueError("Each cost record needs its complete frozen job")
        job = record["job"]
        job_id = job.get("id")
        if job_id in actual or job_id not in expected or job != expected[job_id]:
            raise ValueError("Duplicated, unexpected or changed cost job")
        actual[job_id] = record
    if set(actual) != set(expected):
        raise ValueError("Missing fit, repetition, seed or cell in cost results")

    origins_shared, targets_by_cell = None, {}
    normalized = {}
    for job_id, job in expected.items():
        record = actual[job_id]
        fitted = record.get("fit")
        if not isinstance(fitted, dict):
            raise ValueError("Every job requires its fit or baseline setup report")
        artifact = _number(fitted.get("artifact_bytes"), "Artifact size", integer=True)
        if job["method"] in BASELINES:
            if fitted.get("fit_seconds") is not None or fitted.get("fit_memory") is not None:
                raise ValueError("Baselines must mark fitting as not applicable")
            seconds, memory = None, None
        else:
            seconds = _number(fitted.get("fit_seconds"), "Complete fit duration", positive=True)
            memory = _rss(fitted.get("fit_memory"))
        prediction = record.get("predict")
        latency, predict_memory = [], None
        if job["predict"]:
            target_count, origins, latency, predict_memory = _timings(prediction)
            if origins_shared is None:
                origins_shared = origins
            if origins_shared != origins:
                raise ValueError("All methods and cells must share the same latency origins")
            cell = job["cell"]
            if cell in targets_by_cell and targets_by_cell[cell] != target_count:
                raise ValueError("All methods/seeds must score the same target count in a cell")
            targets_by_cell[cell] = target_count
        elif prediction is not None:
            raise ValueError("Extra deterministic repetitions must not add inference samples")
        normalized[job_id] = {"job": job, "fit_seconds": seconds, "fit_rss": memory,
                              "artifact_bytes": artifact, "latency": latency,
                              "predict_rss": predict_memory, "predict": prediction}

    results = {}
    for condition in conditions:
        method = condition["id"]
        rows = [normalized[key] for key, job in expected.items() if job["method"] == method]
        repetitions = sorted({row["job"]["repeat"] for row in rows})
        learned = method not in BASELINES
        fit_samples = [row["fit_seconds"] for row in rows] if learned else []
        panel_seconds = ([math.fsum(row["fit_seconds"] for row in rows if row["job"]["repeat"] == rep)
                          for rep in repetitions] if learned else [])
        latency = [value for row in rows for value in row["latency"]]
        per_cell = []
        for cell in cells:
            cell_rows = [row for row in rows if row["job"]["cell"] == cell]
            fits = [row["fit_seconds"] for row in cell_rows] if learned else []
            per_cell.append({"cell": cell,
                "fit_seconds": {"observations": fits, "summary": timing_summary(fits) if fits else None},
                "fit_observation_labels": [({"seed": row["job"]["seed"]} if method == "lstm"
                                             else {"repeat": row["job"]["repeat"]}) for row in cell_rows] if learned else [],
                "fit_process_peak_rss": _byte_summary([row["fit_rss"] for row in cell_rows] if learned else []),
                "deployed_artifact_bytes": cell_rows[0]["artifact_bytes"],
                "prediction": [{"repeat": row["job"]["repeat"], "seed": row["job"]["seed"],
                                "timings": row["predict"]["timings"],
                                "process_peak_rss_bytes": row["predict_rss"]}
                               for row in cell_rows if row["predict"] is not None]})
        results[method] = {
            "condition": condition,
            "period": period, "cell_count": len(cells), "per_cell": per_cell,
            "formal_complete_model_cost": period == "final",
            "fit_status": "not_applicable" if not learned else
                          ("formal_final_selected_model_cost" if period == "final" else "confirmation_descriptive_only"),
            "fit_observation_count": len(fit_samples),
            "fit_seconds_all_observations": fit_samples,
            "fit_seconds_per_cell_observation_summary": timing_summary(fit_samples) if fit_samples else None,
            "panel_fit_seconds": {"observations": panel_seconds,
                                  "labels": ([{"seed": SEEDS[r-1]} if method == "lstm" else {"repeat": r}
                                              for r in repetitions] if learned else []),
                                  "summary": timing_summary(panel_seconds) if panel_seconds else None,
                                  "scope": "sum of measured complete fits across cells for each repeat or seed"},
            "fit_process_peak_rss": _byte_summary([row["fit_rss"] for row in rows] if learned else []),
            "prediction_process_peak_rss": _byte_summary([row["predict_rss"] for row in rows if row["predict_rss"] is not None]),
            "memory_scope": "total fresh worker process peak including dependencies/data; not incremental model memory",
            "deployed_artifact_bytes": sum(row["deployed_artifact_bytes"] for row in per_cell),
            "artifact_representative": "seed 2026, one model per cell" if method == "lstm" else "repeat 1, one complete model per cell",
            "common_latency_origins": origins_shared, "common_latency_origin_count": len(origins_shared),
            "single_origin_latency_seconds": latency,
            "single_origin_latency_summary": timing_summary(latency) if latency else None,
            "comparison_latency_seconds": timing_summary(latency)["median_seconds"] if latency else None,
            "latency_status": "available" if latency else "unavailable_no_origins_common_to_all_cells",
            "latency_scope": ("pooled common-origin observations across cells, three passes and all three separate seed models; no prediction ensemble"
                              if method == "lstm" else "pooled common-origin observations across cells and three passes of repeat 1"),
            "batch_target_counts": dict(targets_by_cell),
            "batch_scope": "per-cell eligible batch sizes may differ; each cell has identical targets across methods and seeds",
        }
    return results
