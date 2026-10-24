"""LSTM validation in native standardized coordinates, outside prediction timers."""
import math
import numpy as np

from tools.pilot_measurements import (
    _indices, _predictions, _exact_repeat, elapsed_call, clock_information, timing_summary,
)

# A declared engineering acceptance rule, NOT a universal recurrent-error bound.
# Derived from the unchanged model precision, never from forecast errors or maxima.
TOLERANCE = float(np.sqrt(np.finfo(np.float32).eps))
RULE = "lstm_standardized_sqrt_float32_epsilon_v2"


def check_origins(batch_raw, single_raw, reference_z, mean, sd):
    if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
        raise ValueError("Validation requires the finite original fitting scaler")
    arrays = [np.asarray(a, dtype=np.float64) for a in (batch_raw, single_raw, reference_z)]
    if (any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in arrays)
            or len({a.shape for a in arrays}) != 1):
        raise ValueError("Validation requires aligned finite origin predictions")
    batch, single, reference = (arrays[0]-mean)/sd, (arrays[1]-mean)/sd, arrays[2]
    comparisons = {"single_vs_batch": (single, batch),
                   "batch_vs_float64": (batch, reference),
                   "single_vs_float64": (single, reference)}
    differences = {}
    for name, (actual, expected) in comparisons.items():
        np.testing.assert_allclose(actual, expected, rtol=TOLERANCE, atol=TOLERANCE,
                                   err_msg=f"Amended LSTM numerical check: {name}")
        delta = np.abs(actual-expected)
        differences[name] = {"maximum_absolute_z": float(delta.max()),
            "maximum_bound_fraction": float(np.max(delta/(TOLERANCE*(1+np.abs(expected)))))}
    return {"rule": RULE, "rtol": TOLERANCE, "atol": TOLERANCE,
            "units": "standardized raw model output before clipping",
            "mean": float(mean), "sd": float(sd), "origin_count": len(batch),
            "batch_z": batch.tolist(), "single_z": single.tolist(),
            "float64_reference_z": reference.tolist(), "differences": differences,
            "reference": "tools.check_lstm.predict_saved_weights; exact float32 inputs/weights promoted to float64",
            "engineering_rule_not_universal_error_bound": True}


def measure_lstm(predict, targets, origins, *, mean, sd, reference):
    """Same prediction calls/timers as the pilot; only validation is amended.

    Independent reference arithmetic runs after all timed prediction work. It is
    never used as a forecast and never changes the deployed model's precision.
    """
    targets = _indices(targets, "Targets")
    origins = _indices(origins, "Origins", allow_empty=True)
    if len(origins) > 100 or not np.isin(origins, targets).all():
        raise ValueError("Origins must belong to the unchanged common target list")
    if not len(origins):
        from ensemble_study.measurement import measure_predictions_allow_empty
        return measure_predictions_allow_empty(predict, targets, origins,
                                                rtol=1e-5, atol=1e-5*sd)
    clock = clock_information()
    batch = _predictions(predict(targets), len(targets))
    batch_seconds = []
    for _ in range(3):
        values, seconds = elapsed_call(lambda: predict(targets))
        _exact_repeat(_predictions(values, len(targets)), batch)
        batch_seconds.append(seconds)
    workloads = [np.array([origin], dtype=np.int64) for origin in origins]
    singles = [_predictions(predict(workload), 1) for workload in workloads]
    single_seconds = []
    for _ in range(3):
        row = []
        for workload, expected in zip(workloads, singles):
            values, seconds = elapsed_call(lambda: predict(workload))
            _exact_repeat(_predictions(values, 1), expected)
            row.append(seconds)
        single_seconds.append(row)
    # No additional model calls, input transformations or checks inside timers.
    audit = check_origins(batch[0][np.searchsorted(targets, origins)],
        np.array([x[0][0] for x in singles]), reference(origins), mean, sd)
    return {"raw": batch[0], "final": batch[1], "timings": {
        "clock": clock, "warmup_passes": 1, "timed_passes": 3,
        "target_count": len(targets), "origins": origins.tolist(), "origin_count": len(origins),
        "batch_seconds": batch_seconds, "batch_summary": timing_summary(batch_seconds),
        "single_origin_seconds": single_seconds,
        "single_origin_summary": timing_summary(np.asarray(single_seconds).ravel()),
        "single_origin_status": "available",
        "single_origin_sample_order": "pass then ascending origin",
        "single_vs_batch_tolerance": {"rtol": TOLERANCE, "atol": TOLERANCE,
            "units": "standardized raw model output before clipping", "rule": RULE},
        "identical_workload_repeats_equal": True, "single_vs_batch_checked": True,
        "timer_excludes_output_checks_and_metric_scoring": True,
        "numerical_validation": audit}}


def independent_reference(bundle, data, origins):
    from forecasting.data import lstm_inputs
    from forecasting.pilot_models import prediction_examples
    from tools.check_lstm import predict_saved_weights
    inputs, calendar = lstm_inputs(prediction_examples(data["traffic"], data["observed"],
        data["timestamps"], origins), bundle.scaler)
    weights = {k: v.detach().cpu().numpy() for k, v in bundle.single.model.state_dict().items()}
    return predict_saved_weights(weights, inputs, calendar)


def verify_audit(audit, count):
    if audit.get("rule") != RULE or audit.get("origin_count") != count:
        raise ValueError("Missing amended LSTM numerical evidence")
    # Reconstruct the stated z values in raw coordinates for the common checker.
    # Use zero/one here to avoid introducing inverse-transform roundoff again.
    checked = check_origins(audit["batch_z"], audit["single_z"],
                            audit["float64_reference_z"], 0., 1.)
    if checked["origin_count"] != count:
        raise ValueError("Numerical audit arrays do not cover all common origins")
    for key in ("rtol", "atol", "units", "differences", "reference",
                "engineering_rule_not_universal_error_bound"):
        if audit.get(key) != checked[key]:
            raise ValueError(f"Numerical audit differs: {key}")
    if not math.isfinite(audit["mean"]) or not math.isfinite(audit["sd"]) or audit["sd"] <= 0:
        raise ValueError("Invalid saved audit scaler")
