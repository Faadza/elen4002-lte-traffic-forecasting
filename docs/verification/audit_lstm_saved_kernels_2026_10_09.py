"""Read-only saved LSTM numerical audit; no fitting, scoring or cost measurement."""

if __package__:
    from .audit_cli import run_audit, write_report
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from audit_cli import run_audit, write_report
    finally:
        sys.path.pop(0)


def _audit(paths):
    import json
    import math
    from dataclasses import asdict
    from datetime import datetime, timezone
    from pathlib import Path
    import sys

    WORKSPACE = paths.workspace
    ROOT = paths.repo

    import numpy as np
    from ensemble_study.evaluation_plan import verify_plan, load_series
    from ensemble_study.model import load_model, predict_selected, fingerprint
    from forecasting.data import lstm_inputs
    from forecasting.pilot_models import prediction_examples
    from tools.check_lstm import model_weights, predict_saved_weights
    from tools.cpu_controls import single_thread_cpu
    from tools.development_store import load_success, sha256, _inventory

    DIRECTORY = paths.directory
    OUTPUT = paths.output
    FAILED_JOB = "lstm--Cell_001963--r3"
    EXPECTED_PLAN = "62a2e41e041cb0765b9484d24927da6bf178e43c9f76a7565f81bb2d4efcd44e"
    # Fixed before this audit, derived from machine precision, never fitted to errors.
    TOLERANCE = float(np.sqrt(np.finfo(np.float32).eps))


    def compare(left, right, tolerance):
        delta = np.abs(left - right)
        bound = tolerance + tolerance * np.abs(right)
        ratio = delta / bound
        return {"absolute_difference": delta.tolist(), "allowed_difference": bound.tolist(),
                "ratio": ratio.tolist(), "maximum_absolute_difference": float(delta.max()),
                "maximum_ratio": float(ratio.max()), "violations": int(np.count_nonzero(ratio > 1.))}


    def audit_model(job, attempt, plan, controls):
        files_before = _inventory(attempt)
        fitted = json.loads((attempt / "fit.json").read_text())
        assert fitted["job"] == job and fitted["plan_sha256"] == EXPECTED_PLAN
        assert _inventory(attempt / "model") == fitted["model_files_sha256"]
        assert sha256(attempt / "reference.npz") == fitted["reference_sha256"]
        data, targets = load_series(DIRECTORY, plan, job["cell"])
        origins = np.asarray(plan["common_origins"], dtype=np.int64)
        assert len(origins) == 100 and np.isin(origins, targets).all()
        indices = np.searchsorted(targets, origins)
        bundle = load_model(attempt / "model")
        before_state = fingerprint(bundle)
        assert before_state == fitted["model_state_sha256"]
        controls.verify()
        batch_raw, batch_final = predict_selected(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
        repeated_raw, repeated_final = predict_selected(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
        np.testing.assert_array_equal(batch_raw, repeated_raw)
        np.testing.assert_array_equal(batch_final, repeated_final)
        with np.load(attempt / "reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["targets"], targets)
            np.testing.assert_array_equal(saved["raw"], batch_raw)
            np.testing.assert_array_equal(saved["final"], batch_final)
        np.testing.assert_array_equal(batch_final, np.maximum(batch_raw, 0.))
        single_raw, single_final = [], []
        for i, origin in enumerate(origins):
            one = np.asarray([origin], dtype=np.int64)
            raw, final = predict_selected(bundle, data["traffic"], data["observed"], data["timestamps"], one)
            if i == 0:
                repeated = predict_selected(bundle, data["traffic"], data["observed"], data["timestamps"], one)
                np.testing.assert_array_equal(raw, repeated[0])
                np.testing.assert_array_equal(final, repeated[1])
            np.testing.assert_array_equal(final, np.maximum(raw, 0.))
            single_raw.append(float(raw[0]))
            single_final.append(float(final[0]))
        single_raw, single_final = np.asarray(single_raw), np.asarray(single_final)
        example = prediction_examples(data["traffic"], data["observed"], data["timestamps"], origins)
        sequence, calendar = lstm_inputs(example, bundle.scaler)
        assert sequence.dtype == np.float32 and calendar.dtype == np.float32
        all_example = prediction_examples(data["traffic"], data["observed"], data["timestamps"], targets)
        all_sequence, all_calendar = lstm_inputs(all_example, bundle.scaler)
        np.testing.assert_array_equal(sequence, all_sequence[indices])
        np.testing.assert_array_equal(calendar, all_calendar[indices])
        # This is an independent NumPy float64 recurrence using unchanged saved
        # float32 weights and exactly the rounded float32 inputs of native inference.
        reference_z = predict_saved_weights(model_weights(bundle.single.model), sequence, calendar)
        reference_raw = bundle.scaler.inverse(reference_z)
        batch_raw, batch_final = batch_raw[indices], batch_final[indices]
        batch_z = (batch_raw - bundle.scaler.mean) / bundle.scaler.sd
        single_z = (single_raw - bundle.scaler.mean) / bundle.scaler.sd
        arrays = (batch_raw, batch_final, single_raw, single_final, batch_z, single_z, reference_z, reference_raw)
        assert all(np.isfinite(array).all() for array in arrays)
        checks = {"single_vs_batch": compare(single_z, batch_z, TOLERANCE),
                  "batch_vs_float64_reference": compare(batch_z, reference_z, TOLERANCE),
                  "single_vs_float64_reference": compare(single_z, reference_z, TOLERANCE)}
        historical = {"single_vs_batch_standardized_1e_minus5": compare(single_z, batch_z, 1e-5),
                      "batch_vs_float64_reference_1e_minus5": compare(batch_z, reference_z, 1e-5),
                      "single_vs_float64_reference_1e_minus5": compare(single_z, reference_z, 1e-5)}
        old_difference = np.abs(single_raw - batch_raw)
        old_bound = 1e-5 * bundle.scaler.sd + 1e-5 * np.abs(batch_raw)
        assert fingerprint(bundle) == before_state
        controls.verify()
        assert _inventory(attempt) == files_before
        record = {"job": job, "attempt": str(attempt.relative_to(DIRECTORY)),
                  "status": "passed" if all(c["violations"] == 0 for c in checks.values()) else "failed_numerical_guard",
                  "historical_completion_present": (attempt / "complete.json").exists(),
                  "historical_failure_present": (attempt / "failure.json").exists(),
                  "original_attempt_files_sha256": files_before,
                  "series_sha256": plan["series"][job["cell"]]["sha256"],
                  "fitted_state_sha256": before_state, "fitted_state_unchanged": True,
                  "scaler": asdict(bundle.scaler), "common_origins": origins.tolist(),
                  "saved_batch_reload_exact": True, "repeated_full_batch_exact": True,
                  "repeated_single_origin_exact": int(origins[0]),
                  "each_native_final_equals_maximum_zero_raw": True,
                  "reference_input_dtypes": {"sequence": str(sequence.dtype), "calendar": str(calendar.dtype)},
                  "reference_inputs_equal_full_batch_slices": True,
                  "standardized_native_output_reconstruction": "(original-unit raw - frozen training mean)/frozen training sd; float64 inverse round-trip only",
                  "arrays": {"batch_raw": batch_raw.tolist(), "batch_final": batch_final.tolist(),
                             "single_raw": single_raw.tolist(), "single_final": single_final.tolist(),
                             "batch_z": batch_z.tolist(), "single_z": single_z.tolist(),
                             "float64_reference_z": reference_z.tolist(), "float64_reference_raw": reference_raw.tolist()},
                  "fixed_precision_derived_checks": checks,
                  "historical_1e_minus5_diagnostics": historical,
                  "historical_original_unit_guard": {"absolute_difference": old_difference.tolist(),
                      "allowed_difference": old_bound.tolist(), "maximum_ratio": float(np.max(old_difference/old_bound)),
                      "violations": int(np.count_nonzero(old_difference > old_bound))}}
        path = OUTPUT / (job["id"] + ".json")
        write_report(path, record)
        return {"job_id": job["id"], "status": record["status"], "record": path.name,
                "sha256": sha256(path), "historical_completion_present": record["historical_completion_present"],
                "maximum_ratios": {name: value["maximum_ratio"] for name, value in checks.items()},
                "maximum_standardized_absolute_differences": {name: value["maximum_absolute_difference"] for name,value in checks.items()},
                "historical_guard_violations": {name: value["violations"] for name, value in historical.items()},
                "historical_original_unit_violations": record["historical_original_unit_guard"]["violations"]}


    def run_checks():
        if OUTPUT.exists() and any(OUTPUT.iterdir()):
            raise RuntimeError("Use a fresh audit directory; never overwrite diagnostics")
        OUTPUT.mkdir(exist_ok=True)
        plan, _, digest = verify_plan(DIRECTORY, check_environment=False)
        assert digest == EXPECTED_PLAN
        assigned = []
        for job in plan["jobs"]:
            if job["method"] != "lstm":
                continue
            if job["id"] == FAILED_JOB:
                failing = (job, DIRECTORY / "jobs" / job["id"] / "attempt-0001")
            else:
                saved = load_success(DIRECTORY, job["id"], digest)
                if saved is not None:
                    assigned.append((job, saved[0]))
        assert len(assigned) == 80
        assigned.insert(0, failing)
        frozen = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "Saved-model numerical correctness audit only; no fits, accuracy scores, rankings or formal cost measurement",
                  "audit_script_sha256": sha256(Path(__file__).resolve()), "confirmation_plan_sha256": digest,
                  "source_sha256": plan["code_sha256"], "selection_sha256": plan["selection_sha256"],
                  "publication_commit": plan["publication_commit"], "publication_record": plan["publication_record"],
                  "fixed_guard": {"rtol": TOLERANCE, "atol": TOLERANCE, "formula": "sqrt(np.finfo(np.float32).eps)",
                        "coordinate_system": "standardized native model output", "reference": "independent float64 gate recurrence on exact float32 inputs and saved float32 weights",
                        "checks": ["single_vs_batch", "batch_vs_float64_reference", "single_vs_float64_reference"],
                        "claim": "precision-derived engineering rule fixed before audit; not a universal forward-error bound"},
                  "planned_saved_models": len(assigned), "completed_models": 80, "failed_fit_artifacts": 1,
                  "new_fits": 0, "partial_accuracy_compared": False, "formal_timing": False,
                  "diagnostic_records_are_not_completed_research_attempts": True}
        write_report(OUTPUT / "audit-plan.json", frozen)
        print(json.dumps({"event": "audit_rule_frozen", "saved_models": len(assigned), "tolerance": TOLERANCE}), flush=True)
        results = []
        with single_thread_cpu() as controls:
            before = controls.verify()
            for index, (job, attempt) in enumerate(assigned, 1):
                result = audit_model(job, attempt, plan, controls)
                results.append(result)
                print(json.dumps({"event": "saved_model_audited", "index": index, **result}), flush=True)
            after = controls.verify()
        report = {**frozen, "status": "passed" if all(r["status"] == "passed" for r in results) else "failed_numerical_guard",
                  "finished_at_utc": datetime.now(timezone.utc).isoformat(), "audited_models": len(results),
                  "origin_comparisons_per_kernel": 100 * len(results), "controls_before": before, "controls_after": after,
                  "results": results, "maximum_ratios": {key: max(r["maximum_ratios"][key] for r in results)
                        for key in results[0]["maximum_ratios"]},
                  "maximum_standardized_absolute_differences": {key: max(r["maximum_standardized_absolute_differences"][key] for r in results)
                        for key in results[0]["maximum_standardized_absolute_differences"]}}
        assert sha256(DIRECTORY / "plan.json") == EXPECTED_PLAN
        for name, value in plan["code_sha256"].items():
            assert sha256(ROOT / name) == value
        write_report(OUTPUT / "report.json", report)
        print(json.dumps({"event": "audit_finished", "status": report["status"], "models": len(results),
                          "report_path": str(OUTPUT / "report.json"), "report_sha256": sha256(OUTPUT / "report.json"),
                          "maximum_ratios": report["maximum_ratios"]}), flush=True)

    run_checks()

def main(argv=None):
    run_audit(__file__, "confirmation", _audit, argv, output_directory=True)


if __name__ == "__main__":
    main()
