"""Read-only complete-confirmation provenance, numerical-evidence and cost audit."""

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
    from pathlib import Path
    from datetime import datetime, timezone
    from collections import Counter
    import json
    import math
    import sys

    WORKSPACE = paths.workspace
    ROOT = paths.repo

    import numpy as np
    from tools.development_store import _read_json, _inventory, load_success, sha256
    from ensemble_study.evaluation_plan import verify_plan, load_series
    from ensemble_study.measurement import aggregate_costs
    from validation_revision.continuation import (verify_amendment, snapshot,
        validate_amended_result, amendment_identity, inventory_digest, AMENDMENT, RECEIPT)
    from validation_revision.measurement import verify_audit, check_origins, independent_reference, RULE
    from ensemble_study.model import load_model, fingerprint
    from tools.cpu_controls import single_thread_cpu

    DIRECTORY = paths.directory
    REPORT = paths.output


    def run_checks():
        if REPORT.exists():
            raise RuntimeError("Refuse to overwrite an existing audit")
        amendment, amendment_sha = verify_amendment(DIRECTORY)
        plan, _, checksum = verify_plan(DIRECTORY, check_environment=False)
        state = snapshot(DIRECTORY, plan, amendment, amendment_sha)
        assert len(state["complete"]) == len(plan["jobs"]) == 741
        assert len(state["failures"]) == 6 and state["next_job"] is None
        summary = _read_json(DIRECTORY / "summary.json")
        seal = _read_json(DIRECTORY / "validation-summary.json")
        expected_seal = {"status": "passed", "base_plan_sha256": checksum,
            "validation_amendment": amendment_identity(amendment_sha),
            "base_summary_sha256": sha256(DIRECTORY / "summary.json"),
            "successful_jobs": 741, "failed_attempts": 6,
            "historical_successes_preserved": 422, "final_requires_same_validation_rule": RULE}
        assert seal == expected_seal
        assert summary["status"] == "passed" and summary["period"] == "confirmation"
        assert summary["plan_sha256"] == checksum and summary["selection_sha256"] == plan["selection_sha256"]
        assert summary["primary_model"] == plan["primary_model"]
        assert set(summary["evidence"]) == set(state["complete"])
        historical_report_path = WORKSPACE / "context/reconciliation-review/lstm-float32-numerical-audit-2026-10-09/report.json"
        historical_report = _read_json(historical_report_path)
        historical_records = {r["job_id"]: r for r in historical_report["results"] if r["historical_completion_present"]}
        assert len(historical_records) == 80
        records, numerical = [], []
        counts, component_counts = Counter(), Counter()
        accepted_files = historical_successes = new_successes = readiness_count = 0
        all_readiness_samples = []
        print("Auditing 741 sealed attempts and 171 LSTM numerical-evidence records; no fitting or timing.", flush=True)
        with single_thread_cpu() as controls:
            controls_before = controls.verify()
            for index, job in enumerate(plan["jobs"], 1):
                attempt, marker = load_success(DIRECTORY, job["id"], checksum)
                row = validate_amended_result(attempt, job, plan, checksum, amendment, amendment_sha, read_forecasts=False)
                records.append(row)
                accepted_files += len(marker["files_sha256"]) + 1
                counts[job["method"]] += 1
                component_counts.update(row["fit"]["training"]["component_fit_counts"])
                relative = str(attempt.relative_to(DIRECTORY))
                if relative in amendment["historical_attempts"]:
                    historical_successes += 1
                else:
                    new_successes += 1
                assert summary["evidence"][job["id"]] == {"attempt": relative,
                        "completion_sha256": sha256(attempt / "complete.json")}
                for phase in (row["fit"], row["predict"]):
                    for name in ("readiness_before", "readiness_after"):
                        ready = phase[name]
                        assert ready["operator"] == "user" and ready["host_ready_assertion"] is True
                        all_readiness_samples.extend(ready["background"]["samples_percent"])
                        readiness_count += 1
                if job["method"] == "lstm":
                    if job["id"] in historical_records:
                        entry = historical_records[job["id"]]
                        path = historical_report_path.parent / entry["record"]
                        assert sha256(path) == entry["sha256"]
                        evidence = _read_json(path)
                        assert evidence["fitted_state_sha256"] == row["fit"]["model_state_sha256"]
                        assert evidence["original_attempt_files_sha256"] == _inventory(attempt)
                        arrays, scaler = evidence["arrays"], evidence["scaler"]
                        audit = check_origins(arrays["batch_raw"], arrays["single_raw"], arrays["float64_reference_z"], scaler["mean"], scaler["sd"])
                        source = "independent_preserved_saved_model_diagnostic"
                        reference_recomputed = False
                    else:
                        audit = row["predict"]["timings"]["numerical_validation"]
                        verify_audit(audit, len(plan["common_origins"]))
                        bundle = load_model(attempt / "model")
                        original_state = fingerprint(bundle)
                        assert original_state == row["fit"]["model_state_sha256"]
                        data, _ = load_series(DIRECTORY, plan, job["cell"])
                        reference = independent_reference(bundle, data, np.asarray(plan["common_origins"], dtype=np.int64))
                        np.testing.assert_array_equal(reference, audit["float64_reference_z"])
                        assert fingerprint(bundle) == original_state
                        source = "new_worker_audit_with_independently_recomputed_float64_reference"
                        reference_recomputed = True
                    numerical.append({"job_id": job["id"], "origin_count": audit["origin_count"],
                        "source": source, "float64_reference_recomputed_this_audit": reference_recomputed,
                        "fitted_state_sha256": row["fit"]["model_state_sha256"], "differences": audit["differences"]})
                if index % 100 == 0:
                    print(json.dumps({"checked_jobs": index, "checked_lstm_models": len(numerical)}), flush=True)
            controls_after = controls.verify()
        assert historical_successes == 422 and new_successes == 319 and len(numerical) == 171
        cost = aggregate_costs(records, plan["cells"], "confirmation", plan["selection"]["conditions"])
        assert cost == summary["costs"]
        failures = []
        for relative in state["failures"]:
            path = DIRECTORY / relative
            failure = _read_json(path / "failure.json")
            phase = "predict" if failure["message"].startswith("predict ") else "fit"
            log = (path / (phase + ".log")).read_text()
            if "Background workload exceeds the declared readiness check or is unavailable" in log:
                reason = "cpu_readiness"
                stage = "after_prediction" if "readiness_after = " in log else "before_prediction"
            elif relative == amendment["diagnosed_failure_attempt"]:
                reason, stage = "original_lstm_batch_single_numerical_guard", "prediction_measurement_validation"
            else:
                raise AssertionError("Unknown failure reason")
            failures.append({"attempt": relative, "started_at_utc": _read_json(path/"attempt.json")["started_utc"],
                "reason": reason, "stage": stage, "inventory_sha256": inventory_digest(path),
                "complete_present": (path/"complete.json").exists(),
                "failed_cpu_sample_values_available": False if reason == "cpu_readiness" else None})
        failures.sort(key=lambda r: r["started_at_utc"])
        assert Counter(r["reason"] for r in failures) == {"cpu_readiness": 5, "original_lstm_batch_single_numerical_guard": 1}
        sessions = []
        for path in sorted((DIRECTORY/"sessions").iterdir()):
            started, finished = _read_json(path/"started.json"), _read_json(path/"finished.json")
            assert started["plan_sha256"] == checksum and started["stage"] == "confirmation"
            assert finished["wall_seconds"] > 0 and math.isfinite(finished["wall_seconds"])
            journal = path/"recovery-events.jsonl"
            events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
            sessions.append({"name": path.name, "started": started, "finished": finished,
                "files_sha256": _inventory(path), "recovery_event_counts": dict(Counter(e["event"] for e in events)),
                "retry_attempts": [e["attempt"] for e in events if e["event"] == "readiness_retry"]})
        assert sessions[-1]["finished"]["status"] == "passed"
        last_failure = failures[-1]
        assert last_failure["attempt"] == "jobs/lstm--Cell_039872--r3/attempt-0001"
        assert last_failure["reason"] == "cpu_readiness"
        assert last_failure["attempt"] in sessions[-1]["retry_attempts"]
        last_job = last_failure["attempt"].split("/")[1]
        assert summary["evidence"][last_job]["attempt"].endswith("attempt-0002")
        for name, digest in plan["code_sha256"].items():
            assert sha256(ROOT / name) == digest
        verify_amendment(DIRECTORY)
        report = {"schema_version": 1, "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Complete confirmation integrity, published provenance, worker readiness, numerical evidence and stored computation aggregation",
            "new_model_fits": 0, "new_native_lstm_predictions": 0, "formal_timing_performed": False,
            "final_period_values_read": False, "forecast_accuracy_recomputation_scope": "Delegated separately; this audit does not recompute MASE/MAE/RMSE rankings",
            "audit_script_sha256": sha256(Path(__file__).resolve()), "original_plan_sha256": checksum,
            "original_execution_source_count": len(plan["code_sha256"]), "original_sources_unchanged": True,
            "amendment_sha256": amendment_sha, "amendment_receipt_sha256": sha256(RECEIPT),
            "selection_sha256": plan["selection_sha256"], "summary_sha256": sha256(DIRECTORY/"summary.json"),
            "validation_summary_sha256": sha256(DIRECTORY/"validation-summary.json"),
            "validation_summary_exact": True, "summary_completion_evidence_exact": True,
            "successful_jobs": len(records), "failed_attempts": len(failures), "total_attempts": len(records)+len(failures),
            "accepted_files_including_markers_verified": accepted_files, "historical_successes_preserved": historical_successes,
            "new_amendment_successes": new_successes, "job_counts_by_condition": dict(counts),
            "accepted_fitting_component_counts": dict(component_counts), "readiness_snapshots_verified": readiness_count,
            "maximum_accepted_cpu_sample_percent": max(all_readiness_samples), "all_operators_user": True,
            "cost_aggregation_exact": True, "cost_scope": "Confirmation descriptive only; no formal final-period cost claim",
            "numerical_rule": RULE, "lstm_models_checked": len(numerical), "lstm_common_origin_checks": len(numerical)*100,
            "historical_lstm_diagnostics_verified": 80, "new_lstm_worker_audits_verified": 91,
            "float64_references_recomputed_this_audit": 91, "numerical_results": numerical,
            "maximum_numerical_bound_fractions": {key: max(r["differences"][key]["maximum_bound_fraction"] for r in numerical)
                  for key in numerical[0]["differences"]},
            "failures": failures, "sessions": sessions,
            "last_cpu_failure_retried_and_passed": True, "historical_failure_samples_not_reconstructed": True,
            "primary_model_pin": summary["primary_model"], "primary_comparator_pin": summary["primary_comparator"],
            "final_requires_same_validation_rule": seal["final_requires_same_validation_rule"],
            "controls_before_reference_audit": controls_before, "controls_after_reference_audit": controls_after,
            "limitations": ["Numerical rule is an engineering acceptance check, not a universal recurrent-error bound.",
                "Recomputation uses saved weights; it is not independent retraining.",
                "Readiness is a sampled operational screen, not proof of exclusive cores or fixed frequency."]}
        write_report(REPORT, report)
        print(json.dumps({"status": "passed", "jobs": len(records), "attempts": len(records)+len(failures),
            "lstm_models": len(numerical), "report": str(REPORT), "sha256": sha256(REPORT),
            "maximum_numerical_bound_fractions": report["maximum_numerical_bound_fractions"]}), flush=True)

    run_checks()

def main(argv=None):
    run_audit(__file__, "confirmation", _audit, argv)


if __name__ == "__main__":
    main()
