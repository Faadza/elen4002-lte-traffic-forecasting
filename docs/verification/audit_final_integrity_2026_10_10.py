"""Read-only final provenance, saved-model and numerical validation audit."""

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
    from collections import Counter, defaultdict
    import json
    import math
    import sys

    WORKSPACE = paths.workspace
    ROOT = paths.repo

    import numpy as np
    from tools.development_store import (_read_json, _inventory, _attempts, load_success,
        sha256)
    from ensemble_study.evaluation_plan import verify_plan, load_series
    from final_stage.run import (verify_amendment, verify_publication, validate_amended_result,
        amendment_identity, inventory_digest, CONFIG, RECEIPT)
    from docs.commands.confirmation_recovery import readiness_failure
    from validation_revision.measurement import verify_audit, independent_reference, RULE
    from ensemble_study.model import load_model, fingerprint
    from tools.cpu_controls import single_thread_cpu

    DIRECTORY = paths.directory
    REPORT = paths.output


    def run_checks():
        if REPORT.exists():
            raise RuntimeError("Refuse to overwrite an existing audit")
        config = verify_publication()
        amendment, amendment_sha = verify_amendment(DIRECTORY)
        plan, _, checksum = verify_plan(DIRECTORY, check_environment=False)
        assert len(plan["jobs"]) == 1539 and sum(j["predict"] for j in plan["jobs"]) == 741
        assert len(config["effective_dependencies_sha256"]) == 72
        assert amendment["historical_attempts"] == {} and amendment["settings_retuned"] is False
        summary = _read_json(DIRECTORY/"summary.json")
        seal = _read_json(DIRECTORY/"validation-summary.json")
        expected_seal = {"status": "passed", "base_plan_sha256": checksum,
            "validation_amendment": amendment_identity(amendment_sha),
            "base_summary_sha256": sha256(DIRECTORY/"summary.json"),
            "successful_jobs": 1539, "failed_attempts": 4,
            "historical_successes_preserved": 0, "all_final_lstm_use_same_validation_rule": RULE}
        assert seal == expected_seal
        assert summary["status"] == "passed" and summary["period"] == "final"
        assert summary["plan_sha256"] == checksum and summary["selection_sha256"] == config["selection_sha256"]
        assert summary["primary_model"] == config["primary_model"] == plan["primary_model"]
        assert summary["primary_comparator"] == config["primary_comparator"] == plan["primary_comparator"]
        assert set(summary["evidence"]) == {j["id"] for j in plan["jobs"]}
        assert {p.name for p in (DIRECTORY/"jobs").iterdir()} == {j["id"] for j in plan["jobs"]}
        counts, component_counts, actors = Counter(), Counter(), Counter()
        accepted_files = readiness_count = worker_records = prediction_count = 0
        readiness_samples, failures, numerical = [], [], []
        deterministic = defaultdict(dict)
        evidence_hashes, data_cache = {}, {}
        all_model_records = []
        print("Auditing final sealed artifacts, 171 LSTM references and all deterministic fit repetitions; no training or new native forwards.", flush=True)
        with single_thread_cpu() as controls:
            controls_before = controls.verify()
            for index, job in enumerate(plan["jobs"], 1):
                saved = load_success(DIRECTORY, job["id"], checksum)
                assert saved is not None
                attempt, marker = saved
                row = validate_amended_result(attempt, job, plan, checksum, amendment, amendment_sha, read_forecasts=False)
                relative = str(attempt.relative_to(DIRECTORY))
                assert summary["evidence"][job["id"]] == {"attempt": relative,
                    "completion_sha256": sha256(attempt/"complete.json")}
                evidence_hashes[job["id"]] = sha256(attempt/"complete.json")
                accepted_files += len(marker["files_sha256"]) + 1
                counts[job["method"]] += 1
                component_counts.update(row["fit"]["training"]["component_fit_counts"])
                prediction_count += int(job["predict"])
                phases = [row["fit"]] + ([row["predict"]] if job["predict"] else [])
                for phase in phases:
                    worker_records += 1
                    actors[phase["execution_operator"]["operator"]] += 1
                    assert phase["execution_operator"] == {"operator": "user", "host_ready_assertion": True}
                    for name in ("readiness_before", "readiness_after"):
                        ready = phase[name]
                        assert ready["operator"] == "user" and ready["host_ready_assertion"] is True
                        readiness_samples.extend(ready["background"]["samples_percent"])
                        readiness_count += 1
                # Independently reconstruct fingerprints from each saved deployment
                # artifact; do not merely compare the reported fingerprint strings.
                bundle = load_model(attempt/"model")
                actual_state = fingerprint(bundle)
                assert actual_state == row["fit"]["model_state_sha256"]
                all_model_records.append({"job_id": job["id"], "state_sha256": actual_state})
                if job["condition"]["kind"] in ("ridge", "direct", "blend"):
                    deterministic[(job["method"], job["cell"])][job["repeat"]] = actual_state
                    if job["repeat"] > 1:
                        assert not job["predict"] and row["predict"] is None
                if job["method"] == "lstm":
                    assert job["predict"] and job["seed"] in (2026, 2027, 2028)
                    assert bundle.settings["selected_epochs"] == 23 and bundle.settings["seed"] == job["seed"]
                    audit = row["predict"]["timings"]["numerical_validation"]
                    verify_audit(audit, len(plan["common_origins"]))
                    if job["cell"] not in data_cache:
                        data_cache[job["cell"]] = load_series(DIRECTORY, plan, job["cell"])[0]
                    reference = independent_reference(bundle, data_cache[job["cell"]],
                        np.asarray(plan["common_origins"], dtype=np.int64))
                    np.testing.assert_array_equal(reference, audit["float64_reference_z"])
                    assert fingerprint(bundle) == actual_state
                    numerical.append({"job_id": job["id"], "origin_count": audit["origin_count"],
                        "fitted_state_sha256": actual_state, "reference_recomputed_exact": True,
                        "differences": audit["differences"]})
                for previous in _attempts(DIRECTORY, job["id"]):
                    metadata = _read_json(previous/"attempt.json")
                    assert metadata["job"] == job and metadata["plan_sha256"] == checksum
                    assert _read_json(previous/"validation-amendment.json") == amendment_identity(amendment_sha)
                    if (previous/"failure.json").exists():
                        assert readiness_failure(previous) and not (previous/"complete.json").exists()
                        failed = _read_json(previous/"failure.json")
                        phase = failed["message"].split()[0]
                        log = (previous/(phase+".log")).read_text()
                        assert "Background workload exceeds the declared readiness check or is unavailable" in log
                        failures.append({"attempt": str(previous.relative_to(DIRECTORY)),
                            "started_at_utc": metadata["started_utc"], "phase": phase,
                            "readiness_stage": "after" if "readiness_after = " in log else "before",
                            "failure_inventory_sha256": inventory_digest(previous),
                            "next_successful_attempt": relative,
                            "failed_cpu_sample_values_available": False})
                    else:
                        assert previous == attempt and (previous/"complete.json").exists()
                if index % 200 == 0:
                    print(json.dumps({"checked_jobs": index, "checked_lstm_references": len(numerical)}), flush=True)
            controls_after = controls.verify()
        assert len(deterministic) == 399 and len(numerical) == 171 and len(failures) == 4
        assert prediction_count == 741 and readiness_count == 4560 and worker_records == 2280
        repeated = []
        for (method, cell), states in deterministic.items():
            assert set(states) == {1, 2, 3} and len(set(states.values())) == 1
            repeated.append({"method": method, "cell": cell, "repetitions": [1, 2, 3], "state_sha256": states[1]})
        sessions = []
        for path in sorted((DIRECTORY/"sessions").iterdir()):
            started, finished = _read_json(path/"started.json"), _read_json(path/"finished.json")
            assert started["plan_sha256"] == checksum and started["stage"] == "final"
            assert math.isfinite(finished["wall_seconds"]) and finished["wall_seconds"] > 0
            assert _read_json(path/"operator.json") == {"operator": "user", "host_ready_assertion": True}
            assert _read_json(path/"validation-amendment.json") == amendment_identity(amendment_sha)
            events = [_read_json_line(line) for line in (path/"recovery-events.jsonl").read_text().splitlines()]
            sessions.append({"name": path.name, "started": started, "finished": finished,
                "files_sha256": _inventory(path), "event_counts": dict(Counter(e["event"] for e in events)),
                "retried_attempts": [e["attempt"] for e in events if e["event"] == "readiness_retry"]})
        assert len(sessions) == 1 and sessions[0]["finished"]["status"] == "passed"
        assert sessions[0]["event_counts"]["completed"] == 1539
        assert set(sessions[0]["retried_attempts"]) == {r["attempt"] for r in failures}
        assert sessions[0]["event_counts"]["readiness_retry"] == 4
        for name, digest in config["effective_dependencies_sha256"].items():
            assert sha256(ROOT/name) == digest
        verify_amendment(DIRECTORY)
        assert sha256(DIRECTORY/"summary.json") == expected_seal["base_summary_sha256"]
        report = {"schema_version": 1, "status": "passed", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Final integrity, published provenance, readiness, independent saved-weight numerical audit and deterministic repetition identity",
            "new_fits": 0, "new_native_model_forwards": 0, "formal_timing_performed": False,
            "accuracy_bootstrap_cost_aggregation": "Delegated to separate independent audits; not recomputed here",
            "audit_script_sha256": sha256(Path(__file__).resolve()),
            "plan_sha256": checksum, "validation_plan_sha256": amendment_sha,
            "configuration_sha256": sha256(CONFIG), "publication_receipt_sha256": sha256(RECEIPT),
            "summary_sha256": sha256(DIRECTORY/"summary.json"),
            "validation_summary_sha256": sha256(DIRECTORY/"validation-summary.json"),
            "selection_sha256": plan["selection_sha256"], "confirmation_sha256": config["confirmation_sha256"],
            "original_execution_sources_verified": len(plan["code_sha256"]),
            "effective_dependencies_verified": len(config["effective_dependencies_sha256"]),
            "all_published_source_and_confirmation_pins_unchanged": True,
            "successful_jobs": 1539, "failed_attempts": 4, "total_attempts": 1543,
            "prediction_jobs": prediction_count, "fit_only_jobs": 1539-prediction_count,
            "accepted_artifact_files_including_markers": accepted_files,
            "job_counts_by_condition": dict(counts), "accepted_fitting_component_counts": dict(component_counts),
            "worker_records_verified": worker_records, "worker_operators": dict(actors),
            "readiness_snapshots_verified": readiness_count, "readiness_cpu_samples_verified": len(readiness_samples),
            "maximum_accepted_cpu_sample_percent": max(readiness_samples),
            "deployed_artifact_fingerprints_recomputed": len(all_model_records),
            "deterministic_triplets_verified": len(repeated), "deterministic_additional_repeats_verified": 2*len(repeated),
            "all_final_fits_fresh": True, "all_deterministic_triplet_fingerprints_equal": True,
            "numerical_rule": RULE, "lstm_models_independent_reference_checked": len(numerical),
            "lstm_common_origin_reference_checks": sum(r["origin_count"] for r in numerical),
            "all_float64_reference_outputs_reproduced_exactly": True,
            "maximum_numerical_bound_fractions": {name: max(r["differences"][name]["maximum_bound_fraction"] for r in numerical)
                for name in numerical[0]["differences"]},
            "numerical_results": numerical, "deterministic_triplets": repeated,
            "deployment_states": all_model_records, "completion_marker_sha256": evidence_hashes,
            "failed_attempt_history": sorted(failures, key=lambda r:r["started_at_utc"]), "sessions": sessions,
            "summary_and_validation_seal_exact": True,
            "primary_model_pin": plan["primary_model"], "primary_comparator_pin": plan["primary_comparator"],
            "controls_before_audit": controls_before, "controls_after_audit": controls_after,
            "limitations": ["Saved-weight reference recomputation is not independent retraining.",
                "The precision-derived numerical rule is not a universal recurrent forward-error bound.",
                "CPU readiness samples do not prove exclusive cores or fixed CPU frequency.",
                "Failed readiness sample values were not saved and cannot be reconstructed."]}
        write_report(REPORT, report)
        print(json.dumps({"status": "passed", "successful_jobs": 1539, "failed_attempts": 4,
            "reference_models": len(numerical), "report": str(REPORT), "sha256": sha256(REPORT),
            "maximum_numerical_bound_fractions": report["maximum_numerical_bound_fractions"]}), flush=True)


    def _read_json_line(line):
        return json.loads(line)

    run_checks()

def main(argv=None):
    run_audit(__file__, "final", _audit, argv)


if __name__ == "__main__":
    main()
