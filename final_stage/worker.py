"""Fresh LSTM prediction worker for the published final validation amendment."""
import argparse
from pathlib import Path

from tools.development_store import _read_json, _inventory, sha256, write_json_atomic
from ensemble_study.evaluation_plan import verify_plan, load_series
from ensemble_study.evaluation_run import read_execution_identity, audit_components
from validation_revision.measurement import measure_lstm, independent_reference


def worker(directory, job_id, attempt):
    import numpy as np
    from tools.cpu_controls import single_thread_cpu
    from tools.pilot_measurements import memory_snapshot
    from ensemble_study.common import native_readiness
    from final_stage.run import verify_amendment, amendment_identity
    amendment, amendment_sha = verify_amendment(directory)
    plan, config, checksum = verify_plan(directory)
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    if (job["method"] != "lstm" or not job["predict"]
            or attempt.parent != directory/"jobs"/job_id):
        raise ValueError("Amended worker accepts only its assigned LSTM prediction job")
    metadata = _read_json(attempt/"attempt.json")
    if (metadata["job"] != job or metadata["plan_sha256"] != checksum
            or attempt.name != f"attempt-{metadata['attempt_number']:04d}"
            or any((attempt/n).exists() for n in ("complete.json", "failure.json", "result.json", "predict.json"))
            or _read_json(attempt/"validation-amendment.json") != amendment_identity(amendment_sha)):
        raise ValueError("Worker requires a fresh amendment-bound attempt")
    fitted = _read_json(attempt/"fit.json")
    if (fitted["job"] != job or fitted["plan_sha256"] != checksum
            or _inventory(attempt/"model") != fitted["model_files_sha256"]
            or sha256(attempt/"reference.npz") != fitted["reference_sha256"]):
        raise ValueError("Fitting/model/reference evidence differs")
    data, targets = load_series(directory, plan, job["cell"])
    execution = read_execution_identity(attempt)
    identity = plan["development_evidence"]["environment_identity"]
    readiness_before = native_readiness(identity, operator=execution["operator"],
                                        host_ready=execution["host_ready_assertion"])
    with single_thread_cpu() as controls:
        from ensemble_study.model import load_model, fingerprint, predict_selected, predict_with_components
        before = controls.verify()
        bundle = load_model(attempt/"model")
        if fingerprint(bundle) != fitted["model_state_sha256"]:
            raise ValueError("Reloaded model state differs")
        def predict(indices):
            return predict_selected(bundle, data["traffic"], data["observed"], data["timestamps"], indices)
        # Peak prediction RSS is sampled BEFORE independent gate-audit work so
        # the additional verification does not inflate the deployment metric.
        memory = []
        def reference(indices):
            memory.append(memory_snapshot())
            return independent_reference(bundle, data, indices)
        measured = measure_lstm(predict, targets, np.asarray(plan["common_origins"], dtype=np.int64),
            mean=bundle.scaler.mean, sd=bundle.scaler.sd, reference=reference)
        prediction_memory = memory[0] if memory else memory_snapshot()
        with np.load(attempt/"reference.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["targets"], targets)
            np.testing.assert_array_equal(saved["raw"], measured["raw"])
            np.testing.assert_array_equal(saved["final"], measured["final"])
        component = predict_with_components(bundle, data["traffic"], data["observed"], data["timestamps"], targets)
        audit_components(job["condition"], component)
        with np.load(attempt/"reference.npz", allow_pickle=False) as saved:
            for name, values in component.items():
                if values is not None:
                    np.testing.assert_array_equal(saved[name], values)
        if fingerprint(bundle) != fitted["model_state_sha256"]:
            raise AssertionError("Prediction mutated fitted state")
        from tools.pilot_results import save_predictions, score_predictions
        csv = save_predictions(attempt/"predictions.csv", job["cell"], job["method"], job["seed"],
            targets, data["timestamps"][targets], data["traffic_observed"][targets],
            measured["raw"], measured["final"])
        scored = score_predictions(data["traffic_observed"], data["observed"], data["timestamps"],
            job["fit_end"], job["score_start"], job["score_end"], targets, measured["raw"], measured["final"])
        result = {"status": "passed", "job": job, "plan_sha256": checksum,
            "validation_amendment": amendment_identity(amendment_sha),
            "timings": measured["timings"], "prediction_memory": prediction_memory,
            "predictions_sha256": csv["sha256"], "evaluation": scored,
            "reload_predictions_exact": True, "fitted_state_unchanged": True,
            "controls_before": before, "controls_after": controls.verify()}
    readiness_after = native_readiness(identity, operator=execution["operator"],
                                       host_ready=execution["host_ready_assertion"])
    if read_execution_identity(attempt) != execution:
        raise ValueError("Execution identity changed during worker")
    result.update(readiness_before=readiness_before, readiness_after=readiness_after,
                  execution_operator=execution)
    verify_plan(directory)
    verify_amendment(directory)
    write_json_atomic(attempt/"predict.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    args = parser.parse_args()
    from tools.run_development_experiments import startup_check
    startup_check()
    worker(args.output_dir.absolute(), args.worker, args.attempt.absolute())
