"""Audit saved development evidence and produce the finite ensemble selection.

No model fitting and no confirmation/final traffic reads occur here. Historical
audit helpers retain their native-host/source checks. New fitting belongs only
to the separately authorized missing-capacity runner.
"""
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import pandas as pd

from adaptation.plan import PLAN_SHA256, SUMMARY_SHA256
from ensemble_study import common
from ensemble_study.replay import replay_study
from ensemble_study.selection import BASELINES, CAPACITIES, OBJECTIVES, METRICS, STANDALONES, WEIGHTS
from evaluation.metrics import seasonal_scale
from final_evaluation.analysis import score_cell, aggregate_method
from forecasting.development_plan import FOLDS, TUNING_CELLS
from tools.development_store import (
    _inventory, _read_json, _safe_path, load_success, sha256, verify_inventory, write_json_atomic,
)

L1_PINS = {
    "plan.json": "b5ce3587d3859627215320654660ed35cbbd765dfec2ec179f0ff88a3c2ade13",
    "selection.json": "ed598e316ecd9cb3305c677971e279eedd6148b2503af48ca1351185b9961a4b",
    "l1-summary.json": "3d3bc1ccfbabe0b8f81ec123acc88033aba6ed3e227901d493dd3fdc3ddfca2c",
    "original-inventory.json": "bdf228be27fd1ebe262e4d83ce1d0e8f1f1194a589a1a3d6b7a3f6bc2637fc10",
}
REUSED_L1_CAPACITIES = {(32, 31), (128, 7)}
LSTM_ID = "lstm-h32-d0p2"


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _paths(original_directory, l1_directory):
    original = _safe_path(Path(original_directory).absolute())
    l1 = _safe_path(Path(l1_directory).absolute())
    if original == l1 or original in l1.parents or l1 in original.parents:
        raise ValueError("Historical result sources must be separate directories")
    return original, l1


def _success(directory, job, checksum):
    success = load_success(directory, job["id"], checksum)
    if success is None or success[1]["job"] != job:
        raise ValueError("Missing complete immutable job with its exact declared identity")
    return success[0]


def _frame(path, job, data, targets):
    frame = pd.read_csv(path, float_precision="round_trip", keep_default_na=False,
                        dtype={"seed": str}, parse_dates=["timestamp"])
    np.testing.assert_array_equal(frame.target_index.to_numpy(), targets)
    np.testing.assert_array_equal(frame.actual.to_numpy(), data["traffic_observed"][targets])
    np.testing.assert_array_equal(frame.timestamp.to_numpy(), data["timestamps"][targets].to_numpy())
    raw = frame.prediction_raw.to_numpy(dtype=np.float64)
    if (not np.isfinite(raw).all() or not frame.cell.eq(job["cell"]).all()
            or not frame.method.eq(job["method"]).all()
            or not frame.seed.eq("" if job["seed"] is None else str(job["seed"])).all()):
        raise ValueError("Saved forecasts have the wrong identity or nonfinite outputs")
    np.testing.assert_array_equal(frame.prediction.to_numpy(), np.maximum(0., raw))
    return frame


def derive_fixed_epochs(jobs, results):
    """Re-derive the frozen epoch count from exactly 9×3×3 selected fits."""
    selected = [j for j in jobs if j.get("stage") == "lstm" and j.get("candidate_id") == LSTM_ID]
    expected = {(cell, fold, seed) for cell in TUNING_CELLS for fold in range(3)
                for seed in common.FIXED["seeds"]}
    keys = [(j["cell"], j["fold_index"], j["seed"]) for j in selected]
    if len(keys) != 81 or set(keys) != expected or set(results) != {j["id"] for j in selected}:
        raise ValueError("Fixed epochs require all 81 selected original LSTM cell/fold/seed results")
    epochs = []
    for job in selected:
        result = results[job["id"]]
        value = result["training"]["selected_epochs"]
        if (result["job"] != job or isinstance(value, bool) or not isinstance(value, int)
                or not 1 <= value <= 100):
            raise ValueError("Invalid selected LSTM epoch evidence")
        epochs.append(value)
    derived = math.ceil(statistics.median(epochs))
    if derived != common.FIXED["lstm_epochs"]:
        raise ValueError("The declared 23 epochs no longer reproduce from original evidence")
    return {"selected_result_count": len(epochs), "selected_epochs": epochs,
            "median_selected_epochs": float(statistics.median(epochs)), "fixed_epochs": derived,
            "rule": "ceil(median(all81 selected architecture cell/fold/seed stopping epochs))"}


def _check_beta_zero(attempt, job, result, data):
    """Verify full-target beta-zero identity without refitting any model."""
    from forecasting.data import fit_scaler, make_examples, scaled_targets
    from l1_revision.model import load_model, model_fingerprint
    if job["parameters"]["beta"] != 0 or result["dependency"] is not None:
        raise ValueError("Only independent beta-zero L1 trees may be reused")
    bundle = load_model(attempt / "model")
    if bundle.beta != 0 or bundle.ridge_bundle is not None:
        raise ValueError("A historical L1 control must have no ridge component")
    end = job["fit_end"]
    scaler = fit_scaler(data["traffic"][:end], data["observed"][:end], end)
    examples = make_examples(data["traffic"][:end], data["observed"][:end], data["timestamps"][:end], 0, end)
    if (bundle.scaler != scaler or bundle.settings["training_target_count"] != len(examples.targets)
            or bundle.model.num_feature() != 340
            or result["training"]["component_fit_counts"] != {"direct_lightgbm_l1": 1}
            or result["training"].get("ridge_reused", False)):
        raise ValueError("Historical standalone L1 scaler, labels or feature count differs")
    # Labels were not serialized; the pinned beta-zero fitting code maps these
    # standardized full targets directly to LightGBM's float32 native labels.
    labels = np.asarray(scaled_targets(examples, scaler), dtype="<f4")
    return {"model_state_sha256": model_fingerprint(bundle),
            "fitting_target_indices_sha256": hashlib.sha256(np.asarray(examples.targets, dtype="<i8").tobytes()).hexdigest(),
            "reconstructed_full_target_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            "fitting_target_count": len(examples.targets), "scaler": asdict(scaler)}


def _load_historical(original_directory, l1_directory):
    from adaptation.run_ridge_extension import audit_original
    from l1_revision import run_experiment as historical_l1
    from tools import run_development_experiments as original_runner
    original, l1 = _paths(original_directory, l1_directory)
    for name, expected in L1_PINS.items():
        if sha256(l1 / name) != expected:
            raise ValueError(f"Pinned historical L1 evidence changed: {name}")
    original_plan, config, original_records, original_inventory = audit_original(original)
    l1_inventory = _inventory(l1, exclude=(".run.lock",))
    l1_plan, l1_config, l1_checksum = historical_l1.verify_plan(l1)
    historical_l1.verify_source_inventory(l1, l1_plan)
    if (Path(l1_plan["original_directory"]).absolute() != original
            or original_inventory != _read_json(l1 / "original-inventory.json")
            or l1_config != config or original_plan["environment_identity"] != l1_plan["environment_identity"]):
        raise ValueError("Historical experiments do not share the same immutable original source")
    original_summary = _read_json(original / "development-summary.json")
    l1_summary, l1_selection = _read_json(l1 / "l1-summary.json"), _read_json(l1 / "selection.json")
    if (l1_summary["selection"] != l1_selection or l1_summary["status"] != "passed"
            or l1_summary["holdout_forecasts_evaluated"] is not False
            or l1_summary["new_successful_tree_fit_count"] != 216):
        raise ValueError("Historical L1 completed summary is inconsistent")
    chosen = original_summary["selections"]
    if (chosen["ridge"]["selection"]["selected"]["candidate_id"] != "ridge-a1000"
            or chosen["lstm"]["selection"]["selected"]["candidate_id"] != LSTM_ID
            or chosen["lstm"]["selection"]["fixed_epochs"] != common.FIXED["lstm_epochs"]):
        raise ValueError("Original standalone ridge or LSTM selection changed")
    cells = sorted(TUNING_CELLS)
    data = {cell: original_runner.load_series(original, original_plan, cell) for cell in cells}
    base, tree_rows, selected_lstm, ledger = defaultdict(dict), {}, {}, {}
    for job in original_plan["jobs"]:
        include = (job["stage"] in ("baselines", "direct")
                   or job["candidate_id"] in ("ridge-a1000", LSTM_ID))
        if not include:
            continue
        attempt = _success(original, job, PLAN_SHA256)
        targets = original_runner.targets_for(data[job["cell"]], job)
        frame = _frame(attempt / "predictions" / f"{job['candidate_id']}.csv", job, data[job["cell"]], targets)
        raw = frame.prediction_raw.to_numpy(dtype=np.float64)
        key = (job["cell"], job["fold_index"])
        ledger[f"original/{job['id']}"] = {"attempt": str(attempt.relative_to(original)),
            "completion_sha256": sha256(attempt / "complete.json"), "result_sha256": sha256(attempt / "result.json")}
        if job["stage"] == "direct":
            tree_rows[("l2", job["parameters"]["trees"], job["parameters"]["leaves"], *key)] = raw
        else:
            method = "ridge" if job["stage"] == "ridge" else "lstm" if job["stage"] == "lstm" else job["method"]
            base[method].setdefault(key, {})[str(job["seed"])] = raw
        if job["stage"] == "lstm":
            selected_lstm[job["id"]] = _read_json(attempt / "result.json")
    epochs = derive_fixed_epochs(original_plan["jobs"], selected_lstm)
    beta_zero_ledger = {}
    for job in l1_plan["jobs"]:
        attempt = _success(l1, job, l1_checksum)
        expected = l1_selection["evidence"][job["id"]]
        if expected != {"attempt": str(attempt.relative_to(l1)), "completion_sha256": sha256(attempt / "complete.json")}:
            raise ValueError("Historical L1 selection references changed completion evidence")
        if job["parameters"]["beta"] != 0:
            continue  # Immutable inventory is preserved; retired models are not active conditions.
        result = historical_l1.validate_result(attempt, job, l1_plan, l1_checksum, config)
        key = (job["cell"], job["fold_index"])
        targets = original_runner.targets_for(data[job["cell"]], job)
        frame = _frame(attempt / "predictions.csv", job, data[job["cell"]], targets)
        identity = _check_beta_zero(attempt, job, result, data[job["cell"]])
        tree_rows[("l1", job["parameters"]["trees"], job["parameters"]["leaves"], *key)] = frame.prediction_raw.to_numpy(dtype=np.float64)
        beta_zero_ledger[job["id"]] = {**identity, **expected, "result_sha256": sha256(attempt / "result.json")}
    expected_trees = {(o, t, l, c, f) for o, capacities in (("l2", CAPACITIES), ("l1", REUSED_L1_CAPACITIES))
                      for t, l in capacities for c in cells for f in range(3)}
    if set(tree_rows) != expected_trees or len(beta_zero_ledger) != 54:
        raise ValueError("Historical reuse grid must contain exactly 162 L2 and 54 standalone L1 fits")
    verify_inventory(original, original_inventory, exclude=(".run.lock",))
    verify_inventory(l1, l1_inventory, exclude=(".run.lock",))
    audit = {"status": "passed", "environment_identity": original_plan["environment_identity"],
        "reused_counts": {"l2": 162, "l1": 54, "ridge": 27},
        "sources": {"original": {"directory": str(original), "files_sha256": original_inventory},
                    "l1": {"directory": str(l1), "files_sha256": l1_inventory}},
        "pins": {"original_plan_sha256": PLAN_SHA256, "original_summary_sha256": SUMMARY_SHA256,
                 "historical_l1": L1_PINS, "config_sha256": original_runner.CONFIG_SHA256},
        "historical_execution_source_sha256": {**original_plan["code_sha256"], **l1_plan["revision_and_dependency_code_sha256"]},
        "selected_lstm_epochs": epochs, "selected_original_evidence": ledger,
        "reused_l1_identity_evidence": beta_zero_ledger,
        "full_target_label_provenance": "Reconstructed from pinned beta-zero fit code and fitting prefix; labels were not serialized; no refitting performed",
        "fitting_performed": False, "heldout_values_or_errors_read": False}
    return {"audit": audit, "original_plan": original_plan, "config": config, "data": data,
            "base_raw": dict(base), "tree_raw": tree_rows,
            "original_summary": original_summary, "l1_summary": l1_summary}


def audit_sources(original_directory, l1_directory):
    """Public JSON-safe audit used before preparing the 108 missing fits."""
    return _load_historical(original_directory, l1_directory)["audit"]


def completion_cost_records(directory, checksum):
    """Retain failures/retries and both overlapping search-time boundaries."""
    directory = Path(directory)
    def elapsed(path):
        if not path.exists():
            return None
        value = _read_json(path)
        seconds = value["wall_seconds"]
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("Invalid recorded development elapsed duration")
        return value
    attempts = []
    for path in sorted((directory / "jobs").glob("*/attempt-*")):
        if _read_json(path / "attempt.json")["plan_sha256"] != checksum:
            raise ValueError("Timing attempt does not belong to this completion plan")
        attempts.append({"attempt": str(path.relative_to(directory)),
                         "successful": (path / "complete.json").exists(),
                         "failure_recorded": (path / "failure.json").exists(),
                         "timing": elapsed(path / "elapsed.json")})
    sessions = []
    for path in sorted((directory / "sessions").glob("session-*")):
        start = _read_json(path / "started.json")
        if start["plan_sha256"] != checksum or start["stage"] != "missing_l1_completion":
            raise ValueError("Timing session does not belong to this completion stage")
        sessions.append({"session": path.name, "timing": elapsed(path / "finished.json")})
    return {"attempts_including_failures": attempts, "stage_sessions_including_failures": sessions,
            "successful_attempt_count": sum(a["successful"] for a in attempts),
            "failed_or_interrupted_attempt_count": sum(not a["successful"] for a in attempts),
            "recorded_attempt_wall_seconds": math.fsum(a["timing"]["wall_seconds"] for a in attempts if a["timing"]),
            "recorded_stage_wall_seconds": math.fsum(a["timing"]["wall_seconds"] for a in sessions if a["timing"]),
            "attempts_without_elapsed_time": sum(a["timing"] is None for a in attempts),
            "sessions_without_elapsed_time": sum(a["timing"] is None for a in sessions),
            "scope": "Search cost only; attempt/stage durations overlap and must not be added; failed attempts are not automatically successful model fits"}


def _append_completion(history, completion_directory):
    from ensemble_study.development import completed_records, verify_plan
    directory = _safe_path(Path(completion_directory).absolute())
    plan, _, checksum = verify_plan(directory, check_native=False, full_sources=True)
    if plan["source_audit"] != history["audit"]:
        raise ValueError("New completion does not bind the same audited historical evidence")
    inventory = _inventory(directory, exclude=(".run.lock",))
    rows = completed_records(directory, reload_models=True)
    expected = {("l1", t, l, c, f) for t, l in common.MISSING for c in sorted(TUNING_CELLS) for f in range(3)}
    seen, ledger, fit_seconds = set(), {}, []
    from tools import run_development_experiments as original_runner
    for row in rows:
        job, attempt, result = row["job"], Path(row["attempt"]), row["result"]
        condition = job["condition"]
        key = (condition["objective"], condition["trees"], condition["leaves"], job["cell"], job["fold_index"])
        if key not in expected or key in seen or key in history["tree_raw"]:
            raise ValueError("Duplicate, extra or historically completed capacity in new results")
        if condition["kind"] != "direct" or condition["ridge_weight"] != 0 or job["seed"] is not None:
            raise ValueError("Completion must contain only independent full-target L1 trees")
        data = history["data"][job["cell"]]
        targets = original_runner.targets_for(data, job)
        # Re-read the exact immutable CSV against the original target truth.
        frame = _frame(attempt / "predictions.csv", {**job, "method": condition["id"]}, data, targets)
        history["tree_raw"][key] = frame.prediction_raw.to_numpy(dtype=np.float64)
        seen.add(key)
        ledger[job["id"]] = {"attempt": str(attempt.relative_to(directory)),
            "completion_sha256": sha256(attempt / "complete.json"), "result_sha256": sha256(attempt / "result.json")}
        fit_seconds.append(result["fit_seconds"])
    if seen != expected or len(rows) != 108:
        raise ValueError("Exactly all 108 missing-capacity successes are required")
    verify_inventory(directory, inventory, exclude=(".run.lock",))
    return {"directory": str(directory), "files_sha256": inventory, "plan_sha256": checksum,
            "successful_fit_count": 108, "successful_fit_seconds": fit_seconds,
            "successful_fit_seconds_total": math.fsum(fit_seconds), "evidence": ledger,
            "search_costs": completion_cost_records(directory, checksum)}


def _cohorts(history):
    from tools import run_development_experiments as original_runner
    result = {}
    for (objective, trees, leaves, cell, fold), direct in history["tree_raw"].items():
        data = history["data"][cell]
        start, end = FOLDS[fold]
        job = {"score_start": start, "score_end": end}
        targets = original_runner.targets_for(data, job)
        result[(objective, trees, leaves, cell, fold)] = {
            "targets": targets, "actual": data["traffic_observed"][targets],
            "ridge_raw": history["base_raw"]["ridge"][(cell, fold)]["None"],
            "direct_raw": direct,
            "weekly_scale": seasonal_scale(data["traffic_observed"][:start], data["observed"][:start], start),
        }
    return result


def _condition_diagnostics(condition, history, cohorts, cells):
    per_fold, per_cell = [], {cell: [] for cell in cells}
    for fold, (start, end) in enumerate(FOLDS):
        results = {}
        for cell in cells:
            data = history["data"][cell]
            if condition["kind"] in ("direct", "blend"):
                source = cohorts[(condition["objective"], condition["trees"], condition["leaves"], cell, fold)]
                weight = condition["ridge_weight"]
                values = source["direct_raw"] if weight == 0 else (
                    weight * source["ridge_raw"] + (1-weight) * source["direct_raw"])
                raw = {"None": values}
            else:
                raw = history["base_raw"][condition["id"]][(cell, fold)]
                # Every component shares this data-only cohort and denominator.
                source = next(v for k, v in cohorts.items() if k[3:] == (cell, fold))
            targets = source["targets"]
            eligible = np.zeros(end-start, dtype=bool)
            eligible[targets-start] = True
            padded = {}
            for seed, values in raw.items():
                padded[seed] = np.full(end-start, np.nan)
                padded[seed][targets-start] = values
            fully_observed = np.asarray([data["observed"][t-168:t].all() for t in range(start, end)])
            seeds = common.FIXED["seeds"] if condition["kind"] == "lstm" else ["None"]
            results[cell] = score_cell(data["timestamps"][start:end], data["traffic_observed"][start:end],
                data["observed"][start:end], eligible, padded, source["weekly_scale"], fully_observed,
                expected_seeds=seeds)
            per_cell[cell].append(results[cell]["scores"])
        per_fold.append({"fold_index": fold, "summary": aggregate_method(results, cells)})
    return {"condition": condition,
            "scores": {m: float(np.mean([f["summary"]["scores"][m] for f in per_fold])) for m in METRICS},
            "folds": per_fold,
            "per_cell_scores": {c: {m: float(np.mean([s[m] for s in values])) for m in METRICS}
                                for c, values in per_cell.items()},
            "cell_count": len(cells), "fold_count": 3}


def _prediction_tables(cohorts, components, history, cells):
    """Retain raw components, weighted raw output and final bound for every grid point."""
    tables = {}
    for objective in OBJECTIVES:
        trees, leaves = components[objective]["trees"], components[objective]["leaves"]
        for weight in WEIGHTS:
            parts = []
            for cell in cells:
                for fold in range(3):
                    source = cohorts[(objective, trees, leaves, cell, fold)]
                    ridge, direct = source["ridge_raw"], source["direct_raw"]
                    raw = direct.copy() if weight == 0 else ridge.copy() if weight == 1 else weight*ridge + (1-weight)*direct
                    parts.append(pd.DataFrame({"objective": objective, "ridge_weight": weight,
                        "cell": cell, "fold_index": fold, "target_index": source["targets"],
                        "timestamp": history["data"][cell]["timestamps"][source["targets"]],
                        "actual": source["actual"], "ridge_raw": ridge, "lightgbm_raw": direct,
                        "prediction_raw": raw, "prediction": np.maximum(0., raw)}))
            suffix = format(weight, ".12g").replace(".", "p")
            tables[f"weight-predictions/{objective}-r{suffix}.csv"] = pd.concat(parts, ignore_index=True)
    return tables


def summarize_evidence(original_directory, l1_directory, completion_directory, *, protocol_path=common.PROTOCOL):
    """Audit all 324 capacity cohorts, replay weights and return saveable results."""
    if Path(protocol_path).absolute() != common.PROTOCOL.absolute():
        raise ValueError("Use the approved repository protocol")
    common.protocol()
    protocol_hash, source_hashes = sha256(common.PROTOCOL), common.code_hashes("development")
    history = _load_historical(original_directory, l1_directory)
    completion = _append_completion(history, completion_directory)
    cohorts, cells = _cohorts(history), sorted(TUNING_CELLS)
    replay = replay_study(cohorts, cells)
    selected = replay["selection"]
    diagnostics = {c["id"]: _condition_diagnostics(c, history, cohorts, cells) for c in selected["conditions"]}
    evidence = {"historical_sources": history["audit"], "new_completion": completion,
                "replay_code_sha256": source_hashes, "development_data_only": True}
    wrapper = {"schema_version": 1, "status": "passed", "study_id": common.STUDY_ID,
        **{key: selected[key] for key in ("conditions", "primary_candidate", "selected_components", "selected_weights")},
        "fixed": dict(common.FIXED), "protocol_sha256": protocol_hash, "evidence": evidence}
    summary = {"schema_version": 1, "status": "passed", "study_id": common.STUDY_ID,
        "scope": "Adaptive development selection only; no held-out improvement established",
        "selection": wrapper, "capacity_selection": replay["capacity_selection"],
        "weight_records": replay["weight_records"], "weight_selection": selected,
        "conditions": diagnostics, "nine_equal_condition_ids": [*STANDALONES, "ensemble_l2_r0p5", "ensemble_l1_r0p5"],
        "fixed_50_50_results": {o: diagnostics[f"ensemble_{o}_r0p5"] for o in OBJECTIVES},
        "new_successful_development_fits": 108, "reused_l1_fits": 54, "reused_l2_fits": 162,
        "replayed_weights_per_objective": 8, "replay_performed_new_fitting": False,
        "fixed_epochs_evidence": history["audit"]["selected_lstm_epochs"],
        "development_search_costs": {
            "original": {key: value for key, value in history["original_summary"].items()
                         if "seconds" in key or "timing" in key or key in ("attempts_including_failures", "stage_sessions_including_failures")},
            "historical_l1": history["l1_summary"]["additional_tuning_cost"],
            "new_l1_completion": completion,
            "interpretation": "Search/replay accounting, not formal complete-system cost; attempt and stage durations overlap and must not be added"},
        "formal_comparative_cost_measured": False, "heldout_values_or_errors_read": False,
        "limitations": [
            "This objective/capacity/weight study follows earlier viewed development experiments and is not strict preregistration.",
            "Historical tuning effort differed between model families; completed evidence and prior attempts remain preserved.",
            "Earlier full-year illustration exposure remains disclosed; raw traffic is not claimed wholly unseen.",
            "Residual and partial-ridge architectures are historical only and are excluded from the active independent-ensemble comparison.",
            "Development-selected primary and settings must remain fixed through confirmation/final; endpoints are permitted negative findings about ensembling.",
            "MASE selects settings; MAE, macro RMSE, busiest-hour errors, coverage, clipping and observed-history sensitivity remain supporting outcomes.",
            "Reused standalone L1 training labels are supported by immutable source/metadata and reconstruction, not a fresh fit or a saved-label artifact.",
        ]}
    if sha256(common.PROTOCOL) != protocol_hash or common.code_hashes("development") != source_hashes:
        raise ValueError("Study code/protocol changed during audit or replay")
    for source in (*history["audit"]["sources"].values(), completion):
        verify_inventory(Path(source["directory"]), source["files_sha256"], exclude=(".run.lock",))
    return {"selection": wrapper, "development_summary": summary,
            "prediction_tables": _prediction_tables(cohorts, selected["selected_components"], history, cells)}


def save_evidence(output_directory, payload):
    """Write one new immutable output tree outside all source result folders."""
    destination = _safe_path(Path(output_directory).absolute())
    if destination.exists():
        raise ValueError("Use a new evidence output directory; never overwrite earlier evidence")
    selection, summary = payload["selection"], payload["development_summary"]
    if summary["selection"] != selection or selection["status"] != "passed":
        raise ValueError("Summary and selection do not identify the same completed replay")
    evidence = selection["evidence"]
    sources = [*evidence["historical_sources"]["sources"].values(), evidence["new_completion"]]
    for source in sources:
        directory = Path(source["directory"]).absolute()
        if destination == directory or destination in directory.parents or directory in destination.parents:
            raise ValueError("New evidence must be separate from historical/completion sources")
        verify_inventory(directory, source["files_sha256"], exclude=(".run.lock",))
    if (sha256(common.PROTOCOL) != selection["protocol_sha256"]
            or common.code_hashes("development") != evidence["replay_code_sha256"]):
        raise ValueError("Protocol or replay code changed before saving selection")
    tables = payload["prediction_tables"]
    expected_names = {f"weight-predictions/{o}-r{format(w, '.12g').replace('.', 'p')}.csv"
                      for o in OBJECTIVES for w in WEIGHTS}
    if set(tables) != expected_names:
        raise ValueError("Save every declared objective/weight forecast table")
    expected_columns = {"objective", "ridge_weight", "cell", "fold_index", "target_index", "timestamp",
                        "actual", "ridge_raw", "lightgbm_raw", "prediction_raw", "prediction"}
    if any(set(frame) != expected_columns or frame.empty for frame in tables.values()):
        raise ValueError("Replay forecast tables must preserve components, targets, raw and bounded outputs")
    destination.mkdir(parents=True)
    write_json_atomic(destination / "selection.json", selection)
    write_json_atomic(destination / "development-summary.json", summary)
    (destination / "weight-predictions").mkdir()
    for name, frame in tables.items():
        with (destination / name).open("x", newline="") as stream:
            frame.to_csv(stream, index=False, float_format="%.17g", date_format="%Y-%m-%d %H:%M:%S")
        loaded = pd.read_csv(destination / name, float_precision="round_trip", parse_dates=["timestamp"])
        if list(loaded) != list(frame):
            raise ValueError("Saved replay prediction columns failed round-trip checking")
        for column in frame:
            np.testing.assert_array_equal(loaded[column].to_numpy(), frame[column].to_numpy())
    write_json_atomic(destination / "complete.json", {"schema_version": 1, "status": "passed",
        "study_id": common.STUDY_ID, "files_sha256": _inventory(destination),
        "protocol_sha256": selection["protocol_sha256"], "fitting_performed": False})
    return {"status": "passed", "directory": str(destination),
            "selection_sha256": sha256(destination / "selection.json"),
            "summary_sha256": sha256(destination / "development-summary.json"),
            "primary_candidate": selection["primary_candidate"]}
