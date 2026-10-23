"""Explain two historical weight grids using saved development forecasts only.

The independent blend is the primary frozen method (20% ridge, 80% direct L2
LightGBM). The original residual hybrid is a secondary historical comparison.
No fitting, new weight search, or later-period evaluation occurs.
The eight-weight table reuses the completed ``blend_revision.replay.run`` output.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path, PurePosixPath

from blend_revision.replay import (
    BOUNDARIES, METRICS, WEIGHTS, InputReader, aggregate_scores, decode_json,
    digest, relative_path, require, safe_path,
)


ORIGINAL_SUMMARY_SHA256 = "a3d6905021770195c8ff813c0bbbc4536ecc227371c03420f140e7fe086cb3b2"
ORIGINAL_PLAN_SHA256 = "2e8ac69fa8d596d34311b4e970ed4d87c2d2396e0e0124f32197a157018e63b2"
REPLAY_COMPLETE_SHA256 = "682a0069a45b583cee0165396d1f760377fcc69475d4fa682842fdef5238be5c"
REPLAY_SUMMARY_SHA256 = "019b8490fae05ecdaa554beefc9a2cfe7bcfd5eeff0c1f32996cd87a0c1c067c"
CORRECTION_WEIGHTS = (0, 0.25, 0.5, 0.75, 1)


def close_scores(actual, expected):
    return all(math.isclose(actual[m], expected[m], rel_tol=1e-12, abs_tol=1e-10)
               for m in METRICS)


def _weight_key(weight):
    return str(weight)


def load_verified_replay(replay_dir):
    """Load the specifically audited replay after checking every output hash."""
    reader = InputReader(replay_dir, {"complete.json": REPLAY_COMPLETE_SHA256})
    complete = reader.json("complete.json")
    require(complete["status"] == "passed" and complete["schema_version"] == 1
            and complete["original_plan_sha256"] == ORIGINAL_PLAN_SHA256,
            "Replay completion does not match the original development plan")
    names = {"blend-development-summary.json"} | {
        f"predictions-ridge-w{str(w).replace('.', 'p')}.csv" for w in WEIGHTS}
    require(set(complete["files_sha256"]) == names,
            "Replay inventory does not contain exactly the eight declared weights")
    present = set()
    for path in reader.root.iterdir():
        safe_path(path)
        require(path.is_file(), "Replay inventory contains an unexpected directory or special file")
        present.add(path.name)
    require(present == names | {"complete.json"}, "Replay inventory changed")
    for name, expected in complete["files_sha256"].items():
        reader.read(name, expected)
    summary = reader.json("blend-development-summary.json", REPLAY_SUMMARY_SHA256)
    require(summary["status"] == "passed" and summary["original_plan_sha256"] == ORIGINAL_PLAN_SHA256
            and summary["script_hashes"] == complete["script_hashes"]
            and summary["fitting_performed"] is False
            and summary["confirmation_or_final_values_accessed"] is False,
            "Replay is not the completed development-only arithmetic check")
    require(tuple(summary["declared_weights_before_calculation"]) == WEIGHTS
            and summary["components"] == {"ridge": "ridge-a1000", "direct": "direct-t128-l7"}
            and summary["selected"]["ridge_proportion"] == 0.2,
            "Replay does not reproduce the frozen independent blend")
    return summary, reader


def summarize_candidate(records, direct_records, cells):
    """Macro scores, folds, and paired cell/cohort counts without selection."""
    scores, folds = aggregate_scores(records, cells)
    baseline_scores, baseline_folds = aggregate_scores(direct_records, cells)
    baseline = {(r["cell"], r["fold_index"]): r for r in direct_records}
    differences = [r["MASE"] - baseline[(r["cell"], r["fold_index"])]["MASE"]
                   for r in records]
    per_cell = []
    for cell in cells:
        score = {m: math.fsum(r[m] for r in records if r["cell"] == cell) / 3 for m in METRICS}
        base = {m: math.fsum(r[m] for r in direct_records if r["cell"] == cell) / 3 for m in METRICS}
        per_cell.append({"cell": cell, "scores": score,
                         "mase_change_vs_direct": score["MASE"] - base["MASE"]})
    cell_changes = [r["mase_change_vs_direct"] for r in per_cell]
    require(baseline_scores["MASE"] > 0, "Direct MASE must be positive for relative comparison")
    return {
        "scores": scores, "fold_scores": folds, "per_cell_scores": per_cell,
        "cells_better_than_direct": sum(x < 0 for x in cell_changes),
        "cells_worse_than_direct": sum(x > 0 for x in cell_changes),
        "cells_tied_with_direct": sum(x == 0 for x in cell_changes),
        "cohorts_better_than_direct": sum(x < 0 for x in differences),
        "cohorts_worse_than_direct": sum(x > 0 for x in differences),
        "cohorts_tied_with_direct": sum(x == 0 for x in differences),
        "folds_better_than_direct": sum(a["MASE"] < b["MASE"] for a, b in zip(folds, baseline_folds)),
        "relative_mase_change_percent_vs_direct": 100 * (scores["MASE"] / baseline_scores["MASE"] - 1),
    }


def score_forecast_rows(rows, reference, cell, fold, scale):
    """Recompute one stored cohort, refusing target/mask/actual misalignment."""
    require(math.isfinite(scale) and scale > 0, "Invalid fitting-prefix MASE denominator")
    require(rows and len(rows) == len(reference), "Forecast cohort length differs")
    start, end = BOUNDARIES[fold][1:]
    targets = [int(r["target_index"]) for r in rows]
    require(targets == sorted(set(targets)) and all(start <= t < end for t in targets),
            "Forecast target falls outside the declared development cohort")
    require(targets == [int(r["target_index"]) for r in reference], "Forecast target alignment differs")
    errors = []
    for row, baseline in zip(rows, reference):
        actual, prediction, raw = (float(row[k]) for k in ("actual", "prediction", "prediction_raw"))
        require(row["cell"] == baseline["cell"] == cell
                and int(baseline["fold_index"]) == fold
                and row["method"] == "ridge_lightgbm_hybrid" and row["seed"] == ""
                and all(math.isfinite(v) for v in (actual, prediction, raw))
                and actual == float(baseline["actual"]) and actual >= 0
                and prediction == max(0.0, raw),
                "Forecast identity, target value, or final clipping differs")
        errors.append(prediction - actual)
    mae = math.fsum(abs(e) for e in errors) / len(errors)
    return {"cell": cell, "fold_index": fold, "MAE": mae,
            "RMSE": math.sqrt(math.fsum(e * e for e in errors) / len(errors)), "MASE": mae / scale}


def _correction_records(reader, plan, original_summary, replay, replay_reader):
    selections = original_summary["selections"]
    selected = selections["hybrid"]["selection"]["selected"]
    require(selected["candidate_id"] == "hybrid-t32-l31-w1"
            and selected["parameters"] == {"trees": 32, "leaves": 31, "weight": 1},
            "Historical residual model selection differs")
    selection = selections["hybrid"]
    require(selection["plan_sha256"] == ORIGINAL_PLAN_SHA256
            and selection["confirmation_or_final_evaluated"] is False,
            "Historical residual selection is not development-only")
    # Correction weight zero is exactly the selected ridge forecast, already
    # checked at the weight-one endpoint of the independent blend replay.
    records = {0: replay["per_weight_cohort_scores"]["1"]}
    records.update({w: [] for w in CORRECTION_WEIGHTS[1:]})
    direct_rows = list(csv.DictReader(io.StringIO(replay_reader.read(
        "predictions-ridge-w0.csv", replay_reader.hashes["predictions-ridge-w0.csv"]).decode())))
    reference = {}
    for row in direct_rows:
        reference.setdefault((row["cell"], int(row["fold_index"])), []).append(row)
    scales = {(r["cell"], r["fold_index"]): r["mase_scale"] for r in replay["cohort_evidence"]}
    jobs = [j for j in plan["jobs"] if j["candidate_id"] == "hybrid-t32-l31"]
    require(len(jobs) == 27 and {(j["cell"], j["fold_index"]) for j in jobs} == set(reference),
            "Historical residual jobs differ from the 27 common cohorts")
    for job in jobs:
        cell, fold = job["cell"], job["fold_index"]
        require((job["fit_end"], job["score_start"], job["score_end"]) == BOUNDARIES[fold]
                and job["parameters"] == {"trees": 32, "leaves": 31} and job["seed"] is None,
                "Residual job does not match the historical setting or chronology")
        evidence = selection["evidence"][job["id"]]
        attempt = relative_path(evidence["attempt"])
        require(PurePosixPath(attempt).parent.as_posix() == "jobs/" + job["id"],
                "Residual evidence points to a different job")
        marker = reader.json(attempt + "/complete.json", evidence["completion_sha256"])
        require(marker["status"] == "passed" and marker["schema_version"] == 1
                and marker["job"] == job and marker["plan_sha256"] == ORIGINAL_PLAN_SHA256,
                "Residual completion metadata differs")
        hashes = marker["files_sha256"]
        result = reader.json(attempt + "/result.json", hashes["result.json"])
        require(result["status"] == "passed" and result["job"] == job
                and result["plan_sha256"] == ORIGINAL_PLAN_SHA256,
                "Residual result metadata differs")
        saved = {r["candidate_id"]: r for r in result["records"]}
        require(len(saved) == len(result["records"]) == 4, "Residual records are incomplete or duplicated")
        for weight in CORRECTION_WEIGHTS[1:]:
            cid = "hybrid-t32-l31-w" + str(weight).replace(".", "p")
            name = "predictions/" + cid + ".csv"
            rows = list(csv.DictReader(io.StringIO(reader.read(attempt + "/" + name, hashes[name]).decode())))
            score = score_forecast_rows(rows, reference[(cell, fold)], cell, fold, scales[(cell, fold)])
            record = saved[cid]
            require(record["cell"] == cell and record["fold_index"] == fold
                    and record["seed"] is None and record["status"] == "passed"
                    and math.isclose(record["mase_scale"], scales[(cell, fold)], rel_tol=1e-12)
                    and close_scores(score, record["scores"]), "Saved residual scores do not reproduce forecasts")
            records[weight].append(score)
    return records


def build_substudy(original_dir, replay_dir):
    """Return a reproducible summary; read only pinned development artifacts."""
    replay, replay_reader = load_verified_replay(replay_dir)
    expected = dict(replay["read_file_hashes"])
    expected.update({"development-summary.json": ORIGINAL_SUMMARY_SHA256, "plan.json": ORIGINAL_PLAN_SHA256})
    reader = InputReader(original_dir, expected)
    original_summary = reader.json("development-summary.json")
    plan = reader.json("plan.json")
    require(original_summary["status"] == "passed" and original_summary["holdout_forecasts_evaluated"] is False
            and original_summary["plan_sha256"] == ORIGINAL_PLAN_SHA256
            and plan["holdout_forecasts_enabled"] is False and plan["scope"] == "declared development search only",
            "Only completed original development results are accepted")
    # Recheck all original dependencies used by the completed blend replay.
    for name, expected_hash in replay["read_file_hashes"].items():
        reader.read(name, expected_hash)
    cells = sorted(plan["series"])
    require(len(cells) == 9, "Expected the original nine development cells")
    direct = replay["per_weight_cohort_scores"]["0"]
    blend_table = []
    for weight in WEIGHTS:
        result = summarize_candidate(replay["per_weight_cohort_scores"][_weight_key(weight)], direct, cells)
        prior = next(r for r in replay["table"] if r["ridge_proportion"] == weight)
        require(close_scores(result["scores"], prior["scores"])
                and all(close_scores(a, b) for a, b in zip(result["fold_scores"], prior["fold_scores"])),
                "Replay table does not match its cohort records")
        blend_table.append({"ridge_weight": weight, "direct_weight": 1 - weight,
                            "role": "frozen_primary" if weight == 0.2 else "historical_development_variant", **result})
    correction_records = _correction_records(reader, plan, original_summary, replay, replay_reader)
    correction_table = []
    candidates = {c["candidate_id"]: c for c in original_summary["selections"]["hybrid"]["selection"]["candidates"]}
    for weight in CORRECTION_WEIGHTS:
        result = summarize_candidate(correction_records[weight], direct, cells)
        cid = "hybrid-zero" if weight == 0 else "hybrid-t32-l31-w" + str(weight).replace(".", "p")
        require(close_scores(result["scores"], candidates[cid]["scores"]),
                "Recomputed residual scores differ from the immutable original selection")
        correction_table.append({"correction_weight": weight, "ridge_coefficient": 1,
                                 "role": "frozen_secondary" if weight == 1 else "historical_development_variant", **result})
    reader.verify_unchanged()
    replay_reader.verify_unchanged()
    return {
        "schema_version": 1, "status": "passed", "scope": "Previously viewed development forecasts only",
        "fitting_performed": False, "new_selection_performed": False,
        "confirmation_or_final_values_accessed": False, "active_tree_objective": "squared_error",
        "cohort": {"cells": cells, "cell_count": 9, "fold_count": 3, "cell_fold_count": 27,
                   "fit_score_boundaries": BOUNDARIES, "forecasts_per_variant": 18144},
        "aggregation": "Mean metric across cells within each fold, then mean across folds; RMSE is macro-RMSE, not pooled RMSE",
        "comparison_reference": {"candidate_id": "direct-t128-l7", "loss": "L2", "scores": blend_table[0]["scores"]},
        "independent_blend": {
            "formula_original_units": "max(0, w * ridge_raw + (1-w) * direct_raw)",
            "weight_meaning": "Fraction of an independently trained ridge traffic forecast; zero is direct LightGBM, one is ridge",
            "components": {"ridge_alpha": 1000, "direct_trees": 128, "direct_leaves": 7, "direct_loss": "L2"},
            "frozen_ridge_weight": 0.2, "table": blend_table},
        "original_residual_hybrid": {
            "formula_standardized_units": "ridge_z + lambda * correction_z; correction learns z_actual - ridge_z; inverse scale once, then zero clip",
            "weight_meaning": "Strength of a learned error correction, not a fraction of a second independent traffic forecast; zero is ridge",
            "components": {"ridge_alpha": 1000, "correction_trees": 32, "correction_leaves": 31, "correction_loss": "L2"},
            "frozen_correction_weight": 1, "table": correction_table},
        "provenance": {"original_summary_sha256": ORIGINAL_SUMMARY_SHA256, "original_plan_sha256": ORIGINAL_PLAN_SHA256,
                       "replay_complete_sha256": REPLAY_COMPLETE_SHA256, "replay_summary_sha256": REPLAY_SUMMARY_SHA256,
                       "replay_producer": "blend_revision.replay.run",
                       "original_files_verified": len(reader.hashes), "replay_files_verified": len(replay_reader.hashes),
                       "original_read_inventory_sha256": digest(json.dumps(reader.hashes, sort_keys=True).encode()),
                       "positive_correction_forecasts_recomputed": 4 * 18144, "inputs_unchanged_after_read": True},
        "limitations": [
            "All variants were already evaluated during development; these are not independent validation results.",
            "Weights mean different things in the two architectures and must not be labelled interchangeably.",
            "The frozen primary remains 20% ridge / 80% direct L2 LightGBM; this summary does not reopen selection.",
            "Standalone endpoints reuse verified component results; positive correction variants reuse existing residual-tree forecasts.",
            "No deployment compute, held-out superiority, statistical significance, or equal historical tuning budget is established.",
            "Other development experiments remain preserved separately; this bounded table does not establish the best known model configuration."]}


def write_substudy(original_dir, replay_dir, output_path):
    """Write one small summary only after every input passes verification."""
    output = safe_path(output_path)
    originals = (safe_path(original_dir), safe_path(replay_dir))
    require(not output.exists(), "Refusing to overwrite a prior substudy")
    require(output.parent.is_dir() and all(p != output and p not in output.parents for p in originals),
            "Output must be outside both immutable input directories")
    result = build_substudy(original_dir, replay_dir)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", required=True, type=Path)
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = write_substudy(args.original_dir, args.replay_dir, args.output)
    print(json.dumps({"status": result["status"], "fitting_performed": False,
                      "new_selection_performed": False, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
