"""Replay the declared global blend grid from frozen development forecasts.

No estimator is fitted, no full-year source is opened, and no held-out period
is read. The input manifest deliberately binds this replay to the audited
experiment. To investigate a different experiment, declare a separate study.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform

import numpy as np


WEIGHTS = (0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1)
COMPONENTS = {"ridge": "ridge-a1000", "direct": "direct-t128-l7"}
METRICS = ("MAE", "RMSE", "MASE")
BOUNDARIES = ((4116, 4116, 4788), (4788, 4788, 5460), (5460, 5460, 6132))
SHARED_SOURCES = ("evaluation/__init__.py", "evaluation/metrics.py",
                  "preprocessing/__init__.py", "preprocessing/splits.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "Parent traversal is forbidden")
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink(), f"Symbolic links are forbidden: {ancestor}")
    return path


def relative_path(name):
    require(isinstance(name, str) and bool(name), "Expected a relative path")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and "\\" not in name
            and p.as_posix() == name and name != ".", "Unsafe input path")
    return name


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    result = json.loads(data, object_pairs_hook=unique)
    json.dumps(result, allow_nan=False)
    return result


class InputReader:
    """Read regular input files and preserve the exact bytes' provenance."""

    def __init__(self, original_dir, expected):
        self.root = safe_path(original_dir)
        require(self.root.is_dir(), "Original results directory is missing")
        self.expected = expected
        self.hashes = {}

    def read(self, name, expected=None):
        name = relative_path(name)
        path = safe_path(self.root / name)
        require(path.is_file(), f"Expected a regular input file: {name}")
        data = path.read_bytes()
        actual = digest(data)
        pinned = self.expected.get(name)
        require(expected is not None or pinned is not None,
                f"Unbound input read refused: {name}")
        for value in (pinned, expected, self.hashes.get(name)):
            require(value is None or value == actual, f"Input hash mismatch: {name}")
        self.hashes[name] = actual
        return data

    def json(self, name, expected=None):
        return decode_json(self.read(name, expected))

    def verify_unchanged(self):
        for name, expected in tuple(self.hashes.items()):
            self.read(name, expected)


def aggregate_scores(records, cells):
    """Require one record per fixed cell/fold, then average cells and folds."""
    expected = {(cell, fold) for cell in cells for fold in range(3)}
    actual = [(row["cell"], row["fold_index"]) for row in records]
    require(len(actual) == len(expected) and set(actual) == expected,
            "Missing, extra, or duplicated scoring cohort")
    for row in records:
        require(all(math.isfinite(row[m]) and row[m] >= 0 for m in METRICS),
                "Scores must be finite and nonnegative")
    folds = [{m: math.fsum(row[m] for row in records if row["fold_index"] == f)
              / len(cells) for m in METRICS} for f in range(3)]
    scores = {m: math.fsum(row[m] for row in folds) / 3 for m in METRICS}
    return scores, folds


def select_weight(table):
    """Use unrounded MASE; exact ties prefer weight zero, then lower weight."""
    require(len(table) == len(WEIGHTS)
            and {row["ridge_proportion"] for row in table} == set(WEIGHTS),
            "Selection requires exactly the declared global weight grid")
    require(all(math.isfinite(row["scores"]["MASE"])
                and row["scores"]["MASE"] >= 0 for row in table),
            "Selection MASE must be finite and nonnegative")
    return min(table, key=lambda row: (row["scores"]["MASE"],
                                      row["ridge_proportion"]))


def verify_attempt(reader, selection, job, plan_hash):
    evidence = selection["evidence"][job["id"]]
    attempt = relative_path(evidence["attempt"])
    require(PurePosixPath(attempt).parent.as_posix() == "jobs/" + job["id"],
            "Selection points to the wrong job directory")
    marker = reader.json(attempt + "/complete.json", evidence["completion_sha256"])
    require(marker["status"] == "passed" and marker["schema_version"] == 1
            and marker["job"] == job and marker["plan_sha256"] == plan_hash,
            "Completion metadata does not match the fixed job/plan")
    inventory = marker["files_sha256"]
    required = {"attempt.json", "result.json", "model/model.json",
                "predictions/" + job["candidate_id"] + ".csv"}
    require(required <= set(inventory), "Incomplete selected-attempt inventory")
    present = set()
    for path in (reader.root / attempt).rglob("*"):
        safe_path(path)
        if path.is_dir():
            continue
        require(path.is_file(), "Special files are forbidden in an attempt")
        present.add(path.relative_to(reader.root / attempt).as_posix())
    require(present == set(inventory) | {"complete.json"},
            "Selected-attempt artifact inventory changed")
    for name, expected in inventory.items():
        reader.read(attempt + "/" + relative_path(name), expected)
    metadata = reader.json(attempt + "/attempt.json", inventory["attempt.json"])
    require(metadata["job"] == job and metadata["plan_sha256"] == plan_hash,
            "Attempt metadata differs from its completion")
    return attempt, inventory


def validate_output(original_dir, output_dir):
    original = safe_path(original_dir)
    output = safe_path(output_dir)
    require(not output.exists(), "Output directory already exists; refusing overwrite")
    require(output != original and original not in output.parents,
            "Output must be outside original results")
    require(output.parent.is_dir(), "Output parent directory must already exist")
    return output


def run(original_dir, output_dir):
    output = validate_output(original_dir, output_dir)
    package = Path(__file__).parent
    script_paths = (package / "replay.py", package / "model.py", package / "__init__.py",
                    package / "expected_inputs.json")
    script_hashes = {p.name: digest(safe_path(p).read_bytes()) for p in script_paths}
    manifest_path = safe_path(package / "expected_inputs.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest = decode_json(manifest_bytes)
    require(manifest["schema_version"] == 1 and manifest["components"] == COMPONENTS
            and tuple(manifest["weights"]) == WEIGHTS
            and tuple(map(tuple, manifest["fit_score_boundaries"])) == BOUNDARIES,
            "Frozen replay specification differs from the declared design")
    reader = InputReader(original_dir, manifest["files_sha256"])
    plan = reader.json("plan.json")
    plan_hash = reader.hashes["plan.json"]
    require(plan["holdout_forecasts_enabled"] is False
            and plan["scope"] == "declared development search only",
            "Only the original development plan may be replayed")
    require(all(plan[key] == expected for key, expected in
                manifest["original_provenance"].items()), "Original provenance differs")
    shared_hashes = {name: digest(safe_path(package.parent / name).read_bytes())
                     for name in SHARED_SOURCES}
    require(all(value == plan["code_sha256"][name] for name, value in shared_hashes.items()),
            "Local replay dependency differs from the pinned original source")
    from blend_revision.model import blend_raw_predictions

    for name in SHARED_SOURCES:
        module_name = name[:-3].replace("/", ".").removesuffix(".__init__")
        module = importlib.import_module(module_name)
        require(Path(module.__file__).resolve() == (package.parent / name).resolve(),
                "Replay imported a different local source dependency")
    require(all(importlib.metadata.version(name) == plan["environment_identity"]["packages"][name]
                for name in ("numpy", "pandas")), "Numerical package versions differ from the original run")
    library_modules = (np, importlib.import_module("numpy._core._multiarray_umath"),
                       importlib.import_module("pandas"))
    runtime_library_files = {module.__name__: {"path": str(Path(module.__file__).resolve()),
                             "sha256": digest(Path(module.__file__).read_bytes())}
                             for module in library_modules}
    cells = manifest["cells"]
    require(len(cells) == 9 and cells == sorted(set(cells))
            and cells == sorted(plan["series"]), "Development cohort differs")
    selections = {name: reader.json(f"selections/{name}.json") for name in COMPONENTS}
    for name, selection in selections.items():
        require(selection["plan_sha256"] == plan_hash and selection["stage"] == name
                and selection["confirmation_or_final_evaluated"] is False
                and selection["selection"]["selected"]["candidate_id"] == COMPONENTS[name],
                f"Selected {name} dependency differs")
    selected_jobs = [job for job in plan["jobs"]
                     if job["candidate_id"] in COMPONENTS.values()]
    jobs = {job["id"]: job for job in selected_jobs}
    require(len(jobs) == len(selected_jobs) == 54, "Expected 54 unique component jobs")
    records = {weight: [] for weight in WEIGHTS}
    forecasts = {weight: [] for weight in WEIGHTS}
    cohort_evidence = []
    dependencies = {}
    for cell in cells:
        specification = plan["series"][cell]
        require(specification["path"] == f"series/{cell}.npz", "Unexpected series path")
        data_bytes = reader.read(specification["path"], specification["sha256"])
        with np.load(io.BytesIO(data_bytes), allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        require({"traffic", "traffic_observed", "observed", "timestamps"} <= set(data),
                "Required series arrays are missing")
        require(all(data[key].ndim == 1 and len(data[key]) == 6132 for key in
                    ("traffic", "traffic_observed", "observed", "timestamps")),
                "Replay accepts development prefixes of exactly 6132 hours")
        observed, traffic = data["observed"], data["traffic"]
        actual_series = data["traffic_observed"]
        require(observed.dtype.kind == "b", "Observed mask must be boolean")
        for fold, (fit_end, score_start, score_end) in enumerate(BOUNDARIES):
            pairs = [i for i in range(168, fit_end) if observed[i] and observed[i - 168]]
            require(bool(pairs), "No observed fitting-prefix weekly pairs")
            scale = math.fsum(abs(float(actual_series[i]) - float(actual_series[i - 168]))
                              for i in pairs) / len(pairs)
            require(math.isfinite(scale) and scale > 0, "Invalid fitting-prefix MASE scale")
            targets = [i for i in range(score_start, score_end)
                       if observed[i] and np.isfinite(traffic[i - 168:i]).all()]
            recorded_targets = specification["folds"][fold]["targets"]
            require(targets == recorded_targets and len(targets) == 672,
                    "Observed common target mask differs from original plan")
            y = [float(actual_series[i]) for i in targets]
            require(all(math.isfinite(value) for value in y), "Nonfinite observed target")
            raw = {}
            for name, cid in COMPONENTS.items():
                job_id = f"{cid}--{cell}--f{fold + 1}"
                job = jobs[job_id]
                expected_parameters = {"alpha": 1000} if name == "ridge" else {"trees": 128, "leaves": 7}
                require((job["fit_end"], job["score_start"], job["score_end"]) == BOUNDARIES[fold]
                        and job["cell"] == cell and job["fold_index"] == fold
                        and job["parameters"] == expected_parameters and job["seed"] is None,
                        "Component job differs from the declared design")
                attempt, inventory = verify_attempt(reader, selections[name], job, plan_hash)
                dependencies[job_id] = selections[name]["evidence"][job_id]
                name_csv = f"predictions/{cid}.csv"
                text = reader.read(f"{attempt}/{name_csv}", inventory[name_csv]).decode()
                rows = list(csv.DictReader(io.StringIO(text)))
                require([int(row["target_index"]) for row in rows] == targets,
                        "Forecast indices do not match the observed common mask")
                values = []
                for row, target in zip(rows, targets):
                    value = float(row["prediction_raw"])
                    require(row["cell"] == cell and row["seed"] == ""
                            and float(row["actual"]) == float(actual_series[target])
                            and np.datetime64(row["timestamp"], "ns") == data["timestamps"][target]
                            and math.isfinite(value) and float(row["prediction"]) == max(0.0, value),
                            "Forecast alignment, raw output, or clipping differs")
                    values.append(value)
                raw[name] = np.asarray(values, dtype=float)
            cohort_evidence.append({"cell": cell, "fold_index": fold, "fit_end": fit_end,
                                    "score_start": score_start, "score_end": score_end,
                                    "targets": len(targets), "weekly_pairs": len(pairs),
                                    "mase_scale": scale})
            for weight in WEIGHTS:
                mixed_raw, predictions = blend_raw_predictions(raw["ridge"], raw["direct"], weight)
                errors = [float(p) - a for p, a in zip(predictions, y)]
                mae = math.fsum(abs(e) for e in errors) / len(errors)
                records[weight].append({"cell": cell, "fold_index": fold, "MAE": mae,
                                        "RMSE": math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
                                        "MASE": mae / scale,
                                        "clipped_count": int(np.count_nonzero(mixed_raw < 0))})
                forecasts[weight].extend({"cell": cell, "fold_index": fold,
                                          "target_index": target, "actual": actual,
                                          "ridge_raw": float(r), "direct_raw": float(d),
                                          "prediction_raw": float(m), "prediction": float(p)}
                                         for target, actual, r, d, m, p in
                                         zip(targets, y, raw["ridge"], raw["direct"], mixed_raw, predictions))
    require(set(manifest["files_sha256"]) <= set(reader.hashes),
            "Some frozen diagnostic dependencies were not consumed")
    table = []
    for weight in WEIGHTS:
        scores, folds = aggregate_scores(records[weight], cells)
        base = records[0]
        better = sum(r["MASE"] < d["MASE"] for r, d in zip(records[weight], base))
        worse = sum(r["MASE"] > d["MASE"] for r, d in zip(records[weight], base))
        table.append({"ridge_proportion": weight, "direct_proportion": 1 - weight,
                      "scores": scores, "fold_scores": folds,
                      "cohorts_better_than_direct": better, "cohorts_worse_than_direct": worse,
                      "cohorts_tied_with_direct": 27 - better - worse,
                      "clipped_predictions": sum(row["clipped_count"] for row in records[weight])})
    for weight, name in ((0, "direct"), (1, "ridge")):
        computed = next(row for row in table if row["ridge_proportion"] == weight)
        saved = selections[name]["selection"]["selected"]
        require(all(math.isclose(computed["scores"][m], saved["scores"][m],
                                 rel_tol=1e-12, abs_tol=1e-10) for m in METRICS),
                "Component endpoint scores differ from saved selections")
    for row in table:
        row["relative_mase_change_percent_vs_direct"] = 100 * (row["scores"]["MASE"] / table[0]["scores"]["MASE"] - 1)
        row["fold_relative_mase_change_percent_vs_direct"] = [
            100 * (f["MASE"] / b["MASE"] - 1) for f, b in zip(row["fold_scores"], table[0]["fold_scores"])]
        row["folds_improved"] = sum(f["MASE"] < b["MASE"] for f, b in zip(row["fold_scores"], table[0]["fold_scores"]))
    selected = select_weight(table)
    per_cell = []
    for cell in cells:
        scores = {str(weight): {m: math.fsum(row[m] for row in records[weight] if row["cell"] == cell) / 3
                               for m in METRICS} for weight in WEIGHTS}
        per_cell.append({"cell": cell, "scores_by_ridge_proportion": scores})
    require(script_hashes == {p.name: digest(safe_path(p).read_bytes()) for p in script_paths},
            "Replay implementation changed during calculation")
    require(script_hashes["expected_inputs.json"] == digest(manifest_bytes),
            "Replay input manifest changed before calculation")
    require(shared_hashes == {name: digest(safe_path(package.parent / name).read_bytes())
                              for name in SHARED_SOURCES},
            "Local replay dependency changed during calculation")
    require(all(digest(Path(item["path"]).read_bytes()) == item["sha256"]
                for item in runtime_library_files.values()),
            "Numerical runtime library changed during calculation")
    reader.verify_unchanged()
    result = {
        "schema_version": 1, "status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Reproduction of post-hoc development blend selection; not independent validation",
        "method": "Independent ridge/direct-LightGBM raw forecasts blended in original units; final-only zero clipping",
        "components": COMPONENTS, "declared_weights_before_calculation": WEIGHTS,
        "aggregation": "Equal cells within each fold, then equal fold means; fitting-prefix weekly MASE scales",
        "selection_rule": "Lowest full-precision development MASE; exact ties prefer zero then smaller ridge weight",
        "selected": selected, "table": table,
        "per_weight_cohort_scores": {str(w): records[w] for w in WEIGHTS},
        "per_cell_scores": per_cell, "cohort_evidence": cohort_evidence,
        "cells": 9, "folds": 3, "cohorts": 27, "forecasts_per_candidate": 18144,
        "component_forecasts_checked": 36288, "component_endpoint_scores_reproduce_saved_selections": True,
        "original_plan_sha256": plan_hash, "original_provenance": manifest["original_provenance"],
        "selection_dependencies": dependencies, "read_file_hashes": reader.hashes,
        "script_hashes": script_hashes, "original_inputs_unchanged_after_replay": True,
        "local_execution_source_sha256": shared_hashes,
        "local_execution_sources_match_original_plan": True,
        "runtime_library_files": runtime_library_files,
        "fitting_performed": False, "confirmation_or_final_values_accessed": False,
        "full_year_source_reopened": False, "formal_comparative_timing": False,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": importlib.metadata.version("pandas")},
        "limitations": ["Components and global blend weight were selected on these same development folds.",
                        "This reproduces an already viewed diagnostic; it supplies no new independent evidence.",
                        "The original residual hybrid and the failed ridge extension remain separate historical results.",
                        "No held-out superiority, significance, or complete-model computational advantage is established."]}
    # All reads and validation finish before any output is created. mkdir is
    # exclusive, so competing invocations cannot overwrite another replay.
    output.mkdir()
    summary_path = output / "blend-development-summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for weight in WEIGHTS:
        path = output / f"predictions-ridge-w{str(weight).replace('.', 'p')}.csv"
        with path.open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(forecasts[weight][0]))
            writer.writeheader()
            writer.writerows(forecasts[weight])
    output_hashes = {p.name: digest(p.read_bytes()) for p in sorted(output.iterdir())}
    (output / "complete.json").write_text(json.dumps({"status": "passed", "schema_version": 1,
        "files_sha256": output_hashes, "script_hashes": script_hashes,
        "original_plan_sha256": plan_hash}, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.original_dir, args.output_dir)
    print(json.dumps({"status": result["status"], "selected": result["selected"],
                      "fitting_performed": False, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
