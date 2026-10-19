"""Verify the prepared files and load exact values with a common target mask.

Run from the repository root: python -m tools.check_prepared_data DATA_DIRECTORY
This checks data preparation only; it does not fit or score a forecasting model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.prepare import causal_fill
from preprocessing.splits import CALENDAR_START, N_HOURS, LOOKBACK, TRAIN_END, VAL_END
from preprocessing.validate_data import sha256
from preprocessing.windows import eligible_targets

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_files(directory, project_root=PROJECT_ROOT):
    """Use the committed checklist, rather than trusting a downloaded report."""
    directory, project_root = Path(directory), Path(project_root)
    reference = json.loads((project_root / "docs/data-bundle-checksums.json").read_text())
    # The saved bundle and the current configuration have separate file identities.
    expected_config = reference.get("runtime_config_sha256", reference["config_sha256"])
    if sha256(project_root / "run_config.yaml") != expected_config:
        raise ValueError("Current configuration differs from the verified specification")
    for name, expected in reference["code_sha256"].items():
        if Path(name).name != name or sha256(project_root / "preprocessing" / name) != expected:
            raise ValueError(f"Preparation code changed since validation: {name}")
    for name, expected in reference["files"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"Prepared file differs from the verified copy: {name}")
    report = json.loads((directory / "validation_report.json").read_text())
    if report.get("config_sha256") != reference["config_sha256"]:
        raise ValueError("Prepared data was validated against a different source configuration")
    if not report["raw_unchanged"] or not report["tuning_masks_computable"]:
        raise ValueError("The data report does not establish preparation readiness")
    return report


def load_validated_data(directory, project_root=PROJECT_ROOT):
    """Return (hourly table, report); use report lists for eligible/tuning IDs."""
    report = verify_files(directory, project_root)
    # The default CSV parser can round saved float64 values differently.
    hourly = pd.read_csv(Path(directory) / "hourly.csv", parse_dates=["ts"],
                         float_precision="round_trip")
    columns = ["cell", "ts", "traffic_observed", "traffic", "observed"]
    if list(hourly.columns) != columns or len(hourly) != report["grid_rows"]:
        raise ValueError("Unexpected hourly table schema or row count")
    cells = sorted(hourly.cell.unique())
    calendar = pd.date_range(CALENDAR_START, periods=N_HOURS, freq="h")
    expected_index = pd.MultiIndex.from_product([cells, calendar], names=["cell", "ts"])
    actual_index = pd.MultiIndex.from_frame(hourly[["cell", "ts"]])
    if len(cells) != report["integrity"]["cells"] or not actual_index.equals(expected_index):
        raise ValueError("Hourly rows must contain the complete ordered cell calendars")
    masks = []
    for _, group in hourly.groupby("cell", sort=False):
        expected_inputs, expected_observed = causal_fill(group.traffic_observed.to_numpy())
        if not np.array_equal(group.observed.to_numpy(), expected_observed):
            raise ValueError("Observation flags disagree with original measurements")
        if not np.array_equal(group.traffic.to_numpy(), expected_inputs, equal_nan=True):
            raise ValueError("Prepared inputs disagree with the approved causal filling")
        masks.append(eligible_targets(expected_inputs, expected_observed))
    # Every model will use this observed-target, past-only eligibility rule.
    hourly["eligible_target"] = np.concatenate(masks)
    mask_by_cell = hourly.eligible_target.to_numpy().reshape(len(cells), N_HOURS)
    periods = {"development": (LOOKBACK, TRAIN_END), "confirmation": (TRAIN_END, VAL_END),
               "final": (VAL_END, N_HOURS)}
    for row in report["coverage"]:
        if row["cap_hours"] == 3:
            start, end = periods[row["period"]]
            if int(mask_by_cell[:, start:end].sum()) != row["eligible_targets"]:
                raise ValueError("Reconstructed target mask differs from the validation report")
    return hourly, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory")
    args = parser.parse_args()
    hourly, report = load_validated_data(args.data_directory)
    print(json.dumps({"status": "passed", "scope": "prepared data only; no forecasting",
        "grid_rows": len(hourly), "cells": report["integrity"]["cells"],
        "genuine_observations": int(hourly.observed.sum()),
        "eligible_targets": int(hourly.eligible_target.sum()),
        "source_sha256": report["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
