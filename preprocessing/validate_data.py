"""Reproducible integrity, preparation and development-only quality report.

No forecasting models are fitted. Holdout values are checked for raw validity
and causally prepared, but their traffic distributions and errors are not analysed.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from preprocessing.prepare import load_raw, prepare, causal_fill
from preprocessing.windows import eligible_targets
from preprocessing.select_cells import select_cells, prefix_stats
from preprocessing.splits import N_HOURS, TRAIN_END, VAL_END, FOLDS, LOOKBACK, MAX_FILL_HOURS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def check_config(config):
    """Fail on drift between data validation and approved specification."""
    expected = [
        config["chronology"]["development"] == [0, TRAIN_END],
        config["chronology"]["confirmation"] == [TRAIN_END, VAL_END],
        config["chronology"]["final_test"] == [VAL_END, N_HOURS],
        config["features"]["traffic_lags"] == LOOKBACK,
        config["data"]["calendar_hours_provisional"] == N_HOURS,
        config["data"]["input_fill"]["development_check_caps"] == [0, 1, 3],
        config["tuning_panel"]["maximum_cells"] == 9,
        config["tuning_panel"]["quantiles"] == [i/10 for i in range(1, 10)],
        config["tuning_panel"]["quantile_method"] == "linear",
        config["eligibility"]["arbitrary_observed_fraction_cutoff"] is None,
        config["data"]["require_observed_scoring_and_training_target"] is True,
        config["data"]["require_complete_prepared_history"] is True,
        config["data"]["history_hours"] == LOOKBACK,
        config["metrics"]["mase_lag"] == 168,
        config["features"]["original_missing_flags"] == LOOKBACK,
        config["data"]["input_fill"]["maximum_hours_provisional"] == MAX_FILL_HOURS,
        config["tuning_panel"]["selection_prefix"] == [0, FOLDS[0]["train_end"]],
        config["eligibility"]["main_selection_prefix"] == [0, TRAIN_END],
        config["models"]["lstm"]["inner_stopping_tail_hours"] == 336,
        config["chronology"]["folds"] == [
            {"fit": [0, f["train_end"]], "score": [f["eval_start"], f["eval_end"]]} for f in FOLDS],
    ]
    if not all(expected):
        raise ValueError("Data-validation constants differ from the scientific configuration")


def coverage_rows(prepared):
    # Holdout checks count availability only; no traffic distribution is inspected.
    rows = []
    periods = [("development", LOOKBACK, TRAIN_END),
               ("confirmation", TRAIN_END, VAL_END), ("final", VAL_END, N_HOURS)]
    for cell, group in prepared.groupby("cell", sort=True):
        original = group.traffic_observed.to_numpy()
        for cap in [0, 1, 3]:
            filled, observed = causal_fill(original, cap)
            usable = eligible_targets(filled, observed)
            for period, start, end in periods:
                count = int(usable[start:end].sum())
                actual = int(observed[start:end].sum())
                rows.append({"cell": cell, "cap_hours": cap, "period": period,
                    "calendar_targets": end-start, "observed_targets": actual,
                    "eligible_targets": count, "fraction_calendar": count/(end-start),
                    "fraction_observed": count/actual if actual else np.nan})
    return pd.DataFrame(rows)


def gap_rows(prepared):
    rows = []
    for cell, group in prepared.groupby("cell", sort=True):
        missing = ~group.observed.to_numpy().astype(bool)
        edges = np.diff(np.r_[False, missing, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            rows.append({"cell": cell, "start_index": int(start), "end_exclusive": int(end),
                "hours": int(end-start), "start": str(group.ts.iloc[start]),
                "last_missing": str(group.ts.iloc[end-1])})
    return pd.DataFrame(rows)


def development_checks(prepared):
    # Limit all numerical traffic diagnostics to the development period.
    prefixes, variation, days = [], [], []
    for cell, group in prepared.groupby("cell", sort=True):
        values = group.traffic.to_numpy()[:TRAIN_END]
        observed = group.observed.to_numpy()[:TRAIN_END].astype(bool)
        ok = eligible_targets(values, observed)
        scale = prefix_stats(values, observed, TRAIN_END)["weekly_scale"]
        for fold_index, fold in enumerate(FOLDS, 1):
            fit_end = fold["train_end"]
            inner_end = fit_end-336
            for kind, end in [("outer_fit", fit_end), ("inner_fit", inner_end)]:
                prefixes.append({"cell": cell, "fold": fold_index, "segment": kind,
                    **prefix_stats(values, observed, end),
                    "stopping_targets": int(ok[inner_end:fit_end].sum()),
                    "outer_score_targets": int(ok[fold["eval_start"]:fold["eval_end"]].sum())})
        for lag in [1, 2, 3, 24, 168]:
            pairs = observed[lag:] & observed[:-lag]
            difference = np.abs(values[lag:]-values[:-lag])[pairs]
            variation.append({"cell": cell, "lag_hours": lag, "observed_pairs": int(pairs.sum()),
                "mean_absolute_change": float(difference.mean()) if pairs.any() else np.nan,
                "change_over_weekly_scale": float(difference.mean()/scale) if pairs.any() and scale > 0 else np.nan})
        complete, eligible_days = 0, 0
        # Calendar starts at midnight; discard an incomplete final calendar day.
        for start in range(0, TRAIN_END-23, 24):
            if observed[start:start+24].all():
                complete += 1
                day = values[start:start+24]
                maxima = np.flatnonzero(day == day.max()) + start
                eligible_days += int(ok[maxima].all())
        days.append({"cell": cell, "complete_observed_days": complete,
                     "eligible_busiest_hour_days": eligible_days})
    return pd.DataFrame(prefixes), pd.DataFrame(variation), pd.DataFrame(days)


def protect_sources(destinations, sources):
    """Reject aliases, symlinks and hard links to any input before writing."""
    for destination in map(Path, destinations):
        for source in map(Path, sources):
            same_path = destination.resolve() == source.resolve()
            same_file = destination.exists() and source.exists() and destination.samefile(source)
            if same_path or same_file:
                raise ValueError("Output would overwrite a source file; use a separate output directory")


def check_tuning_panel(config, selected_ids, computable):
    """Stop before saving a bundle that disagrees with the experiment rules."""
    if selected_ids != config["tuning_panel"]["ids"]:
        raise ValueError("Selected tuning cells differ from run_config.yaml; review before use")
    if not computable:
        raise ValueError("Tuning cells lack usable fitting, stopping or scoring data")


def run(raw_path, output_dir, config_path, manifest_path):
    raw_path, output_dir = Path(raw_path).resolve(), Path(output_dir).resolve()
    config = yaml.safe_load(Path(config_path).read_text())
    check_config(config)
    before = sha256(raw_path)
    if before != config["data"]["raw_sha256"]:
        raise ValueError("Raw SHA-256 differs from the approved source; review before use")
    raw = load_raw(raw_path)
    prepared = prepare(raw)
    main, early, panel = select_cells(prepared)
    coverage = coverage_rows(prepared)
    gaps = gap_rows(prepared)
    prefixes, variation, days = development_checks(prepared)
    panel_prefixes = prefixes[prefixes.cell.isin(panel.cell)]
    panel_valid = bool(len(panel) and (panel_prefixes.exclusion_reason == "").all()
                       and (panel_prefixes.stopping_targets > 0).all()
                       and (panel_prefixes.outer_score_targets > 0).all())
    check_tuning_panel(config, panel.cell.tolist(), panel_valid)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"hourly.csv": prepared, "eligible_cells.csv": main,
        "early_cells.csv": early, "tuning_cells.csv": panel, "coverage.csv": coverage,
        "missing_runs.csv": gaps, "development_prefixes.csv": prefixes,
        "development_variation.csv": variation, "development_days.csv": days}
    destinations = list(outputs) + ["validation_report.json", "data_manifest.json"]
    sources = {raw_path, Path(config_path).resolve(), Path(manifest_path).resolve()}
    protect_sources([output_dir/name for name in destinations], sources)
    for name, frame in outputs.items():
        destination = output_dir/name
        if destination.resolve() in [raw_path, Path(config_path).resolve(), Path(manifest_path).resolve()]:
            raise ValueError("Output would overwrite a source file")
        # Seventeen significant digits preserve float64 values on a round-trip read.
        frame.to_csv(destination, index=False, float_format="%.17g")
    summary = coverage.groupby(["cap_hours", "period"], sort=True)[
        ["calendar_targets", "observed_targets", "eligible_targets"]].sum().reset_index()
    summary["fraction_calendar"] = summary.eligible_targets/summary.calendar_targets
    summary["fraction_observed"] = summary.eligible_targets/summary.observed_targets
    report = {"schema_version": 1, "source_sha256": before,
        "scope": "Integrity and preparation only; numerical quality diagnostics restricted to development.",
        "integrity": raw.attrs["integrity"], "grid_rows": len(prepared),
        "missing_cell_hours": int((prepared.observed == 0).sum()),
        "filled_input_hours": int(((prepared.observed == 0) & prepared.traffic.notna()).sum()),
        "unresolved_input_hours": int(prepared.traffic.isna().sum()),
        "eligible_main_cells": main.loc[main.exclusion_reason == "", "cell"].tolist(),
        "tuning_cells": panel.cell.tolist(), "tuning_masks_computable": panel_valid,
        "coverage": summary.to_dict("records"),
        "complete_observed_development_days": int(days.complete_observed_days.sum()),
        "eligible_busiest_hour_development_days": int(days.eligible_busiest_hour_days.sum()),
        "config_sha256": sha256(config_path),
        "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "PyYAML": yaml.__version__},
        "output_sha256": {name: sha256(output_dir/name) for name in outputs},
        "raw_unchanged": before == sha256(raw_path),
        "unknown_metadata": ["raw physical units", "timezone", "interval start/end convention",
          "reporting latency", "collection operator/location", "original release checksum", "dataset licence"]}
    if not report["raw_unchanged"]:
        raise RuntimeError("Raw source changed during validation")
    write_json(output_dir/"validation_report.json", json_safe(report))
    # Keep acquisition fields intact. A rerun must not replace them with placeholders.
    manifest = json.loads(Path(manifest_path).read_text())
    manifest["validation"] = {"source_sha256": before,
        "report_sha256": sha256(output_dir/"validation_report.json"),
        "historical_metadata_status": "Acquisition/source statements inherited; see current provenance review.",
        "current_units": "dataset traffic units; physical conversion unverified",
        "current_clock": "recorded clock; timezone and interval convention unspecified",
        "current_licence": "not independently verified", "raw_redistribution": False}
    destination = output_dir/"data_manifest.json"
    if destination.resolve() in [raw_path, Path(config_path).resolve(), Path(manifest_path).resolve()]:
        raise ValueError("Use a separate output directory; source metadata must remain unchanged")
    write_json(destination, manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="run_config.yaml")
    parser.add_argument("--manifest", default="data_manifest.json")
    args = parser.parse_args()
    report = run(args.raw_csv, args.output_dir, args.config, args.manifest)
    print(json.dumps({key: report[key] for key in ["integrity", "grid_rows", "missing_cell_hours",
          "filled_input_hours", "unresolved_input_hours", "tuning_cells", "tuning_masks_computable", "raw_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
