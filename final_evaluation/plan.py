"""Bind later evaluation to reviewed development choices and chronological data."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from tools.development_store import sha256, write_json_atomic, load_success, _read_json
from tools.run_development_experiments import environment_identity, read_config
from tools.run_cpu import CPU_ENV

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_SHA = "a3d6905021770195c8ff813c0bbbc4536ecc227371c03420f140e7fe086cb3b2"
ORIGINAL_PLAN_SHA = "2e8ac69fa8d596d34311b4e970ed4d87c2d2396e0e0124f32197a157018e63b2"
PERIODS = {"confirmation": (6132, 7446), "final": (7446, 8760)}
STANDALONES = ("persistence", "daily_seasonal_naive", "weekly_seasonal_naive",
               "ridge", "direct_lightgbm", "lstm")
BASELINES = STANDALONES[:3]
BLEND = "ridge_direct_lightgbm_blend"
RESIDUAL = "ridge_lightgbm_hybrid"
METHODS = (*STANDALONES[:5], RESIDUAL, BLEND, "lstm")
SEEDS = (2026, 2027, 2028)
FROZEN = {"ridge_alpha": 1000, "direct_trees": 128, "direct_leaves": 7,
          "residual_trees": 32, "residual_leaves": 31, "residual_weight": 1,
          "blend_ridge_weight": .2, "lstm_hidden": 32, "lstm_dropout": .2,
          "lstm_epochs": 23, "tree_objective": "regression"}

# Exact revisions to data/configuration verification in the public code.
# Each pair is (archived source hash, reviewed runtime source hash).
REVIEWED_SOURCE_REVISIONS = {
    "tools/check_prepared_data.py": (
        "74a1a4d624d92ba5b31ee782b2bbbb481766a865fe1435f218a9f4b5db2eb303",
        "bc74fafc0badacc80754c05bafd582684ba252c524032568d1d4c13b7560f437"),
    "tools/run_development_capacity.py": (
        "c117a517bcac2cf082e91344620fba424629894f776526f3f2ebd76f833182b7",
        "252ebd0cc631509416dc8b2cf8e4601037bc61e84bb59122af3215315c3c55a2"),
    "tools/run_development_experiments.py": (
        "61ded17a1e31e9c58f8f279ffbf55cb22793610b68e1d51b456ac77c67adeefc",
        "b3f8ecd1cf8ae8f40a7d2cdac4192ba9ed3965f4eaff4491460a0c9e7e87b8f2"),
    "tools/run_development_pilot.py": (
        "861c957bb4693b05881d8c00139a2a3be7e7adc5beee9a86e0494abfbca43657",
        "802948ee3997f001bc670481f61f4444ed38ba763e12410cad1e3d434d271ae9"),
}


def verify_development_code(archived_hashes):
    """Accept exact archived code or a reviewed path-and-hash pair only."""
    revisions = {}
    for name, archived in archived_hashes.items():
        current = sha256(ROOT / name)
        if current == archived:
            continue
        if REVIEWED_SOURCE_REVISIONS.get(name) != (archived, current):
            raise ValueError(f"Unreviewed development source change: {name}")
        revisions[name] = {"original_sha256": archived, "current_sha256": current}
    return revisions


def code_hashes():
    directories = ("tools", "models", "forecasting", "evaluation", "preprocessing",
                   "blend_revision", "final_evaluation")
    return {str(p.relative_to(ROOT)): sha256(p) for d in directories
            for p in sorted((ROOT / d).glob("*.py"))}


def jobs_for(period, cells):
    if period not in PERIODS or not cells or len(set(cells)) != len(cells):
        raise ValueError("Expected a declared period and unique nonempty cell panel")
    start, end = PERIODS[period]
    jobs = []
    for method in METHODS:
        for cell in cells:
            repetitions = 3 if method == "lstm" or (period == "final" and method not in BASELINES) else 1
            for repeat in range(1, repetitions + 1):
                jobs.append({"id": f"{method}--{cell}--r{repeat}", "method": method,
                    "cell": cell, "repeat": repeat, "seed": SEEDS[repeat-1] if method == "lstm" else None,
                    "fit_end": start, "score_start": start, "score_end": end,
                    "predict": repeat == 1 or method == "lstm", "period": period})
    return jobs


def development_evidence(original):
    """Read selected immutable outputs only; never change original results."""
    original = Path(original).absolute()
    if sha256(original / "development-summary.json") != SUMMARY_SHA:
        raise ValueError("Original audited development summary differs")
    if sha256(original / "plan.json") != ORIGINAL_PLAN_SHA:
        raise ValueError("Original audited development plan differs")
    summary, plan = _read_json(original / "development-summary.json"), _read_json(original / "plan.json")
    expected = {"ridge": "ridge-a1000", "direct": "direct-t128-l7",
                "hybrid": "hybrid-t32-l31-w1", "lstm": "lstm-h32-d0p2"}
    pins = {"development-summary.json": SUMMARY_SHA, "plan.json": ORIGINAL_PLAN_SHA}
    epochs = []
    for stage, cid in expected.items():
        selection = _read_json(original / f"selections/{stage}.json")
        if selection != summary["selections"][stage] or selection["selection"]["selected"]["candidate_id"] != cid:
            raise ValueError("Frozen settings differ from audited global development selections")
        pins[f"selections/{stage}.json"] = sha256(original / f"selections/{stage}.json")
        selected = [j for j in plan["jobs"] if j["stage"] == stage
                    and j["candidate_id"] == (cid[:-3] if stage == "hybrid" else cid)]
        if len(selected) != (81 if stage == "lstm" else 27):
            raise ValueError("Incomplete selected development cohort")
        for job in selected:
            saved = load_success(original, job["id"], ORIGINAL_PLAN_SHA)
            if saved is None:
                raise ValueError("Missing immutable selected development fit")
            attempt, marker = saved
            evidence = selection["evidence"][job["id"]]
            if (str(attempt.relative_to(original)) != evidence["attempt"]
                    or sha256(attempt / "complete.json") != evidence["completion_sha256"]):
                raise ValueError("Development selection evidence changed")
            result = _read_json(attempt / "result.json")
            if result["job"] != job or result["plan_sha256"] != ORIGINAL_PLAN_SHA:
                raise ValueError("Wrong selected-fit provenance")
            if stage == "lstm":
                epochs.append(result["training"]["selected_epochs"])
    import statistics
    if (math.ceil(statistics.median(epochs)) != 23
            or summary["selections"]["lstm"]["selection"]["fixed_epochs"] != 23):
        raise ValueError("Fixed LSTM epochs no longer match all 81 selected records")
    source_revisions = verify_development_code(plan["code_sha256"])
    return {"directory": str(original), "pins": pins, "selected_lstm_epoch_count": len(epochs),
            "fixed_epochs": 23, "environment_identity": plan["environment_identity"],
            "source_revisions": source_revisions}


def verify_development_pins(evidence):
    for name, digest in evidence["pins"].items():
        if sha256(Path(evidence["directory"]) / name) != digest:
            raise ValueError("Original development evidence changed")
    archived_plan = _read_json(Path(evidence["directory"]) / "plan.json")
    if evidence.get("source_revisions", {}) != verify_development_code(archived_plan["code_sha256"]):
        raise ValueError("Recorded development source revisions differ")


def host_check(machine):
    """Fail rather than guess if formal Mac power conditions are unavailable."""
    if machine.get("system") != "Darwin":
        raise ValueError("Formal execution requires the recorded Mac")
    power, profiles = machine.get("power_source", {}), machine.get("power_profiles", {})
    if power.get("returncode") != 0 or "AC Power" not in power.get("stdout", ""):
        raise ValueError("Connect the formal Mac to AC power")
    import re
    modes = re.findall(r"lowpowermode\s+(\d+)", profiles.get("stdout", ""))
    if profiles.get("returncode") != 0 or not modes or any(x != "0" for x in modes):
        raise ValueError("Low Power Mode must be verifiably disabled")


def load_series(directory, plan, cell):
    import numpy as np
    import pandas as pd
    from forecasting.data import _hourly_clock
    from preprocessing.windows import eligible_targets
    start, end = PERIODS[plan["period"]]
    entry = plan["series"][cell]
    path = Path(directory) / entry["path"]
    if entry["path"] != f"series/{cell}.npz" or sha256(path) != entry["sha256"]:
        raise ValueError("Evaluation series changed")
    with np.load(path, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files}
    if set(data) != {"traffic", "traffic_observed", "observed", "timestamps"}:
        raise ValueError("Unexpected evaluation arrays")
    if any(x.ndim != 1 or len(x) != end for x in data.values()):
        raise ValueError("Data must end at this period boundary")
    flags = data["observed"]
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("Invalid observation mask")
    data["observed"] = flags.astype(bool)
    flags = data["observed"]
    if not np.array_equal(np.isfinite(data["traffic_observed"]), flags):
        raise ValueError("Genuine observations and mask differ")
    np.testing.assert_array_equal(data["traffic"][flags], data["traffic_observed"][flags])
    data["timestamps"] = _hourly_clock(pd.DatetimeIndex(data["timestamps"]), end)
    targets = np.flatnonzero(eligible_targets(data["traffic"], flags)[start:end]) + start
    if targets.tolist() != entry["targets"] or len(targets) == 0:
        raise ValueError("Common target cohort differs")
    return data, targets


def verify_plan(directory, *, check_environment=True):
    directory = Path(directory)
    plan = _read_json(directory / "plan.json")
    checksum = sha256(directory / "plan.json")
    if _read_json(directory / "plan-checksum.json") != {"sha256": checksum}:
        raise ValueError("Sealed evaluation plan changed")
    config = read_config()
    if (plan["schema_version"] != 1 or plan["code_sha256"] != code_hashes()
            or plan["frozen"] != FROZEN or plan["cpu_startup_profile"] != CPU_ENV
            or plan["config_sha256"] != sha256(ROOT / "run_config.yaml")
            or len(plan["cells"]) != 57 or sorted(plan["cells"]) != plan["cells"]
            or set(plan["series"]) != set(plan["cells"])
            or plan["jobs"] != jobs_for(plan["period"], plan["cells"])):
        raise ValueError("Evaluation code, settings, panel or workload drift")
    verify_development_pins(plan["development_evidence"])
    if check_environment:
        identity, _ = environment_identity()
        if identity != plan["development_evidence"]["environment_identity"]:
            raise ValueError("Use the original formal Mac/interpreter/packages")
    if plan["period"] == "final":
        freeze = Path(plan["confirmation_freeze"]["path"])
        if sha256(freeze) != plan["confirmation_freeze"]["sha256"]:
            raise ValueError("Frozen confirmation comparator changed")
        sealed = _read_json(freeze)
        if (sealed["period"] != "confirmation" or sealed["status"] != "passed"
                or sealed["primary_model"] != BLEND or sealed["frozen"] != FROZEN
                or sealed["primary_comparator"] not in STANDALONES
                or sealed["primary_comparator"] != plan["primary_comparator"]):
            raise ValueError("Final comparator differs from confirmation")
    elif plan["primary_comparator"] is not None or plan["confirmation_freeze"] is not None:
        raise ValueError("Confirmation cannot preset its winning comparator")
    expected = {f"{cell}.npz" for cell in plan["cells"]}
    if {p.name for p in (directory / "series").iterdir()} != expected:
        raise ValueError("Missing or additional evaluation series")
    if (directory / "jobs").exists():
        known = {j["id"] for j in plan["jobs"]}
        if any(p.name not in known for p in (directory / "jobs").iterdir()):
            raise ValueError("Unexpected evaluation job")
    return plan, config, checksum


def prepare(data_directory, original_directory, directory, period, publication_commit, confirmation=None):
    """Export only the currently authorized horizon; never fit or forecast."""
    import numpy as np
    import re
    from preprocessing.windows import eligible_targets
    from tools.check_prepared_data import load_validated_data
    from tools.pilot_measurements import select_origins
    if period not in PERIODS or not re.fullmatch(r"[0-9a-f]{40}", publication_commit or ""):
        raise ValueError("Supply the published 40-character code commit and declared period")
    directory = Path(directory).absolute()
    if directory.exists():
        raise ValueError("Preparation needs a new directory; preserve earlier evidence")
    evidence = development_evidence(original_directory)
    identity, machine = environment_identity()
    if identity != evidence["environment_identity"]:
        raise ValueError("Use the original formal Mac/interpreter/packages for preparation")
    config = read_config()
    freeze_info, primary = None, None
    if period == "final":
        if confirmation is None:
            raise ValueError("Final preparation requires sealed confirmation results")
        from final_evaluation.run import summarize
        previous = summarize(Path(confirmation), seal=False)
        if (previous["period"] != "confirmation" or previous["status"] != "passed"
                or previous["primary_model"] != BLEND or previous["frozen"] != FROZEN
                or previous["primary_comparator"] not in STANDALONES):
            raise ValueError("Final preparation must follow this study's sealed confirmation")
        freeze_path = Path(confirmation).absolute() / "summary.json"
        if not freeze_path.exists() or _read_json(freeze_path) != previous:
            raise ValueError("Confirmation must be complete and sealed before final data export")
        primary = previous["primary_comparator"]
        freeze_info = {"path": str(freeze_path), "sha256": sha256(freeze_path)}
    elif confirmation is not None:
        raise ValueError("Confirmation preparation has no preceding holdout results")
    hourly, report = load_validated_data(data_directory)
    cells = sorted(report["eligible_main_cells"])
    if len(cells) != 57:
        raise ValueError("The fixed main panel must contain all 57 eligible cells")
    start, end = PERIODS[period]
    groups, series = {}, {}
    common = None
    for cell in cells:
        group = hourly.loc[hourly.cell == cell].iloc[:end]
        if len(group) != end:
            raise ValueError("Incomplete evaluation calendar")
        values, flags = group.traffic.to_numpy(), group.observed.to_numpy(dtype=bool)
        mask = eligible_targets(values, flags)
        targets = np.flatnonzero(mask[start:end]) + start
        if len(targets) == 0 or not mask[:start].any():
            raise ValueError("Resolve empty fixed-panel cohorts before running models")
        common = targets if common is None else np.intersect1d(common, targets)
        groups[cell] = group
        series[cell] = {"path": f"series/{cell}.npz", "targets": targets.tolist()}
    # All integrity and chronology checks precede creating a new output tree.
    directory.mkdir(parents=True)
    (directory / "series").mkdir()
    for cell, group in groups.items():
        path = directory / series[cell]["path"]
        np.savez(path, traffic=group.traffic.to_numpy(), traffic_observed=group.traffic_observed.to_numpy(),
                 observed=group.observed.to_numpy(dtype=bool), timestamps=group.ts.to_numpy())
        series[cell]["sha256"] = sha256(path)
    plan = {"schema_version": 1, "scope": "fixed original-loss blend and residual comparison",
            "period": period, "created_utc": datetime.now(timezone.utc).isoformat(),
            "publication_commit": publication_commit, "code_sha256": code_hashes(),
            "config_sha256": sha256(ROOT / "run_config.yaml"), "frozen": FROZEN,
            "cells": cells, "series": series, "jobs": jobs_for(period, cells),
            "common_origins": select_origins(common).tolist(), "cpu_startup_profile": CPU_ENV,
            "development_evidence": evidence, "primary_comparator": primary,
            "confirmation_freeze": freeze_info, "machine_at_preparation": machine,
            "prepared_data_directory": str(Path(data_directory).absolute()),
            "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
            "cost_scope": "formal complete selected-model cost on final only; confirmation prediction latency supports exact ties",
            "design_limits": ["Scope chosen after adaptive development; original-loss configurations only",
                              "Development-selected blend is not guaranteed to improve later accuracy",
                              "Secondary original residual hybrid cannot replace the frozen main blend",
                              "No weight selection on confirmation or final errors"]}
    write_json_atomic(directory / "plan.json", plan)
    write_json_atomic(directory / "plan-checksum.json", {"sha256": sha256(directory / "plan.json")})
    return plan
