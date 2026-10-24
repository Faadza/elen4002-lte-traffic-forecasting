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
STUDY_ID = "independent_l1_l2_2026_10_09"
STANDALONES = ("persistence", "daily_seasonal_naive", "weekly_seasonal_naive",
               "ridge", "lightgbm_l2", "lightgbm_l1", "lstm")
BASELINES = STANDALONES[:3]
SEEDS = (2026, 2027, 2028)
FIXED = {"ridge_alpha": 1000, "lstm_hidden": 32, "lstm_dropout": .2,
         "lstm_epochs": 23, "seeds": list(SEEDS)}
WEIGHTS = (0, .05, .1, .2, .3, .5, .75, 1)


def code_hashes():
    from ensemble_study.common import code_hashes as current_hashes
    return current_hashes()


def condition_for(objective, weight, capacity):
    """Canonical endpoints are shared with the standalone panel, never hybrids."""
    from ensemble_study.selection import condition_for_weight
    return condition_for_weight(objective, **capacity, ridge_weight=weight)


def expected_conditions(components, weights):
    conditions = [{"id": name, "kind": "baseline", "objective": None,
                   "trees": None, "leaves": None, "ridge_weight": None} for name in BASELINES]
    conditions.append(condition_for("l2", 1, components["l2"]))
    conditions.extend(condition_for(o, 0, components[o]) for o in ("l2", "l1"))
    conditions.append({"id": "lstm", "kind": "lstm", "objective": None,
                       "trees": None, "leaves": None, "ridge_weight": None})
    conditions.extend(condition_for(o, .5, components[o]) for o in ("l2", "l1"))
    for objective in ("l2", "l1"):
        candidate = condition_for(objective, weights[objective], components[objective])
        if candidate not in conditions:
            conditions.append(candidate)
    return conditions


def validate_selection(selection):
    """Reject old PR20 defaults, incomplete grids, renamed endpoints or panel drift."""
    from numbers import Integral, Real
    if (not isinstance(selection, dict) or selection.get("schema_version") != 1
            or selection.get("status") != "passed" or selection.get("study_id") != STUDY_ID
            or selection.get("fixed") != FIXED):
        raise ValueError("Expected a passed development selection for this complete study")
    components, weights = selection.get("selected_components"), selection.get("selected_weights")
    if not isinstance(components, dict) or set(components) != {"l1", "l2"}:
        raise ValueError("Both objective component selections are required")
    if not isinstance(weights, dict) or set(weights) != {"l1", "l2"}:
        raise ValueError("Both objective weight selections are required")
    for objective in ("l2", "l1"):
        capacity = components[objective]
        if (not isinstance(capacity, dict) or set(capacity) != {"trees", "leaves"}
                or any(isinstance(x, bool) or not isinstance(x, Integral) for x in capacity.values())
                or capacity["trees"] not in (32, 128, 512) or capacity["leaves"] not in (7, 31)):
            raise ValueError("Selected capacity is outside the matched six-capacity grid")
        weight = weights[objective]
        if isinstance(weight, bool) or not isinstance(weight, Real) or weight not in WEIGHTS:
            raise ValueError("Selected weight is outside the exact declared grid")
    expected = expected_conditions(components, weights)
    if selection.get("conditions") != expected:
        raise ValueError("Frozen conditions must contain all standalones, equal mixtures and selected weights once")
    from ensemble_study.selection import validate_condition
    for condition in selection["conditions"]:
        validate_condition(condition)
    choices = {condition_for(o, weights[o], components[o])["id"] for o in ("l2", "l1")}
    if selection.get("primary_candidate") not in choices:
        raise ValueError("The frozen primary must be one of the development-selected objective candidates")
    import re
    if not re.fullmatch(r"[0-9a-f]{64}", selection.get("protocol_sha256", "")):
        raise ValueError("Missing protocol fingerprint")
    if not isinstance(selection.get("evidence"), dict) or not selection["evidence"]:
        raise ValueError("Selection must retain its development evidence")
    return selection


def read_selection(path, digest):
    from ensemble_study.common import protocol
    protocol()
    if sha256(path) != digest:
        raise ValueError("Frozen selection bytes differ from the published fingerprint")
    selection = validate_selection(_read_json(path))
    if selection["protocol_sha256"] != sha256(ROOT / "ensemble_study/protocol.json"):
        raise ValueError("Selection and active study protocol differ")
    return selection


def jobs_for(period, cells, conditions):
    if period not in PERIODS or not cells or len(set(cells)) != len(cells):
        raise ValueError("Expected a declared period and unique nonempty cell panel")
    start, end = PERIODS[period]
    jobs = []
    if (not conditions or len({c["id"] for c in conditions}) != len(conditions)
            or any(not isinstance(c["id"], str) or not c["id"].replace("_", "").isalnum() for c in conditions)):
        raise ValueError("Expected unique canonical condition IDs")
    from ensemble_study.selection import validate_condition
    for condition in conditions:
        validate_condition(condition)
        method = condition["id"]
        for cell in cells:
            repetitions = 3 if method == "lstm" or (period == "final" and method not in BASELINES) else 1
            for repeat in range(1, repetitions + 1):
                jobs.append({"id": f"{method}--{cell}--r{repeat}", "method": method,
                    "cell": cell, "repeat": repeat, "seed": SEEDS[repeat-1] if method == "lstm" else None,
                    "fit_end": start, "score_start": start, "score_end": end,
                    "predict": repeat == 1 or method == "lstm", "period": period,
                    "condition": condition.copy()})
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
    for name, digest in plan["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Original numerical implementation has changed")
    return {"directory": str(original), "pins": pins, "selected_lstm_epoch_count": len(epochs),
            "fixed_epochs": 23, "environment_identity": plan["environment_identity"]}


def verify_development_pins(evidence):
    for name, digest in evidence["pins"].items():
        if sha256(Path(evidence["directory"]) / name) != digest:
            raise ValueError("Original development evidence changed")


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


def cost_plan_projection(plan, plan_sha256):
    """Separate immutable cost plan; shares final workers and adds no fits.

The cost plan references the accuracy plan's digest. The accuracy plan does not
reference this projection, so their provenance has no circular digest.
    """
    if plan["period"] != "final":
        raise ValueError("Formal repeated cost belongs only to the final period")
    return {"schema_version": 1, "study_id": STUDY_ID, "period": "final",
        "accuracy_plan_sha256": plan_sha256,
        "selection_sha256": plan["selection_sha256"],
        "conditions": plan["selection"]["conditions"], "cells": plan["cells"],
        "jobs": [{"id": j["id"], "method": j["method"], "cell": j["cell"],
                  "repeat": j["repeat"], "seed": j["seed"],
                  "learned_fit": j["method"] not in BASELINES,
                  "measure_prediction": j["predict"],
                  "accuracy_candidate": j["predict"],
                  "artifact_representative": j["repeat"] == 1}
                 for j in plan["jobs"]],
        "common_origins": plan["common_origins"],
        "conventions": {
            "execution": "same sequential fresh fit/predict workers as final accuracy; no duplicate cost-only study",
            "deterministic_fit_repetitions": 3,
            "deterministic_repeat_identity": "repeats 2/3 must match repeat 1 model fingerprint",
            "deterministic_accuracy_and_prediction": "repeat 1 only",
            "lstm_fit_observations": list(SEEDS),
            "lstm_prediction": "three separate seed models; never an averaged-prediction ensemble",
            "artifact_representative": "repeat 1 or LSTM seed 2026; one complete deployed system per cell",
            "fit_boundary": "loaded raw arrays; include features, scaler and all fresh components; exclude imports and files",
            "prediction_boundary": "loaded models; include features, scaling, every component, inversion and final clipping",
            "batch_warmup_passes": 1, "batch_timed_passes": 3,
            "single_origin_warmup_passes": 1, "single_origin_timed_passes": 3,
            "single_origin_summary": "median and empirical p95 over identical all-cell origins; unavailable if no common origins",
            "memory": "separate fresh-worker peak process RSS for fit/predict; includes dependencies/data, not incremental model memory",
            "search_cost": "excluded; historical search and bounded L1 completion reported separately",
            "baseline_fitting": "not applicable, not a measured zero-duration fit",
        }}


def verify_cost_plan(directory, plan, plan_sha256):
    path = Path(directory) / "cost-plan.json"
    if plan["period"] == "final":
        if _read_json(path) != cost_plan_projection(plan, plan_sha256):
            raise ValueError("Immutable final cost plan differs from the frozen accuracy workload")
    elif path.exists():
        raise ValueError("Confirmation cannot claim a formal repeated final-cost plan")


def verify_plan(directory, *, check_environment=True):
    directory = Path(directory)
    plan = _read_json(directory / "plan.json")
    checksum = sha256(directory / "plan.json")
    if _read_json(directory / "plan-checksum.json") != {"sha256": checksum}:
        raise ValueError("Sealed evaluation plan changed")
    config = read_config()
    selection = read_selection(plan["selection_file"], plan["selection_sha256"])
    verify_publication(plan["publication_record"]["path"], selection,
                       plan["selection_sha256"], plan["publication_commit"])
    if (plan["schema_version"] != 1 or plan["code_sha256"] != code_hashes()
            or plan["study_id"] != STUDY_ID or plan["selection"] != selection
            or plan["publication_record"]["sha256"] != sha256(plan["publication_record"]["path"])
            or plan["cpu_startup_profile"] != CPU_ENV
            or plan["config_sha256"] != sha256(ROOT / "run_config.yaml")
            or plan["prepared_checklist_sha256"] != sha256(ROOT / "docs/data-bundle-checksums.json")
            or len(plan["cells"]) != 57 or sorted(plan["cells"]) != plan["cells"]
            or set(plan["series"]) != set(plan["cells"])
            or plan["jobs"] != jobs_for(plan["period"], plan["cells"], selection["conditions"])
            or plan["primary_model"] != selection["primary_candidate"]):
        raise ValueError("Evaluation code, settings, panel or workload drift")
    verify_cost_plan(directory, plan, checksum)
    verify_development_pins(plan["development_evidence"])
    from ensemble_study.common import verify_saved_readiness
    verify_saved_readiness(plan["readiness_at_preparation"], plan["development_evidence"]["environment_identity"])
    if plan["readiness_at_preparation"]["operator"] != plan["operator"]:
        raise ValueError("Preparation operator differs from the sealed plan")
    import numpy as np
    from tools.pilot_measurements import select_origins
    common = None
    for cell in plan["cells"]:
        targets = np.asarray(plan["series"][cell]["targets"], dtype=np.int64)
        start, end = PERIODS[plan["period"]]
        if not len(targets) or np.any(targets < start) or np.any(targets >= end) or np.any(np.diff(targets) <= 0):
            raise ValueError("Invalid fixed target cohort")
        common = targets if common is None else np.intersect1d(common, targets)
    if plan["common_origins"] != select_origins(common).tolist():
        raise ValueError("Latency origins differ from the complete fixed-panel mask")
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
                or sealed["study_id"] != STUDY_ID
                or sealed["selection_sha256"] != plan["selection_sha256"]
                or sealed["primary_model"] != selection["primary_candidate"]
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


def verify_publication(path, selection, selection_sha256, publication_commit):
    """Check the recorded remote-byte audit; a digest alone is not authentication."""
    from ensemble_study.common import verify_publication as verify_receipt
    receipt = verify_receipt(path, scope="evaluation", selection_sha256=selection_sha256,
                             commit=publication_commit)
    if receipt["protocol_sha256"] != selection["protocol_sha256"]:
        raise ValueError("Published-byte verification record differs from frozen selection")
    return receipt


def prepare(data_directory, original_directory, directory, period, publication_commit,
            selection_file, selection_sha256, publication_record, confirmation=None,
            *, operator="automated", host_ready=False):
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
    selection_file, publication_record = Path(selection_file).absolute(), Path(publication_record).absolute()
    selection = read_selection(selection_file, selection_sha256)
    verify_publication(publication_record, selection, selection_sha256, publication_commit)
    evidence = development_evidence(original_directory)
    from ensemble_study.common import native_readiness
    readiness = native_readiness(evidence["environment_identity"], operator=operator, host_ready=host_ready)
    config = read_config()
    freeze_info, primary = None, None
    if period == "final":
        if confirmation is None:
            raise ValueError("Final preparation requires sealed confirmation results")
        from ensemble_study.evaluation_run import summarize
        previous = summarize(Path(confirmation), seal=False)
        if (previous["period"] != "confirmation" or previous["status"] != "passed"
                or previous["study_id"] != STUDY_ID
                or previous["selection_sha256"] != selection_sha256
                or previous["primary_model"] != selection["primary_candidate"]
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
    plan = {"schema_version": 1, "study_id": STUDY_ID,
            "scope": "frozen matched L1/L2 independent ensembles and seven standalones",
            "period": period, "created_utc": datetime.now(timezone.utc).isoformat(),
            "publication_commit": publication_commit, "code_sha256": code_hashes(),
            "publication_record": {"path": str(publication_record), "sha256": sha256(publication_record)},
            "selection_file": str(selection_file), "selection_sha256": selection_sha256,
            "selection": selection, "primary_model": selection["primary_candidate"],
            "config_sha256": sha256(ROOT / "run_config.yaml"),
            "cells": cells, "series": series, "jobs": jobs_for(period, cells, selection["conditions"]),
            "common_origins": select_origins(common).tolist(), "cpu_startup_profile": CPU_ENV,
            "development_evidence": evidence, "primary_comparator": primary,
            "confirmation_freeze": freeze_info, "readiness_at_preparation": readiness,
            "operator": operator, "host_ready_assertion": host_ready,
            "prepared_data_directory": str(Path(data_directory).absolute()),
            "prepared_checklist_sha256": sha256(ROOT / "docs/data-bundle-checksums.json"),
            "cost_scope": "formal complete selected-model cost on final only; confirmation prediction latency supports exact ties",
            "design_limits": ["Protocol amendment made after adaptive development",
                              "Primary condition selected only on development; an endpoint is not a successful mixture",
                              "All fixed equal-weight and selected-weight conditions reported; no holdout redesign",
                              "Confirmation selects only the standalone comparator; final does not select a new primary",
                              "Runtime is not an energy measurement or LTE packet latency",
                              "Conditional calendar-block intervals over a short final period; no guaranteed coverage"]}
    write_json_atomic(directory / "plan.json", plan)
    plan_sha256 = sha256(directory / "plan.json")
    write_json_atomic(directory / "plan-checksum.json", {"sha256": plan_sha256})
    if period == "final":
        write_json_atomic(directory / "cost-plan.json", cost_plan_projection(plan, plan_sha256))
    return plan
