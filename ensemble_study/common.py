"""Immutable configuration, publication receipts and truthful native readiness."""
from datetime import datetime, timezone
import math
from pathlib import Path
import re

from tools.development_store import _read_json, sha256
from tools.run_development_experiments import environment_identity
from final_evaluation.plan import host_check

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ensemble_study/protocol.json"
STUDY_ID = "independent_l1_l2_2026_10_09"
FIXED = {"ridge_alpha": 1000, "lstm_hidden": 32, "lstm_dropout": .2,
         "lstm_epochs": 23, "seeds": [2026, 2027, 2028]}
CAPACITIES = ((32, 7), (32, 31), (128, 7), (128, 31), (512, 7), (512, 31))
MISSING = ((32, 7), (128, 31), (512, 7), (512, 31))
WEIGHTS = (0, .05, .1, .2, .3, .5, .75, 1)


def protocol():
    value = _read_json(PROTOCOL)
    if (value["study_id"] != STUDY_ID or value["fixed"] != FIXED
            or value["capacities"] != [list(x) for x in CAPACITIES]
            or value["missing_l1_capacities"] != [list(x) for x in MISSING]
            or value["ridge_weights"] != list(WEIGHTS)):
        raise ValueError("Approved protocol parameters changed")
    return value


def code_hashes(scope="evaluation"):
    if scope not in ("development", "evaluation"):
        raise ValueError("Unknown source scope")
    directories = ("tools", "models", "forecasting", "evaluation", "preprocessing",
                   "adaptation", "blend_revision", "l1_revision", "final_evaluation", "ensemble_study")
    excluded = {"evaluation_plan.py", "evaluation_run.py", "measurement.py", "report.py"}
    hashes = {str(p.relative_to(ROOT)): sha256(p) for name in directories
            for p in sorted((ROOT/name).glob("*.py"))
            if not (scope == "development" and name == "ensemble_study" and p.name in excluded)}
    handoff = ROOT / "tools/run_ensemble_development_mac.sh"
    if handoff.exists():
        hashes[str(handoff.relative_to(ROOT))] = sha256(handoff)
    hashes["docs/source-compatibility.json"] = sha256(ROOT / "docs/source-compatibility.json")
    return hashes


def verify_publication(path, *, scope, selection_sha256=None, commit=None):
    value = _read_json(path)
    if (value.get("verified") is not True
            or not re.fullmatch(r"[0-9a-f]{40}", value.get("commit", ""))
            or (commit is not None and value["commit"] != commit)
            or value.get("selection_sha256") != selection_sha256
            or value.get("protocol_sha256") != sha256(PROTOCOL)
            or value.get("code_sha256") != code_hashes(scope)):
        raise ValueError("Published-byte verification receipt does not match this execution")
    return value


def verify_saved_readiness(snapshot, expected_identity):
    if (snapshot.get("status") != "passed" or snapshot.get("identity") != expected_identity
            or snapshot.get("operator") not in ("automated", "user")
            or (snapshot["operator"] == "user" and snapshot.get("host_ready_assertion") is not True)
            or not isinstance(snapshot.get("checked_at_utc"), str)):
        raise ValueError("Native host identity or truthful operator readiness is missing")
    machine = snapshot.get("machine", {})
    host_check(machine)
    if (machine.get("cpu_model", {}).get("returncode") != 0
            or machine["cpu_model"].get("stdout") != expected_identity.get("cpu_model")):
        raise ValueError("Native CPU identity was not read successfully")
    background = snapshot.get("background", {})
    samples = background.get("samples_percent", [])
    if (background.get("method") != "two 1-second totalCPU samples"
            or background.get("maximum_allowed_percent") != 20.0
            or background.get("passed") is not True or len(samples) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) or not 0 <= x <= 20 for x in samples)):
        raise ValueError("Background workload exceeds the declared readiness check or is unavailable")
    return snapshot


def native_readiness(expected_identity, operator="automated", host_ready=False):
    """Sample host conditions outside timers; never impersonate user confirmation.

    The two total-CPU samples are an operational screen, not proof of exclusive
    cores or constant frequency. The test applies equally before/after workers.
    """
    import psutil
    identity, machine = environment_identity()
    if identity != expected_identity:
        raise ValueError("Native Mac/interpreter/packages differ or a required host query is restricted")
    host_check(machine)
    samples = [psutil.cpu_percent(interval=1) for _ in range(2)]
    value = {"status": "passed", "operator": operator,
             "checked_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
             "machine": machine, "host_ready_assertion": bool(host_ready),
             "background": {"method": "two 1-second totalCPU samples", "samples_percent": samples,
                            "maximum_allowed_percent": 20.0, "passed": all(x <= 20 for x in samples)},
             "exclusive_cores_or_fixed_frequency_established": False}
    return verify_saved_readiness(value, expected_identity)
