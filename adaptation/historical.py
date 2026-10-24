"""Read-only checks for the pinned development archives.

An archived source hash identifies the code that produced a result. A current
hash identifies the code reading it. Only the exact reviewed pairs in the
catalog may differ; these checks never authorise resuming an archived run.
"""
from pathlib import Path

from adaptation.plan import PLAN_SHA256, SUMMARY_SHA256
from tools.development_store import _read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/source-compatibility.json"
L1_PLAN_SHA256 = "b5ce3587d3859627215320654660ed35cbbd765dfec2ec179f0ff88a3c2ade13"


def catalog():
    sha256(CATALOG)
    value = _read_json(CATALOG)
    if value["schema_version"] != 1:
        raise ValueError("Unknown source compatibility catalog")
    return value


def catalog_sha256():
    return sha256(CATALOG)


def verify_source_hashes(archived, current, revisions, additions=None):
    """Require a complete inventory and exact path, old-hash, new-hash pairs."""
    additions = additions or {}
    if set(archived) & set(additions) or set(current) != set(archived) | set(additions):
        raise ValueError("Development source inventory changed")
    applied = {}
    for name, old in archived.items():
        new = current[name]
        if old == new:
            continue
        if revisions.get(name) != {"original_sha256": old, "current_sha256": new}:
            raise ValueError(f"Unreviewed development source change: {name}")
        applied[name] = {"original_sha256": old, "current_sha256": new}
    for name, expected in additions.items():
        if current[name] != expected:
            raise ValueError(f"Unreviewed added archive verifier: {name}")
    return applied


def verify_development_code(archived, current=None):
    if current is None:
        from tools.run_development_pilot import code_hashes
        current = code_hashes()
    return verify_source_hashes(archived, current, catalog()["development_revisions"])


def verify_archived_config(archived):
    expected = catalog()["config_revision"]
    if expected != {"original_sha256": archived,
                    "current_sha256": sha256(ROOT / "run_config.yaml")}:
        raise ValueError("Unreviewed archived/current configuration pair")


def verify_l1_package(archived, current):
    value = catalog()
    return verify_source_hashes(archived, current, value["l1_package_revisions"],
                                value["l1_package_additions"])


def verify_original_plan(directory):
    """Check archived provenance and all data cohorts without a live-host check.

    This is for reading saved evidence only. The execution verifier in
    tools.run_development_experiments remains strict about current code/host.
    """
    from forecasting.development_plan import STAGES, TUNING_CELLS, build_jobs
    from tools import run_development_experiments as original
    from tools.run_cpu import CPU_ENV
    directory = Path(directory)
    checksum = sha256(directory / "plan.json")
    if (checksum != PLAN_SHA256
            or sha256(directory / "development-summary.json") != SUMMARY_SHA256
            or _read_json(directory / "plan-checksum.json") != {"sha256": checksum}):
        raise ValueError("Original archived plan or summary differs from the reviewed run")
    plan = _read_json(directory / "plan.json")
    config = original.read_config()
    verify_archived_config(plan["config_sha256"])
    verify_development_code(plan["code_sha256"])
    checklist = {"original_sha256": plan["prepared_checklist_sha256"],
                 "current_sha256": sha256(ROOT / "docs/data-bundle-checksums.json")}
    if checklist != catalog()["checklist_revision"]:
        raise ValueError("Unreviewed archived/current data checklist pair")
    if (plan["schema_version"] != 1 or plan["jobs"] != original.all_jobs(config)
            or set(plan["series"]) != set(TUNING_CELLS)
            or plan["source_sha256"] != config["data"]["raw_sha256"]
            or plan["stage_counts"] != {s: len(build_jobs(config, s)) for s in STAGES}
            or plan["cpu_startup_profile"] != CPU_ENV
            or plan["holdout_forecasts_enabled"] is not False
            or plan["formal_comparative_timing"] is not False):
        raise ValueError("Archived plan differs from the fixed development workload")
    if {p.name for p in (directory / "series").iterdir()} != {f"{c}.npz" for c in TUNING_CELLS}:
        raise ValueError("Unexpected development series files")
    for cell in TUNING_CELLS:
        if plan["series"][cell]["path"] != f"series/{cell}.npz":
            raise ValueError("Unexpected series path")
        data = original.load_series(directory, plan, cell)
        for job in build_jobs(config, "ridge"):
            if job["cell"] == cell and job["parameters"]["alpha"] == 0.1:
                if original.targets_for(data, job).tolist() != plan["series"][cell]["folds"][job["fold_index"]]["targets"]:
                    raise ValueError("Shared target cohort changed")
    known = {job["id"] for job in plan["jobs"]}
    if (directory / "jobs").exists() and any(p.name not in known or p.is_symlink() for p in (directory / "jobs").iterdir()):
        raise ValueError("Unexpected archived job directory")
    return plan, config, checksum
