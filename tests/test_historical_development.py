"""Archived provenance versus fresh execution, using invented disk evidence.

No saved model is loaded, fitted or used for inference in these checks.
"""
from copy import deepcopy

import pytest

from adaptation import historical
from adaptation import run_ridge_extension as ridge
from l1_revision import run_experiment as l1
from tools import run_development_experiments as original
from tools.development_store import _inventory, _read_json, sha256
from tools.run_development_pilot import code_hashes
from tests.test_development_experiments import (
    config, prepared as current_development, put_json, reseal_plan,
)


def archived_hashes(current, revisions, additions=()):
    result = {name: digest for name, digest in current.items() if name not in additions}
    for name, pair in revisions.items():
        result[name] = pair["original_sha256"]
    return result


@pytest.fixture
def original_archive(current_development, monkeypatch):
    directory, plan, _ = current_development
    plan = deepcopy(plan)
    mapping = historical.catalog()
    plan["config_sha256"] = mapping["config_revision"]["original_sha256"]
    plan["prepared_checklist_sha256"] = mapping["checklist_revision"]["original_sha256"]
    plan["code_sha256"] = archived_hashes(plan["code_sha256"], mapping["development_revisions"])
    reseal_plan(directory, plan)
    summary = {"selections": {
        stage: {"selection": {"selected": {"candidate_id": candidate}}}
        for stage, candidate in (("ridge", "ridge-a1000"), ("direct", "direct-t128-l7"))}}
    put_json(directory / "development-summary.json", summary)
    for module in (historical, ridge, l1):
        monkeypatch.setattr(module, "PLAN_SHA256", sha256(directory / "plan.json"))
        monkeypatch.setattr(module, "SUMMARY_SHA256", sha256(directory / "development-summary.json"))
    return directory, plan


def test_archive_verification_reads_data_without_host_query_or_writes(original_archive, monkeypatch):
    directory, plan = original_archive
    before = _inventory(directory)
    monkeypatch.setattr(original, "environment_identity", lambda: pytest.fail("Archive queried live host"))
    actual, _, checksum = historical.verify_original_plan(directory)
    assert actual == plan and checksum == sha256(directory / "plan.json")
    assert _inventory(directory) == before
    # The same archive is deliberately ineligible for the current-run resumer.
    with pytest.raises(ValueError, match="fixed code, data or development workload"):
        original.verify_plan(directory)


@pytest.mark.parametrize("damage", ["plan", "summary", "series"])
def test_archive_rejects_changed_evidence(original_archive, damage):
    directory, plan = original_archive
    path = {"plan": directory / "plan.json", "summary": directory / "development-summary.json",
            "series": directory / next(iter(plan["series"].values()))["path"]}[damage]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        historical.verify_original_plan(directory)


@pytest.mark.parametrize("damage", ["jobs", "series", "holdout", "config", "checklist", "code"])
def test_archive_checks_scientific_and_source_contracts_after_pin(original_archive, monkeypatch, damage):
    directory, plan = original_archive
    plan = deepcopy(plan)
    if damage == "jobs": plan["jobs"].pop()
    elif damage == "series": next(iter(plan["series"].values()))["folds"][0]["targets"].pop()
    elif damage == "holdout": plan["holdout_forecasts_enabled"] = True
    elif damage == "config": plan["config_sha256"] = original.CONFIG_SHA256
    elif damage == "checklist": plan["prepared_checklist_sha256"] = "0" * 64
    else: plan["code_sha256"]["models/ridge.py"] = "0" * 64
    reseal_plan(directory, plan)
    # Exercise the contract below the exact-pin gate with invented evidence.
    monkeypatch.setattr(historical, "PLAN_SHA256", sha256(directory / "plan.json"))
    with pytest.raises(ValueError):
        historical.verify_original_plan(directory)


@pytest.mark.parametrize("damage", ["old", "current", "path", "missing", "extra", "addition"])
def test_only_exact_source_pairs_and_explicit_additions_are_allowed(damage):
    archived = {"changed.py": "a" * 64, "same.py": "b" * 64}
    current = {"changed.py": "c" * 64, "same.py": "b" * 64, "reader.py": "d" * 64}
    revisions = {"changed.py": {"original_sha256": "a" * 64, "current_sha256": "c" * 64}}
    additions = {"reader.py": "d" * 64}
    assert historical.verify_source_hashes(archived, current, revisions, additions) == revisions
    if damage == "old": archived["changed.py"] = "0" * 64
    elif damage == "current": current["changed.py"] = "0" * 64
    elif damage == "path": revisions["other.py"] = revisions.pop("changed.py")
    elif damage == "missing": current.pop("same.py")
    elif damage == "extra": current["unexpected.py"] = "e" * 64
    else: current["reader.py"] = "0" * 64
    with pytest.raises(ValueError):
        historical.verify_source_hashes(archived, current, revisions, additions)


@pytest.fixture
def prepared_l1(original_archive, monkeypatch, config):
    source, initial = original_archive
    before = _inventory(source)
    # Fitting and completed-result replay are outside this plan-compatibility test.
    monkeypatch.setattr(l1, "audit_original", lambda _: (initial, config, {}, before))
    monkeypatch.setattr(l1, "historical_blend", lambda *args: {"invented": True})
    directory = source.parent / "new-l1"
    plan = l1.prepare(source, directory)
    assert _inventory(source) == before
    return directory, plan


def test_new_l1_plan_preserves_old_sources_and_freezes_current_runtime(prepared_l1):
    directory, plan = prepared_l1
    actual, _, checksum = l1.verify_plan(directory)
    assert actual == plan and checksum == sha256(directory / "plan.json")
    assert plan["runtime_development_code_sha256"] == code_hashes()
    assert plan["original_code_sha256"] != plan["runtime_development_code_sha256"]
    assert plan["source_compatibility_sha256"] == historical.catalog_sha256()
    assert not (directory / "jobs").exists()
    with pytest.raises(ValueError, match="not the reviewed historical plan"):
        l1.verify_archived_plan(directory)


@pytest.mark.parametrize("field", ["runtime_development_code_sha256", "source_compatibility_sha256",
                                  "revision_and_dependency_code_sha256", "original_code_sha256"])
def test_new_l1_rejects_resealed_runtime_drift(prepared_l1, field):
    directory, plan = prepared_l1
    plan = deepcopy(plan)
    plan[field] = "0" * 64 if field == "source_compatibility_sha256" else {}
    reseal_plan(directory, plan)
    with pytest.raises(ValueError):
        l1.verify_plan(directory)


def test_new_l1_execution_keeps_original_host_guard(prepared_l1, monkeypatch):
    directory, plan = prepared_l1
    monkeypatch.setattr(original, "environment_identity", lambda: ({"different_host": True}, {}))
    with pytest.raises(ValueError, match="original machine"):
        l1.verify_current_identity(directory, plan, sha256(directory / "plan.json"))


def test_new_l1_preparation_rejects_other_host_before_creating_output(original_archive, monkeypatch, config):
    source, initial = original_archive
    monkeypatch.setattr(l1, "audit_original", lambda _: (initial, config, {}, _inventory(source)))
    monkeypatch.setattr(l1, "historical_blend", lambda *args: {})
    monkeypatch.setattr(original, "environment_identity", lambda: ({"different_host": True}, {}))
    destination = source.parent / "wrong-host"
    with pytest.raises(ValueError, match="original host"):
        l1.prepare(source, destination)
    assert not destination.exists()


@pytest.fixture
def archived_l1(prepared_l1, monkeypatch):
    directory, plan = prepared_l1
    plan = deepcopy(plan)
    mapping = historical.catalog()
    plan["config_sha256"] = mapping["config_revision"]["original_sha256"]
    plan["revision_and_dependency_code_sha256"] = archived_hashes(
        plan["revision_and_dependency_code_sha256"], mapping["l1_package_revisions"],
        mapping["l1_package_additions"])
    del plan["runtime_development_code_sha256"], plan["source_compatibility_sha256"]
    reseal_plan(directory, plan)
    monkeypatch.setattr(historical, "L1_PLAN_SHA256", sha256(directory / "plan.json"))
    return directory, plan


def test_l1_archive_is_read_only_and_cannot_resume(archived_l1, monkeypatch):
    directory, plan = archived_l1
    before = _inventory(directory)
    monkeypatch.setattr(original, "environment_identity", lambda: pytest.fail("Archive queried live host"))
    actual, _, _ = l1.verify_archived_plan(directory)
    assert actual == plan and _inventory(directory) == before
    with pytest.raises(ValueError, match="current code or configuration"):
        l1.verify_plan(directory)


@pytest.mark.parametrize("field", ["config_sha256", "revision_and_dependency_code_sha256",
                                  "environment_identity", "jobs"])
def test_l1_archive_checks_contract_below_exact_pin(archived_l1, monkeypatch, field):
    directory, plan = archived_l1
    plan = deepcopy(plan)
    plan[field] = original.CONFIG_SHA256 if field == "config_sha256" else {}
    reseal_plan(directory, plan)
    monkeypatch.setattr(historical, "L1_PLAN_SHA256", sha256(directory / "plan.json"))
    with pytest.raises(ValueError):
        l1.verify_archived_plan(directory)


def test_l1_archive_rejects_unknown_resealed_plan(archived_l1):
    directory, plan = archived_l1
    plan["created_at_utc"] = "different"
    reseal_plan(directory, plan)
    with pytest.raises(ValueError, match="not the reviewed historical plan"):
        l1.verify_archived_plan(directory)
