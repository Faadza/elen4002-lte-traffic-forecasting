"""Audit resume, crash preservation and inherited locking using invented files."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import development_store as store


PLAN = "a" * 64
JOB = {"id": "ridge-cell-1-fold-0-alpha-0.1", "method": "ridge", "alpha": 0.1}


def new_attempt(tmp_path):
    path, cached = store.begin_attempt(tmp_path, JOB, PLAN)
    assert not cached
    (path / "worker.log").write_text("invented worker output\n")
    store.write_json_atomic(path / "result.json", {"status": "passed", "value": 3.25})
    return path


def complete(tmp_path):
    path = new_attempt(tmp_path)
    marker = store.complete_attempt(path, JOB, PLAN, {"checked": True})
    return path, marker


def test_new_attempt_is_bound_to_full_job_plan_and_start_time(tmp_path):
    path = new_attempt(tmp_path)
    data = json.loads((path / "attempt.json").read_text())
    assert path.relative_to(tmp_path).as_posix() == f"jobs/{JOB['id']}/attempt-0001"
    assert data["job"] == JOB
    assert data["plan_sha256"] == PLAN
    assert data["attempt_number"] == 1
    assert data["started_utc"].endswith("+00:00")
    assert store.load_success(tmp_path, JOB["id"], PLAN) is None


def test_success_is_cached_only_with_explicit_resume_even_if_retry_requested(tmp_path):
    path, marker = complete(tmp_path)
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    with pytest.raises(ValueError, match="already succeeded"):
        store.begin_attempt(tmp_path, JOB, PLAN)
    assert store.begin_attempt(tmp_path, JOB, PLAN, resume=True) == (path, True)
    assert store.begin_attempt(tmp_path, JOB, PLAN, resume=True, retry_failed=True) == (path, True)
    assert store.load_success(tmp_path, JOB["id"], PLAN) == (path, marker)
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    assert len(list(path.parent.iterdir())) == 1


@pytest.mark.parametrize("failed", [False, True])
def test_interrupted_or_failed_attempt_requires_explicit_retry_and_preserves_bytes(tmp_path, failed):
    first = new_attempt(tmp_path)
    if failed:
        store.write_json_atomic(first / "failure.json", {"status": "failed", "reason": "invented"})
    previous = {p.name: p.read_bytes() for p in first.iterdir()}
    for options in ({}, {"resume": True}, {"retry_failed": True}):
        with pytest.raises(ValueError, match="resume"):
            store.begin_attempt(tmp_path, JOB, PLAN, **options)
    second, cached = store.begin_attempt(tmp_path, JOB, PLAN, resume=True, retry_failed=True)
    assert second.name == "attempt-0002" and not cached
    assert {p.name: p.read_bytes() for p in first.iterdir()} == previous
    store.complete_attempt(second, JOB, PLAN)
    assert store.load_success(tmp_path, JOB["id"], PLAN)[0] == second


def test_completion_manifest_includes_nested_outputs_and_closed_log(tmp_path):
    path = new_attempt(tmp_path)
    (path / "model").mkdir()
    (path / "model" / "weights.bin").write_bytes(b"invented weights")
    marker = store.complete_attempt(path, JOB, PLAN)
    expected = {"attempt.json", "worker.log", "result.json", "model/weights.bin"}
    assert set(marker["files_sha256"]) == expected
    assert all(marker["files_sha256"][name] == store.sha256(path / name) for name in expected)
    assert "complete.json" not in marker["files_sha256"]


@pytest.mark.parametrize("change", ["modified", "missing", "added", "nested_added"])
def test_success_inventory_detects_changed_missing_and_extra_files(tmp_path, change):
    path, _ = complete(tmp_path)
    if change == "modified":
        (path / "worker.log").write_text("changed")
    elif change == "missing":
        (path / "result.json").unlink()
    elif change == "added":
        (path / "extra.txt").write_text("extra")
    else:
        (path / "extra").mkdir()
        (path / "extra" / "file.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory differs"):
        store.begin_attempt(tmp_path, JOB, PLAN, resume=True, retry_failed=True)
    assert len(list(path.parent.iterdir())) == 1


@pytest.mark.parametrize("where", ["output", "jobs", "attempt", "file", "directory", "complete"])
def test_symlink_is_rejected_even_if_it_resolves_to_correct_bytes(tmp_path, where):
    root = tmp_path / "run"
    path, _ = complete(root)
    if where == "output":
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    elif where in {"jobs", "attempt"}:
        target = root / "jobs" if where == "jobs" else path
        moved = tmp_path / "moved"
        target.rename(moved)
        target.symlink_to(moved, target_is_directory=True)
    elif where == "directory":
        (path / "nested").symlink_to(tmp_path, target_is_directory=True)
    else:
        target = path / ("complete.json" if where == "complete" else "worker.log")
        moved = tmp_path / "moved-file"
        target.rename(moved)
        target.symlink_to(moved)
    with pytest.raises(ValueError, match="Symbolic"):
        store.load_success(root, JOB["id"], PLAN)


@pytest.mark.parametrize("marker_content", ["{", '{"status":"passed","x":NaN}',
    '{"status":"passed","x":1e999}', '{"status":"passed","status":"passed"}', '[]'])
def test_malformed_completion_is_never_silently_retried(tmp_path, marker_content):
    path = new_attempt(tmp_path)
    (path / "complete.json").write_text(marker_content)
    with pytest.raises((ValueError, TypeError)):
        store.begin_attempt(tmp_path, JOB, PLAN, resume=True, retry_failed=True)
    assert len(list(path.parent.iterdir())) == 1


@pytest.mark.parametrize("change", ["plan", "job", "status", "metadata_omitted"])
def test_marker_binding_and_manifest_are_checked(tmp_path, change):
    path, marker = complete(tmp_path)
    if change == "plan":
        marker["plan_sha256"] = "b" * 64
    elif change == "job":
        marker["job"]["alpha"] = 2.0
    elif change == "status":
        marker["status"] = "failed"
    else:
        del marker["files_sha256"]["attempt.json"]
    (path / "complete.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="Completion marker"):
        store.load_success(tmp_path, JOB["id"], PLAN)


@pytest.mark.parametrize("change", ["job", "plan"])
def test_changed_requested_job_or_plan_is_rejected_before_new_attempt(tmp_path, change):
    path, _ = complete(tmp_path)
    altered = dict(JOB, alpha=3.0) if change == "job" else JOB
    digest = "b" * 64 if change == "plan" else PLAN
    with pytest.raises(ValueError, match="job|plan"):
        store.begin_attempt(tmp_path, altered, digest, resume=True, retry_failed=True)
    assert len(list(path.parent.iterdir())) == 1


def test_no_attempt_may_follow_a_successful_one(tmp_path):
    first, _ = complete(tmp_path)
    second = first.parent / "attempt-0002"
    second.mkdir()
    metadata = json.loads((first / "attempt.json").read_text())
    metadata["attempt_number"] = 2
    store.write_json_atomic(second / "attempt.json", metadata)
    with pytest.raises(ValueError, match="after an immutable successful"):
        store.load_success(tmp_path, JOB["id"], PLAN)


def test_failed_attempt_cannot_be_marked_successful(tmp_path):
    path = new_attempt(tmp_path)
    store.write_json_atomic(path / "failure.json", {"status": "cancelled"})
    with pytest.raises(ValueError, match="failed or completed"):
        store.complete_attempt(path, JOB, PLAN)
    assert not (path / "complete.json").exists()


def test_success_cannot_be_resealed_or_mix_failure_and_success(tmp_path):
    path, _ = complete(tmp_path)
    with pytest.raises(ValueError, match="failed or completed"):
        store.complete_attempt(path, JOB, PLAN)
    store.write_json_atomic(path / "failure.json", {"status": "failed"})
    with pytest.raises(ValueError, match="both failed and successful"):
        store.load_success(tmp_path, JOB["id"], PLAN)


@pytest.mark.parametrize("name", ["attempt-0000", "attempt-00001", "attempt-2", "unexpected"])
def test_unexpected_attempt_names_fail(tmp_path, name):
    directory = tmp_path / "jobs" / JOB["id"]
    (directory / name).mkdir(parents=True)
    with pytest.raises(ValueError, match="attempt|Unexpected"):
        store.load_success(tmp_path, JOB["id"], PLAN)


def test_gap_in_attempt_numbers_fails(tmp_path):
    path = new_attempt(tmp_path)
    path.rename(path.parent / "attempt-0002")
    with pytest.raises(ValueError, match="missing attempt numbers"):
        store.load_success(tmp_path, JOB["id"], PLAN)


@pytest.mark.parametrize("identity", ["../job", "/job", "job/name", "", ".hidden", None])
def test_job_paths_are_plain_identifiers(tmp_path, identity):
    with pytest.raises(ValueError, match="plain filename"):
        store.begin_attempt(tmp_path, dict(JOB, id=identity), PLAN)
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize("digest", ["abc", "G" * 64, "A" * 64, 1, None])
def test_plan_hash_must_be_lowercase_sha256(tmp_path, digest):
    with pytest.raises(ValueError, match="SHA256"):
        store.begin_attempt(tmp_path, JOB, digest)


@pytest.mark.parametrize("payload", [{"value": float("nan")}, {"x": float("inf")}, {1: "key"}])
def test_nonfinite_and_nonstring_key_json_never_reaches_disk(tmp_path, payload):
    with pytest.raises(ValueError):
        store.write_json_atomic(tmp_path / "bad.json", payload)
    assert list(tmp_path.iterdir()) == []


def test_atomic_json_never_overwrites_without_explicit_replace(tmp_path):
    path = tmp_path / "state.json"
    store.write_json_atomic(path, {"n": 1})
    first = path.read_bytes()
    with pytest.raises(FileExistsError):
        store.write_json_atomic(path, {"n": 2})
    assert path.read_bytes() == first
    store.write_json_atomic(path, {"n": 2}, replace=True)
    assert json.loads(path.read_text()) == {"n": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_atomic_publish_failure_cleans_temporary_file(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("invented publish failure")
    monkeypatch.setattr(store.os, "link", fail)
    with pytest.raises(OSError, match="invented"):
        store.write_json_atomic(tmp_path / "never.json", {"n": 1})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a//b", "a/./b", "a\\b", "."])
def test_inventory_rejects_untrusted_paths(tmp_path, name):
    with pytest.raises(ValueError, match="inventory|Inventory"):
        store.verify_inventory(tmp_path, {name: PLAN})


def test_lock_rejects_concurrent_parent_and_releases_after_close(tmp_path):
    with store.run_lock(tmp_path) as descriptor:
        assert os.get_inheritable(descriptor)
        with pytest.raises(RuntimeError, match="Another parent or worker"):
            with store.run_lock(tmp_path):
                pytest.fail("A second lock must not be acquired")
    assert (tmp_path / ".run.lock").is_file()
    with store.run_lock(tmp_path):
        pass


def test_inherited_worker_retains_lock_when_parent_context_closes(tmp_path):
    worker = None
    try:
        with store.run_lock(tmp_path) as descriptor:
            worker = subprocess.Popen([sys.executable, "-c",
                "import sys; print('ready',flush=True); sys.stdin.read(1)"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                pass_fds=(descriptor,))
            assert worker.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeError, match="Another parent or worker"):
            with store.run_lock(tmp_path):
                pytest.fail("Worker must retain its inherited lock")
        worker.communicate("x", timeout=5)
        assert worker.returncode == 0
        with store.run_lock(tmp_path):
            pass
    finally:
        if worker is not None and worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)


def test_stale_lock_file_is_not_a_stale_pid_bypass(tmp_path):
    (tmp_path / ".run.lock").write_text("obsolete PID is irrelevant")
    with store.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another parent or worker"):
            with store.run_lock(tmp_path):
                pass


def test_lock_symlink_rejected(tmp_path):
    target = tmp_path / "outside.txt"
    target.write_text("preserve")
    root = tmp_path / "run"
    root.mkdir()
    (root / ".run.lock").symlink_to(target)
    with pytest.raises(ValueError, match="Symbolic"):
        with store.run_lock(root):
            pass
    assert target.read_text() == "preserve"


def test_failed_report_cannot_be_sealed_as_passed(tmp_path):
    path = new_attempt(tmp_path)
    with pytest.raises(ValueError, match="failed job"):
        store.complete_attempt(path, JOB, PLAN, {"status": "failed"})
    assert not (path / "complete.json").exists()


def test_special_file_is_rejected_without_trying_to_read_it(tmp_path):
    path = new_attempt(tmp_path)
    os.mkfifo(path / "unexpected-pipe")
    with pytest.raises(ValueError, match="Only regular files"):
        store.complete_attempt(path, JOB, PLAN)
    assert not (path / "complete.json").exists()


def test_missing_start_record_is_not_silently_reconstructed(tmp_path):
    path = new_attempt(tmp_path)
    (path / "attempt.json").unlink()
    with pytest.raises(FileNotFoundError):
        store.begin_attempt(tmp_path, JOB, PLAN, resume=True, retry_failed=True)
    assert len(list(path.parent.iterdir())) == 1
