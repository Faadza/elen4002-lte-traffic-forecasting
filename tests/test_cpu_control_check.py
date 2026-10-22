"""Ensure a diagnostic cannot save a pass after a failed control or cleanup."""
from contextlib import contextmanager
import json

import pytest

from tools import check_cpu_controls as check


def test_existing_output_stops_before_runtime_changes(monkeypatch, tmp_path):
    output = tmp_path / "checks.json"
    output.write_text("keep previous evidence")
    monkeypatch.setattr(check, "single_thread_cpu", lambda: pytest.fail("must not configure"))
    with pytest.raises(ValueError, match="new output"):
        check.check_controls(output)
    assert output.read_text() == "keep previous evidence"


@pytest.mark.parametrize("failure", ["entry", "operation", "cleanup"])
def test_failed_diagnostic_never_saves_pass(monkeypatch, tmp_path, failure):
    class Controls:
        def verify(self):
            return {"runtime_library_controls_verified": True}

    @contextmanager
    def context():
        if failure == "entry":
            raise RuntimeError("entry failed")
        yield Controls()
        if failure == "cleanup":
            raise RuntimeError("cleanup failed")

    def operations(controls):
        if failure == "operation":
            raise RuntimeError("operation failed")
        return {"traffic_data_loaded": False}

    monkeypatch.setattr(check, "single_thread_cpu", context)
    monkeypatch.setattr(check, "exercise_operations", operations)
    output = tmp_path / "checks.json"
    with pytest.raises(RuntimeError, match=failure):
        check.check_controls(output)
    assert not output.exists()


def test_saved_scope_does_not_authorise_timing(monkeypatch, tmp_path):
    class Controls:
        def verify(self):
            return {"runtime_library_controls_verified": True,
                    "formal_timing_eligible": False}

    @contextmanager
    def context():
        yield Controls()

    monkeypatch.setattr(check, "single_thread_cpu", context)
    monkeypatch.setattr(check, "exercise_operations", lambda controls: {"traffic_data_loaded": False})
    monkeypatch.setattr(check, "machine_information", lambda: {"background_workload_confirmed_by_user": False})
    output = tmp_path / "nested" / "checks.json"
    result = check.check_controls(output)
    saved = json.loads(output.read_text())
    assert saved == result
    assert result["status"] == "passed"
    assert result["controls_after"]["runtime_library_controls_verified"]
    assert not result["formal_timing_eligible"]
    assert not result["formal_timing_measured"]
    assert not result["operations"]["traffic_data_loaded"]
    assert result["code_sha256"]["tools/check_cpu_controls.py"] == check._sha256(check.__file__)
