"""Audit entrypoint safety using invented paths; never open study archives."""
import builtins
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from docs.verification import audit_cli


ROOT = Path(__file__).resolve().parents[1]
AUDITS = (
    "audit_confirmation_integrity_2026_10_09.py",
    "audit_final_integrity_2026_10_10.py",
    "audit_lstm_saved_kernels_2026_10_09.py",
    "confirmation-independent-arithmetic-2026-10-09.py",
    "final-independent-arithmetic-2026-10-10.py",
)


def load_audit(filename, *, packaged=True):
    name = "audit_test_" + filename.replace("-", "_").removesuffix(".py")
    if packaged:
        name = "docs.verification." + name
    spec = importlib.util.spec_from_file_location(name, ROOT / "docs/verification" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", AUDITS)
@pytest.mark.parametrize("packaged", [True, False])
def test_import_does_not_read_archives_or_load_numerical_runtime(monkeypatch, filename, packaged):
    original_import = builtins.__import__
    forbidden = {"numpy", "pandas", "tools", "ensemble_study", "final_stage",
                 "models", "forecasting", "validation_revision"}

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            pytest.fail("Import attempted a numerical or study module: " + name)
        return original_import(name, *args, **kwargs)

    def reject_read(*args, **kwargs):
        pytest.fail("Import attempted to read an archive")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "read_text", reject_read)
    monkeypatch.setattr(Path, "read_bytes", reject_read)
    before = sys.path[:]
    module = load_audit(filename, packaged=packaged)
    assert callable(module.main) and callable(module._audit)
    assert sys.path == before


@pytest.mark.parametrize("filename", AUDITS)
def test_main_routes_explicit_paths_without_running_audit(tmp_path, monkeypatch, filename):
    module = load_audit(filename)
    observed = []
    monkeypatch.setattr(module, "_audit", observed.append)
    output = tmp_path / "new" / "output"
    workspace, directory = tmp_path / "workspace", tmp_path / "saved-study"
    module.main(["--repo", str(ROOT), "--workspace", str(workspace),
                 "--directory", str(directory), "--output", str(output)])
    assert observed == [audit_cli.AuditPaths(ROOT, workspace.resolve(), directory.resolve(), output.resolve())]
    assert output.is_dir() == filename.startswith("audit_lstm")


@pytest.mark.parametrize("filename", AUDITS)
def test_existing_output_stops_before_audit(tmp_path, monkeypatch, filename):
    module = load_audit(filename)
    output = tmp_path / "existing"
    if filename.startswith("audit_lstm"):
        output.mkdir()  # Even an empty existing directory is not a fresh output.
    else:
        output.write_text("previous result")
    monkeypatch.setattr(module, "_audit", lambda paths: pytest.fail("Audit should not start"))
    with pytest.raises(FileExistsError, match="already exists"):
        module.main(["--output", str(output)])
    if output.is_file():
        assert output.read_text() == "previous result"


def test_default_paths_use_checkout_and_declared_period(tmp_path):
    observed = []
    output = tmp_path / "report.json"
    audit_cli.run_audit(ROOT / "docs/verification/example.py", "final", observed.append,
                       ["--output", str(output)])
    assert observed == [audit_cli.AuditPaths(ROOT, ROOT.parent,
        ROOT / "results/local_runs/l1-l2-independent-ensemble/final", output.resolve())]
    assert not output.exists()


def test_output_inside_archive_or_dangling_symlink_is_rejected(tmp_path):
    archive = tmp_path / "study"
    with pytest.raises(ValueError, match="outside"):
        audit_cli.run_audit(ROOT / "docs/verification/example.py", "final", lambda p: pytest.fail(),
            ["--directory", str(archive), "--output", str(archive / "new.json")])
    output = tmp_path / "link"
    output.symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileExistsError):
        audit_cli.run_audit(ROOT / "docs/verification/example.py", "final", lambda p: pytest.fail(),
                           ["--output", str(output)])
    assert not (tmp_path / "missing-target").exists()


def test_report_write_is_exclusive_and_retains_serialisation(tmp_path):
    output = tmp_path / "new" / "report.json"
    value = {"b": 1.25, "a": [2, 3]}
    audit_cli.write_report(output, value, sort_keys=False)
    assert output.read_text() == json.dumps(value, indent=2, allow_nan=False) + "\n"
    with pytest.raises(FileExistsError):
        audit_cli.write_report(output, {"changed": True})
    assert json.loads(output.read_text()) == value
    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        audit_cli.write_report(invalid, {"value": float("nan")})
    assert not invalid.exists()


def test_repository_imports_reject_mixed_checkout_and_restore_path(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "models.foreign_audit_fixture",
                        SimpleNamespace(__file__=str(tmp_path / "model.py")))
    with pytest.raises(RuntimeError, match="fresh Python process"):
        with audit_cli.repository_imports(ROOT):
            pytest.fail("Must reject mixed runtime")
    monkeypatch.delitem(sys.modules, "models.foreign_audit_fixture")
    before = sys.path[:]
    with pytest.raises(ValueError):
        with audit_cli.repository_imports(ROOT):
            assert sys.path[0] == str(ROOT)
            raise ValueError("invented audit failure")
    assert sys.path == before


@pytest.mark.parametrize("filename", AUDITS)
def test_direct_cli_help_and_optimisation_guard(tmp_path, filename):
    script = ROOT / "docs/verification" / filename
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    help_result = subprocess.run([sys.executable, "-B", str(script), "--help"], cwd=tmp_path,
        env=environment, capture_output=True, text=True, timeout=10)
    assert help_result.returncode == 0, help_result.stderr
    for flag in ("--repo", "--workspace", "--directory", "--output"):
        assert flag in help_result.stdout
    output = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, "-B", "-O", str(script), "--output", str(output)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "-O disables verification assertions" in result.stderr
    assert not output.exists()
