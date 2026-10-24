"""Explicit input paths and fresh outputs for the dated research audits.

An archived runtime retains its own publication-receipt locations and pinned
paths. Selecting a workspace here changes diagnostic lookups, not those pins.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class AuditPaths:
    repo: Path
    workspace: Path
    directory: Path
    output: Path


def write_report(path, value, *, sort_keys=True):
    """Write a new report; never replace a previous audit or its evidence."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


@contextmanager
def repository_imports(repo):
    """Use the selected runtime, rejecting a process that mixed checkouts."""
    names = ("adaptation", "blend_revision", "ensemble_study", "evaluation",
             "final_evaluation", "final_stage", "forecasting", "l1_revision",
             "models", "preprocessing", "tools", "validation_revision")
    for name, module in tuple(sys.modules.items()):
        if name.split(".", 1)[0] not in names and not name.startswith("docs.commands"):
            continue
        filename = getattr(module, "__file__", None)
        if filename is not None and not Path(filename).resolve().is_relative_to(repo):
            raise RuntimeError("Start a fresh Python process for the selected --repo")
    docs = sys.modules.get("docs")
    if docs is not None and repo / "docs" not in {
            Path(path).resolve() for path in getattr(docs, "__path__", ())}:
        raise RuntimeError("Use a fresh direct-script invocation to select another --repo")
    original = sys.path[:]
    sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        sys.path[:] = original


def run_audit(script, period, audit, argv=None, *, output_directory=False):
    parser = argparse.ArgumentParser(
        description="Check the original archived study without changing its evidence.")
    parser.add_argument("--repo", type=Path, default=Path(script).resolve().parents[2],
                        help="Runtime source checkout; defaults to this checkout")
    parser.add_argument("--workspace", type=Path,
                        help="Diagnostic workspace; defaults to --repo's parent. Does not relocate publication receipts")
    parser.add_argument("--directory", type=Path,
                        help="Archived study directory; defaults to --repo/results/local_runs/l1-l2-independent-ensemble/" + period)
    parser.add_argument("--output", type=Path, required=True,
                        help="New diagnostic directory (must not exist, even empty)" if output_directory else "New JSON report (must not exist)")
    args = parser.parse_args(argv)
    if not __debug__:
        raise RuntimeError("Audits require normal Python; -O disables verification assertions")
    repo = args.repo.resolve()
    workspace = (args.workspace or repo.parent).resolve()
    directory = (args.directory or repo / "results/local_runs/l1-l2-independent-ensemble" / period).resolve()
    # Check the unresolved leaf too: a dangling symlink is an existing output.
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Audit output already exists; choose a new --output")
    output = args.output.resolve()
    if output.is_relative_to(directory):
        raise ValueError("Audit output must be outside the archived study directory")
    if not repo.is_dir():
        raise NotADirectoryError("The selected --repo does not exist")
    paths = AuditPaths(repo, workspace, directory, output)
    with repository_imports(repo):
        if output_directory:
            output.mkdir(parents=True, exist_ok=False)
        audit(paths)
