"""Small, resumable job folders; all writers must hold ``run_lock``.

A successful attempt is an immutable set of files with a completion manifest.
Interrupted/failed attempts are kept, and retrying them needs explicit consent.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile


_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ATTEMPT = re.compile(r"attempt-([0-9]{4,})\Z")


def _safe_path(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Parent traversal is not allowed")
    # Existing ancestors must be real directories, never symbolic links.
    for item in reversed((path, *path.parents)):
        if item.is_symlink():
            raise ValueError(f"Symbolic links are not allowed: {item}")
    return path


def _plain_id(value):
    if not isinstance(value, str) or not _JOB_ID.fullmatch(value):
        raise ValueError("Job ID must be a plain filename identifier")
    return value


def _digest(value):
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("Expected a lowercase SHA256 digest")
    return value


def _json_bytes(payload):
    def check_keys(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("JSON object keys must be strings")
            for child in value.values():
                check_keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_keys(child)

    check_keys(payload)
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Repeated JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    value = json.loads(_safe_path(path).read_text(), object_pairs_hook=unique_keys,
                       parse_constant=invalid_constant)
    _json_bytes(value)  # Also reject overflow such as 1e999.
    return value


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256(path):
    """Hash one regular file without following a symbolic link."""
    path = _safe_path(path)
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json_atomic(path, payload, *, replace=False):
    """Publish complete finite JSON; replacing an existing file is opt-in."""
    if not isinstance(replace, bool):
        raise ValueError("replace must be a boolean")
    path = _safe_path(path)
    content = _json_bytes(payload)
    if not path.parent.is_dir():
        raise ValueError("The destination directory must already exist")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # link() fails atomically if the destination already exists.
            os.link(temporary, path)
            temporary.unlink()
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_file(name):
    if not isinstance(name, str) or not name:
        raise ValueError("Inventory paths must be nonempty relative strings")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
        raise ValueError(f"Unsafe inventory path: {name}")
    if name == ".":
        raise ValueError("Inventory entries must name files")
    return name


def _inventory(directory, exclude=()):
    directory = _safe_path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory: {directory}")
    excluded = {_relative_file(name) for name in exclude}
    result = {}
    for parent, directories, files in os.walk(directory, followlinks=False):
        for name in sorted(directories + files):
            path = Path(parent) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symbolic links are not allowed: {path}")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ValueError(f"Only regular files are allowed: {path}")
            relative = path.relative_to(directory).as_posix()
            if relative not in excluded:
                result[relative] = sha256(path)
    return dict(sorted(result.items()))


def verify_inventory(directory, expected, *, exclude=()):
    """Require the exact recorded files, not just an unchanged subset."""
    if not isinstance(expected, dict):
        raise ValueError("Inventory must be an object mapping paths to SHA256")
    for name, digest in expected.items():
        _relative_file(name)
        _digest(digest)
    actual = _inventory(directory, exclude)
    if actual != expected:
        raise ValueError("Attempt inventory differs: missing, added or changed files")
    return actual


@contextmanager
def run_lock(outputdir):
    """Hold an OS lock; pass this fd to workers with ``pass_fds=(fd,)``.

    Closing the parent's fd alone leaves an inherited worker lock in place.
    The lock file stays on disk so another process cannot lock a new inode.
    """
    directory = _safe_path(outputdir)
    directory.mkdir(parents=True, exist_ok=True)
    lockfile = _safe_path(directory / ".run.lock")
    descriptor = os.open(lockfile, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Run lock must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another parent or worker still owns this run") from error
        os.set_inheritable(descriptor, True)
        yield descriptor
    finally:
        # Do not explicitly unlock: a still-running worker owns this fd too.
        os.close(descriptor)


def _job(job):
    if not isinstance(job, dict):
        raise ValueError("Job must be a JSON object")
    _plain_id(job.get("id"))
    _json_bytes(job)
    return job


def _attempts(outputdir, job_id):
    directory = _safe_path(Path(outputdir) / "jobs" / _plain_id(job_id))
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError("Job location must be a directory")
    attempts = []
    for path in directory.iterdir():
        _safe_path(path)
        match = _ATTEMPT.fullmatch(path.name)
        if not match or not path.is_dir():
            raise ValueError(f"Unexpected entry in job directory: {path.name}")
        number = int(match[1])
        if path.name != f"attempt-{number:04d}" or number < 1:
            raise ValueError("Invalid attempt number")
        attempts.append((number, path))
    attempts.sort()
    if [number for number, _ in attempts] != list(range(1, len(attempts) + 1)):
        raise ValueError("Attempt history has missing attempt numbers")
    return [path for _, path in attempts]


def _attempt_metadata(attempt, job_id, plan_sha256):
    metadata = _read_json(attempt / "attempt.json")
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
            or metadata.get("plan_sha256") != plan_sha256
            or metadata.get("attempt_number") != int(attempt.name.split("-")[1])
            or not isinstance(metadata.get("started_utc"), str)):
        raise ValueError("Attempt metadata does not match the requested plan")
    job = _job(metadata.get("job"))
    if job["id"] != job_id:
        raise ValueError("Attempt job ID differs from the requested job")
    # Traversal rejects symlinks/special files even in an unfinished attempt.
    _inventory(attempt)
    return metadata


def load_success(outputdir, job_id, plan_sha256):
    """Return (attempt path, marker), or None; reject corrupt/duplicate success."""
    _plain_id(job_id)
    _digest(plan_sha256)
    success = None
    previous_job = None
    for attempt in _attempts(outputdir, job_id):
        metadata = _attempt_metadata(attempt, job_id, plan_sha256)
        if previous_job is not None and _json_bytes(metadata["job"]) != previous_job:
            raise ValueError("Attempts disagree about the job settings")
        previous_job = _json_bytes(metadata["job"])
        if success is not None:
            raise ValueError("An attempt exists after an immutable successful attempt")
        if not (attempt / "complete.json").exists():
            if (attempt / "failure.json").exists():
                if not isinstance(_read_json(attempt / "failure.json"), dict):
                    raise ValueError("Failure record must be a JSON object")
            continue
        if (attempt / "failure.json").exists():
            raise ValueError("Attempt cannot be both failed and successful")
        marker = _read_json(attempt / "complete.json")
        if (not isinstance(marker, dict) or marker.get("schema_version") != 1
                or marker.get("status") != "passed"
                or marker.get("plan_sha256") != plan_sha256
                or _json_bytes(marker.get("job")) != previous_job):
            raise ValueError("Completion marker does not match the requested job/plan")
        inventory = marker.get("files_sha256")
        if not isinstance(inventory, dict) or "attempt.json" not in inventory:
            raise ValueError("Completion marker must include attempt metadata")
        verify_inventory(attempt, inventory, exclude=("complete.json",))
        success = (attempt, marker)
    return success


def begin_attempt(outputdir, job, plan_sha256, *, resume=False, retry_failed=False):
    """Reuse one verified success, or create the next preserved attempt."""
    _job(job)
    _digest(plan_sha256)
    if not isinstance(resume, bool) or not isinstance(retry_failed, bool):
        raise ValueError("resume and retry_failed must be booleans")
    if retry_failed and not resume:
        raise ValueError("Retry requires explicit resume and retry_failed")
    existing = _attempts(outputdir, job["id"])
    success = load_success(outputdir, job["id"], plan_sha256)
    for attempt in existing:
        metadata = _attempt_metadata(attempt, job["id"], plan_sha256)
        if _json_bytes(metadata["job"]) != _json_bytes(job):
            raise ValueError("Requested job differs from the recorded attempt")
    if success is not None:
        if not resume:
            raise ValueError("Job already succeeded; explicit resume is required")
        return success[0], True
    if existing and not (resume and retry_failed):
        raise ValueError("Failed or incomplete attempt requires resume and retry_failed")
    directory = _safe_path(Path(outputdir) / "jobs" / job["id"])
    directory.mkdir(parents=True, exist_ok=True)
    number = len(existing) + 1
    attempt = directory / f"attempt-{number:04d}"
    attempt.mkdir()  # Existing evidence is never overwritten.
    write_json_atomic(attempt / "attempt.json", {
        "schema_version": 1, "job": job, "plan_sha256": plan_sha256,
        "attempt_number": number, "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    _sync_directory(directory)
    return attempt, False


def complete_attempt(attempt, job, plan_sha256, report=None):
    """Seal after worker exit, closed log and caller validation, never before."""
    attempt = _safe_path(attempt)
    _job(job)
    _digest(plan_sha256)
    if not _ATTEMPT.fullmatch(attempt.name) or attempt.parent.name != job["id"]:
        raise ValueError("Attempt path does not match the job")
    metadata = _attempt_metadata(attempt, job["id"], plan_sha256)
    if _json_bytes(metadata["job"]) != _json_bytes(job):
        raise ValueError("Completion job differs from the recorded attempt")
    if (attempt / "failure.json").exists() or (attempt / "complete.json").exists():
        raise ValueError("A failed or completed attempt cannot be completed again")
    if report is not None:
        if not isinstance(report, dict):
            raise ValueError("Optional completion report must be a JSON object")
        if report.get("status", "passed") != "passed":
            raise ValueError("A completion report cannot describe a failed job")
    marker = {"schema_version": 1, "status": "passed", "job": job,
              "plan_sha256": plan_sha256, "files_sha256": _inventory(attempt),
              "report": report}
    write_json_atomic(attempt / "complete.json", marker)
    return marker
