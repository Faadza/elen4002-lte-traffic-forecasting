"""Scoped numerical-library controls; use inside each CPU experiment process.

Importing this module does not import numerical libraries or change their settings.
Apple's BLAS control applies to subsequent BLAS/LAPACK calls on this caller thread;
it is not an observation of every native kernel or every operating-system thread.
"""
from contextlib import contextmanager
import ctypes
import os
import platform
import threading

from tools.run_cpu import CPU_ENV

ACCELERATE_FRAMEWORK = "/System/Library/Frameworks/Accelerate.framework/Accelerate"
BLAS_MULTI_THREADED = 0
BLAS_SINGLE_THREADED = 1


def _check_startup():
    wrong = [key for key, value in CPU_ENV.items() if os.environ.get(key) != value]
    if wrong:
        raise RuntimeError("Use tools/run_cpu.py; missing CPU controls: " + ", ".join(wrong))


def _load_libraries():
    # Import all project numerical backends before applying threadpoolctl limits.
    import numpy
    import scipy
    import scipy.linalg
    import sklearn
    import lightgbm
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    return numpy, scipy, torch, threadpool_info, threadpool_limits


class _AppleBLAS:
    """Bind the macOS 15+ public API with its unsigned-enum/int-return ABI."""

    def __init__(self):
        try:
            self.library = ctypes.CDLL(ACCELERATE_FRAMEWORK)
            self.get = self.library.BLASGetThreading
            self.set = self.library.BLASSetThreading
        except (OSError, AttributeError) as error:
            raise RuntimeError("The required Apple BLAS threading API is unavailable") from error
        self.get.argtypes, self.get.restype = [], ctypes.c_uint
        self.set.argtypes, self.set.restype = [ctypes.c_uint], ctypes.c_int

    def read(self):
        mode = self.get()
        if mode not in (BLAS_MULTI_THREADED, BLAS_SINGLE_THREADED):
            raise RuntimeError(f"Unrecognised Apple BLAS threading mode: {mode}")
        return mode

    def apply(self, mode):
        result = self.set(mode)
        if result != 0:
            raise RuntimeError(f"Apple BLASSetThreading({mode}) failed with status {result}")
        if self.read() != mode:
            raise RuntimeError("Apple BLAS threading readback differs from the requested mode")


@contextmanager
def _apple_limit(system):
    if system != "Darwin":
        yield None, None
        return
    apple = _AppleBLAS()
    previous = apple.read()
    try:
        # Apply on Darwin even if NumPy uses another backend: SciPy/Torch may use it.
        apple.apply(BLAS_SINGLE_THREADED)
        yield apple, previous
    finally:
        apple.apply(previous)


def _pool_identity(pool):
    keys = ("filepath", "user_api", "internal_api", "prefix")
    values = tuple(pool.get(key) for key in keys)
    if not all(isinstance(value, str) and value for value in values):
        raise RuntimeError("A detected numerical pool has an incomplete identity")
    return values


def _build_dependencies(module):
    dependencies = module.show_config(mode="dicts").get("Build Dependencies", {})
    return {kind: dependencies.get(kind, {}) for kind in ("blas", "lapack")}


def _verify_backend(build, pools, system, apple_verified, name):
    """Require an identified backend and its matching available control API."""
    visible = {pool.get("internal_api") for pool in pools if pool.get("user_api") == "blas"}
    aliases = {"openblas": "openblas", "scipy-openblas": "openblas", "mkl": "mkl",
               "mkl-sdl": "mkl", "mkl-static-lp64-iomp": "mkl", "blis": "blis"}
    for kind, dependency in build.items():
        backend = str(dependency.get("name", "")).lower().replace("_", "-")
        if dependency.get("found") is not True:
            raise RuntimeError(f"{name} {kind.upper()} build dependency was not identified")
        if backend == "accelerate":
            if system != "Darwin" or not apple_verified:
                raise RuntimeError(f"{name} {kind.upper()} Accelerate control is unverified")
        elif backend not in aliases or aliases[backend] not in visible:
            raise RuntimeError(f"{name} {kind.upper()} backend {backend!r} lacks a matched control")


class CPUControls:
    """A live context's inspection handle; it cannot be reused in another thread."""

    def __init__(self, numpy, scipy, torch, pool_info, system, apple, previous_mode):
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._torch, self._pool_info = torch, pool_info
        self._system, self._apple, self._previous_mode = system, apple, previous_mode
        self._numpy_build, self._scipy_build = _build_dependencies(numpy), _build_dependencies(scipy)
        self._reference_pools = frozenset(_pool_identity(pool) for pool in pool_info())
        self._active = True

    def _check_owner(self):
        if not self._active:
            raise RuntimeError("CPU controls must be verified inside their active context")
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise RuntimeError("CPU controls belong to a different process or caller thread")

    def verify(self):
        """Read back limits now; call before and after the intended numerical work."""
        self._check_owner()
        _check_startup()
        pools = self._pool_info()
        identities = frozenset(_pool_identity(pool) for pool in pools)
        if identities != self._reference_pools:
            raise RuntimeError("Loaded numerical pools changed; enter a new CPU-control context")
        if not pools:
            raise RuntimeError("No supported numerical pools are detected")
        if any(pool.get("num_threads") != 1 for pool in pools):
            raise RuntimeError("Detected numerical pools do not all report one thread")
        apple_verified = self._apple is not None and self._apple.read() == BLAS_SINGLE_THREADED
        if self._system == "Darwin" and not apple_verified:
            raise RuntimeError("Apple BLAS single-thread caller setting is unverified")
        _verify_backend(self._numpy_build, pools, self._system, apple_verified, "NumPy")
        _verify_backend(self._scipy_build, pools, self._system, apple_verified, "SciPy")
        torch = self._torch
        torch_report = {
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_build": torch.version.cuda,
            "hip_build": getattr(torch.version, "hip", None),
        }
        if (torch_report["intraop_threads"] != 1 or torch_report["interop_threads"] != 1
                or not torch_report["deterministic_algorithms"]
                or torch_report["cuda_build"] is not None or torch_report["hip_build"] is not None):
            raise RuntimeError("PyTorch CPU-build/thread/determinism controls are unverified")
        return {
            "status": "runtime_library_controls_verified",
            "runtime_library_controls_verified": True,
            "requested_threads": 1,
            "startup_controls_verified": True,
            "process_id": self._pid,
            "caller_thread_id": self._thread,
            "system": self._system,
            "numpy_build_dependencies": self._numpy_build,
            "scipy_build_dependencies": self._scipy_build,
            "detected_pool_count": len(pools),
            "detected_pool_limits_verified": True,
            "native_pools": pools,
            "pool_identities": [list(item) for item in sorted(identities)],
            "apple_blas": {
                "required": self._system == "Darwin",
                "framework": ACCELERATE_FRAMEWORK if self._apple is not None else None,
                "previous_mode": self._previous_mode,
                "current_mode": BLAS_SINGLE_THREADED if apple_verified else None,
                "caller_thread_setting_verified": apple_verified if self._apple is not None else None,
                "scope": "Subsequent BLAS/LAPACK calls on this caller thread",
            },
            "torch": torch_report,
            "limitations": [
                "Library settings are verified; native execution of every kernel is not traced.",
                "Use this context in each process and on the same caller thread as numerical work.",
                "PyTorch process controls remain applied after context exit; RNG seeds are unchanged.",
                "This check does not establish workload boundaries, host stability or formal timing readiness.",
            ],
            "formal_timing_eligible": False,
        }


@contextmanager
def single_thread_cpu():
    """Apply and verify library controls, then restore native scoped limits on exit.

    PyTorch's process-level inter-op setting cannot be repeatedly reset after work;
    its one-thread/deterministic controls remain applied. No random seed is set.
    """
    _check_startup()
    numpy, scipy, torch, pool_info, pool_limits = _load_libraries()
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            raise RuntimeError("Start a fresh process to set PyTorch inter-op threads to one") from error
    torch.use_deterministic_algorithms(True)
    system = platform.system()
    with _apple_limit(system) as (apple, previous_mode), pool_limits(limits=1):
        controls = CPUControls(numpy, scipy, torch, pool_info, system, apple, previous_mode)
        try:
            controls.verify()
            yield controls
            controls.verify()
        finally:
            controls._active = False
