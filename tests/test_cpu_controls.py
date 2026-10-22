"""Check fail-closed library controls without forecasting or benchmark claims."""
from contextlib import contextmanager
import ctypes
import json
from types import SimpleNamespace

import pytest

from tools import cpu_controls as controls


class CFunction:
    """A callable carrying ctypes signature attributes for native API fixtures."""

    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class FakeApple:
    def __init__(self, mode=0):
        self.mode, self.calls = mode, []
        self.fail_mode = None
        self.ignore_set = False
        self.BLASGetThreading = CFunction(lambda: self.mode)
        self.BLASSetThreading = CFunction(self.set)

    def set(self, mode):
        self.calls.append(mode)
        if mode == self.fail_mode:
            return -1
        if not self.ignore_set:
            self.mode = mode
        return 0


class FakeTorch:
    def __init__(self):
        self.intra, self.inter, self.deterministic = 4, 4, False
        self.version = SimpleNamespace(cuda=None, hip=None)
        self.inter_failure = False

    def set_num_threads(self, threads):
        self.intra = threads

    def get_num_threads(self):
        return self.intra

    def set_num_interop_threads(self, threads):
        if self.inter_failure:
            raise RuntimeError("inter-op work already began")
        self.inter = threads

    def get_num_interop_threads(self):
        return self.inter

    def use_deterministic_algorithms(self, value):
        self.deterministic = value

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic


def build(backend="scipy-openblas", found=True):
    dependencies = {kind: {"name": backend, "found": found, "version": "test"}
                    for kind in ("blas", "lapack")}
    return SimpleNamespace(show_config=lambda *, mode: {"Build Dependencies": dependencies})


def pool(path="/test/libopenblas.so", threads=1):
    return {"filepath": path, "prefix": "libopenblas", "user_api": "blas",
            "internal_api": "openblas", "num_threads": threads, "version": "test"}


@pytest.fixture
def fake_runtime(monkeypatch):
    for key, value in controls.CPU_ENV.items():
        monkeypatch.setenv(key, value)
    state = SimpleNamespace(numpy=build(), scipy=build(), torch=FakeTorch(),
                            pools=[pool(threads=4)], enters=0, exits=0, apple=FakeApple())

    @contextmanager
    def limits(*, limits):
        assert limits == 1
        state.enters += 1
        previous = [entry["num_threads"] for entry in state.pools]
        for entry in state.pools:
            entry["num_threads"] = 1
        try:
            yield
        finally:
            for entry, count in zip(state.pools, previous):
                entry["num_threads"] = count
            state.exits += 1

    monkeypatch.setattr(controls, "_load_libraries", lambda:
        (state.numpy, state.scipy, state.torch, lambda: [dict(p) for p in state.pools], limits))
    monkeypatch.setattr(controls.platform, "system", lambda: "Linux")
    monkeypatch.setattr(controls.ctypes, "CDLL", lambda path: state.apple)
    return state


def test_startup_failure_precedes_any_numerical_import(monkeypatch, fake_runtime):
    monkeypatch.delenv("VECLIB_MAXIMUM_THREADS")
    monkeypatch.setattr(controls, "_load_libraries", lambda: pytest.fail("Numerical import reached"))
    with pytest.raises(RuntimeError, match="missing CPU controls.*VECLIB"):
        with controls.single_thread_cpu():
            pytest.fail("Context entered")


def test_controls_scope_report_and_restore(fake_runtime):
    state = fake_runtime
    with controls.single_thread_cpu() as handle:
        result = handle.verify()
        assert result["runtime_library_controls_verified"] is True
        assert result["formal_timing_eligible"] is False
        assert result["torch"]["intraop_threads"] == result["torch"]["interop_threads"] == 1
        assert result["apple_blas"]["required"] is False
        assert result["native_pools"][0]["filepath"] == "/test/libopenblas.so"
        json.dumps(result)
    assert state.pools[0]["num_threads"] == 4
    assert state.enters == state.exits == 1
    assert state.torch.intra == state.torch.inter == 1
    with pytest.raises(RuntimeError, match="active context"):
        handle.verify()


def test_apple_correct_abi_and_fixed_system_framework(monkeypatch):
    apple, paths = FakeApple(), []
    monkeypatch.setattr(controls.ctypes, "CDLL", lambda path: paths.append(path) or apple)
    bound = controls._AppleBLAS()
    assert paths == ["/System/Library/Frameworks/Accelerate.framework/Accelerate"]
    assert bound.get.argtypes == [] and bound.get.restype is ctypes.c_uint
    assert bound.set.argtypes == [ctypes.c_uint] and bound.set.restype is ctypes.c_int


@pytest.mark.parametrize("problem", ["load", "getter", "setter"])
def test_missing_apple_api_fails_closed(monkeypatch, problem):
    apple = FakeApple()
    if problem == "load":
        def load(path):
            raise OSError("not found")
    else:
        delattr(apple, "BLASGetThreading" if problem == "getter" else "BLASSetThreading")
        load = lambda path: apple
    monkeypatch.setattr(controls.ctypes, "CDLL", load)
    with pytest.raises(RuntimeError, match="unavailable"):
        with controls._apple_limit("Darwin"):
            pytest.fail("Context entered")


@pytest.mark.parametrize("old_mode", [0, 1])
def test_apple_native_restore_after_success_and_exception(fake_runtime, old_mode):
    apple = fake_runtime.apple
    apple.mode = old_mode
    with pytest.raises(ValueError, match="body failed"):
        with controls._apple_limit("Darwin") as (bound, previous):
            assert previous == old_mode and bound.read() == 1
            raise ValueError("body failed")
    assert apple.mode == old_mode
    assert apple.calls == [1, old_mode]


def test_apple_setter_failure_restores_original(fake_runtime):
    fake_runtime.apple.fail_mode = 1
    with pytest.raises(RuntimeError, match="failed with status -1"):
        with controls._apple_limit("Darwin"):
            pytest.fail("Context entered")
    assert fake_runtime.apple.mode == 0
    assert fake_runtime.apple.calls == [1, 0]


def test_apple_readback_mismatch_restores_original(fake_runtime):
    fake_runtime.apple.ignore_set = True
    with pytest.raises(RuntimeError, match="readback differs"):
        with controls._apple_limit("Darwin"):
            pytest.fail("Context entered")
    assert fake_runtime.apple.calls == [1, 0]


@pytest.mark.parametrize("mode", [2, 55])
def test_unrecognised_initial_apple_mode_is_rejected(fake_runtime, mode):
    fake_runtime.apple.mode = mode
    with pytest.raises(RuntimeError, match="Unrecognised"):
        with controls._apple_limit("Darwin"):
            pytest.fail("Context entered")
    assert fake_runtime.apple.calls == []


def test_unrecognised_setter_readback_restores_original(fake_runtime):
    readings = iter([0, 2, 0])
    fake_runtime.apple.BLASGetThreading.function = lambda: next(readings)
    with pytest.raises(RuntimeError, match="Unrecognised"):
        with controls._apple_limit("Darwin"):
            pytest.fail("Context entered")
    assert fake_runtime.apple.calls == [1, 0]


def test_native_restoration_failure_is_not_suppressed(fake_runtime):
    with pytest.raises(RuntimeError, match="failed with status -1"):
        with controls._apple_limit("Darwin"):
            fake_runtime.apple.fail_mode = 0


@pytest.mark.parametrize("numpy_backend", ["accelerate", "scipy-openblas"])
def test_darwin_controls_apple_even_when_numpy_uses_openblas(monkeypatch, fake_runtime, numpy_backend):
    state = fake_runtime
    monkeypatch.setattr(controls.platform, "system", lambda: "Darwin")
    state.numpy, state.scipy = build(numpy_backend), build("accelerate")
    with controls.single_thread_cpu() as handle:
        result = handle.verify()
        assert result["apple_blas"]["caller_thread_setting_verified"] is True
        assert result["apple_blas"]["previous_mode"] == 0
        assert result["apple_blas"]["current_mode"] == 1
    assert state.apple.calls == [1, 0]
    assert state.apple.mode == 0


def test_context_error_restores_both_native_controls(monkeypatch, fake_runtime):
    monkeypatch.setattr(controls.platform, "system", lambda: "Darwin")
    with pytest.raises(ValueError, match="body failed"):
        with controls.single_thread_cpu():
            raise ValueError("body failed")
    assert fake_runtime.apple.mode == 0
    assert fake_runtime.pools[0]["num_threads"] == 4
    assert fake_runtime.exits == 1


@pytest.mark.parametrize("wrong_count", [0, 2, None])
def test_changed_pool_limit_rejected_and_restored(fake_runtime, wrong_count):
    with pytest.raises(RuntimeError, match="one thread"):
        with controls.single_thread_cpu():
            fake_runtime.pools[0]["num_threads"] = wrong_count
    assert fake_runtime.pools[0]["num_threads"] == 4
    assert fake_runtime.exits == 1


def test_new_library_pool_requires_fresh_context(fake_runtime):
    with pytest.raises(RuntimeError, match="pools changed"):
        with controls.single_thread_cpu() as handle:
            fake_runtime.pools.append(pool(path="/test/new-openblas.so"))
            handle.verify()
    assert fake_runtime.exits == 1
    with controls.single_thread_cpu() as handle:
        assert handle.verify()["detected_pool_count"] == 2


def test_removed_library_pool_is_rejected(fake_runtime):
    with pytest.raises(RuntimeError, match="pools changed"):
        with controls.single_thread_cpu() as handle:
            fake_runtime.pools.clear()
            handle.verify()


@pytest.mark.parametrize("attribute", ["getpid", "get_ident"])
def test_wrong_process_or_thread_cannot_reuse_handle(monkeypatch, fake_runtime, attribute):
    with controls.single_thread_cpu() as handle:
        module = controls.os if attribute == "getpid" else controls.threading
        original = getattr(module, attribute)
        monkeypatch.setattr(module, attribute, lambda: original() + 1)
        with pytest.raises(RuntimeError, match="different process or caller thread"):
            handle.verify()
        monkeypatch.setattr(module, attribute, original)


@pytest.mark.parametrize("package", ["numpy", "scipy"])
@pytest.mark.parametrize("bad_backend,found", [("mystery-blas", True), ("scipy-openblas", False),
                                              ("mkl", True), ("accelerate", True)])
def test_unknown_or_uncontrolled_backend_rejected(fake_runtime, package, bad_backend, found):
    setattr(fake_runtime, package, build(bad_backend, found))
    with pytest.raises(RuntimeError, match="identified|matched control|Accelerate control"):
        with controls.single_thread_cpu():
            pytest.fail("Context entered")
    assert fake_runtime.exits == 1


def test_lapack_backend_also_checked(fake_runtime):
    fake_runtime.scipy.show_config(mode="dicts")["Build Dependencies"]["lapack"]["name"] = "unknown"
    with pytest.raises(RuntimeError, match="SciPy LAPACK"):
        with controls.single_thread_cpu():
            pytest.fail("Context entered")


def test_empty_pool_visibility_does_not_pass(fake_runtime):
    fake_runtime.pools.clear()
    with pytest.raises(RuntimeError, match="No supported numerical pools"):
        with controls.single_thread_cpu():
            pytest.fail("Context entered")


def test_incomplete_pool_identity_does_not_pass(fake_runtime):
    del fake_runtime.pools[0]["filepath"]
    with pytest.raises(RuntimeError, match="incomplete identity"):
        with controls.single_thread_cpu():
            pytest.fail("Context entered")
    assert fake_runtime.exits == 1


def test_environment_change_inside_context_is_rejected(monkeypatch, fake_runtime):
    with pytest.raises(RuntimeError, match="missing CPU controls"):
        with controls.single_thread_cpu():
            monkeypatch.setenv("OMP_NUM_THREADS", "2")


def test_apple_setting_change_inside_context_is_rejected(monkeypatch, fake_runtime):
    monkeypatch.setattr(controls.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="Apple BLAS single-thread"):
        with controls.single_thread_cpu():
            fake_runtime.apple.mode = 0
    assert fake_runtime.apple.mode == 0
    assert fake_runtime.exits == 1


@pytest.mark.parametrize("setting", ["intra", "inter", "deterministic", "cuda", "hip"])
def test_torch_control_drift_rejected(fake_runtime, setting):
    with pytest.raises(RuntimeError, match="PyTorch CPU-build/thread/determinism"):
        with controls.single_thread_cpu():
            if setting in ("cuda", "hip"):
                setattr(fake_runtime.torch.version, setting, "test-gpu")
            else:
                setattr(fake_runtime.torch, setting, False if setting == "deterministic" else 2)


def test_interop_already_one_does_not_attempt_reset(fake_runtime):
    fake_runtime.torch.inter = 1
    fake_runtime.torch.inter_failure = True
    with controls.single_thread_cpu() as handle:
        assert handle.verify()["torch"]["interop_threads"] == 1


def test_interop_failure_requires_new_process(fake_runtime):
    fake_runtime.torch.inter_failure = True
    with pytest.raises(RuntimeError, match="fresh process"):
        with controls.single_thread_cpu():
            pytest.fail("Context entered")
    assert fake_runtime.enters == 0


def test_real_linux_library_context_preserves_rng_and_checks_toy_product():
    # A tiny numerical operation verifies integration; this is not an LTE fit/timing run.
    if controls.platform.system() != "Linux":
        pytest.skip("This independent real-backend fixture targets the recorded Linux environment")
    import numpy as np
    import torch
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    with controls.single_thread_cpu() as handle:
        result = handle.verify()
        np.testing.assert_array_equal(np.array([[1.0, 2.0]]) @ np.array([[3.0], [4.0]]), [[11.0]])
        assert result["runtime_library_controls_verified"]
        assert not result["formal_timing_eligible"]
        assert all(item["num_threads"] == 1 for item in result["native_pools"])
    assert torch.equal(torch_state, torch.get_rng_state())
    new_state = np.random.get_state()
    assert numpy_state[0] == new_state[0] and numpy_state[2:] == new_state[2:]
    np.testing.assert_array_equal(numpy_state[1], new_state[1])
