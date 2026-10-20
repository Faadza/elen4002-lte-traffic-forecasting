"""Regression checks for observable pools versus the macOS Accelerate blind spot."""
import json
import sys

import pytest

from tools import check_forecasting_pipeline as check


@pytest.fixture
def mac_accelerate(monkeypatch):
    for key, value in check.CPU_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(check.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(check, "threadpool_info", lambda: [])
    monkeypatch.setattr(check.np, "show_config", lambda **kwargs: {
        "Build Dependencies": {"blas": {"name": "accelerate", "found": True, "version": "unknown"}}})


def test_empty_accelerate_records_unverified_runtime(mac_accelerate):
    report = check.check_thread_controls()
    assert report["status"] == "partial_accelerate_visibility"
    assert report["startup_controls_verified"]
    assert report["detected_pool_count"] == 0
    assert not report["detected_pool_limits_verified"]
    assert report["accelerate_runtime_limit_verified"] is False
    assert not report["formal_timing_eligible"]
    assert report["limitations"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("key", ["VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"])
def test_missing_startup_control_is_not_excused(mac_accelerate, monkeypatch, key):
    monkeypatch.delenv(key)
    with pytest.raises(RuntimeError, match=key):
        check.check_thread_controls()


@pytest.mark.parametrize("threads", [2, 0, None])
def test_detected_non_one_pool_still_fails(mac_accelerate, monkeypatch, threads):
    monkeypatch.setattr(check, "threadpool_info", lambda: [{"prefix": "libomp", "num_threads": threads}])
    with pytest.raises(RuntimeError, match="libomp"):
        check.check_thread_controls()


@pytest.mark.parametrize("system,name,found", [
    ("Linux", "accelerate", True),
    ("Darwin", "unknown", True),
    ("Darwin", "accelerate", False),
    ("Linux", "scipy-openblas", True),
])
def test_unexplained_empty_pools_fail(mac_accelerate, monkeypatch, system, name, found):
    monkeypatch.setattr(check.platform, "system", lambda: system)
    monkeypatch.setattr(check.np, "show_config", lambda **kwargs: {
        "Build Dependencies": {"blas": {"name": name, "found": found}}})
    with pytest.raises(RuntimeError, match="No supported numerical pool"):
        check.check_thread_controls()


def test_openblas_one_thread_is_observed(mac_accelerate, monkeypatch):
    monkeypatch.setattr(check.platform, "system", lambda: "Linux")
    monkeypatch.setattr(check.np, "show_config", lambda **kwargs: {
        "Build Dependencies": {"blas": {"name": "scipy-openblas", "found": True}}})
    monkeypatch.setattr(check, "threadpool_info", lambda: [{"prefix": "libopenblas", "num_threads": 1}])
    report = check.check_thread_controls()
    assert report["status"] == "detected_pools_checked"
    assert report["detected_pool_limits_verified"]
    assert report["accelerate_runtime_limit_verified"] is None
    assert not report["formal_timing_eligible"]


def test_other_one_thread_pool_does_not_verify_accelerate(mac_accelerate, monkeypatch):
    monkeypatch.setattr(check, "threadpool_info", lambda: [{"prefix": "libomp", "num_threads": 1}])
    report = check.check_thread_controls()
    assert report["detected_pool_limits_verified"]
    assert report["status"] == "partial_accelerate_visibility"
    assert report["accelerate_runtime_limit_verified"] is False


def test_cli_stops_before_loading_data_when_pool_is_bad(mac_accelerate, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["check", "unused", "--module", "shared", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(check, "threadpool_info", lambda: [{"prefix": "libomp", "num_threads": 8}])
    monkeypatch.setattr(check, "check_module", lambda *args: pytest.fail("Data path must not run"))
    with pytest.raises(RuntimeError, match="libomp"):
        check.main()


def test_shared_path_and_saved_report_keep_accelerate_limitation(mac_accelerate, monkeypatch, tmp_path, capsys):
    # A complete invented calendar exercises the real shared checker and JSON save.
    length = check.TRAIN_END
    traffic = 1.0 + check.np.arange(length, dtype=float)
    hourly = check.pd.DataFrame({
        "cell": "Cell_003791", "ts": check.pd.date_range("2020-01-01", periods=length, freq="h"),
        "traffic": traffic, "traffic_observed": traffic, "observed": check.np.ones(length, dtype=int),
        "eligible_target": check.np.arange(length) >= check.LOOKBACK,
    })
    monkeypatch.setattr(check, "load_validated_data", lambda path: (hourly, {"source_sha256": "invented-test"}))
    monkeypatch.setattr(sys, "argv", ["check", "unused", "--module", "shared", "--output-dir", str(tmp_path)])
    check.main()
    console = json.loads(capsys.readouterr().out)
    saved = json.loads((tmp_path / "checks.json").read_text())
    assert console["status"] == saved["status"] == "passed"
    assert saved["native_pools"] == []
    assert console["thread_controls"] == saved["thread_controls"]
    assert saved["thread_controls"]["accelerate_runtime_limit_verified"] is False
    assert saved["thread_controls"]["formal_timing_eligible"] is False
    assert saved["shared_checks"]["scoring_targets"] == 72
    assert not saved["formal_timing_measured"]
    assert not (tmp_path / "predictions.csv").exists()
