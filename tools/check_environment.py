"""Verify installed libraries using small invented arrays, never the LTE dataset.

Checks native loading, CPU math, a tiny LSTM gradient update and same-seed
repeatability. This is installation evidence, not forecast/pipeline validation.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from tools.run_cpu import CPU_ENV


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def fingerprint(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def check_environment():
    # Fail before importing numerical libraries if startup controls are absent.
    wrong = [name for name, value in CPU_ENV.items() if os.environ.get(name) != value]
    if wrong:
        raise RuntimeError("Use tools/run_cpu.py first; missing settings: " + ", ".join(wrong))
    if platform.python_implementation() != "CPython" or platform.python_version() != "3.12.14":
        raise RuntimeError("Reference profile requires CPython 3.12.14; record/review a different interpreter first")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use a dedicated virtual environment, not the system/runtime Python")
    venv_cfg = read_optional(Path(sys.prefix)/"pyvenv.cfg") or ""
    if "include-system-site-packages = false" not in venv_cfg.lower():
        raise RuntimeError("The environment must not inherit system packages")

    import numpy as np
    import pandas as pd
    import scipy
    import yaml
    import lightgbm as lgb
    import matplotlib
    import psutil
    import sklearn
    import torch
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_info, threadpool_limits

    root = Path(__file__).resolve().parents[1]
    expected = {}
    for line in (root/"requirements/core.in").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            expected[name] = version
    versions = {name: metadata.version(name) for name in expected}
    mismatches = {name: [expected[name], value] for name, value in versions.items() if expected[name] != value}
    torch_version = metadata.version("torch")
    if torch_version.split("+")[0] != "2.9.1":
        mismatches["torch"] = ["2.9.1 (CPU execution)", torch_version]
    if mismatches:
        raise RuntimeError("Unexpected direct-package versions: " + str(mismatches))
    if torch.version.cuda is not None:
        raise RuntimeError("Use the declared CPU PyTorch build; CUDA build found")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)

    # threadpool_limits supplements startup variables for already loaded native pools.
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(2026)
        X = rng.normal(size=(64, 4)).astype(np.float64)
        y = 2*X[:, 0] - X[:, 1] + .1
        r1 = Ridge(alpha=1.0, solver="svd").fit(X, y)
        r2 = Ridge(alpha=1.0, solver="svd").fit(X, y)
        ridge_pred = r1.predict(X)
        np.testing.assert_array_equal(ridge_pred, r2.predict(X))

        # 'regression' is LightGBM's native name for the specified squared-error loss.
        tree_params = dict(objective="regression", n_estimators=4, num_leaves=7,
                           min_child_samples=4, n_jobs=1, device_type="cpu",
                           deterministic=True, force_col_wise=True,
                           random_state=2026, verbosity=-1)
        t1 = lgb.LGBMRegressor(**tree_params).fit(X, y)
        t2 = lgb.LGBMRegressor(**tree_params).fit(X, y)
        tree_pred = t1.booster_.predict(X, num_threads=1)
        np.testing.assert_array_equal(tree_pred, t2.booster_.predict(X, num_threads=1))
        native_text = t1.booster_.model_to_string().split("parameters:\n", 1)[1].split("end of parameters", 1)[0]
        # The native model also records defaults omitted by the sklearn wrapper.
        native_tree_parameters = {}
        for line in native_text.splitlines():
            if line.startswith("[") and line.endswith("]"):
                name, value = line[1:-1].split(":", 1)
                native_tree_parameters[name.strip()] = value.strip()

        def lstm_once():
            # Reset the seed before both initialisation and the invented inputs.
            torch.manual_seed(2026)
            sequence = torch.randn(4, 168, 2, dtype=torch.float32, device="cpu")
            calendar = torch.randn(4, 4, dtype=torch.float32, device="cpu")
            target = torch.randn(4, 1, dtype=torch.float32, device="cpu")
            layer = torch.nn.LSTM(2, 16, num_layers=1, batch_first=True, device="cpu")
            dropout = torch.nn.Dropout(.2)
            head = torch.nn.Linear(20, 1, device="cpu")
            params = list(layer.parameters()) + list(head.parameters())
            optimizer = torch.optim.Adam(params, lr=.001)
            optimizer.zero_grad()
            hidden, _ = layer(sequence)
            pred = head(torch.cat((dropout(hidden[:, -1, :]), calendar), dim=1))
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError("LSTM/Adam gradient check failed")
            optimizer.step()
            # Disable output dropout while comparing the updated predictions.
            dropout.eval()
            with torch.no_grad():
                hidden, _ = layer(sequence)
                result = head(torch.cat((dropout(hidden[:, -1, :]), calendar), dim=1))
            if result.shape != (4, 1) or not torch.isfinite(result).all():
                raise RuntimeError("LSTM output check failed")
            return result.numpy(), optimizer.defaults

        lstm_pred, adam_defaults = lstm_once()
        lstm_repeat, _ = lstm_once()
        np.testing.assert_array_equal(lstm_pred, lstm_repeat)
        if not (np.isfinite(ridge_pred).all() and np.isfinite(tree_pred).all()):
            raise RuntimeError("Nonfinite native-library output")
        pools = threadpool_info()
        if not pools or any(pool["num_threads"] != 1 for pool in pools):
            raise RuntimeError("Loaded BLAS/OpenMP pools did not respect the one-thread limit")
        # All computation above happened inside this controlled context.
        pool_report = [{key: pool.get(key) for key in
            ["user_api", "internal_api", "prefix", "version", "num_threads", "architecture"]} for pool in pools]

    cpu_name = platform.processor() or None
    if sys.platform.startswith("linux"):
        lines = (read_optional("/proc/cpuinfo") or "").splitlines()
        cpu_name = next((line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")), cpu_name)
    if sys.platform == "darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, capture_output=True)
        cpu_name = result.stdout.strip() or cpu_name
    installed = sorted({(dist.metadata["Name"], dist.version) for dist in metadata.distributions()})
    checks = {
        "isolated_venv": True, "direct_versions_match": True,
        "cpu_build_and_device": True, "ridge_repeat_equal": True,
        "lightgbm_repeat_equal": True, "lstm_forward_backward_adam": True,
        "lstm_repeat_equal": True, "single_thread_pools": True,
    }
    return {
        "schema_version": 1,
        "scope": "Installation smoke check on invented arrays only; no LTE observations or forecasting experiment.",
        "status": "passed", "checks": checks,
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "compiler": platform.python_compiler()},
        "machine": {"os": platform.system(), "os_release": platform.release(), "architecture": platform.machine(),
                    "macos_version": platform.mac_ver()[0] or None,
                    "windows_version": platform.win32_ver() if sys.platform == "win32" else None,
                    "libc": platform.libc_ver(), "cpu_name": cpu_name,
                    "visible_logical_cpus": os.cpu_count(),
                    "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                    "host_visible_memory_bytes": psutil.virtual_memory().total,
                    "cgroup_cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
                    "cgroup_memory_max": read_optional("/sys/fs/cgroup/memory.max")},
        "packages": dict(installed), "startup_controls": CPU_ENV,
        "native_pools_during_checks": pool_report,
        "torch": {"version": torch_version, "device": "cpu", "cuda_build": torch.version.cuda,
                  "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                  "mps_available_but_unused": bool(torch.backends.mps.is_available())},
        "toy_output_hashes": {"ridge": fingerprint(ridge_pred), "lightgbm": fingerprint(tree_pred),
                              "lstm_after_one_update": fingerprint(lstm_pred)},
        "library_defaults": {"ridge_sklearn_defaults": Ridge().get_params(),
            "lightgbm_sklearn_defaults": lgb.LGBMRegressor().get_params(),
            "adam_defaults": adam_defaults,
            "lightgbm_toy_booster_parameters": t1.booster_.params,
            "lightgbm_toy_effective_native_parameters": native_tree_parameters,
            "lstm": {"bias": True, "num_layers": 1, "bidirectional": False,
                     "batch_first": True, "internal_dropout": 0.0, "proj_size": 0,
                     "dtype": "float32", "default_weight_initialization": "uniform +/- 1/sqrt(hidden_size)"}},
        "limitations": ["Same-environment toy repeatability only; no guarantee across OS, CPUs, library builds or releases.",
                        "Shared host: exclusive CPU allocation, clock frequency and power mode are not established.",
                        "Formal timing host remains to be designated and checked before the pilot.",
                        "Framework default report is not verification of the future model implementation."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = check_environment()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "torch": report["torch"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
