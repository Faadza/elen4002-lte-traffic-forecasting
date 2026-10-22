"""Check numerical runtime controls with invented inputs, never traffic data.

Run through tools/run_cpu.py. A passing report applies to this diagnostic
process; every future experiment worker must enter the same control context.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path

from tools.cpu_controls import single_thread_cpu

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_command(arguments):
    """Record optional host information without changing system settings."""
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": type(error).__name__}


def machine_information():
    import psutil
    machine = {"system": platform.system(), "release": platform.release(),
        "macos": platform.mac_ver()[0], "machine": platform.machine(),
        "python": platform.python_version(), "python_compiler": platform.python_compiler(),
        "executable": sys.executable, "isolated_venv": sys.prefix != sys.base_prefix,
        "physical_cpus": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "background_workload_confirmed_by_user": False,
        "exclusive_core_or_fixed_frequency_established": False}
    if platform.system() == "Darwin":
        machine["cpu_model"] = _read_command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
        machine["power_profiles"] = _read_command(["/usr/bin/pmset", "-g", "custom"])
        machine["power_source"] = _read_command(["/usr/bin/pmset", "-g", "batt"])
    return machine


def exercise_operations(controls):
    """Exercise BLAS/LAPACK and our eager CPU LSTM without a fitting run."""
    import numpy as np
    import scipy.linalg
    import torch
    import yaml
    from models.lightgbm import build_parameters
    from models.lstm import TrafficLSTM

    left = np.array([[1., 2.], [3., 4.], [5., 6.]])
    right = np.array([[2., -1.], [1., 3.]])
    np.testing.assert_array_equal(left @ right, [[4., 5.], [10., 9.], [16., 13.]])
    # Both libraries may load their own LAPACK backend; exercise both paths.
    matrix = np.diag([3., 2., 1.])
    for svd in (np.linalg.svd, scipy.linalg.svd):
        u, values, vh = svd(matrix, full_matrices=False)
        np.testing.assert_allclose(values, [3., 2., 1.], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose((u * values) @ vh, matrix, rtol=1e-12, atol=1e-12)
    controls.verify()

    owner = (os.getpid(), threading.get_ident())
    hook_count = 0

    def check_gradient(gradient):
        nonlocal hook_count
        if (os.getpid(), threading.get_ident()) != owner:
            raise RuntimeError("CPU gradient hook ran outside the controlled calling thread")
        controls.verify()  # Includes Apple mode readback on the actual hook thread.
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("Nonfinite invented-input LSTM gradient")
        hook_count += 1
        return gradient

    # Preserve the caller's RNG state; no optimizer update or measured-data fit.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model = TrafficLSTM(16, 0.0)
        handles = [p.register_hook(check_gradient) for p in model.parameters()]
        try:
            sequence = torch.linspace(-1, 1, 2 * 168 * 2,
                                      dtype=torch.float32, device="cpu").reshape(2, 168, 2)
            calendar = torch.zeros((2, 4), dtype=torch.float32, device="cpu")
            predictions = model(sequence, calendar)
            if predictions.shape != (2,) or not bool(torch.isfinite(predictions).all()):
                raise RuntimeError("Invalid invented-input LSTM prediction")
            predictions.square().mean().backward()
            if hook_count != len(list(model.parameters())):
                raise RuntimeError("Not every LSTM parameter received a checked gradient")
        finally:
            for handle in handles:
                handle.remove()

    config = yaml.safe_load((ROOT / "run_config.yaml").read_text())
    tree_settings = config["models"]["trees"]
    parameters = build_parameters(tree_settings, tree_settings["tree_counts"][0],
                                  tree_settings["leaf_caps"][0])
    if parameters["num_threads"] != 1 or parameters["device_type"] != "cpu":
        raise RuntimeError("LightGBM configuration is not single-thread CPU")
    return {"numpy_matmul": True, "numpy_and_scipy_svd": True,
        "lstm_cpu_forward_and_backward": True, "gradient_hooks_checked": hook_count,
        "gradient_hooks_on_controlled_thread": True,
        "lightgbm_import_and_declared_controls_checked": True,
        "lightgbm_num_threads": parameters["num_threads"],
        "lightgbm_fit_performed": False, "optimizer_updates": 0,
        "traffic_data_loaded": False}


def check_controls(output_file):
    output = Path(output_file)
    if output.exists():
        raise ValueError("Use a new output file to preserve previous evidence")
    with single_thread_cpu() as controls:
        before = controls.verify()
        operations = exercise_operations(controls)
        after = controls.verify()
        import torch
        framework_build = {"configuration": torch.__config__.show(),
                           "parallel_information": torch.__config__.parallel_info()}
    # Write only after verification and successful context cleanup.
    result = {"status": "passed", "scope": "invented-input CPU-control diagnostic only",
        "controls_before": before, "controls_after": after, "operations": operations,
        "machine": machine_information(), "torch_build": framework_build,
        "versions": {name: metadata.version(name) for name in
            ["numpy", "scipy", "scikit-learn", "lightgbm", "torch", "threadpoolctl"]},
        "code_sha256": {name: _sha256(ROOT / name) for name in
            ["tools/cpu_controls.py", "tools/check_cpu_controls.py", "tools/run_cpu.py",
             "models/lstm.py", "models/lightgbm.py"]},
        "config_sha256": _sha256(ROOT / "run_config.yaml"),
        "formal_timing_eligible": False, "formal_timing_measured": False,
        "limitations": [
            "Readbacks verify documented library settings, not every internal kernel or OS thread.",
            "Apple BLAS/LAPACK mode belongs to the calling thread; each worker must configure itself.",
            "This process does not configure another process or certify machine workload conditions.",
            "The experiment runner, measurement boundaries and pilot remain separate gates."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = check_controls(args.output)
    print(json.dumps({"status": result["status"], "scope": result["scope"],
        "controls": result["controls_after"], "operations": result["operations"],
        "formal_timing_eligible": False, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
