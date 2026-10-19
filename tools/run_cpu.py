"""Launch Python with the declared CPU/thread settings before library imports.

Example: python tools/run_cpu.py -m tools.check_environment --output report.json
This launcher controls execution settings; it does not authorise experiments.
"""
import os
import subprocess
import sys

CPU_ENV = {
    "PYTHONHASHSEED": "2026",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "CUDA_VISIBLE_DEVICES": "-1",
    "MPLBACKEND": "Agg",
}


def main():
    if len(sys.argv) == 1:
        raise SystemExit("Usage: python tools/run_cpu.py -m MODULE [arguments]")
    environment = os.environ.copy()
    environment.update(CPU_ENV)
    # A child process receives these controls before any numerical library imports.
    raise SystemExit(subprocess.run([sys.executable, *sys.argv[1:]], env=environment).returncode)


if __name__ == "__main__":
    main()
