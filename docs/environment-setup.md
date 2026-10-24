# Python environment

The project uses CPython 3.12.14 with pinned packages so changes in library versions do not silently change numerical behaviour. Use a dedicated virtual environment and the package file for the actual operating system and processor.

| Platform | Package file |
|---|---|
| Linux x86_64 | `requirements/linux-x86_64-py312.lock.txt` |
| Apple Silicon macOS | `requirements/macos-arm64-py312.candidate.txt` |
| Windows x86_64 | `requirements/windows-x86_64-py312.candidate.txt` |

The Linux file records the reference package environment. The macOS and Windows files retain their candidate labels; availability of a package file does not establish that it has been installed and tested on another computer. Intel Macs and other architectures need a separately checked package specification. Do not substitute a different platform's file.

The main packages are NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, LightGBM 4.6.0 and PyTorch 2.9.1. `requirements/core.in` records the direct dependencies; the platform files include dependency hashes. PyTorch implements the LSTM on the CPU.

Create `.venv` using a CPython 3.12.14 interpreter. On macOS, check the processor with `uname -m`; the Apple Silicon file requires `arm64`. LightGBM also needs the native OpenMP library. Where Homebrew is used, install it with `brew install libomp` and retain the installed version.

From the repository root, choose one package installation command:

```bash
# Linux x86_64
.venv/bin/python -m pip install --require-hashes -r requirements/linux-x86_64-py312.lock.txt --report install-report.json

# Apple Silicon macOS
.venv/bin/python -m pip install --require-hashes -r requirements/macos-arm64-py312.candidate.txt --report install-report.json
```

On Windows, use PowerShell:

```powershell
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements/windows-x86_64-py312.candidate.txt --report install-report.json
```

Check dependencies and the data-processing tests with the environment's interpreter:

```bash
.venv/bin/python -m pip check
.venv/bin/python tools/run_cpu.py -m pytest tests/test_preprocessing.py tests/test_data_validation.py tests/test_prepared_data.py -q
```

Windows uses `.venv\Scripts\python.exe` in place of `.venv/bin/python`. Keep installation reports local. If installation fails, retain the error and identify the incompatible package or native dependency before changing a pinned version.

`tools/run_cpu.py` sets numerical thread variables before Python loads the libraries. Each later experiment process must also apply its runtime thread controls. The launcher alone does not reserve a processor core or establish a quiet timing environment.

An optional installation check exercises the packages using small invented arrays:

```bash
.venv/bin/python tools/run_cpu.py -m tools.check_environment --output environment-check.json
```

This check does not read traffic data or measure forecasting accuracy. The recorded study used one Mac for comparative timing; a successful installation on another computer does not reproduce those timings. Use [the data instructions](../data/README.md) for local data and [the data contract](data-contract.md) for preparation behaviour.

See [running and checking the code](execution.md) for the test command, development entrypoints and the external evidence required by historical audits and continuation runs.
