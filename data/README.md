# Local traffic data

The pipeline uses the original hourly traffic CSV and keeps both raw and prepared data outside Git. The source record and its limitations are in [data_manifest.json](../data_manifest.json). Use an authorised local copy with SHA-256:

```text
2578a993553c0f21b00a5d2ae9cd84312bc33d692b0df00dc789016892f6c872
```

Place the original CSV at `data/raw/combined_data.csv`. To generate preparation outputs in a separate directory, run from the repository root:

```bash
.venv/bin/python tools/run_cpu.py -m preprocessing.validate_data data/raw/combined_data.csv --output-dir data/processed/validation
```

The command checks the source hash, rejects invalid or conflicting observations, constructs complete hourly calendars and writes the prepared values, masks and validation report. It leaves the original CSV unchanged. [The data contract](../docs/data-contract.md) defines the columns and target eligibility rule.

To load the existing validated bundle, place its 11 files at `data/processed/validated` and run:

```bash
.venv/bin/python tools/run_cpu.py -m tools.check_prepared_data data/processed/validated
```

The loader checks the original bundle against [its recorded checksums](../docs/data-bundle-checksums.json). The current configuration has a separate checksum because execution notes were removed without changing the scientific settings. A newly generated validation report records the current configuration hash, so it is a new output and must not replace the original bundle or its recorded checksums.

Traffic is reported in dataset units. Physical conversion, timezone, original-release identity and redistribution rights have not been independently established. Keep the data and archives local.
