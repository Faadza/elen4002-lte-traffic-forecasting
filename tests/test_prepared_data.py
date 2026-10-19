"""Reject altered files, inconsistent configuration and loss of precision."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.splits import CALENDAR_START, N_HOURS
from preprocessing.validate_data import sha256, write_json
from tools.check_prepared_data import load_validated_data, verify_files


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        for name in ["data", "docs", "preprocessing"]:
            (self.root / name).mkdir()
        (self.root / "run_config.yaml").write_text("fixture configuration\n")
        (self.root / "preprocessing/prepare.py").write_text("fixture source\n")
        self.values = np.arange(N_HOURS, dtype=float) + .30000000000000004
        self.values[20:25] = np.nan
        inputs = self.values.copy()
        inputs[20:23] = self.values[19]
        self.hourly = pd.DataFrame({"cell": "Cell_1",
            "ts": pd.date_range(CALENDAR_START, periods=N_HOURS, freq="h"),
            "traffic_observed": self.values, "traffic": inputs,
            "observed": np.isfinite(self.values).astype(int)})
        self.hourly.to_csv(self.data / "hourly.csv", index=False, float_format="%.17g")
        write_json(self.data / "validation_report.json", {"raw_unchanged": True,
            "config_sha256": sha256(self.root / "run_config.yaml"),
            "tuning_masks_computable": True, "grid_rows": N_HOURS, "integrity": {"cells": 1},
            "coverage": [{"cap_hours": 3, "period": "development", "eligible_targets": 6132-193},
                {"cap_hours": 3, "period": "confirmation", "eligible_targets": 1314},
                {"cap_hours": 3, "period": "final", "eligible_targets": 1314}]})
        self.reference = {"config_sha256": sha256(self.root / "run_config.yaml"),
            "code_sha256": {"prepare.py": sha256(self.root / "preprocessing/prepare.py")}}
        self.record_files()

    def record_files(self):
        self.reference["files"] = {p.name: sha256(p) for p in self.data.iterdir()}
        write_json(self.root / "docs/data-bundle-checksums.json", self.reference)

    def test_exact_values_and_manually_counted_target_mask(self):
        loaded, _ = load_validated_data(self.data, self.root)
        np.testing.assert_array_equal(loaded.traffic_observed.to_numpy(), self.values)
        expected = np.arange(N_HOURS) >= 193
        np.testing.assert_array_equal(loaded.eligible_target.to_numpy(), expected)
        self.assertEqual(loaded.traffic_observed.iloc[0], .30000000000000004)

    def test_modified_or_missing_file_rejected(self):
        path = self.data / "hourly.csv"
        with path.open("a") as handle:
            handle.write("\n")
        with self.assertRaises(ValueError):
            verify_files(self.data, self.root)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            verify_files(self.data, self.root)

    def test_changed_configuration_and_source_rejected(self):
        path = self.root / "run_config.yaml"
        path.write_text("different rules\n")
        with self.assertRaises(ValueError):
            verify_files(self.data, self.root)
        path.write_text("fixture configuration\n")
        (self.root / "preprocessing/prepare.py").write_text("different source\n")
        with self.assertRaises(ValueError):
            verify_files(self.data, self.root)

    def use_current_configuration(self):
        # The original report still identifies the configuration used to prepare it.
        path = self.root / "run_config.yaml"
        path.write_text("current specification without execution notes\n")
        self.reference["runtime_config_sha256"] = sha256(path)
        self.record_files()

    def test_current_configuration_loads_unchanged_original_bundle(self):
        original_files = self.reference["files"].copy()
        original_config = self.reference["config_sha256"]
        self.use_current_configuration()
        loaded, report = load_validated_data(self.data, self.root)
        np.testing.assert_array_equal(loaded.traffic_observed.to_numpy(), self.values)
        self.assertEqual(self.reference["files"], original_files)
        self.assertEqual(report["config_sha256"], original_config)
        self.assertNotEqual(self.reference["runtime_config_sha256"], original_config)

    def test_changed_current_configuration_rejected(self):
        self.use_current_configuration()
        (self.root / "run_config.yaml").write_text("different scientific settings\n")
        with self.assertRaisesRegex(ValueError, "Current configuration"):
            verify_files(self.data, self.root)

    def test_current_configuration_requires_its_own_hash(self):
        self.use_current_configuration()
        del self.reference["runtime_config_sha256"]
        self.record_files()
        with self.assertRaisesRegex(ValueError, "Current configuration"):
            verify_files(self.data, self.root)

    def test_current_hash_cannot_replace_original_report_configuration(self):
        self.use_current_configuration()
        path = self.data / "validation_report.json"
        import json
        report = json.loads(path.read_text())
        report["config_sha256"] = self.reference["runtime_config_sha256"]
        write_json(path, report)
        self.record_files()
        with self.assertRaisesRegex(ValueError, "different source configuration"):
            verify_files(self.data, self.root)

    def test_missing_original_report_configuration_rejected(self):
        self.use_current_configuration()
        path = self.data / "validation_report.json"
        import json
        report = json.loads(path.read_text())
        del report["config_sha256"]
        write_json(path, report)
        self.record_files()
        with self.assertRaisesRegex(ValueError, "different source configuration"):
            verify_files(self.data, self.root)

    def test_current_configuration_keeps_original_preparation_code_check(self):
        self.use_current_configuration()
        (self.root / "preprocessing/prepare.py").write_text("different preparation\n")
        with self.assertRaisesRegex(ValueError, "Preparation code changed"):
            verify_files(self.data, self.root)

    def test_calendar_error_rejected_even_if_file_hash_is_refreshed(self):
        self.hourly.loc[1, "ts"] = self.hourly.ts.iloc[0]
        self.hourly.to_csv(self.data / "hourly.csv", index=False, float_format="%.17g")
        self.record_files()
        with self.assertRaises(ValueError):
            load_validated_data(self.data, self.root)

    def test_imputed_target_rejected_even_if_file_hash_is_refreshed(self):
        self.hourly.loc[20, "observed"] = 1
        self.hourly.to_csv(self.data / "hourly.csv", index=False, float_format="%.17g")
        self.record_files()
        with self.assertRaises(ValueError):
            load_validated_data(self.data, self.root)

    def test_noncausal_fill_rejected_even_if_file_hash_is_refreshed(self):
        self.hourly.loc[20, "traffic"] = self.hourly.traffic_observed.iloc[25]
        self.hourly.to_csv(self.data / "hourly.csv", index=False, float_format="%.17g")
        self.record_files()
        with self.assertRaises(ValueError):
            load_validated_data(self.data, self.root)


if __name__ == "__main__":
    unittest.main()
