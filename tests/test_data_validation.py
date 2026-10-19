"""Regression tests for actual data-integrity and leakage risks; builtin unittest."""
import tempfile
import copy
import os
import yaml
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from preprocessing.validate_data import check_config, protect_sources, check_tuning_panel
from preprocessing.prepare import load_raw, prepare, causal_fill
from preprocessing.windows import eligible_targets
from preprocessing.select_cells import prefix_stats, choose_panel, select_cells, weekly_scale


class DataValidationTests(unittest.TestCase):
    def test_tuning_panel_must_match_config_and_be_computable(self):
        config = {"tuning_panel": {"ids": ["Cell_1", "Cell_2"]}}
        check_tuning_panel(config, ["Cell_1", "Cell_2"], True)
        with self.assertRaises(ValueError):
            check_tuning_panel(config, ["Cell_2", "Cell_1"], True)
        with self.assertRaises(ValueError):
            check_tuning_panel(config, ["Cell_1", "Cell_2"], False)

    def load_text(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"raw.csv"
            path.write_text("Date,Hour,CellName,Traffic\n" + rows)
            return load_raw(path)

    def test_source_aliases_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"raw.csv"
            source.write_text("unchanged evidence")
            linked = Path(directory)/"hourly.csv"
            os.link(source, linked)
            with self.assertRaises(ValueError):
                protect_sources([linked], [source])
            self.assertEqual(source.read_text(), "unchanged evidence")
            with self.assertRaises(ValueError):
                protect_sources([source], [source])

    def test_config_drift_rejected(self):
        path = Path(__file__).resolve().parents[1]/"run_config.yaml"
        config = yaml.safe_load(path.read_text())
        check_config(config)
        for section, key, value in [("tuning_panel", "maximum_cells", 3),
                                    ("tuning_panel", "quantiles", [.25, .5, .75]),
                                    ("eligibility", "arbitrary_observed_fraction_cutoff", .95)]:
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                check_config(changed)
        changed = copy.deepcopy(config)
        changed["data"]["input_fill"]["development_check_caps"] = [0, 1]
        with self.assertRaises(ValueError):
            check_config(changed)

    def test_exact_duplicates_only(self):
        raw = self.load_text("2017-10-23,0,Cell_1,4\n"*2)
        self.assertEqual(len(raw), 1)
        self.assertEqual(raw.attrs["integrity"]["exact_duplicates"], 1)
        for second in ["5", "4.0"]:
            with self.assertRaises(ValueError):
                self.load_text("2017-10-23,0,Cell_1,4\n2017-10-23,0,Cell_1,"+second+"\n")

    def test_reject_invalid_source_fields(self):
        cases = ["2017-10-23,24,Cell_1,4", "2017-10-23,-1,Cell_1,4",
            "2017-10-23,0.5,Cell_1,4", "2017-02-30,0,Cell_1,4",
            "2017-10-22,0,Cell_1,4", "2017-10-23,0,,4",
            "2017-10-23,0,Cell_1,inf", "2017-10-23,0,Cell_1,-1",
            "2017-10-23,0,Cell_1,", "2017-10-23,0,Cell_1,NaN"]
        for row in cases:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.load_text(row+"\n")

    def test_original_and_filled_columns_are_distinct(self):
        raw = self.load_text("2017-10-23,1,Cell_1,4\n2017-10-23,6,Cell_1,8\n")
        out = prepare(raw)
        self.assertEqual(len(out), 8760)
        self.assertTrue(np.isnan(out.traffic.iloc[0]))
        np.testing.assert_equal(out.traffic.iloc[1:7].to_numpy(), [4, 4, 4, 4, np.nan, 8])
        self.assertEqual(out.traffic_observed.notna().sum(), 2)
        self.assertEqual(out.observed.sum(), 2)

    def test_fill_is_causal_and_does_not_reset_at_split(self):
        values = np.array([np.nan, 2, np.nan, np.nan, np.nan, np.nan, 9])
        full, mask = causal_fill(values)
        for end in range(1, len(values)+1):
            partial, _ = causal_fill(values[:end])
            np.testing.assert_equal(full[:end], partial)
        self.assertEqual(full[4], 2)
        self.assertTrue(np.isnan(full[5]))
        np.testing.assert_equal(mask, [0, 1, 0, 0, 0, 0, 1])
        unfilled, _ = causal_fill(values, 0)
        np.testing.assert_equal(unfilled, values)

    def test_common_mask_observed_target_and_past_only(self):
        values = [1, 2, 2, 4, np.nan, 6, 7, 8]
        observed = [1, 1, 0, 1, 0, 1, 1, 1]
        np.testing.assert_equal(eligible_targets(values, observed, 2),
                               [0, 0, 0, 1, 0, 0, 0, 1])
        for end in range(1, len(values)+1):
            np.testing.assert_equal(eligible_targets(values, observed, 2)[:end],
                eligible_targets(values[:end], observed[:end], 2))

    def test_bad_array_shapes_flags_and_nonfinite_rejected(self):
        for values, flags in [([[1], [2]], [1, 1]), ([1, 2], [1]),
                               ([1, 2], [1, 2]), ([1, np.inf], [1, 0]),
                               ([1, np.nan], [1, 1])]:
            with self.assertRaises(ValueError):
                eligible_targets(values, flags, 1)
        for cap in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                causal_fill([1, np.nan], cap)

    def test_prefix_stats_exclude_fills_and_future_values(self):
        values = np.arange(400, dtype=float)
        flags = np.ones(400, int)
        flags[10] = 0
        values[10] = 100000  # filled inputs must not enter SD or weekly pairs
        stats = prefix_stats(values, flags, 200)
        expected = np.delete(np.arange(200, dtype=float), 10)
        self.assertEqual(stats["mean"], expected.mean())
        self.assertEqual(stats["sd"], expected.std(ddof=0))
        self.assertEqual(stats["weekly_scale"], 168)
        values[200:] *= 999
        self.assertEqual(stats, prefix_stats(values, flags, 200))
        self.assertTrue(np.isnan(weekly_scale(values, flags, 100)))

    def test_no_arbitrary_coverage_cutoff(self):
        values = np.arange(1000, dtype=float)
        flags = np.ones(1000, int)
        flags[:100] = 0
        values[:100] = np.nan
        self.assertEqual(prefix_stats(values, flags, 1000)["exclusion_reason"], "")

    def test_panel_small_empty_and_full_precision_ties(self):
        small = pd.DataFrame({"cell": ["B", "A"], "cv": [1., 2.], "exclusion_reason": ["", ""]})
        self.assertEqual(choose_panel(small).cell.tolist(), ["A", "B"])
        self.assertTrue(choose_panel(small.iloc[:0]).empty)
        pool = pd.DataFrame({"cell": [f"C{i}" for i in range(10)],
                             "cv": np.linspace(1, 1.000009, 10), "exclusion_reason": [""]*10})
        result = choose_panel(pool)
        # Premature rounding would make every CV equal and choose C0 first.
        self.assertEqual(result.cell.iloc[0], "C1")
        pool["cv"] = 1.
        self.assertEqual(choose_panel(pool).cell.tolist(), [f"C{i}" for i in range(9)])

    def test_tuning_pool_independent_of_later_main_quality(self):
        values = np.arange(8760, dtype=float)
        observed = np.ones(8760, int)
        table = pd.DataFrame({"cell": "A", "ts": pd.date_range("2017-10-23", periods=8760, freq="h"),
                              "traffic": values, "observed": observed})
        _, early, panel = select_cells(table)
        table.loc[4116:, "traffic"] = 1e9
        _, changed_early, changed_panel = select_cells(table)
        pd.testing.assert_frame_equal(early, changed_early)
        pd.testing.assert_frame_equal(panel, changed_panel)


if __name__ == "__main__":
    unittest.main()
