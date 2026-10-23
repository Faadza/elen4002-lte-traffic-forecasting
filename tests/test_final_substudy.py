"""Invented-input checks for development-only arithmetic and evidence binding."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from final_evaluation import substudy


class SubstudyArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.reference = [dict(cell="A", fold_index="0", target_index=str(4116+i), actual=str(a))
                          for i, a in enumerate((10, 20))]
        self.rows = [dict(cell="A", method="ridge_lightgbm_hybrid", seed="",
                          target_index=str(4116+i), actual=str(a),
                          prediction_raw=str(p), prediction=str(max(0, p)))
                     for i, (a, p) in enumerate(((10, 8), (20, 24)))]

    def score(self, rows=None):
        return substudy.score_forecast_rows(self.rows if rows is None else rows,
                                            self.reference, "A", 0, 2)

    def test_error_arithmetic_uses_absolute_error_and_training_scale(self):
        result = self.score()
        self.assertEqual(result["MAE"], 3)
        self.assertAlmostEqual(result["RMSE"], 10**.5)
        self.assertEqual(result["MASE"], 1.5)

    def test_clipping_happens_only_on_saved_final_prediction(self):
        rows = copy.deepcopy(self.rows)
        rows[0].update(prediction_raw="-2", prediction="0")
        self.assertEqual(self.score(rows)["MAE"], 7)
        rows[0]["prediction"] = "-2"
        with self.assertRaisesRegex(ValueError, "clipping"):
            self.score(rows)

    def test_rejects_future_target(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["target_index"] = "6132"
        with self.assertRaisesRegex(ValueError, "outside"):
            self.score(rows)

    def test_rejects_duplicate_or_reordered_targets(self):
        with self.assertRaises(ValueError):
            self.score(list(reversed(self.rows)))
        rows = copy.deepcopy(self.rows)
        rows[1]["target_index"] = rows[0]["target_index"]
        with self.assertRaises(ValueError):
            self.score(rows)

    def test_rejects_target_value_change_or_missing_forecast(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["actual"] = "12"
        with self.assertRaisesRegex(ValueError, "target value"):
            self.score(rows)
        with self.assertRaisesRegex(ValueError, "length"):
            self.score(self.rows[:-1])

    def test_rejects_nonfinite_prediction_and_invalid_scale(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["prediction_raw"] = "nan"
        with self.assertRaises(ValueError):
            self.score(rows)
        with self.assertRaisesRegex(ValueError, "denominator"):
            substudy.score_forecast_rows(self.rows, self.reference, "A", 0, 0)

    def test_paired_cell_wins_are_averaged_across_folds(self):
        # A wins two folds but loses the cell mean; B wins the cell mean.
        base, model = [], []
        for cell, values in (("A", (1, 1, 7)), ("B", (1, 1, 1))):
            for fold, value in enumerate(values):
                base.append(dict(cell=cell, fold_index=fold, MAE=2, RMSE=4, MASE=2))
                model.append(dict(cell=cell, fold_index=fold, MAE=value, RMSE=2*value, MASE=value))
        result = substudy.summarize_candidate(model, list(reversed(base)), ["A", "B"])
        self.assertEqual(result["scores"]["MASE"], 2)
        self.assertEqual(result["scores"]["RMSE"], 4)
        self.assertEqual(result["cells_better_than_direct"], 1)
        self.assertEqual(result["cells_worse_than_direct"], 1)
        self.assertEqual(result["cohorts_better_than_direct"], 5)
        self.assertEqual(result["cohorts_worse_than_direct"], 1)
        self.assertEqual(result["folds_better_than_direct"], 2)

    def test_missing_cohort_cannot_improve_a_candidate(self):
        records = [dict(cell="A", fold_index=f, MAE=1, RMSE=1, MASE=1) for f in range(3)]
        with self.assertRaisesRegex(ValueError, "cohort"):
            substudy.summarize_candidate(records[:-1], records, ["A"])


class SubstudyEvidenceTests(unittest.TestCase):
    def test_unbound_and_tampered_original_reads_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            path = Path(directory)/"plan.json"
            path.write_bytes(b"original")
            reader = substudy.InputReader(directory, {"plan.json": substudy.digest(b"original")})
            self.assertEqual(reader.read("plan.json"), b"original")
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                reader.verify_unchanged()
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                reader.read("../plan.json")

    def test_replay_complete_cannot_be_replaced_with_new_self_consistent_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            (Path(directory)/"complete.json").write_text(json.dumps({"status":"passed"}))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                substudy.load_verified_replay(directory)

    def test_output_cannot_overwrite_input_or_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original, replay = root/"original", root/"replay"
            original.mkdir(); replay.mkdir()
            with patch.object(substudy, "build_substudy") as builder:
                with self.assertRaisesRegex(ValueError, "outside"):
                    substudy.write_substudy(original, replay, original/"new.json")
                output = root/"summary.json"
                output.write_text("keep")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    substudy.write_substudy(original, replay, output)
                self.assertEqual(output.read_text(), "keep")
                builder.assert_not_called()

    def test_reused_direct_endpoint_has_only_ties(self):
        records = [dict(cell="A", fold_index=f, MAE=1+f, RMSE=2+f, MASE=1+f) for f in range(3)]
        result = substudy.summarize_candidate(records, records, ["A"])
        self.assertEqual(result["cells_tied_with_direct"], 1)
        self.assertEqual(result["cohorts_tied_with_direct"], 3)
        self.assertEqual(result["relative_mase_change_percent_vs_direct"], 0)


if __name__ == "__main__":
    unittest.main()
