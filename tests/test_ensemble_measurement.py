"""Invented workloads: missing latency, complete seed panels and cost arithmetic."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from ensemble_study.measurement import aggregate_costs as _aggregate_costs, measure_predictions_allow_empty
from ensemble_study.evaluation_plan import BASELINES, expected_conditions, jobs_for

CONDITIONS = expected_conditions({"l1": {"trees": 32, "leaves": 31}, "l2": {"trees": 128, "leaves": 7}}, {"l1": .3, "l2": .2})
METHODS = [c["id"] for c in CONDITIONS]

def aggregate_costs(records, cells, period):
    return _aggregate_costs(records, cells, period, CONDITIONS)
from tools.pilot_measurements import timing_summary


def memory(value):
    return {"process_peak_rss_bytes": value, "incremental_model_memory": False}


def records(period="final", cells=("A", "B"), empty=False):
    result = []
    for job in jobs_for(period, list(cells), CONDITIONS):
        cell_number = cells.index(job["cell"]) + 1
        repeat = job["repeat"]
        baseline = job["method"] in BASELINES
        fit = {"fit_seconds": None if baseline else cell_number * repeat,
               "fit_memory": None if baseline else memory(1000 * cell_number * repeat),
               "artifact_bytes": cell_number * 100 + repeat}
        prediction = None
        if job["predict"]:
            single = [] if empty else [[cell_number / 100 + repeat / 1000, cell_number / 100 + repeat / 1000 + .01]] * 3
            flat = [x for row in single for x in row]
            batch = [1., 2., 3.]
            prediction = {"prediction_memory": memory(500 + cell_number), "timings": {
                "warmup_passes": 1, "timed_passes": 3, "target_count": 10 + cell_number,
                "origins": [] if empty else [10, 20], "origin_count": 0 if empty else 2,
                "batch_seconds": batch, "batch_summary": timing_summary(batch),
                "single_origin_seconds": single,
                "single_origin_summary": timing_summary(flat) if flat else None,
                "single_vs_batch_checked": not empty,
                "identical_workload_repeats_equal": True}}
        result.append({"job": job, "fit": fit, "predict": prediction})
    return result


class MeasurementTests(unittest.TestCase):
    def test_empty_origins_retains_batch_work_without_substitution(self):
        seen = []
        def predict(targets):
            seen.append(targets.tolist())
            raw = targets.astype(float) - 11.
            return raw, np.maximum(raw, 0.)
        outcome = measure_predictions_allow_empty(predict, np.array([10, 11, 12]), [], rtol=0., atol=0.)
        self.assertEqual(seen, [[10, 11, 12]] * 4)
        self.assertEqual(len(outcome["timings"]["batch_seconds"]), 3)
        self.assertIsNone(outcome["timings"]["single_origin_summary"])
        self.assertEqual(outcome["timings"]["origins"], [])
        self.assertFalse(outcome["timings"]["single_vs_batch_checked"])
        np.testing.assert_array_equal(outcome["final"], [0., 0., 1.])

    def test_available_origins_delegate_to_verified_pilot(self):
        wanted = {"raw": np.array([1.]), "final": np.array([1.]), "timings": {}}
        with patch("ensemble_study.measurement.measure_predictions", return_value=wanted) as routine:
            got = measure_predictions_allow_empty(object(), np.array([10]), [10], rtol=.1, atol=.2)
        self.assertIs(got, wanted)
        routine.assert_called_once()
        self.assertEqual(got["timings"]["single_origin_status"], "available")

    def test_empty_origin_path_checks_exact_repeats(self):
        calls = 0
        def changing(_):
            nonlocal calls
            calls += 1
            return np.array([calls]), np.array([calls])
        with self.assertRaises(AssertionError):
            measure_predictions_allow_empty(changing, [10], [], rtol=0., atol=0.)

    def test_empty_origin_path_rejects_bad_clipping_and_tolerance(self):
        with self.assertRaises(ValueError):
            measure_predictions_allow_empty(lambda _: (np.array([-1.]), np.array([-1.])), [10], [], rtol=0., atol=0.)
        with self.assertRaises(ValueError):
            measure_predictions_allow_empty(lambda _: None, [10], [], rtol=True, atol=0.)


class CostTests(unittest.TestCase):
    def test_complete_final_sums_and_prespecified_artifacts(self):
        costs = aggregate_costs(records(), ["A", "B"], "final")
        self.assertEqual(set(costs), set(METHODS))
        ridge = costs["ridge"]
        self.assertEqual(ridge["fit_observation_count"], 6)
        self.assertEqual(ridge["panel_fit_seconds"]["observations"], [3., 6., 9.])
        self.assertEqual(ridge["panel_fit_seconds"]["summary"]["median_seconds"], 6.)
        self.assertEqual(ridge["deployed_artifact_bytes"], 302)
        self.assertEqual(ridge["single_origin_latency_summary"]["count"], 12)
        self.assertAlmostEqual(ridge["comparison_latency_seconds"], .021)
        self.assertEqual(ridge["batch_target_counts"], {"A": 11, "B": 12})
        lstm = costs["lstm"]
        self.assertEqual(lstm["single_origin_latency_summary"]["count"], 36)
        self.assertEqual(lstm["panel_fit_seconds"]["labels"], [{"seed": 2026}, {"seed": 2027}, {"seed": 2028}])
        self.assertEqual(lstm["deployed_artifact_bytes"], 302)
        self.assertEqual(lstm["per_cell"][0]["fit_process_peak_rss"]["maximum_bytes"], 3000)
        self.assertIsNone(costs["persistence"]["panel_fit_seconds"]["summary"])

    def test_confirmation_has_one_deterministic_fit_three_seeds(self):
        data = records("confirmation")
        costs = aggregate_costs({row["job"]["id"]: row for row in data}, ["A", "B"], "confirmation")
        self.assertEqual(costs["ridge"]["fit_observation_count"], 2)
        self.assertEqual(costs["lstm"]["fit_observation_count"], 6)
        self.assertFalse(costs["ridge"]["formal_complete_model_cost"])
        self.assertEqual(costs["ridge"]["fit_status"], "confirmation_descriptive_only")

    def test_no_common_origin_means_no_latency_for_any_method(self):
        costs = aggregate_costs(records(empty=True), ["A", "B"], "final")
        for cost in costs.values():
            self.assertIsNone(cost["comparison_latency_seconds"])
            self.assertEqual(cost["common_latency_origin_count"], 0)
            self.assertEqual(len(cost["per_cell"][0]["prediction"][0]["timings"]["batch_seconds"]), 3)

    def test_missing_cell_repeat_seed_and_duplicate_rejected(self):
        original = records()
        for altered in (original[:-1], original + [original[0]],
                        [r for r in original if r["job"]["cell"] != "B"]):
            with self.assertRaises(ValueError):
                aggregate_costs(altered, ["A", "B"], "final")

    def test_changed_frozen_job_rejected(self):
        data = records()
        data[-1]["job"]["seed"] = 9000
        with self.assertRaises(ValueError):
            aggregate_costs(data, ["A", "B"], "final")

    def test_missing_prediction_or_extra_repeat_prediction_rejected(self):
        data = records()
        data[0]["predict"] = None
        with self.assertRaises(ValueError):
            aggregate_costs(data, ["A", "B"], "final")
        data = records()
        ref = copy.deepcopy(data[0]["predict"])
        next(r for r in data if r["job"]["method"] == "ridge" and r["job"]["repeat"] == 2)["predict"] = ref
        with self.assertRaises(ValueError):
            aggregate_costs(data, ["A", "B"], "final")

    def test_global_origins_and_cell_target_counts_are_common(self):
        for key, value in (("origins", [11, 20]), ("target_count", 99)):
            data = records()
            next(r for r in reversed(data) if r["predict"] is not None)["predict"]["timings"][key] = value
            with self.assertRaises(ValueError):
                aggregate_costs(data, ["A", "B"], "final")

    def test_invalid_memory_or_duration_is_rejected(self):
        for key, value in (("fit_seconds", float("nan")), ("fit_seconds", True),
                           ("fit_memory", {"process_peak_rss_bytes": 1000, "incremental_model_memory": True}),
                           ("artifact_bytes", -1)):
            data = records()
            next(r for r in data if r["job"]["method"] == "ridge")["fit"][key] = value
            with self.assertRaises(ValueError):
                aggregate_costs(data, ["A", "B"], "final")

    def test_tampered_latency_summary_and_missing_pass_rejected(self):
        data = records()
        data[0]["predict"]["timings"]["single_origin_summary"]["median_seconds"] = 0
        with self.assertRaises(ValueError):
            aggregate_costs(data, ["A", "B"], "final")
        data = records()
        data[0]["predict"]["timings"]["single_origin_seconds"].pop()
        with self.assertRaises(ValueError):
            aggregate_costs(data, ["A", "B"], "final")


if __name__ == "__main__":
    unittest.main()
