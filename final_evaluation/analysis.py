"""Pure scoring of saved forecasts; never fits or chooses model settings.

All arrays cover the complete evaluation calendar before masks. The runner
provides one shared target mask and fitting-only weekly scale per cell. LSTM
seeds are scored separately and their scores averaged, not their forecasts.
"""

from collections.abc import Mapping
import math

import numpy as np

from evaluation.metrics import busiest_hour_measures, finalize_predictions, mae, rmse


STANDALONES = (
    "persistence", "daily_seasonal_naive", "weekly_seasonal_naive",
    "ridge", "direct_lightgbm", "lstm",
)
METRICS = ("MAE", "RMSE", "MASE")


def _cells(expected_cells):
    if isinstance(expected_cells, str):
        raise ValueError("expected_cells must be a sequence")
    cells = list(expected_cells)
    if not cells or any(not isinstance(c, str) or not c for c in cells):
        raise ValueError("expected_cells must contain nonempty cell IDs")
    if len(set(cells)) != len(cells):
        raise ValueError("expected_cells must be unique")
    return cells


def _mask(value, size, name):
    value = np.asarray(value)
    if value.shape != (size,) or not np.isin(value, [0, 1]).all():
        raise ValueError(f"{name} must be a matching 0/1 mask")
    return value.astype(bool)


def _targets(actual, eligible):
    actual = np.asarray(actual, dtype=np.float64)
    if actual.ndim != 1 or np.isinf(actual).any():
        raise ValueError("actual must be a one-dimensional calendar without infinities")
    eligible = _mask(eligible, len(actual), "eligible")
    if not np.isfinite(actual[eligible]).all():
        raise ValueError("actual must be finite at every eligible target")
    return actual, eligible


def _scale(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 0:
        raise ValueError("weekly_scale must be a scalar")
    return float(value) if np.isfinite(value) and value > 0 else None


def _predictions(predictions, eligible, expected_seeds=None):
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("raw_predictions_by_seed must be a nonempty mapping")
    result = {}
    for seed, values in predictions.items():
        key = str(seed)
        if key in result:
            raise ValueError("seed identifiers must remain unique as strings")
        values = np.asarray(values, dtype=np.float64)
        if values.shape != eligible.shape or np.isinf(values).any():
            raise ValueError("each forecast must match the complete calendar without infinities")
        if not np.isfinite(values[eligible]).all():
            raise ValueError("failed or missing eligible forecast; do not prune targets")
        result[key] = values
    if expected_seeds is not None:
        expected = [str(seed) for seed in expected_seeds]
        if len(set(expected)) != len(expected) or set(result) != set(expected):
            raise ValueError("forecast seeds do not match the frozen seed list")
    return result


def _scores(actual, final, eligible, scale):
    if not eligible.any():
        return {metric: None for metric in METRICS}
    absolute = mae(actual[eligible], final[eligible])
    return {
        "MAE": absolute,
        "RMSE": rmse(actual[eligible], final[eligible]),
        "MASE": absolute / scale if scale is not None else None,
    }


def _mean(values):
    values = list(values)
    if not values or any(value is None for value in values):
        return None
    if not np.isfinite(values).all():
        raise ValueError("nonfinite computed score")
    return float(np.mean(values))


def score_cell(timestamps, actual, observed, eligible, raw_predictions_by_seed,
               weekly_scale, fully_observed_history=None, *, expected_seeds=None):
    """Score every seed on identical targets and return JSON-safe diagnostics.

    ``fully_observed_history`` is a data-only full-calendar boolean mask: all
    168 preceding inputs were genuinely observed. It is intersected with the
    original eligibility mask only for the separate sensitivity calculation.
    An undefined scale/empty target cohort is explicit, never an epsilon.
    """
    actual, eligible = _targets(actual, eligible)
    observed = _mask(observed, len(actual), "observed")
    if np.any(eligible & ~observed) or not np.isfinite(actual[observed]).all():
        raise ValueError("targets must be genuine finite observations")
    scale = _scale(weekly_scale)
    raw = _predictions(raw_predictions_by_seed, eligible, expected_seeds)
    sensitivity = None if fully_observed_history is None else (
        eligible & _mask(fully_observed_history, len(actual), "fully_observed_history")
    )
    seeds = {}
    for seed, values in raw.items():
        final = np.full(len(actual), np.nan)
        clipping = {"prediction_count": 0, "clipped_count": 0,
                    "clipped_fraction": None, "mean_upward_adjustment": None}
        if eligible.any():
            final[eligible], clipping = finalize_predictions(values[eligible])
        busiest = busiest_hour_measures(timestamps, actual, observed, eligible, final)
        seeds[seed] = {
            "scores": _scores(actual, final, eligible, scale),
            "busiest_hour": busiest,
            "clipping": clipping,
            "fully_observed_history_scores": None if sensitivity is None else
                _scores(actual, final, sensitivity, scale),
        }
    count, observed_count = int(eligible.sum()), int(observed.sum())
    first = next(iter(seeds.values()))
    busiest = dict(first["busiest_hour"])
    for key in ("absolute_error", "signed_error"):
        busiest[key] = _mean(s["busiest_hour"][key] for s in seeds.values())
    return {
        "status": "available" if count else "no_eligible_targets",
        "scores": {m: _mean(s["scores"][m] for s in seeds.values()) for m in METRICS},
        "weekly_scale": scale,
        "mase_status": "available" if count and scale is not None else
            ("no_eligible_targets" if not count else "unavailable_fitting_scale"),
        "seed_count": len(seeds),
        "seed_scores": seeds,
        "coverage": {
            "calendar_hours": len(actual), "observed_targets": observed_count,
            "eligible_targets": count,
            "fraction_observed_scored": count / observed_count if observed_count else None,
            "fraction_calendar_scored": count / len(actual) if len(actual) else None,
        },
        "busiest_hour": busiest,
        "clipping": {key: _mean(s["clipping"][key] for s in seeds.values())
                     for key in first["clipping"]},
        "fully_observed_history": {
            "status": "not_supplied" if sensitivity is None else
                ("available" if sensitivity.any() else "no_eligible_targets"),
            "target_count": None if sensitivity is None else int(sensitivity.sum()),
            "scores": None if sensitivity is None else {
                m: _mean(s["fully_observed_history_scores"][m] for s in seeds.values())
                for m in METRICS
            },
        },
    }


def aggregate_method(cell_results, expected_cells):
    """Equal-cell summaries with named structural exclusions and contributor IDs.

    Missing cells/seeds raise. A data-unavailable scale or empty cohort is
    reported separately and can never be treated as a failed model's omission.
    Model comparisons must verify identical metric contributor IDs.
    """
    cells = _cells(expected_cells)
    if not isinstance(cell_results, Mapping) or set(cell_results) != set(cells):
        raise ValueError("cell_results must contain the complete declared panel")
    seed_sets = [set(cell_results[cell]["seed_scores"]) for cell in cells]
    if any(seeds != seed_sets[0] or not seeds for seeds in seed_sets):
        raise ValueError("all cells require the same complete seed list")
    contributors = {m: [c for c in cells if cell_results[c]["scores"][m] is not None]
                    for m in METRICS}
    excluded = {m: {c: cell_results[c]["mase_status"] if m == "MASE" else
                    cell_results[c]["status"] for c in cells if c not in contributors[m]}
                for m in METRICS}
    for metric in METRICS:
        if any(reason not in {"no_eligible_targets", "unavailable_fitting_scale"}
               for reason in excluded[metric].values()):
            raise ValueError("a missing score cannot be dropped as model failure")
    busiest_cells = [c for c in cells if cell_results[c]["busiest_hour"]["status"] == "available"]
    sensitivity = {}
    for metric in METRICS:
        members = [c for c in cells if cell_results[c]["fully_observed_history"]["scores"]
                   is not None and cell_results[c]["fully_observed_history"]["scores"][metric]
                   is not None]
        sensitivity[metric] = {
            "score": _mean(cell_results[c]["fully_observed_history"]["scores"][metric]
                           for c in members), "contributors": members,
        }
    return {
        "status": "complete",
        "cell_count": len(cells), "expected_cells": cells,
        "seed_count": len(seed_sets[0]),
        "scores": {m: _mean(cell_results[c]["scores"][m] for c in contributors[m])
                   for m in METRICS},
        "contributors": contributors, "structural_exclusions": excluded,
        "busiest_hour": {
            **{key: _mean(cell_results[c]["busiest_hour"][key] for c in busiest_cells)
               for key in ("absolute_error", "signed_error")},
            "contributors": busiest_cells,
        },
        "clipping": {
            key: _mean(cell_results[c]["clipping"][key] for c in contributors["MAE"])
            for key in ("clipped_fraction", "mean_upward_adjustment")
        },
        "fully_observed_history": sensitivity,
        "cells": {c: cell_results[c] for c in cells},
    }


def compare_methods(hybrid_summary, comparator_summary):
    """Describe paired cell differences; negative means lower hybrid error."""
    if hybrid_summary["expected_cells"] != comparator_summary["expected_cells"]:
        raise ValueError("comparison requires the same declared cell panel")
    result = {}
    for metric in METRICS:
        cells = hybrid_summary["contributors"][metric]
        if cells != comparator_summary["contributors"][metric]:
            raise ValueError("comparison requires identical common metric contributors")
        differences = {
            c: hybrid_summary["cells"][c]["scores"][metric]
            - comparator_summary["cells"][c]["scores"][metric] for c in cells
        }
        result[metric] = {
            "hybrid_minus_comparator": _mean(differences.values()),
            "fraction_cells_improved": (sum(d < 0 for d in differences.values()) / len(cells))
                if cells else None,
            "improved_cell_count": sum(d < 0 for d in differences.values()),
            "tied_cell_count": sum(d == 0 for d in differences.values()),
            "cell_count": len(cells), "cell_differences": differences,
        }
    return result


def select_primary_comparator(method_summaries, latency_by_method=None,
                              candidate_order=STANDALONES):
    """Confirmation selection only; exact ties use comparable latency or order.

    ``latency_by_method`` must contain measured medians from the same frozen
    common workload, never estimates or partial workloads. If any tied
    candidate lacks a finite nonnegative latency, use the configured order.
    """
    order = _cells(candidate_order)
    if any(name not in method_summaries for name in order):
        raise ValueError("all standalone candidates must be complete before selection")
    first = method_summaries[order[0]]
    for name in order:
        summary = method_summaries[name]
        if summary["status"] != "complete" or summary["expected_cells"] != first["expected_cells"]:
            raise ValueError("incomplete or mismatched comparator panel")
        if summary["contributors"]["MASE"] != first["contributors"]["MASE"]:
            raise ValueError("comparators must have identical common MASE contributors")
        score = summary["scores"]["MASE"]
        if score is None or not np.isfinite(score):
            raise ValueError("primary comparator requires an available finite MASE")
    best = min(method_summaries[name]["scores"]["MASE"] for name in order)
    tied = [name for name in order if method_summaries[name]["scores"]["MASE"] == best]
    winner, tie_break = tied[0], "none" if len(tied) == 1 else "candidate_order"
    if len(tied) > 1 and latency_by_method is not None:
        latencies = [latency_by_method.get(name) for name in tied]
        if all(value is not None and np.ndim(value) == 0 and np.isfinite(value)
               and value >= 0 for value in latencies):
            winner = min(tied, key=lambda name: (latency_by_method[name], order.index(name)))
            tie_break = "common_workload_latency_then_candidate_order"
    return {"method": winner, "MASE": float(best), "exact_tied_candidates": tied,
            "tie_break": tie_break, "selected_on": "confirmation",
            "mase_contributors": list(first["contributors"]["MASE"])}


def paired_calendar_intervals(panel, expected_cells, *, replicates=2000,
                              block_lengths=(168, 336), seed=20260928):
    """Paired moving-block intervals on a fixed panel, no fitting or redraw.

    Each panel entry has actual, eligible, weekly_scale, hybrid_by_seed and
    comparator_by_seed. The last two mappings contain raw full-calendar
    forecasts. The same calendar indices jointly sample cells/models/seeds.
    Averaging absolute losses over seeds before sampling is algebraically
    identical to recomputing each seed's MASE and then its cell mean.
    Any undefined replicate makes that block-size interval unavailable.
    """
    cells = _cells(expected_cells)
    if not isinstance(panel, Mapping) or set(panel) != set(cells):
        raise ValueError("uncertainty requires the complete fixed cell panel")
    if isinstance(replicates, bool) or not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    lengths = list(block_lengths)
    if not lengths or len(set(lengths)) != len(lengths) or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in lengths
    ):
        raise ValueError("block_lengths must be unique positive integers")
    masks, differences, valid_scales, point_scores = [], [], [], []
    length, seed_ids = None, None
    for cell in cells:
        entry = panel[cell]
        actual, eligible = _targets(entry["actual"], entry["eligible"])
        if length is not None and len(actual) != length:
            raise ValueError("all cells must have the same complete calendar length")
        length = len(actual)
        scale = _scale(entry["weekly_scale"])
        hybrid = _predictions(entry["hybrid_by_seed"], eligible)
        comparator = _predictions(entry["comparator_by_seed"], eligible)
        current_ids = (set(hybrid), set(comparator))
        if seed_ids is not None and current_ids != seed_ids:
            raise ValueError("seed lists must be identical across cells within each method")
        seed_ids = current_ids
        delta = np.zeros(length)
        if eligible.any():
            h_loss = np.mean([np.abs(np.maximum(0, p[eligible]) - actual[eligible])
                              for p in hybrid.values()], axis=0)
            c_loss = np.mean([np.abs(np.maximum(0, p[eligible]) - actual[eligible])
                              for p in comparator.values()], axis=0)
            if not np.isfinite(h_loss).all() or not np.isfinite(c_loss).all():
                raise ValueError("error calculation overflow")
            if scale is not None:
                delta[eligible] = (h_loss - c_loss) / scale
        masks.append(eligible)
        differences.append(delta)
        valid_scales.append(scale is not None)
        point_scores.append(float(np.mean(delta[eligible])) if eligible.any()
                            and scale is not None else None)
    masks = np.asarray(masks, dtype=np.float64)
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    blocks = {}
    for block_length in lengths:
        base = {"block_hours": int(block_length), "replicates": int(replicates),
                "interval": None, "confidence_level": 0.95,
                "quantile_method": "linear", "percentiles": [0.025, 0.975]}
        if block_length > length:
            blocks[str(block_length)] = {**base, "status": "unavailable",
                "reason": "block_longer_than_calendar", "undefined_replicates": int(replicates)}
            continue
        # Counts preserve repeated calendar hours and each cell's own target mask.
        counts = np.zeros((replicates, length), dtype=np.float64)
        for replicate in range(replicates):
            starts = rng.integers(0, length - block_length + 1,
                                  size=math.ceil(length / block_length))
            indices = (starts[:, None] + np.arange(block_length)).ravel()[:length]
            counts[replicate] = np.bincount(indices, minlength=length)
        denominators = counts @ masks.T
        valid = (denominators > 0).all(axis=1) & all(valid_scales)
        undefined = int((~valid).sum())
        if undefined:
            blocks[str(block_length)] = {**base, "status": "unavailable",
                "reason": "undefined_fixed_panel_replicate", "undefined_replicates": undefined}
            continue
        values = np.mean((counts @ differences.T) / denominators, axis=1)
        if not np.isfinite(values).all():
            raise ValueError("bootstrap statistic overflow")
        bounds = np.quantile(values, [0.025, 0.975], method="linear")
        blocks[str(block_length)] = {**base, "status": "available",
            "undefined_replicates": 0, "interval": [float(v) for v in bounds]}
    return {
        "statistic": "hybrid_minus_frozen_comparator_macro_MASE",
        "scope": "approximate_conditional_interval_not_a_success_gate",
        "point_difference": _mean(point_scores),
        "point_status": "available" if all(p is not None for p in point_scores) else "unavailable_fixed_panel",
        "cell_count": len(cells), "expected_cells": cells, "calendar_hours": length,
        "seed": int(seed), "resample_seeds_or_cells": False, "refit_models": False,
        "denominators_fixed": True, "sampling": "uniform_nonwrapping_calendar_blocks",
        "blocks": blocks,
    }
