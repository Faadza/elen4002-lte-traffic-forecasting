"""The full search can be counted and guarded without loading any traffic."""
import copy
from pathlib import Path
import re

import pytest
import yaml

from forecasting.development_plan import build_jobs, candidates, score_candidates, TUNING_CELLS


@pytest.fixture
def config():
    return yaml.safe_load((Path(__file__).parents[1] / "run_config.yaml").read_text())


@pytest.mark.parametrize("stage,count,candidate_count", [
    ("baselines", 81, 3), ("ridge", 135, 5), ("direct", 162, 6),
    ("hybrid", 162, 6), ("lstm", 324, 4),
])
def test_exact_fixed_workload_and_no_holdout(config, stage, count, candidate_count):
    before = copy.deepcopy(config)
    jobs = build_jobs(config, stage)
    assert len(jobs) == count
    assert len(candidates(config, stage)) == candidate_count
    assert len({job["id"] for job in jobs}) == count
    assert {job["cell"] for job in jobs} == set(TUNING_CELLS)
    assert {job["fold_index"] for job in jobs} == {0, 1, 2}
    assert {(job["fit_end"], job["score_start"], job["score_end"]) for job in jobs} == {
        (4116, 4116, 4788), (4788, 4788, 5460), (5460, 5460, 6132)}
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", job["id"]) for job in jobs)
    assert all(set(job) == {"id", "stage", "method", "cell", "fold_index", "fit_end",
                           "score_start", "score_end", "candidate_id", "parameters", "seed"}
               for job in jobs)
    assert {job["seed"] for job in jobs} == ({2026, 2027, 2028} if stage == "lstm" else {None})
    assert jobs == build_jobs(config, stage)
    assert config == before


def test_hybrid_scores_reuse_six_corrections_and_include_zero_once(config):
    fit = candidates(config, "hybrid")
    score = score_candidates(config, "hybrid")
    assert len(fit) == 6 and len(score) == 25
    assert score[0]["parameters"] == {"weight": 0}
    assert sum(c["parameters"]["weight"] == 0 for c in score) == 1
    for correction in fit:
        shared = [c for c in score if c["candidate_id"].startswith(correction["candidate_id"] + "-")]
        assert [c["parameters"]["weight"] for c in shared] == [.25, .5, .75, 1]
    assert sum(c["parameters"]["weight"] == 1 for c in score) == 6
    assert all("alpha" not in c["parameters"] for c in fit)  # Runner supplies the global ridge dependency.


@pytest.mark.parametrize("section,key", [("ridge", "alphas"), ("trees", "tree_counts"),
    ("trees", "leaf_caps"), ("hybrid", "weights"), ("lstm", "hidden_units"),
    ("lstm", "output_dropout"), ("lstm", "seeds_all_formal_periods")])
def test_undeclared_grids_rejected(config, section, key):
    config["models"][section][key].reverse()
    with pytest.raises(ValueError, match="unchanged declared grids"):
        build_jobs(config, "ridge")


@pytest.mark.parametrize("mutation", ["panel", "holdout", "development", "seed_type", "rounding", "metric"])
def test_changed_panel_chronology_or_selection_rejected(config, mutation):
    if mutation == "panel":
        config["tuning_panel"]["ids"][0] = "DifferentCell"
    elif mutation == "holdout":
        config["chronology"]["folds"][2]["score"][1] = 7446
    elif mutation == "development":
        config["chronology"]["development"][1] = 7446
    elif mutation == "seed_type":
        config["models"]["lstm"]["seeds_all_formal_periods"][0] = 2026.0
    elif mutation == "rounding":
        config["selection"]["rounding_before_tie_detection"] = True
    else:
        config["metrics"]["primary"] = "RMSE"
    with pytest.raises(ValueError):
        build_jobs(config, "lstm")


def test_job_parameter_dictionaries_are_independent(config):
    jobs = build_jobs(config, "ridge")
    jobs[0]["parameters"]["alpha"] = 999
    assert jobs[1]["parameters"]["alpha"] == .1
    assert config["models"]["ridge"]["alphas"][0] == .1


def test_unknown_stage_cannot_request_final_evaluation(config):
    with pytest.raises(ValueError, match="Unknown development stage"):
        build_jobs(config, "final")
