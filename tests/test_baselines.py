import numpy as np
import pytest

from models.baselines import persistence, daily_naive, weekly_naive


def test_exact_lookups():
    traffic = np.arange(500, dtype=float)   # value equals its index
    targets = np.array([200, 300, 450])
    assert list(persistence(traffic, targets)) == [199.0, 299.0, 449.0]
    assert list(daily_naive(traffic, targets)) == [176.0, 276.0, 426.0]
    assert list(weekly_naive(traffic, targets)) == [32.0, 132.0, 282.0]


@pytest.mark.parametrize("method,lag", [(persistence, 1), (daily_naive, 24), (weekly_naive, 168)])
def test_each_comparator_uses_only_its_past_value(method, lag):
    traffic = np.arange(240, dtype=float)
    expected = 200-lag
    assert method(traffic, [200]).tolist() == [expected]
    traffic[200:] = 99999
    assert method(traffic, [200]).tolist() == [expected]
    assert method(traffic[:200], [200]).tolist() == [expected]


@pytest.mark.parametrize("method,lag", [(persistence, 1), (daily_naive, 24), (weekly_naive, 168)])
def test_each_comparator_rejects_insufficient_history(method, lag):
    with pytest.raises(ValueError):
        method(np.arange(200, dtype=float), [lag-1])


@pytest.mark.parametrize("targets", [[200.5], [True], [[200]], [-1], [201], [190, 190], [190, 180]])
def test_unsafe_indices_rejected(targets):
    with pytest.raises(ValueError):
        persistence(np.arange(200, dtype=float), targets)


def test_missing_requested_lag_is_a_failure_not_a_smaller_cohort():
    traffic = np.arange(200, dtype=float)
    traffic[199] = np.nan
    with pytest.raises(ValueError):
        persistence(traffic, [199, 200])
    assert persistence(traffic, []).shape == (0,)


def test_constant_and_periodic_examples_are_exact():
    traffic = np.full(400, 5.)
    assert persistence(traffic, [168, 200]).tolist() == [5., 5.]
    daily = np.tile(np.arange(24, dtype=float), 15)
    assert np.array_equal(daily_naive(daily, [168, 200]), daily[[168, 200]])
    weekly = np.tile(np.arange(168, dtype=float), 3)
    assert np.array_equal(weekly_naive(weekly, [168, 200]), weekly[[168, 200]])
