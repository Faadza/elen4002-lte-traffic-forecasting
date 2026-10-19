import numpy as np

from preprocessing.prepare import causal_fill
from preprocessing.windows import eligible_targets
from preprocessing.splits import N_HOURS, TRAIN_END, VAL_END, FOLDS

nan = np.nan


def test_fill_short_gap():
    values = [5.0, nan, nan, nan, 9.0]
    filled, observed = causal_fill(values)
    assert list(filled) == [5.0, 5.0, 5.0, 5.0, 9.0]
    assert list(observed) == [1, 0, 0, 0, 1]


def test_fill_stops_at_fourth_hour():
    values = [5.0, nan, nan, nan, nan, nan, 9.0]
    filled, _ = causal_fill(values)
    assert list(filled[:4]) == [5.0, 5.0, 5.0, 5.0]
    assert np.isnan(filled[4]) and np.isnan(filled[5])
    assert filled[6] == 9.0


def test_fill_never_uses_future():
    # gap at the start has no past value, so nothing may be filled
    values = [nan, nan, 7.0, 8.0]
    filled, _ = causal_fill(values)
    assert np.isnan(filled[0]) and np.isnan(filled[1])


def test_fill_long_gap_first_three_only():
    # the fill decision must not depend on how long the gap turns out to be
    short = [1.0] + [nan] * 3 + [2.0]
    long = [1.0] + [nan] * 10 + [2.0]
    fs, _ = causal_fill(short)
    fl, _ = causal_fill(long)
    assert list(fs[1:4]) == list(fl[1:4]) == [1.0, 1.0, 1.0]


def test_split_boundaries():
    assert TRAIN_END == 6132 and VAL_END == 7446 and N_HOURS == 8760
    assert TRAIN_END / N_HOURS == 0.7
    assert FOLDS[-1]["eval_end"] == TRAIN_END


def test_eligibility_requires_full_window_and_observed_target():
    n = 400
    traffic = np.ones(n)
    observed = np.ones(n, dtype=int)
    traffic[200] = nan       # unresolved hole
    observed[200] = 0
    observed[300] = 0        # filled hour, has a value but is not observed
    ok = eligible_targets(traffic, observed, lookback=168)

    assert not ok[100]           # window reaches before the series start
    assert ok[199]               # everything before the hole is fine
    assert not ok[200]           # target itself missing
    assert not ok[250]           # hole sits inside the input window
    assert not ok[300]           # filled target is not a real observation
    assert not ok[301]           # the hole at 200 is still inside this window
    assert ok[369]               # first window clear of the hole; the filled
                                 # hour at 300 is fine as an input
