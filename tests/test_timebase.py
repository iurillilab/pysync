import numpy as np
import pytest
from pynapple import Tsd

from pysynch.core.timebase import IndexedTimeBase, TimeBaseTs


@pytest.mark.parametrize(
    "timebase_b_array, resolution_strategy",
    [
        (np.array([1.5, 2, 2.5, 3]), "None"),
        (np.array([1.5, 2, 2.5]), "assume_synch_start"),
        (np.array([2, 2.5, 3]), "assume_synch_end"),
    ],
)
def test_timebase(timebase_b_array, resolution_strategy):
    test_signal_a = TimeBaseTs(np.array([0, 1, 2, 3]))
    test_signal_b = TimeBaseTs(timebase_b_array)

    new_timesig_times = np.arange(2, 3, 0.1)
    signal_timebase_a = Tsd(np.random.rand(len(new_timesig_times)), new_timesig_times)
    signal_timebase_b = test_signal_a.map_ts_to(
        test_signal_b,
        signal_timebase_a,
        mismatch_resolution_strategy=resolution_strategy,
    )

    # Make sure error is trown
    if resolution_strategy != "None":
        with pytest.raises(ValueError):
            test_signal_a.map_ts_to(
                test_signal_b,
                signal_timebase_a,
                mismatch_resolution_strategy="unknown_strategy",
            )

    assert np.allclose(
        signal_timebase_b.index.values,
        np.array([2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95]),
    )


@pytest.mark.parametrize(
    "timebase_b_array, timebase_b_times",
    [
        ([0, 1, 2, 3], np.array([1.5, 2, 2.5, 3])),
        ([0, 1, 2], np.array([1.5, 2, 2.5])),
        ([1, 2], np.array([2, 2.5])),
        ([1, 2, 3], np.array([2, 2.5, 3])),
    ],
)
def test_indexed_timebase(timebase_b_array, timebase_b_times):
    test_signal_a = IndexedTimeBase([0, 1, 2, 3], np.array([0, 1, 2, 3]))
    test_signal_b = IndexedTimeBase(timebase_b_array, timebase_b_times)

    new_timesig_times = np.arange(2, 3, 0.1)
    signal_timebase_a = Tsd(np.random.rand(len(new_timesig_times)), new_timesig_times)
    signal_timebase_b = test_signal_a.map_ts_to(test_signal_b, signal_timebase_a)

    # Make sure error is trown
    with pytest.raises(AssertionError):
        test_signal_a.map_ts_to(TimeBaseTs(timebase_b_array), signal_timebase_a)

    assert np.allclose(
        signal_timebase_b.index.values,
        np.array([2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95]),
    )
