from pysynch import TimeBaseTs
from pynapple import Tsd


def test_timebase():
    import numpy as np

    test_signal_a = TimeBaseTs(np.array([0, 1, 2, 3]))
    test_signal_b = TimeBaseTs(np.array([1.5, 2, 2.5, 3]))

    new_timesig_times = np.arange(2, 3, 0.1)
    signal_timebase_a = Tsd(new_timesig_times, np.random.rand(len(new_timesig_times)))
    signal_timebase_b = test_signal_a.map_ts_to(test_signal_b, signal_timebase_a)
    assert np.allclose(
        signal_timebase_b.index.values,
        np.array([2.5, 2.55, 2.6, 2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95]),
    )
