from __future__ import annotations
import numpy as np
from pynapple import Ts, Tsd, TsGroup


class TimeBaseTs(Ts):
    """TimeBase class for mapping between timebases."""

    def _map_to(self, other_timebase: "TimeBaseTs") -> tuple[float, float]:
        # TODO maybe get intersection first
        # linear regression using least squares in numpy:
        coef, offset = np.polyfit(self.index, other_timebase.index, 1)
        return coef, offset

    def map_times_to(
        self, other_timebase: "TimeBaseTs", times: np.ndarray
    ) -> int | np.ndarray:
        """Core index conversion function, without type conversion."""
        coef, off = self._map_to(other_timebase)
        return (times * coef) + off

    def map_ts_to(self, other_timebase: "TimeBaseTs", tsd_obj: "Ts") -> "Ts":
        """Map a Ts object to another timebase."""
        new_tsd_obj = tsd_obj.copy()
        new_index = self.map_times_to(other_timebase, new_tsd_obj.index)
        new_tsd_obj.index = new_index
        return new_tsd_obj


if __name__ == "__main__":
    import numpy as np

    test_signal_a = TimeBaseTs(np.array([0, 1, 2, 3]))
    test_signal_b = TimeBaseTs(np.array([1.5, 2, 2.5, 3]))

    new_timesig_times = np.arange(2, 3, 0.1)
    signal_timebase_a = Tsd(new_timesig_times, np.random.rand(len(new_timesig_times)))
    print(signal_timebase_a)
    print(test_signal_a.map_ts_to(test_signal_b, signal_timebase_a).index.values)
