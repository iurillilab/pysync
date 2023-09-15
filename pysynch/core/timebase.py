from __future__ import annotations
import numpy as np
from pynapple import Tsd


class TimeBaseTs(Tsd):
    """TimeBase class for mapping signals between timebases.
    The data that should populate this object is detected events from a clock
    signal that gets acquired by multiple acquitision devices, eg device A and B.

    The synch events can then be used to move data acquired in one of the two systems
    to the timebase of the other.
    """

    def _map_to(self, other_timebase: "TimeBaseTs") -> tuple[float, float]:
        # TODO maybe get intersection first
        # linear regression using least squares in numpy:
        coef, offset = np.polyfit(self.index, other_timebase.index, 1)
        return coef, offset

    def map_times_to(
        self, other_timebase: "TimeBaseTs", times: np.ndarray
    ) -> int | np.ndarray:
        """Core index conversion function"""
        coef, off = self._map_to(other_timebase)
        return (times * coef) + off

    def map_ts_to(self, other_timebase: "TimeBaseTs", tsd_obj: "Ts") -> "Ts":
        """Map a Ts object to another timebase."""
        new_tsd_obj = tsd_obj.copy()
        new_index = self.map_times_to(other_timebase, new_tsd_obj.index)
        new_tsd_obj.index = new_index
        return new_tsd_obj

