from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pynapple import Ts, Tsd


class TimeBaseMixin(ABC):
    """TimeBase class for mapping signals between timebases.
    The data that should populate this object is detected events from a clock
    signal that gets acquired by multiple acquitision devices, eg device A and B.

    The synch events can then be used to move data acquired in one of the two systems
    to the timebase of the other."""

    @abstractmethod
    def _map_to(self, other_timebase: TimeBaseTs | IndexedTimeBaseTsd):
        """Core index conversion function"""
        pass

    def map_times_to(
        self, other_timebase: TimeBaseTs | IndexedTimeBaseTsd, times: np.ndarray, **kwargs
    ) -> int | np.ndarray:
        """Core index conversion function"""
        coef, off = self._map_to(other_timebase, **kwargs)
        return (times * coef) + off

    def map_ts_to(
        self, other_timebase: TimeBaseTs | IndexedTimeBaseTsd, tsd_obj: Ts | Tsd, **kwargs
    ):
        """Map a Ts object to another timebase."""

        new_tsd_obj = tsd_obj#.copy()
        new_index = self.map_times_to(other_timebase, new_tsd_obj.index, **kwargs)
        new_tsd_obj.index = new_index
        return new_tsd_obj


class TimeBaseTs(Ts, TimeBaseMixin):
    """TimeBase class for mapping signals between timebases. Assumes timebase
    is built from indistinguishable events (eg. clock ticks) that are detected.
    """

    def _map_to(
        self, other_timebase: "TimeBaseTs", mismatch_resolution_strategy="None"
    ) -> tuple[float, float]:
        """Map two timebases to each other. If the timebases have different
        number of events, use the resolution strategy specifies

        Parameters
        ----------
        other_timebase : TimeBaseTs
            Target timebase to map to.
        mismatch_resolution_strategy : str
            One of "None", "assume_synch_start" or "assume_synch_end".

        Returns
        -------
        tuple[float, float]
            coefficient and offset of the linear regression mapping the two timebases.
        """
        # TODO maybe get intersection first
        # linear regression using least squares in numpy:

        other_index = other_timebase.index
        own_index = self.index

        # Resolve mismatch by resolution strategy:
        if len(own_index) != len(other_index):
            n_events_to_consider = min(len(other_timebase), len(self))
            if mismatch_resolution_strategy == "assume_synch_end":
                # assume that the last event is the synch event
                other_index = other_timebase.index[-n_events_to_consider:]
                own_index = self.index[-n_events_to_consider:]

            elif mismatch_resolution_strategy == "assume_synch_start":
                # assume that the first event is the synch event
                other_index = other_timebase.index[:n_events_to_consider]
                own_index = self.index[:n_events_to_consider]
            else:
                raise ValueError(
                    f"Unknown mismatch resolution strategy for timebases of length {len(self)} and {len(other_timebase)}"
                )

        coef, offset = np.polyfit(own_index, other_index, 1)
        return coef, offset


class IndexedTimeBaseTsd(Tsd, TimeBaseMixin):
    """Timebase class for mapping signals between timebases. Assumes timebase
    is built from events that can be distinguishes (eg, barcodes).
    """

    def _map_to(self, other_timebase: "IndexedTimeBaseTsd") -> tuple[float, float]:
        """Map two timebases to each other.

        Parameters
        ----------
        other_timebase : IndexedTimeBase
            Target timebase to map to.

        Returns
        -------
        tuple[float, float]
            coefficient and offset of the linear regression mapping the two timebases.
        """

        # linear regression using least squares in numpy:
        assert isinstance(
            other_timebase, IndexedTimeBaseTsd
        ), "other_timebase must be an IndexedTimeBase!"
        _, own_index, other_index = np.intersect1d(
            self.values, other_timebase.values, return_indices=True
        )
        coef, offset = np.polyfit(
            self.index[own_index], other_timebase.index[other_index], 1
        )

        return coef, offset
