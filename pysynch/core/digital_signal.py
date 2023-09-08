from __future__ import annotations
import numpy as np
from pysynch.class_utils import lazy_property
import pandas as pd
from pynapple import Tsd, Ts


class DigitalTsd(Tsd):
    """Class to handle digital signals. Features onsets and ofrateets
    detection, and conversion to timebase."""

    def __init__(
        self,
        array: list | np.ndarray | pd.DataFrame,
        time_array: list | np.ndarray = None,
        rate: float | int = None,
        signal_range=None,
        **kwargs,
    ) -> None:
        """Initialize a DigitalTsd object.

        Parameters
        ----------
        array : list | np.ndarray | pd.DataFrame
           Array of digital data.
        time_array : list | np.ndarray, optional
           Array of time values; could be None if rate is provided, by default None
        rate : float | int, optional
            Sampling frequency, if no time_array is provided. Otherwise, computed (average)
        signal_range : _type_, optional
            _description_, by default None
        """
        array = np.array(array)
        if array.dtype != bool:
            if array.max() == 1:
                array = array.astype(bool)
            else:
                print(array.max())
                assert (
                    signal_range is not None
                ), "signal_range must be provided if array is not 0/1"
                array = array > signal_range / 2
        assert (
            (time_array is not None) ^ (rate is not None)
        ), "Either time_array or rate must be provided"

        if time_array is None:
            time_array = np.arange(len(array)) / rate
        super().__init__(time_array, array)

    @property
    def n_pts(self) -> int:
        return len(array)

    @lazy_property
    def onsets(self) -> np.ndarray:
        onsets_arr = np.insert(self.values[1:] & ~self.values[:-1], 0, False)
        return np.nonzero(onsets_arr)[0]

    @lazy_property
    def onsets_times(self) -> np.ndarray:
        return Ts(self.index[self.onsets].values)

    @lazy_property
    def offsets(self) -> np.ndarray:
        offsets_arr = np.insert(~self.values[1:] & self.values[:-1], 0, False)
        return np.nonzero(offsets_arr)[0]

    @lazy_property
    def offsets_times(self) -> np.ndarray:
        return Ts(self.index[self.offsets].values)

    @lazy_property
    def all_events(self) -> np.ndarray:
        all_events = np.concatenate([self.onsets, self.offsets])
        all_events.sort()
        return all_events

    @lazy_property
    def all_events_times(self) -> np.ndarray:
        return Ts(self.index[self.all_events].values)
