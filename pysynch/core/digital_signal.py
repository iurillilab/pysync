from __future__ import annotations
import numpy as np
import pandas as pd
import functools
from pynapple import Tsd, Ts
from pandas.core.internals import BlockManager, SingleBlockManager


class DigitalTsd(Tsd):
    """Class to handle digital signals. Features onsets and offsets
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
        if isinstance(array, SingleBlockManager):
            time_array = array.index.values
            array = array.array

        elif isinstance(array, pd.Series):
            time_array = array.index.values
            array = array.values

        array = np.array(array)

        # convert to bool if not already:
        if array.dtype != bool:
            if array.max() == 1:
                array = array.astype(bool)
            else:
                assert (
                    signal_range is not None
                ), "signal_range must be provided if array is not 0/1"
                array = array > signal_range / 2

        assert (time_array is not None) ^ (
            rate is not None
        ), "Either time_array or rate must be provided"

        if time_array is None:
            time_array = np.arange(len(array)) / rate
        super().__init__(array, time_array)

    @property
    def n_pts(self) -> int:
        # TODO deprecate
        return len(self)

    @functools.cached_property
    def onsets(self) -> np.ndarray:
        print(self.values)
        onsets_arr = np.insert(self.values[1:] & ~self.values[:-1], 0, False)
        return np.nonzero(onsets_arr)[0]

    @functools.cached_property
    def onsets_times(self) -> np.ndarray:
        return Ts(self.index[self.onsets].values)

    @functools.cached_property
    def offsets(self) -> np.ndarray:
        offsets_arr = np.insert(~self.values[1:] & self.values[:-1], 0, False)
        return np.nonzero(offsets_arr)[0]

    @functools.cached_property
    def offsets_times(self) -> np.ndarray:
        return Ts(self.index[self.offsets].values)

    @functools.cached_property
    def all_events(self) -> np.ndarray:
        all_events = np.concatenate([self.onsets, self.offsets])
        all_events.sort()
        return all_events

    @functools.cached_property
    def all_events_times(self) -> np.ndarray:
        return Ts(self.index[self.all_events].values)
