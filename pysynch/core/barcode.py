"""Adapted from:
    Optogenetics and Neural Engineering Core ONE Core, University of Colorado, School of Medicine
    18.Nov.2021 (See bit.ly/onecore for more information, including a more detailed write up.)
"""

import numpy as np

from pysynch.core.digital_signal import DigitalTsd


class BarcodeTsd(DigitalTsd):
    """Class for extracting and analyzing barcodes from a digital signal.
    
    Attributes:
    -----------
    barcodes : np.ndarray
        Array of barcodes and their index.

    Parameters (class attributes - based on your Arduino barcode generator settings):
        NBITS = (int) the number of bits (bars) that are in each barcode (not
              including wrappers).
        INTER_BC_INTERVAL_MS = (int) The duration of time (in milliseconds)
                               between each barcode's start.
        WRAP_DURATION_MS = (int) The duration of time (in milliseconds) of the
                          ON wrapper portion (default = 10 ms) in the barcodes.
        BAR_DURATION_MS = (int) The duration of time (in milliseconds) of each
                         bar (bit) in the barcode.

        GLOBAL_TOLERANCE = (float) The fraction (in %/100) of tolerance allowed
                         for duration measurements (ex: BAR_DURATION_MS).

    """

    # General variables; make sure these align with the timing format of
    # your Arduino-generated barcodes.
    NBITS = 32  # Number of bits in each barcode
    INTER_BC_INTERVAL_MS = 5000  # Distance of barcodes, in milliseconds
    WRAP_DURATION_MS = 10  # Duration of the wrapper pulses, in milliseconds
    BAR_DURATION_MS = 30  # Duration of the barcode, in milliseconds
    GLOBAL_TOLERANCE = 0.20  # In %/100

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the BarcodeSignal class."""
        super().__init__(*args, **kwargs)
        assert self.rate is not None, "Sampling rate must be specified."
        self._barcodes, self._barcodes_idx = None, None

    @property
    def barcodes(self) -> np.ndarray:
        """Array of barcodes."""
        if self._barcodes is None:
            self._read_barcodes()
        return self._barcodes

    @property
    def barcodes_idx(self) -> np.ndarray:
        """Array of barcodes' index."""
        if self._barcodes_idx is None:
            self._read_barcodes()
        return self._barcodes_idx

    @property
    def barcodes_times(self) -> np.ndarray:
        """Array of barcodes' times."""
        return self.barcodes_idx / self.fs

    def _read_barcodes(self) -> None:
        """Analyzes the digital signal to extract the barcodes. Lengthy function inherited from the barcode
        system developers, could be optimized in the future. A bit slow - lots of loops involved.
        """
        wrap_duration = 3 * self.WRAP_DURATION_MS  # Off-On-Off
        total_barcode_duration = self.NBITS * self.BAR_DURATION_MS + 2 * wrap_duration

        # Tolerance conversions
        min_wrap_duration = (
            self.WRAP_DURATION_MS - self.WRAP_DURATION_MS * self.GLOBAL_TOLERANCE
        )
        max_wrap_duration = (
            self.WRAP_DURATION_MS + self.WRAP_DURATION_MS * self.GLOBAL_TOLERANCE
        )
        sample_conversion = 1000 / self.fs  # Convert sampling rate to msec

        # Signal extraction and barcode analysis
        indexed_times = self.all_events

        # Find time difference between index values (ms), and extract barcode wrappers.
        events_time_diff = np.diff(indexed_times) * sample_conversion  # convert to ms

        wrapper_array = indexed_times[
            np.where(
                np.logical_and(
                    min_wrap_duration < events_time_diff,
                    events_time_diff < max_wrap_duration,
                )
            )[0]
        ]

        # Isolate the wrapper_array to wrappers with ON values, to avoid any
        # "OFF wrappers" created by first binary value.
        false_wrapper_check = (
            np.diff(wrapper_array) * sample_conversion
        )  # Convert to ms
        # Locate indices where two wrappers are next to each other.
        false_wrappers = np.where(false_wrapper_check < max_wrap_duration)[0]
        # Delete the "second" wrapper (it's an OFF wrapper going into an ON bar)
        wrapper_array = np.delete(wrapper_array, false_wrappers + 1)

        # Find the barcode "start" wrappers, set these to wrapper_start_times, then
        # save the "real" barcode start times to signals_barcode_start_idxs, which
        # will be combined with barcode values for the output .npy file.
        wrapper_time_diff = np.diff(wrapper_array) * sample_conversion  # convert to ms
        barcode_index = np.where(wrapper_time_diff < total_barcode_duration)[0]
        wrapper_start_times = wrapper_array[barcode_index]
        signals_barcode_start_idxs = (
            wrapper_start_times - self.WRAP_DURATION_MS / sample_conversion
        )
        # Actual barcode start is 10 ms before first 10 ms ON value.

        # Convert wrapper_start_times, on_times, and off_times to ms
        wrapper_start_times = wrapper_start_times * sample_conversion
        on_times = self.onsets * sample_conversion
        off_times = self.offsets * sample_conversion

        signals_barcodes = []

        for start_time in wrapper_start_times:
            oncode = on_times[
                np.where(
                    np.logical_and(
                        on_times > start_time,
                        on_times < start_time + total_barcode_duration,
                    )
                )[0]
            ]
            offcode = off_times[
                np.where(
                    np.logical_and(
                        off_times > start_time,
                        off_times < start_time + total_barcode_duration,
                    )
                )[0]
            ]
            curr_time = (
                offcode[0] + self.WRAP_DURATION_MS
            )  # Jumps ahead to start of barcode
            bits = np.zeros((self.NBITS,))
            interbit_on = False  # Changes to "True" during multiple ON bars

            for bit in range(0, self.NBITS):
                next_on = np.where(
                    oncode >= (curr_time - self.BAR_DURATION_MS * self.GLOBAL_TOLERANCE)
                )[0]
                next_off = np.where(
                    offcode
                    >= (curr_time - self.BAR_DURATION_MS * self.GLOBAL_TOLERANCE)
                )[0]

                if next_on.size > 1:  # Don't include the ending wrapper
                    next_on = oncode[next_on[0]]
                else:
                    next_on = start_time + self.INTER_BC_INTERVAL_MS

                if next_off.size > 1:  # Don't include the ending wrapper
                    next_off = offcode[next_off[0]]
                else:
                    next_off = start_time + self.INTER_BC_INTERVAL_MS

                # Recalculate min/max bar duration around curr_time
                min_bar_duration = (
                    curr_time - self.BAR_DURATION_MS * self.GLOBAL_TOLERANCE
                )
                max_bar_duration = (
                    curr_time + self.BAR_DURATION_MS * self.GLOBAL_TOLERANCE
                )

                if min_bar_duration <= next_on <= max_bar_duration:
                    bits[bit] = 1
                    interbit_on = True
                elif min_bar_duration <= next_off <= max_bar_duration:
                    interbit_on = False
                elif interbit_on == True:
                    bits[bit] = 1

                curr_time += self.BAR_DURATION_MS

            barcode = 0

            for bit in range(0, self.NBITS):  # least sig left
                barcode += bits[bit] * pow(2, bit)

            signals_barcodes.append(barcode)

        # Create merged array with timestamps stacked above their barcode values
        self._barcodes = np.array(signals_barcodes)
        self._barcodes_idx = np.array(signals_barcode_start_idxs)

    def _map_to(self, other_barcode_signal: "BarcodeTsd") -> tuple[float, float]:
        """Core method to map the barcode values of one BarcodeSignal to another BarcodeSignal.
        #TODO consider caching this operation if ends up used frequently - it is approx 0.01 ms for real data
        """
        # Pull the index values from barcodes shared by both groups of data:
        shared_barcodes, own_index, other_index = np.intersect1d(
            self.barcodes, other_barcode_signal.barcodes, return_indices=True
        )

        own_shared_barcode_times = self.barcodes_idx[own_index]
        other_shared_barcode_times = other_barcode_signal.barcodes_idx[other_index]

        # Determine slope between main/secondary timestamps
        coef = (other_shared_barcode_times[-1] - other_shared_barcode_times[0]) / (
            own_shared_barcode_times[-1] - own_shared_barcode_times[0]
        )
        # Determine offset between main and secondary barcode timestamps
        offset = other_shared_barcode_times[0] - own_shared_barcode_times[0] * coef

        return coef, offset

    def _map_indexes_to(
        self, other_barcode_signal: "BarcodeTsd", indexes: int | np.ndarray
    ) -> int | np.ndarray:
        """Core index conversion function, without type conversion."""
        coef, off = self._map_to(other_barcode_signal)
        return (indexes * coef) + off

    def map_indexes_to(
        self, other_barcode_signal: "BarcodeTsd", indexes: int | np.ndarray
    ) -> int | np.ndarray:
        """Map the indexes in self timebase to another BarcodeSignal.
        This is a wrapper around _map_indexes_to that converts the output to int or np.ndarray (as the conversion)
        is not necessary for the resampling function).

        Parameters
        ----------
        other_barcode_signal : BarcodeSignal object
            The BarcodeSignal object describing the timebase to which the indexes will be mapped.
        indexes : np.ndarray
            The indexes (in the self timebase) to be mapped to the new timebase

        Returns
        -------
        np.ndarray :
            The mapped indexes.

        """
        mapped_idxs = self._map_indexes_to(other_barcode_signal, indexes)
        try:
            return int(mapped_idxs)
        except TypeError:
            return mapped_idxs.astype(int)

    def map_times_to(
        self, other_barcode_signal: "BarcodeTsd", times: int | np.ndarray
    ) -> int | np.ndarray:
        """Map the times in self timebase to another BarcodeSignal. timebase
        Parameters
        ----------
        other_barcode_signal
        times

        Returns
        -------

        """
        return (
            self._map_indexes_to(other_barcode_signal, times * self.fs)
            / other_barcode_signal.fs
        )

    def resample_to(
        self, other_barcode_signal: "BarcodeTsd", own_timebase_data: np.ndarray
    ) -> np.ndarray:
        """Resample the data in self timebase to another BarcodeSignal.

        Parameters
        ----------
        other_barcode_signal : BarcodeSignal object
            The BarcodeSignal object describing the timebase to which the data will be resampled.
        own_timebase_data : np.ndarray
            The data (in the self timebase) to be resampled to the new timebase

        Returns
        -------
        np.ndarray :
            The resampled data.

        """

        # TODO check if this could be somehow made faster
        own_idxs = np.arange(self.n_pts)
        other_idxs = np.arange(other_barcode_signal.n_pts)

        mapped_idxs = self._map_indexes_to(other_barcode_signal, own_idxs)
        return np.interp(other_idxs, mapped_idxs, own_timebase_data)


if __name__ == "__main__":
    import time
    from pathlib import Path

    import numpy as np

    from labnpx.intan_reader import IntanData

    data_path = Path("/Users/vigji/Downloads/exampleephysdata")
    intan_path = data_path / "IntanData"
    intan_data = IntanData(intan_path)
    data = intan_data.dig_in_array
    headers = [
        None,
        "barcodes",
        None,
        "laser",
        "photodiode",
        None,
        None,
        None,
        "cameras",
        "servo_forw",
        "servo_back",
        None,
    ]
    data_dict = {k: data[i] for i, k in enumerate(headers) if k is not None}

    t = time.time()
    r = BarcodeTsd(data_dict["barcodes"], fs=4000)

    print(time.time() - t)
    print(r.barcodes.shape)
    print(r.barcodes[:, :2])
    print(r.barcodes[:, -2:])
    assert r.barcodes.shape == (2, 1776)
    assert np.allclose(
        r.barcodes[:, :2], np.array([[1583.0, 21586.0], [29000.0, 29001.0]]).T
    )
    assert np.allclose(
        r.barcodes[:, -2:],
        np.array([[3.5487055e07, 3.5507058e07], [3.0774000e04, 3.0775000e04]]).T,
    )
