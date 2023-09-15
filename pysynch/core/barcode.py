"""Adapted from:
    Optogenetics and Neural Engineering Core ONE Core, University of Colorado, School of Medicine
    18.Nov.2021 (See bit.ly/onecore for more information, including a more detailed write up.)
"""

import numpy as np

from pysynch.core.digital_signal import DigitalTsd
from pysynch.core.timebase import TimeBaseTs


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
        # assert self.rate is not None, "Sampling rate must be specified."
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
        return self.barcodes_idx / self.rate

    @property
    def timebase(self) -> np.ndarray:
        """Generate timebase object from detected barcodes."""
        return TimeBaseTs(self.barcodes, self.barcodes_times)

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
        sample_conversion = 1000 / self.rate  # Convert sampling rate to msec

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
                elif interbit_on:
                    bits[bit] = 1

                curr_time += self.BAR_DURATION_MS

            barcode = 0

            for bit in range(0, self.NBITS):  # least sig left
                barcode += bits[bit] * pow(2, bit)

            signals_barcodes.append(barcode)

        # Create merged array with timestamps stacked above their barcode values
        self._barcodes = np.array(signals_barcodes)
        self._barcodes_idx = np.array(signals_barcode_start_idxs)
