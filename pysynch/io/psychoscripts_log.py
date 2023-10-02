# %%
import functools
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pysynch.core import TimeBaseTs


def find_screen_barcode_onsets_times(photodiode_sig):
    """Truly ugly function to fix in the future to find first element of the barcode."""
    # At the moment this is a magic number titrated to be low enough to include broker barcodes
    # wit less pulses, and high enough to filter any noise:
    CONSECUTIVE_TRUES_N = 7
    MIN_INTERBARCODE_DIST = 2.9
    VALS_FOR_STEP_CALC_N = 40
    TOLERANCE_FRACT = 0.05

    times_intervals = photodiode_sig.offsets_times - photodiode_sig.onsets_times
    shortest_steps = np.sort(times_intervals)[:VALS_FOR_STEP_CALC_N]
    min_step_t = np.median(shortest_steps)

    is_fast_step = (
        times_intervals - min_step_t
    ) < min_step_t + min_step_t * TOLERANCE_FRACT
    consecutive_true_counts = np.convolve(
        is_fast_step, np.ones(CONSECUTIVE_TRUES_N), mode="valid"
    )
    consecutive_true_idxs = np.nonzero(consecutive_true_counts == CONSECUTIVE_TRUES_N)[
        0
    ]

    # Filter first events of the series
    barcode_start_times = photodiode_sig.onsets_times[consecutive_true_idxs]
    valid_starts_idxs = np.insert(
        np.diff(barcode_start_times) > MIN_INTERBARCODE_DIST, 0, True
    )
    return barcode_start_times[valid_starts_idxs]


# TODO all the timestamping/barcode here as well as the TimeBase class should
# be refactored to a more rational unified structure in the future.
class PsychoScriptsLogFolder:
    # Stimuli for which to expect a photodiode logging:
    VISUAL_STIMS = [
        "gratings_omr",
        "visual_mapping_w_on_b_grid",
        "servo_visual_looming",
        "visual_moving_2d_target",
    ]

    def __init__(
        self,
        folder,
    ):
        self.folder = Path(folder)

    @functools.cached_property
    def stim_epochs_df(self):
        DATASET_FORMAT = "%Y.%m.%d-%H.%M.%S"

        # create dictionary of log files, each describing an epoch:
        dict_list = []
        for stim_epoch_file in self.file_list:
            parts = stim_epoch_file.name.split("_")
            stimname = "_".join(parts[1:-1])
            tstamp = datetime.strptime(parts[0], DATASET_FORMAT)
            epoch_dict = dict(time=tstamp, stimulus=stimname, filename=stim_epoch_file)
            dict_list.append(epoch_dict)

        stim_epochs_df = pd.DataFrame(dict_list)

        # Add repetition number:
        stim_epochs_df["repetition"] = 0
        for stimulus in stim_epochs_df["stimulus"].unique():
            selector = stim_epochs_df["stimulus"] == stimulus
            stim_epochs_df.loc[selector, "repetition"] = np.arange(sum(selector))

        stim_epochs_df["isvisual"] = stim_epochs_df["stimulus"].isin(
            PsychoScriptsLogFolder.VISUAL_STIMS
        )

        all_barcodes_t_ns = []
        for log in self.stim_logs_list:
            if len(log.barcode_lines) > 1:
                t = float(log.barcode_lines[1].split(" \t")[0])
                t = log.psy_timebase.map_times_to(log.cpu_timebase, t)
            else:
                t = None
            all_barcodes_t_ns.append(t)
        stim_epochs_df["barcode_cpu_time"] = all_barcodes_t_ns

        return stim_epochs_df

    @property
    def file_list(self):
        return sorted(self.folder.glob("*.log"))

    @property
    def stim_logs_list(self):
        return [PsychoScriptsLog(f) for f in self.file_list]


class PsychoScriptsLog:
    BARCODE_LINE_MATCH_PATTERN = "\tEXP \tunnamed Rect: color = "
    BARCODE_NUMBERS_PARSING_REGEX = r"(\d+\.\d+).*\((-?\d+),"

    def __init__(self, filename):
        self.filename = filename

    @functools.cached_property
    def all_lines(self) -> list[str]:
        with open(self.filename) as f:
            all_lines = f.readlines()
        return all_lines

    @functools.cached_property
    def barcode_lines(self):
        return [
            line for line in self.all_lines if self.BARCODE_LINE_MATCH_PATTERN in line
        ]

    @functools.cached_property
    def squarelogger_transitions(self):
        """Return array of times of barcode changes. First element will
        be the time of a transition on, last element the time of a transition
        off. Time units are pyschopy clock time, in seconds.

        Returns
        -------
        np.ndarray
            array with all the transitions times.
        """

        # Read lines looking like barcodes.
        # TODO might improve with better barcodes logs!

        times, vals = [], []
        for line in self.barcode_lines:
            time, val = re.findall(self.BARCODE_NUMBERS_PARSING_REGEX, line)[0]
            time, val = float(time), (val == "1")
            times.append(time)
            vals.append(val)
        vals, times = np.array(vals), np.array(times)
        if len(vals) == 0:
            return

        vals_ext = np.insert(
            vals, len(vals), ~vals[-1]
        )  # add number to ensure last element is considered transition
        transitions_idxs = vals_ext[1:] != vals_ext[:-1]

        vals_changes = vals[transitions_idxs][1:]
        times_changes = times[transitions_idxs][1:]

        # Check barcode elements start with True and end with False, and we have even numbers of elements:
        assert vals_changes[0] and not vals_changes[-1]
        assert len(vals_changes) % 2 == 0

        return times_changes  # [::2], times_changes[1::2]

    @functools.lru_cache
    def _get_matched_times(self):
        """Return array of times of stimuli changes in psychopy clock time and
        in cpu time, that allows to map to treadmill data.
        """
        MATCHING_TIMEBASES_REGEX = r"(\d+\.\d+).*\((-?\d+)"

        cpu_time = []
        psy_time = []

        for line in self.all_lines:
            matches = re.findall(MATCHING_TIMEBASES_REGEX, line)
            if len(matches) > 0:
                num_1, num_2 = matches[0]
                # This will match also lines with barcode changes described
                # by 1/-1, so we need to filter those out (apperently this is more)
                # efficient than any other regex filtering I tried
                if int(num_2) > 1:
                    psy_time.append(float(num_1))
                    cpu_time.append(int(num_2))

        return np.array(psy_time), np.array(cpu_time)

    @functools.cached_property
    def psy_timebase(self):
        psy_time, _ = self._get_matched_times()

        return TimeBaseTs(psy_time)

    @functools.cached_property
    def cpu_timebase(self):
        _, cpu_time = self._get_matched_times()

        return TimeBaseTs(cpu_time)
