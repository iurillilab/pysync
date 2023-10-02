import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def crop_around_times(trace, times, window, fs):
    idxs = (times * fs).astype(int)
    window = (np.array(window) * fs).astype(int)

    return crop_around_idxs(trace, idxs, window)


def crop_around_idxs(trace, idxs, window):
    window_idxs = np.arange(window[0], window[1], dtype=int)
    idxs_mat = idxs + window_idxs[:, np.newaxis]

    return trace[idxs_mat]


def _get_windows_lims_array(events: np.ndarray, wnd: tuple) -> np.ndarray:
    """From an array of events [t0, t1, ...], and a window, return an array
    [t0-wnd[0], t0+wnd[1], t1-wnd[0], t1+wnd[1], ...].
    """
    windows_limits = np.zeros([len(events) * 2])
    windows_limits[::2] = np.array(events + wnd[0])
    windows_limits[1::2] = np.array(events + wnd[1])

    return windows_limits


def get_times_in_windows(times, events, wnd):
    # Event windows CAN'T BE OVERLAPPING!! Every spike will end up assigned to only one window (the last valid one)

    windows_limits = _get_windows_lims_array(events, wnd)

    searchsorted_indices = np.searchsorted(windows_limits, times) - 1
    event_indices = searchsorted_indices // 2  # , searchsorted_indices.shape

    in_windows_idxs = searchsorted_indices % 2 == 0

    return in_windows_idxs, event_indices[in_windows_idxs]


def get_perievent_spikes(spike_df, events, wnd):
    """Return a dataframe with only the spikes in the windows around the events."""
    in_windows_idxs, valid_event_indices = get_times_in_windows(
        spike_df.time.values, events, wnd
    )
    valid_times = spike_df.time.values[in_windows_idxs]
    valid_nids = spike_df.nid.values[in_windows_idxs]
    valid_event_times = events[valid_event_indices]
    relative_times = valid_times - valid_event_times

    return pd.DataFrame(
        dict(time=relative_times, nid=valid_nids, event=valid_event_indices)
    )


def resample_matrix(matrix, old_time_axis, new_time_axis):
    """
    Resamples a NumPy matrix over its first dimension to a new time axis using linear interpolation with extrapolation.

    Parameters:
        matrix (numpy.ndarray):
            The input matrix to be resampled. Shape: (n, ...), where n is the number of time steps.
        old_time_axis (numpy.ndarray)
            The old time axis corresponding to the input matrix. Shape: (n,).
        new_time_axis (numpy.ndarray)
            The new time axis to which the matrix will be resampled. Shape: (m,).

    Returns:
        numpy.ndarray
            The resampled matrix. Shape: (m, ...), where m is the number of time steps in the new time axis.
    """
    if len(old_time_axis) != matrix.shape[0]:
        raise ValueError("The length of 'old_time_axis' must match the first dimension of the 'matrix'.")

    if not np.all(np.diff(old_time_axis) > 0):
        raise ValueError("'old_time_axis' must be monotonically increasing.")

    if not np.all(np.diff(new_time_axis) > 0):
        raise ValueError("'new_time_axis' must be monotonically increasing.")

    # Calculate the interpolation function with extrapolation
    interp_func = interp1d(old_time_axis, matrix, axis=0, kind='linear', fill_value='extrapolate')

    # Resample the matrix to the new time axis
    resampled_matrix = interp_func(new_time_axis)

    return resampled_matrix
