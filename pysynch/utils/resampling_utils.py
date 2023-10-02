from scipy.interpolate import interp1d
import numpy as np


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