"""
Fundamental Frequency Estimator.

Estimates the physical fundamental frequency (Hz) of a signal
using least-squares sine fitting algorithm.

MATLAB counterpart: findfin.m
"""

from adctoolbox.aout.fit_sine_4param import fit_sine_4param


def estimate_frequency(data, fs=1.0):
    """
    Estimate the physical fundamental frequency (Hz).

    This is a wrapper around the robust `fit_sine` algorithm.
    It converts the normalized frequency (0 ~ 0.5) returned by fit_sine
    into physical frequency (Hz) based on the sampling rate.

    Args:
        data (np.ndarray): Input signal data. 1D or 2D array.
        fs (float): Sampling frequency in Hz (default: 1.0).

    Returns:
        float or np.ndarray: Estimated frequency in Hz.
                             (Scalar if input is 1D, Array if input is 2D)
    """
    result = fit_sine_4param(data)

    freq_norm = result['frequency']

    fin_hz = freq_norm * fs

    return fin_hz
