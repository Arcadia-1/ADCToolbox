"""
Lean SNDR + ENOB computation.

Uses a direct FFT ratio path for optimization loops and spec gates. This keeps
the metric definition aligned with compute_spectrum for explicit coherent
side-bin settings without running harmonic, spur, or auto side-bin analysis.
"""

import numpy as np

from adctoolbox.spectrum._bin_ranges import rfft_inband_bin_count
from adctoolbox.spectrum._locate_fundamental import _locate_fundamental
from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input
from adctoolbox.spectrum._spectrum_averaging import _power_average
from adctoolbox.spectrum._window import (
    _calculate_power_correction,
    _create_window,
    _get_default_side_bin,
)


def quick_sndr(data, fs=1.0, win_type="hann", side_bin=None, max_scale_range=None):
    """
    SNDR + ENOB from a single 1-D capture (same SNDR definition as analyze_spectrum).

    Parameters
    ----------
    data
        Time-domain samples, shape (N,)
    fs
        Sample rate (Hz)
    win_type
        Window name ('hann', 'rectangular', ...)
    side_bin
        Side bins around fundamental; None uses the window's coherent default
    max_scale_range
        Optional full-scale range for input normalization

    Returns
    -------
    dict
        ``{'sndr_dbc': float, 'enob': float}``
    """
    data_normalized = _prepare_fft_input(data, max_scale_range)
    _, n_fft = data_normalized.shape
    n_inband = rfft_inband_bin_count(n_fft, osr=1)

    window_vector, window_gain, equiv_noise_bw_factor = _create_window(win_type, n_fft)
    power_spectrum, _ = _power_average(data_normalized * window_vector)
    power_spectrum *= _calculate_power_correction(window_gain, equiv_noise_bw_factor)

    fundamental_bin, _ = _locate_fundamental(power_spectrum, n_inband)
    if side_bin is None:
        side_bin = _get_default_side_bin(win_type)
    side_bin = int(max(side_bin, 0))

    sig_bin_start = max(fundamental_bin - side_bin, 0)
    sig_bin_end = min(fundamental_bin + side_bin + 1, n_inband)
    sig_linear = float(np.sum(power_spectrum[sig_bin_start:sig_bin_end]))

    noise_distortion_spectrum = power_spectrum.copy()
    noise_distortion_spectrum[: min(side_bin + 1, len(noise_distortion_spectrum))] = 0.0
    noise_distortion_spectrum[sig_bin_start:sig_bin_end] = 0.0
    noise_distortion = float(np.sum(noise_distortion_spectrum[:n_inband]))

    sndr_dbc = 10 * np.log10(sig_linear / (noise_distortion + 1e-20))
    enob = (sndr_dbc - 1.76) / 6.02
    return {"sndr_dbc": float(sndr_dbc), "enob": float(enob)}
