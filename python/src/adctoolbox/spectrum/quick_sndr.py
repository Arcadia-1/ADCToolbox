"""
Lean SNDR + ENOB computation.

Uses a direct FFT ratio path for optimization loops and spec gates. This keeps
the metric definition aligned with compute_spectrum for explicit coherent
side-bin settings without running harmonic or spur analysis. Pass
``side_bin="auto"`` to opt into the heavier auto side-bin detector used by
``compute_spectrum(..., side_bin=None)``.
"""

import numpy as np

from adctoolbox.spectrum._bin_ranges import rfft_inband_bin_count
from adctoolbox.spectrum._locate_fundamental import _locate_fundamental
from adctoolbox.spectrum._prepare_fft_input import _prepare_fft_input
from adctoolbox.spectrum._side_bin_auto import _detect_side_bin_auto
from adctoolbox.spectrum._spectrum_averaging import _power_average
from adctoolbox.spectrum._window import (
    _calculate_power_correction,
    _create_window,
    _get_auto_side_bin_fallback,
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
        Side bins around fundamental. None uses the window's coherent default
        fast path. "auto" uses waveform-based side-bin detection matching
        ``compute_spectrum(..., side_bin=None)``; this adds an extra ideal-tone
        FFT pass and assumes a dominant single-tone capture. For multi-tone,
        strongly distorted, or very noisy captures, pass an explicit integer.
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
    power_correction = _calculate_power_correction(window_gain, equiv_noise_bw_factor)
    power_spectrum *= power_correction

    fundamental_bin, fundamental_bin_fractional = _locate_fundamental(power_spectrum, n_inband)
    if isinstance(side_bin, str) and side_bin == "auto":
        side_bin = _detect_side_bin_auto(
            power_spectrum,
            fundamental_bin,
            fundamental_bin_fractional,
            n_inband,
            n_fft,
            window_vector,
            power_correction,
            fallback_side_bin=_get_auto_side_bin_fallback(win_type),
            minimum_side_bin=_get_default_side_bin(win_type),
        )
    elif side_bin is None:
        side_bin = _get_default_side_bin(win_type)
    side_bin = int(max(side_bin, 0))

    sig_bin_start = max(fundamental_bin - side_bin, 0)
    sig_bin_end = min(fundamental_bin + side_bin + 1, n_inband)
    sig_linear = float(np.sum(power_spectrum[sig_bin_start:sig_bin_end]))

    noise_distortion_spectrum = power_spectrum.copy()
    noise_distortion_spectrum[: min(side_bin, len(noise_distortion_spectrum))] = 0.0
    noise_distortion_spectrum[sig_bin_start:sig_bin_end] = 0.0
    noise_distortion = float(np.sum(noise_distortion_spectrum[:n_inband]))

    sndr_dbc = 10 * np.log10(sig_linear / (noise_distortion + 1e-20))
    enob = (sndr_dbc - 1.76) / 6.02
    return {"sndr_dbc": float(sndr_dbc), "enob": float(enob)}
