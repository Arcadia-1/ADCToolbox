"""Auto-detect coherent main-lobe width (MATLAB plotspec sideBin='auto')."""

import numpy as np

from adctoolbox.spectrum._spectrum_averaging import _power_average


def _detect_side_bin_auto(
    power_spectrum: np.ndarray,
    fundamental_bin: int,
    fundamental_bin_fractional: float,
    n_inband: int,
    n_fft: int,
    window_vector: np.ndarray,
    power_correction: float,
) -> int:
    """Match ``plotspec.m`` sideBin auto: ideal leakage vs median noise floor."""
    t = np.arange(n_fft, dtype=float)
    ideal = np.sin(2 * np.pi * fundamental_bin_fractional * t / n_fft)
    win_norm = window_vector / np.sqrt(np.mean(window_vector**2))
    ideal_spec, _ = _power_average((ideal * win_norm).reshape(1, -1))
    ideal_spec *= power_correction

    peak = max(power_spectrum[fundamental_bin], 1e-30)
    ideal_spec *= peak / max(ideal_spec[fundamental_bin], 1e-30)

    inband = power_spectrum[:n_inband]
    positive = inband[inband > 1e-20]
    if len(positive) >= 3:
        noise_floor_per_bin = float(np.median(positive))
    else:
        noise_floor_per_bin = float(np.median(inband))

    max_sidebin = min(fundamental_bin, n_inband - 1 - fundamental_bin)
    side_bin = max_sidebin
    for sb in range(1, max_sidebin + 1):
        left = fundamental_bin - sb
        right = fundamental_bin + sb
        left_below = left >= 0 and ideal_spec[left] <= noise_floor_per_bin
        right_below = right < n_inband and ideal_spec[right] <= noise_floor_per_bin
        if left_below and right_below:
            side_bin = sb - 1
            break
    return int(max(side_bin, 0))
