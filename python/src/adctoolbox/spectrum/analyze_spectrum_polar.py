"""Analyze spectrum with polar phase visualization (FFT coherent mode).

This module provides a high-level wrapper that combines FFT coherent spectrum
calculation with polar phase visualization.

Part of the modular ADC analysis architecture.
Matches MATLAB plotphase.m FFT mode functionality.
"""

import numpy as np
from typing import Any

from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.spectrum.plot_spectrum_polar import plot_spectrum_polar

def analyze_spectrum_polar(
    data: np.ndarray,
    max_code: float | None = None,
    harmonic: int = 5,
    osr: int = 1,
    cutoff_freq: float = 0,
    fs: float = 1.0,
    win_type: str = 'boxcar',
    create_plot: bool = True, ax=None,
    fixed_radial_range: float | None = None
) -> dict[str, Any]:
    """
    Polar phase spectrum analysis and plotting. (Wrapper function for modular core and plotting)

    This function calculates coherent spectrum with phase alignment and optionally plots it in polar format.

    Parameters:
        data: Input ADC data, shape (N,) or (M, N)
        max_code: Maximum code level for normalization. If None, uses (max - min)
        harmonic: Number of harmonics to mark on polar plot
        osr: Oversampling ratio
        cutoff_freq: High-pass cutoff frequency in Hz
        fs: Sampling frequency in Hz
        win_type: Window function type ('boxcar', 'hann', 'hamming')
        create_plot: Plot the polar spectrum (True) or not (False)
        ax: Optional matplotlib polar axes object. If None and create_plot=True, uses current axes.
        fixed_radial_range: Fixed radial range in dB. If None, auto-scales.

    Returns:
        dict: Full results dictionary from compute_spectrum with coherent_averaging=True
    """

    # 1. --- Core Calculation ---
    results = compute_spectrum(
        data=data,
        max_scale_range=max_code,
        osr=osr,
        cutoff_freq=cutoff_freq,
        fs=fs,
        win_type=win_type,
        coherent_averaging=True,
    )

    # 2. --- Optional Plotting ---
    if create_plot:
        plot_spectrum_polar(results, harmonic=harmonic, ax=ax, fixed_radial_range=fixed_radial_range)

    return results['metrics']
