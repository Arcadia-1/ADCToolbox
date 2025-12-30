"""
Two-tone spectrum analysis for intermodulation distortion (IMD).

Wrapper function combining calculation and plotting following the modular pattern.
Measures IMD2, IMD3 and other spectral metrics with two-tone input.

MATLAB counterpart: specPlot2Tone.m
"""

import numpy as np

from adctoolbox.spectrum.compute_two_tone_spectrum import compute_two_tone_spectrum
from adctoolbox.spectrum.plot_two_tone_spectrum import plot_two_tone_spectrum

def analyze_two_tone_spectrum(
    data: np.ndarray,
    fs: float = 1.0,
    max_scale_range: float | None = None,
    harmonic: int = 3,
    win_type: str = 'hann',
    side_bin: int = 1,
    coherent_averaging: bool = False,
    create_plot: bool = True,
    show_title: bool = True,
    show_labels: bool = True, ax=None
) -> dict:
    """
    Two-tone spectrum analysis with IMD calculation. (Wrapper function for modular core and plotting)

    This function calculates IMD metrics and optionally plots the two-tone spectrum.

    Parameters:
        data: ADC output data, shape (M, N) for M runs or (N,) for single run
        fs: Sampling frequency (Hz)
        max_scale_range: Full scale range (max-min) for normalization
        harmonic: Number of harmonics to mark on plot
        win_type: Window function type ('hann', 'blackman', 'hamming', 'boxcar')
        side_bin: Number of side bins around fundamental
        coherent_averaging: If True, performs coherent averaging with phase alignment
        create_plot: Plot the spectrum (True) or not (False)
        show_title: Display title (True) or not (False)
        show_labels: Add labels and annotations (True) or not (False)
        ax: Optional matplotlib axes object. If None and create_plot=True, a new figure is created.

    Returns:
        dict: Dictionary with performance metrics:
            - enob: Effective Number of Bits
            - sndr_db: Signal-to-Noise and Distortion Ratio (dB)
            - sfdr_db: Spurious-Free Dynamic Range (dB)
            - snr_db: Signal-to-Noise Ratio (dB)
            - thd_db: Total Harmonic Distortion (dB)
            - signal_power_1_dbfs: Power of first tone (dBFS)
            - signal_power_2_dbfs: Power of second tone (dBFS)
            - noise_floor_db: Noise floor (dB)
            - imd2_dbc: 2nd order intermodulation distortion (dBc)
            - imd3_dbc: 3rd order intermodulation distortion (dBc)
    """

    # Step 1: Calculate spectrum data (pure computation)
    results = compute_two_tone_spectrum(
        data=data,
        fs=fs,
        max_scale_range=max_scale_range,
        win_type=win_type,
        side_bin=side_bin,
        harmonic=harmonic,
        coherent_averaging=coherent_averaging
    )

    # Step 2: Plot if requested (pure visualization)
    if create_plot:
        plot_two_tone_spectrum(
            analysis_results=results,
            harmonic=harmonic,
            ax=ax,
            show_title=show_title,
            show_labels=show_labels
        )

    # Step 3: Return metrics dictionary
    return results['metrics']

