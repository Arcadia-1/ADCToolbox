"""
ADC spectrum analysis with coherent averaging (complex spectrum mode).

This wrapper combines compute_spectrum with complex_spectrum=True
and optional plotting for coherent spectrum averaging analysis.

MATLAB counterpart: Coherent FFT averaging analysis
"""

import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.spectrum.plot_spectrum import plot_spectrum


def analyze_spectrum_coherent_averaging(data, fs=1.0, osr=1, max_scale_range=None,
                                       win_type='boxcar', side_bin=1, n_thd=5,
                                       cutoff_freq=0, show_plot=True, show_label=True,
                                       plot_harmonics_up_to=5, ax=None):
    """
    Coherent spectrum analysis with averaging and optional plotting.

    This function performs coherent (complex) spectrum averaging, which:
    - Aligns signal phase across multiple runs (coherent summation)
    - Reduces noise incoherently (√M improvement for M runs)
    - Preserves harmonic relationships (SFDR/THD constant)

    Parameters:
        data: Input data (N,) or (M, N)
        fs: Sampling frequency
        osr: Oversampling ratio
        max_scale_range: Full scale range (max-min) for normalization
        win_type: Window function type ('boxcar', 'hann', 'hamming')
                  Default 'boxcar' for coherent mode (no spectral leakage)
        side_bin: Number of side bins around fundamental
        n_thd: Number of harmonics for THD calculation
        cutoff_freq: High-pass cutoff frequency (Hz) to remove low-frequency noise
        show_plot: Plot the spectrum (True) or not (False)
        show_label: Add labels and annotations (True) or not (False)
        plot_harmonics_up_to: Number of harmonics to mark on the plot
        ax: Optional matplotlib axes object. If None and show_plot=True, a new figure is created.

    Returns:
        dict: Dictionary with performance metrics:
            - enob: Effective Number of Bits
            - sndr_db: Signal-to-Noise and Distortion Ratio (dB)
            - sfdr_db: Spurious-Free Dynamic Range (dB)
            - snr_db: Signal-to-Noise Ratio (dB)
            - thd_db: Total Harmonic Distortion (dB)
            - sig_pwr_dbfs: Signal power (dBFS)
            - noise_floor_db: Noise floor (dB)
            - nsd_dbfs_hz: Noise Spectral Density (dBFS/Hz)
    """

    # 1. --- Core Calculation (Coherent/Complex Mode) ---
    # Use compute_spectrum with complex_spectrum=True for coherent averaging
    results = compute_spectrum(
        data=data,
        fs=fs,
        max_scale_range=max_scale_range,
        win_type=win_type,
        side_bin=side_bin,
        osr=osr,
        n_thd=n_thd,
        complex_spectrum=True,  # Enable coherent averaging mode
        cutoff_freq=cutoff_freq
    )

    # 2. --- Optional Plotting ---
    if show_plot:
        # Pass the analysis results to the pure plotting function.
        plot_spectrum(
            analysis_results=results,
            show_label=show_label,
            plot_harmonics_up_to=plot_harmonics_up_to,
            ax=ax
        )

    return results['metrics']
