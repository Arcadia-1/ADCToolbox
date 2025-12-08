"""
ADC spectrum analysis with coherent averaging (complex spectrum mode).

This wrapper combines calculate_spectrum_data with complex_spectrum=True
and optional plotting for coherent spectrum averaging analysis.

MATLAB counterpart: Coherent FFT averaging analysis
"""

import numpy as np
from .calculate_spectrum_data import calculate_spectrum_data
from .plot_spectrum import plot_spectrum


def analyze_spectrum_coherent_averaging(data, fs=1.0, max_scale_range=None, harmonic=5,
                                       win_type='boxcar', side_bin=1, freq_scale='linear',
                                       show_label=True, is_plot=1, n_thd=5, osr=1,
                                       cutoff_freq=0, ax=None, log_sca=None, label=None):
    """
    Coherent spectrum analysis with averaging and optional plotting.

    This function performs coherent (complex) spectrum averaging, which:
    - Aligns signal phase across multiple runs (coherent summation)
    - Reduces noise incoherently (√M improvement for M runs)
    - Preserves harmonic relationships (SFDR/THD constant)

    Parameters:
        data: Input data (N,) for single run or (M, N) for M runs
        fs: Sampling frequency (Hz)
        max_scale_range: Maximum scale range for normalization
        harmonic: Number of harmonics to analyze (default: 5)
        win_type: Window function type ('boxcar', 'hann', 'hamming')
                  Default 'boxcar' for coherent mode (no spectral leakage)
        side_bin: Number of side bins around fundamental for signal exclusion
        freq_scale: Frequency scale - 'linear' or 'log'
        show_label: Add labels and annotations (True) or not (False)
        is_plot: Plot the spectrum (1) or not (0)
        n_thd: Number of harmonics for THD calculation
        osr: Oversampling ratio
        cutoff_freq: High-pass cutoff frequency (Hz) to remove low-frequency noise
        ax: Optional matplotlib axes object. If None and is_plot=1, creates new figure.
        log_sca: Deprecated. Use freq_scale instead.
        label: Deprecated. Use show_label instead.

    Returns:
        dict: Dictionary containing:
            - Complex spectrum data:
              - complex_spec_coherent: Phase-aligned complex FFT
              - spec_mag_db: Magnitude spectrum in dBFS
              - freq: Frequency axis (Hz)
              - bin_r: Refined fundamental frequency bin
            - Metrics (if available):
              - metrics: Dictionary with enob, sndr_db, sfdr_db, snr_db, thd_db, etc.
              - minR_dB: Noise floor level
            - Plot data:
              - All data needed by plot_spectrum()

    Example:
        >>> # Single run coherent analysis
        >>> result = analyze_spectrum_coherent_averaging(signal_data, fs=100e6)
        >>>
        >>> # Multiple runs for noise reduction
        >>> result = analyze_spectrum_coherent_averaging(signal_matrix, fs=100e6)
        >>> print(f"Noise floor: {result['minR_dB']:.1f} dB")
    """

    # Handle deprecated parameters for backward compatibility
    if log_sca is not None:
        freq_scale = 'log' if log_sca else 'linear'
    if label is not None:
        show_label = bool(label)

    # 1. --- Core Calculation (Coherent/Complex Mode) ---
    # Use calculate_spectrum_data with complex_spectrum=True for coherent averaging
    result = calculate_spectrum_data(
        data=data,
        fs=fs,
        max_scale_range=max_scale_range,
        complex_spectrum=True,  # Enable coherent averaging mode
        win_type=win_type,
        cutoff_freq=cutoff_freq
    )

    # 2. --- Optional Plotting ---
    if is_plot:
        # plot_spectrum expects a single analysis_results dict with all data and metrics
        plot_spectrum(
            analysis_results=result,
            show_label=show_label,
            plot_harmonics_up_to=harmonic,
            ax=ax
        )

    return result
