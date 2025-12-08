"""
ADC spectrum analysis with coherent averaging (complex spectrum mode).

This wrapper combines calculate_spectrum_data with complex_spectrum=True
and optional plotting for coherent spectrum averaging analysis.

MATLAB counterpart: Coherent FFT averaging analysis
"""

import numpy as np
from adctoolbox.aout.calculate_spectrum_data import calculate_spectrum_data
from adctoolbox.aout.plot_spectrum import plot_spectrum


def analyze_spectrum_coherent_averaging(data, fs=1.0, max_code=None, harmonic=5,
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
        max_code: Maximum code level for normalization
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
        max_code=max_code,
        complex_spectrum=True,  # Enable coherent averaging mode
        win_type=win_type,
        cutoff_freq=cutoff_freq,
        calc_metrics=True  # Calculate performance metrics
    )

    # 2. --- Optional Plotting ---
    if is_plot:
        # Build metrics dict for plot_spectrum (use calculated metrics if available)
        if 'metrics' in result:
            metrics = result['metrics']
        else:
            # Placeholder metrics if calculation unavailable
            metrics = {
                'enob': 0.0,
                'sndr_db': 0.0,
                'sfdr_db': 0.0,
                'snr_db': 0.0,
                'thd_db': 0.0,
                'sig_pwr_dbfs': 0.0,
                'noise_floor_db': -100.0,
                'nsd_dbfs_hz': np.nan
            }

        # Use calculate_spectrum_data result directly (it now has all required keys)
        # No need to build plot_data manually - use result which already has the format plot_spectrum expects
        plot_spectrum(
            metrics=metrics,
            plot_data=result,
            harmonic=harmonic,
            freq_scale=freq_scale,
            show_label=show_label,
            ax=ax
        )

    return result
