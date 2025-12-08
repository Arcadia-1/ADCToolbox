"""
ADC spectrum analysis with ENOB, SNDR, SFDR, SNR, THD, Noise Floor, NSD calculations.

MATLAB counterpart: specPlot.m, plotspec.m

This is a wrapper function that combines core FFT calculations and plotting
for backward compatibility with existing code.
"""

import numpy as np
from adctoolbox.aout.calculate_spectrum_data import calculate_spectrum_data
from adctoolbox.aout.plot_spectrum import plot_spectrum


def analyze_spectrum(data, fs=1.0, max_code=None, harmonic=3, win_type='hann',
              side_bin=1, freq_scale='linear', show_label=True, assumed_signal=np.nan, is_plot=1,
              n_thd=5, osr=1, co_avg=0, nf_method=0, ax=None, log_sca=None, label=None):
    """
    Spectral analysis and plotting. (Wrapper function for modular core and plotting)

    This function first calculates all metrics and then conditionally plots the spectrum.

    Parameters:
        data: Input data (N,) or (M, N)
        fs: Sampling frequency
        max_code: Maximum code level for normalization
        harmonic: Number of harmonics to analyze
        win_type: Window function type ('hann', 'hamming', 'boxcar')
        side_bin: Number of side bins around fundamental
        freq_scale: Frequency scale - 'linear' or 'log'
        show_label: Add labels and annotations (True) or not (False)
        assumed_signal: Pre-defined signal level in dBFS
        is_plot: Plot the spectrum (1) or not (0)
        n_thd: Number of harmonics for THD calculation
        osr: Oversampling ratio
        co_avg: Coherent averaging flag
        nf_method: Noise floor calculation method (0=median, 1=trimmed mean, 2=sum without harmonics)
        ax: Optional matplotlib axes object. If None and is_plot=1, a new figure is created.
        log_sca: Deprecated. Use freq_scale instead.
        label: Deprecated. Use show_label instead.

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

    # Handle deprecated parameters for backward compatibility
    if log_sca is not None:
        freq_scale = 'log' if log_sca else 'linear'
    if label is not None:
        show_label = bool(label)

    # 1. --- Core Calculation ---
    # Pass all relevant parameters to the pure calculation kernel.
    results = calculate_spectrum_data(
        data=data, fs=fs, max_code=max_code, win_type=win_type,
        side_bin=side_bin, assumed_signal=assumed_signal, n_thd=n_thd,
        osr=osr, calc_metrics=True
    )

    # Extract metrics and plot_data from results
    metrics = results.get('metrics', {})
    plot_data = results

    # 2. --- Optional Plotting ---
    if is_plot:
        # Pass the calculated metrics and plot data to the pure plotting function.
        plot_spectrum(
            metrics=metrics,
            plot_data=plot_data,
            harmonic=harmonic,
            freq_scale=freq_scale,
            show_label=show_label,
            ax=ax
        )

    return metrics