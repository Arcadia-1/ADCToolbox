"""
ADC spectrum analysis with ENOB, SNDR, SFDR, SNR, THD, Noise Floor, NSD calculations.

MATLAB counterpart: specPlot.m, plotspec.m

This is a wrapper function that combines core FFT calculations and plotting
for backward compatibility with existing code.
"""

import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.spectrum.plot_spectrum import plot_spectrum

def analyze_spectrum(data, fs=1.0, osr=1, max_scale_range=None, win_type='hann', side_bin=None,
                     max_harmonic=5, nf_method=2, assumed_sig_pwr_dbfs=np.nan, coherent_averaging=False,
                     create_plot: bool = True, show_title=True, show_label=True, plot_harmonics_up_to=3, ax=None):
    """
    Spectral analysis and plotting. (Wrapper function for modular core and plotting)

    This function first calculates all metrics and then conditionally plots the spectrum.

    Parameters:
        data: Input data (N,) or (M, N)
        fs: Sampling frequency
        max_scale_range: Full scale range for normalization.
            Can be: scalar (direct range), tuple/list [min, max], or None (auto-detect)
        win_type: Window function type ('hann', 'hamming', 'boxcar')
        side_bin: Number of side bins around fundamental (None for automatic selection)
        osr: Oversampling ratio
        max_harmonic: Number of harmonics for THD calculation
        nf_method: Noise floor calculation method (0=median, 1=trimmed mean, 2=exclude harmonics)
        assumed_sig_pwr_dbfs: Pre-defined signal level in dBFS
        create_plot: Plot the spectrum (True) or not (False)
        show_title: Display auto-generated title (True) or not (False)
        show_label: Add labels and annotations (True) or not (False)
        plot_harmonics_up_to: Number of harmonics to mark on the plot
        ax: Optional matplotlib axes object. If None and create_plot=True, a new figure is created.

    Returns:
        dict: Dictionary with performance metrics:
            - enob: Effective Number of Bits
            - sndr_dbc: Signal-to-Noise and Distortion Ratio (dBc)
            - sfdr_dbc: Spurious-Free Dynamic Range (dBc)
            - snr_dbc: Signal-to-Noise Ratio (dBc)
            - thd_dbc: Total Harmonic Distortion (dBc)
            - sig_pwr_dbfs: Signal power (dBFS)
            - noise_floor_dbfs: Noise floor (dBFS)
            - nsd_dbfs_hz: Noise Spectral Density (dBFS/Hz)
    """

    # 1. --- Core Calculation ---
    # Pass all relevant parameters to the pure calculation kernel.
    results = compute_spectrum(
        data=data,
        fs=fs,
        max_scale_range=max_scale_range,
        win_type=win_type,
        side_bin=side_bin,
        osr=osr,
        max_harmonic=max_harmonic,
        nf_method=nf_method,
        coherent_averaging=coherent_averaging,
        assumed_sig_pwr_dbfs=assumed_sig_pwr_dbfs
    )

    # Print warning if harmonics collide with fundamental
    collided = results['plot_data'].get('collided_harmonics', [])
    if collided and show_label:
        print(f"[Warning from analyze_spectrum]: Harmonics {collided} alias to fundamental (excluded from THD)")

    # 2. --- Optional Plotting ---
    if create_plot:
        # Pass the analysis results to the pure plotting function.
        plot_spectrum(
            compute_results=results,
            show_title=show_title,
            show_label=show_label,
            plot_harmonics_up_to=plot_harmonics_up_to,
            ax=ax
        )

    return results['metrics']