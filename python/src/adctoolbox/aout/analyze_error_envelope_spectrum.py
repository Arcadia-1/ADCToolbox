"""
Error envelope spectrum analysis using Hilbert transform.

Extracts envelope spectrum to reveal amplitude modulation patterns.

MATLAB counterpart: errevspec.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.signal import hilbert
from adctoolbox.spectrum import analyze_spectrum


def analyze_error_envelope_spectrum(err_data, fs=1, show_plot=True,
                                     ax: Optional[plt.Axes] = None, title: str = None):
    """
    Compute envelope spectrum using Hilbert transform.

    Parameters
    ----------
    err_data : array_like
        Error signal (1D array)
    fs : float, default=1
        Sampling frequency in Hz
    show_plot : bool, default=True
        If True, plot the envelope spectrum on current axes
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes (plt.gca())
    title : str, optional
        Title for the plot. If None, no title is set

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'enob': Effective Number of Bits
        - 'sndr_db': Signal-to-Noise and Distortion Ratio (dB)
        - 'sfdr_db': Spurious-Free Dynamic Range (dB)
        - 'snr_db': Signal-to-Noise Ratio (dB)
        - 'thd_db': Total Harmonic Distortion (dB)
        - 'sig_pwr_dbfs': Signal power (dBFS)
        - 'noise_floor_db': Noise floor (dB)
    """
    # Ensure column data
    e = np.asarray(err_data).flatten()

    # Envelope extraction via Hilbert transform
    env = np.abs(hilbert(e))

    # Analyze envelope spectrum
    if show_plot:
        # Use provided axes or set current axes
        if ax is not None:
            plt.sca(ax)

        result = analyze_spectrum(env, fs=fs, show_label=False, n_thd=5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Envelope Spectrum (dB)")
        plt.grid(True, alpha=0.3)

        # Set title if provided
        if title is not None:
            plt.gca().set_title(title, fontsize=10, fontweight='bold')
    else:
        # Analyze without plotting
        import matplotlib
        backend_orig = matplotlib.get_backend()
        matplotlib.use('Agg')  # Non-interactive backend

        result = analyze_spectrum(env, fs=fs, show_label=False, n_thd=5)
        plt.close()

        matplotlib.use(backend_orig)  # Restore original backend

    return result
