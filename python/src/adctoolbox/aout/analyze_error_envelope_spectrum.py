"""
Error envelope spectrum analysis using Hilbert transform.

Extracts envelope spectrum to reveal amplitude modulation patterns.

MATLAB counterpart: errevspec.m
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import hilbert
from adctoolbox.spectrum import analyze_spectrum
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param

def analyze_error_envelope_spectrum(signal, fs=1, frequency=None, create_plot: bool = True,
                                     ax=None, title: str = None):
    """
    Compute envelope spectrum using Hilbert transform.

    This function fits an ideal sine to the signal, computes the error,
    extracts the error envelope using Hilbert transform, and analyzes
    its spectrum to reveal amplitude modulation patterns.

    Parameters
    ----------
    signal : np.ndarray
        ADC output signal (1D array)
    fs : float, default=1
        Sampling frequency in Hz
    frequency : float, optional
        Normalized frequency (0-0.5). If None, auto-detected
    create_plot : bool, default=True
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
        - 'noise_floor_dbfs': Noise floor (dBFS)
        - 'error_signal': Error signal (signal - fitted sine)
        - 'envelope': Error envelope extracted via Hilbert transform

    Notes
    -----
    - Error = signal - ideal_sine (fitted using fit_sine_4param)
    - Envelope = |Hilbert(error)|
    - Analyzes spectrum of envelope to reveal AM patterns
    """
    # Fit ideal sine to extract reference
    if frequency is None:
        fit_result = fit_sine_4param(signal)
    else:
        fit_result = fit_sine_4param(signal, frequency_estimate=frequency)

    sig_ideal = fit_result['fitted_signal']

    # Compute error
    error_signal = signal - sig_ideal

    # Ensure column data
    e = np.asarray(error_signal).flatten()

    # Envelope extraction via Hilbert transform
    env = np.abs(hilbert(e))

    # Analyze envelope spectrum
    if create_plot:
        # Use provided axes or set current axes
        if ax is not None:
            plt.sca(ax)

        result = analyze_spectrum(env, fs=fs, show_label=False, max_harmonic=5)
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

        result = analyze_spectrum(env, fs=fs, show_label=False, max_harmonic=5)
        plt.close()

        matplotlib.use(backend_orig)  # Restore original backend

    # Add error signal and envelope to result
    result['error_signal'] = error_signal
    result['envelope'] = env

    return result
