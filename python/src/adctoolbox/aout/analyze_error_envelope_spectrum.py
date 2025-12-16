"""
Error envelope spectrum analysis using Hilbert transform.

Extracts envelope spectrum to reveal amplitude modulation patterns.

MATLAB counterpart: errevspec.m
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from adctoolbox.spectrum import analyze_spectrum


def analyze_error_envelope_spectrum(err_data, fs=1):
    """
    Compute envelope spectrum using Hilbert transform.

    Parameters:
        err_data: Error signal (1D array)
        fs: Sampling frequency (default: 1)

    Returns:
        dict: Dictionary with keys:
            - enob: Effective Number of Bits
            - sndr_db: Signal-to-Noise and Distortion Ratio (dB)
            - sfdr_db: Spurious-Free Dynamic Range (dB)
            - snr_db: Signal-to-Noise Ratio (dB)
            - thd_db: Total Harmonic Distortion (dB)
            - sig_pwr_dbfs: Signal power (dBFS)
            - noise_floor_db: Noise floor (dB)
    """
    # Ensure column data
    e = np.asarray(err_data).flatten()

    # Envelope extraction via Hilbert transform
    env = np.abs(hilbert(e))

    # Use spec_plot for spectrum analysis (spec_plot will handle closing its own figure)
    # Use n_thd=5 to match MATLAB's default
    result = analyze_spectrum(env, fs=fs, show_label=False, n_thd=5)

    # Update labels with larger fonts to match MATLAB
    plt.grid(True)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Envelope Spectrum (dB)", fontsize=14)
    plt.gca().tick_params(labelsize=14)

    # Note: spec_plot already closes its figure, so no need to close here

    return result
