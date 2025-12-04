import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from .spec_plot import spec_plot


def err_envelope_spectrum(err_data, fs=1):
    """
    Compute envelope spectrum using Hilbert transform.

    Parameters:
        err_data: Error signal (1D array)
        fs: Sampling frequency (default: 1)

    Returns:
        enob: Effective Number of Bits
        sndr: Signal-to-Noise and Distortion Ratio (dB)
        sfdr: Spurious-Free Dynamic Range (dB)
        snr: Signal-to-Noise Ratio (dB)
        thd: Total Harmonic Distortion (dB)
        signal_power: Signal power (dB)
        noise_floor: Noise floor (dB)
    """
    # Ensure column data
    e = np.asarray(err_data).flatten()

    # Envelope extraction via Hilbert transform
    env = np.abs(hilbert(e))

    # Use spec_plot for spectrum analysis (spec_plot will handle closing its own figure)
    enob, sndr, sfdr, snr, thd, signal_power, noise_floor, noise_spectral_density, _ = spec_plot(env, fs=fs, label=0)

    # Update labels with larger fonts to match MATLAB
    plt.grid(True)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Envelope Spectrum (dB)", fontsize=14)
    plt.gca().tick_params(labelsize=14)

    # Note: spec_plot already closes its figure, so no need to close here

    return enob, sndr, sfdr, snr, thd, signal_power, noise_floor
