import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from .spec_plot import spec_plot


def errEnvelopeSpectrum(err_data, Fs=1):
    """
    Compute envelope spectrum using Hilbert transform.

    Parameters:
        err_data: Error signal (1D array)
        Fs: Sampling frequency (default: 1)

    Returns:
        None (generates plot via spec_plot)
    """
    # Ensure column data
    e = np.asarray(err_data).flatten()

    # Envelope extraction via Hilbert transform
    env = np.abs(hilbert(e))

    # Use spec_plot for spectrum analysis (spec_plot will handle closing its own figure)
    spec_plot(env, Fs=Fs, label=0)

    # Update labels with larger fonts to match MATLAB
    plt.grid(True)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Envelope Spectrum (dB)", fontsize=14)
    plt.gca().tick_params(labelsize=14)

    # Note: spec_plot already closes its figure, so no need to close here
