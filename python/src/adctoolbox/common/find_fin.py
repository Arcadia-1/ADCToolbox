from .sine_fit import sine_fit


def find_fin(data, fs=1):
    """
    Find the fundamental frequency of the given signal.

    Parameters:
    - data: The input signal data.
    - fs: Sampling frequency (default is 1).

    Returns:
    - fin: The fundamental frequency of the signal (in Hz).
    """
    # Use sine_fit to fit sine wave and get frequency
    data_fit, freq, mag, dc, phi = sine_fit(data)

    # Multiply the estimated frequency by the sampling frequency
    fin = freq * fs

    return fin
