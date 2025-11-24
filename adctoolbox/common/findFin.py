from .sineFit import sine_fit


def findFin(data, Fs=1):
    """
    Find the fundamental frequency of the given signal.

    Parameters:
    - data: The input signal data.
    - Fs: Sampling frequency (default is 1).

    Returns:
    - fin: The fundamental frequency of the signal (in Hz).
    """
    # Use sine_fit to fit sine wave and get frequency
    data_fit, freq, mag, dc, phi = sine_fit(data)
    
    # 将估算的频率乘以采样频率
    fin = freq * Fs
    
    return fin
