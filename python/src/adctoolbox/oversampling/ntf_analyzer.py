# Not yet verified with both MATLAB and Python testbenches
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def ntf_analyzer(ntf, flow, fhigh, is_plot=None):
    """
    Analyze the performance of NTF (Noise Transfer Function)

    Args:
        ntf: The noise transfer function (in z domain) - scipy.signal.TransferFunction or tuple (num, den)
        flow: Low bound frequency of signal band (relative to Fs)
        fhigh: High bound frequency of signal band (relative to Fs)
        is_plot: Optional plotting flag (1 to plot, None or 0 to skip)

    Returns:
        noiSup: Integrated noise suppression of NTF in signal band in dB (compared to NTF=1)
    """
    w = np.linspace(0, 0.5, 2**16)

    # Convert NTF to transfer function if needed
    if isinstance(ntf, tuple):
        # ntf is (numerator, denominator) tuple
        num, den = ntf
        tf = signal.TransferFunction(num, den, dt=1)  # dt=1 for discrete time
    else:
        tf = ntf

    # Calculate frequency response
    w_rad = w * 2 * np.pi
    _, mag = signal.freqresp(tf, w_rad)
    mag = np.abs(mag)

    # Calculate noise suppression in signal band
    band_mask = (w > flow) & (w < fhigh)
    np_val = np.sum(mag[band_mask]**2) / len(w)
    noiSup = -10 * np.log10(np_val)

    # Plot if requested
    if is_plot == 1:
        plt.semilogx(w, 20 * np.log10(mag))
        plt.hold(True)

        if flow > 0:
            plt.semilogx([flow, flow], 20 * np.log10([np.min(mag), np.max(mag)]), 'k--')

        plt.semilogx([fhigh, fhigh], 20 * np.log10([np.min(mag), np.max(mag)]), 'k--')
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Magnitude (dB)')
        plt.title('NTF Frequency Response')
        plt.grid(True)
        plt.show()

    return noiSup
