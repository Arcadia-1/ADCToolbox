# Not yet verified with both MATLAB and Python testbenches
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def ntf_analyzer(NTF, Flow, Fhigh, isPlot=None):
    """
    Analyze the performance of NTF (Noise Transfer Function)
    
    Args:
        NTF: The noise transfer function (in z domain) - scipy.signal.TransferFunction or tuple (num, den)
        Flow: Low bound frequency of signal band (relative to Fs)
        Fhigh: High bound frequency of signal band (relative to Fs)
        isPlot: Optional plotting flag (1 to plot, None or 0 to skip)
        
    Returns:
        noiSup: Integrated noise suppression of NTF in signal band in dB (compared to NTF=1)
    """
    w = np.linspace(0, 0.5, 2**16)
    
    # Convert NTF to transfer function if needed
    if isinstance(NTF, tuple):
        # NTF is (numerator, denominator) tuple
        num, den = NTF
        tf = signal.TransferFunction(num, den, dt=1)  # dt=1 for discrete time
    else:
        tf = NTF
    
    # Calculate frequency response
    w_rad = w * 2 * np.pi
    _, mag = signal.freqresp(tf, w_rad)
    mag = np.abs(mag)
    
    # Calculate noise suppression in signal band
    band_mask = (w > Flow) & (w < Fhigh)
    np_val = np.sum(mag[band_mask]**2) / len(w)
    noiSup = -10 * np.log10(np_val)
    
    # Plot if requested
    if isPlot == 1:
        plt.semilogx(w, 20 * np.log10(mag))
        plt.hold(True)
        
        if Flow > 0:
            plt.semilogx([Flow, Flow], 20 * np.log10([np.min(mag), np.max(mag)]), 'k--')
        
        plt.semilogx([Fhigh, Fhigh], 20 * np.log10([np.min(mag), np.max(mag)]), 'k--')
        plt.xlabel('Normalized Frequency')
        plt.ylabel('Magnitude (dB)')
        plt.title('NTF Frequency Response')
        plt.grid(True)
        plt.show()
    
    return noiSup
