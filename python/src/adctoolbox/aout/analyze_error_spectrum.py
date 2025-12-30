"""Error spectrum analysis.

Analyzes the spectrum of the fitting error signal directly to reveal
frequency components and error characteristics.
"""

import matplotlib.pyplot as plt

from adctoolbox.spectrum import analyze_spectrum
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param

def analyze_error_spectrum(signal, fs=1, frequency=None, create_plot: bool = True,
                           ax=None, title: str = None):
    """
    Compute error spectrum directly from the error signal.

    This function fits an ideal sine to the signal, computes the error,
    and analyzes the spectrum of the error signal (not envelope).

    Parameters
    ----------
    signal : np.ndarray
        ADC output signal (1D array)
    fs : float, default=1
        Sampling frequency in Hz
    frequency : float, optional
        Normalized frequency (0-0.5). If None, auto-detected
    create_plot : bool, default=True
        If True, plot the error spectrum on current axes
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes (plt.gca())
    title : str, optional
        Title for the plot. If None, no title is set

    Returns
    -------
    result : dict
        Dictionary containing spectrum analysis results:
        - 'enob': Effective Number of Bits
        - 'sndr_db': Signal-to-Noise and Distortion Ratio (dB)
        - 'sfdr_db': Spurious-Free Dynamic Range (dB)
        - 'snr_db': Signal-to-Noise Ratio (dB)
        - 'thd_db': Total Harmonic Distortion (dB)
        - 'sig_pwr_dbfs': Signal power (dBFS)
        - 'noise_floor_dbfs': Noise floor (dBFS)
        - 'error_signal': Error signal (signal - fitted sine)

    Notes
    -----
    - Error = signal - ideal_sine (fitted using fit_sine_4param)
    - Analyzes spectrum of error directly (no envelope extraction)
    - Reveals frequency components in the error signal
    """

    # Fit ideal sine to extract reference
    if frequency is None:
        fit_result = fit_sine_4param(signal)
    else:
        fit_result = fit_sine_4param(signal, frequency_estimate=frequency)

    sig_ideal = fit_result['fitted_signal']

    # Compute error
    error_signal = signal - sig_ideal

    # Analyze error spectrum directly (not envelope)
    if create_plot:
        # Use provided axes or set current axes
        if ax is not None:
            plt.sca(ax)

        result = analyze_spectrum(error_signal, fs=fs, show_label=False, max_harmonic=5)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Error Spectrum (dB)")
        plt.grid(True, alpha=0.3)

        # Set title if provided
        if title is not None:
            plt.gca().set_title(title, fontsize=10, fontweight='bold')
    else:
        # Analyze without plotting
        import matplotlib
        backend_orig = matplotlib.get_backend()
        matplotlib.use('Agg')  # Non-interactive backend

        result = analyze_spectrum(error_signal, fs=fs, show_label=False, max_harmonic=5)
        plt.close()

        matplotlib.use(backend_orig)  # Restore original backend

    # Add error signal to results
    result['error_signal'] = error_signal

    return result
