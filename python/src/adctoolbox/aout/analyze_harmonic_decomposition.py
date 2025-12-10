"""
Analyze harmonic decomposition of signal.

Wrapper function combining computation and optional plotting for convenience.
"""

from .compute_harmonic_decomposition import compute_harmonic_decomposition
from .plot_harmonic_decomposition_time import plot_harmonic_decomposition_time


def analyze_harmonic_decomposition(data, normalized_freq=None, order=10, show_plot=True):
    """
    Analyze and decompose signal into fundamental and harmonic errors.

    Wrapper function that combines core computation and optional plotting for convenience.

    Parameters
    ----------
    data : array_like
        ADC output data, 1D numpy array
    normalized_freq : float, optional
        Normalized frequency (f_in / f_sample), auto-detect if None
    order : int, default=10
        Harmonic order for fitting (fits fundamental + harmonics 2 through order)
    show_plot : bool, default=True
        Whether to display result plot

    Returns
    -------
    fundamental_signal : ndarray
        Fundamental sinewave component (including DC)
    total_error : ndarray
        Total error (data - fundamental_signal)
    harmonic_error : ndarray
        Harmonic distortions (2nd through nth harmonics)
    other_error : ndarray
        All other errors (data - all harmonics)

    Notes
    -----
    The decomposition uses the following model:

    - fundamental_signal = DC + weight_i*cos(ωt) + weight_q*sin(ωt)
    - signal_all = DC + Σ[weight_i_k*cos(kωt) + weight_q_k*sin(kωt)]
    - total_error = data - fundamental_signal
    - harmonic_error = signal_all - fundamental_signal
    - other_error = data - signal_all
    """

    # 1. --- Core Computation ---
    results = compute_harmonic_decomposition(data, normalized_freq, order)

    # 2. --- Optional Plotting ---
    if show_plot:
        plot_harmonic_decomposition_time(results)

    # 3. --- Return Results ---
    return (results['fundamental_signal'], results['total_error'],
            results['harmonic_error'], results['other_error'])
