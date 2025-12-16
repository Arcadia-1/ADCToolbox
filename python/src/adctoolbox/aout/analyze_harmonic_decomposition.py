"""
Analyze harmonic decomposition of signal.

Wrapper function combining computation and optional plotting for convenience.
"""

from .compute_harmonic_decomposition import compute_harmonic_decomposition

try:
    from .plot_decomposition_time import plot_decomposition_time
except ImportError:
    plot_decomposition_time = None


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
        Fundamental sinewave component
    harmonic_signal : ndarray
        Harmonic distortions (2nd through nth harmonics)
    signal_reconstructed : ndarray
        Complete reconstructed signal (fundamental + harmonics)
    residual : ndarray
        Residual/noise (data - signal_reconstructed)

    Notes
    -----
    The decomposition uses the following model:

    - fundamental_signal = weight_i*cos(ωt) + weight_q*sin(ωt)
    - signal_reconstructed = Σ[weight_i_k*cos(kωt) + weight_q_k*sin(kωt)] k=1..order
    - harmonic_signal = signal_reconstructed - fundamental_signal
    - residual = data - signal_reconstructed
    """

    # 1. --- Core Computation ---
    # Note: compute_harmonic_decomposition uses 'harmonic' parameter, not 'order'
    results = compute_harmonic_decomposition(data, max_code=None, harmonic=order)

    # 2. --- Optional Plotting ---
    if show_plot:
        if plot_decomposition_time is not None:
            # Prepare plot_data with required keys
            plot_data = results.copy()
            plot_data['signal'] = data
            plot_data['fundamental_freq'] = 0.1  # Default value, could be improved
            plot_decomposition_time(plot_data)
        else:
            import warnings
            warnings.warn("plot_decomposition_time not available, skipping plot")

    # 3. --- Return Results ---
    return (results['fundamental_signal'], results['harmonic_signal'],
            results['signal_reconstructed'], results['residual'])
