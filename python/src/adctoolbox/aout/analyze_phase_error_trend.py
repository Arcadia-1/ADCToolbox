"""
Analyze phase error using binned approach (trend analysis).

Wrapper function combining computation and optional plotting for phase error
analysis using binned data.
"""

from typing import Tuple
import numpy as np
from .compute_phase_error_from_binned import compute_phase_error_from_binned


def analyze_phase_error_trend(
    signal: np.ndarray,
    normalized_freq: float = None,
    bin_count: int = 100,
    show_plot: bool = True
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Analyze phase error using binned approach for trend analysis.

    Wrapper function that combines core computation and optional plotting
    for convenience. This method bins error by phase to separate amplitude
    and phase noise components.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    normalized_freq : float, optional
        Normalized frequency (f/fs), range 0-0.5. If None, auto-detected.
    bin_count : int, default=100
        Number of phase bins for binning error data.
    show_plot : bool, default=True
        Whether to display result plot.

    Returns
    -------
    am_param : float
        Amplitude modulation parameter (amplitude noise).
    pm_param : float
        Phase modulation parameter (phase noise in radians).
    baseline : float
        Baseline noise floor.
    erms : np.ndarray
        RMS error per phase bin.
    emean : np.ndarray
        Mean error per phase bin.

    Notes
    -----
    The binned approach provides robust trend estimates by averaging error
    within each phase bin. It separates:

    - AM (amplitude modulation): Noise constant across phase
    - PM (phase modulation): Phase jitter varying as sin(phase)

    The model is:
        RMS²(φ) = am_param² * cos²(φ) + pm_param² * sin²(φ) + baseline

    Examples
    --------
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000)) + 0.01*np.random.randn(1000)
    >>> am, pm, bl, erms, emean = analyze_phase_error_trend(sig, 0.1)
    >>> print(f"Amplitude noise: {am:.3e}")
    >>> print(f"Phase noise: {pm:.3e} rad")
    """
    # Auto-detect frequency if not provided
    if normalized_freq is None or np.isnan(normalized_freq):
        try:
            from findFin import findFin
            normalized_freq = findFin(signal)
        except ImportError:
            # FFT-based frequency detection
            signal = np.asarray(signal).flatten()
            spec = np.abs(np.fft.fft(signal))
            spec[0] = 0
            normalized_freq = np.argmax(spec[:len(signal)//2]) / len(signal)
            print(f"Warning: findFin not found, using FFT detection: freq={normalized_freq:.6f}")

    # 1. --- Core Computation ---
    results = compute_phase_error_from_binned(signal, normalized_freq, bin_count)

    # 2. --- Optional Plotting ---
    if show_plot:
        try:
            from .plot_error_binned_phase import plot_error_binned_phase
            plot_error_binned_phase(results)
        except ImportError:
            print("Warning: plot_error_binned_phase not available, skipping plot")

    # 3. --- Return Results ---
    return (
        results['am_param'],
        results['pm_param'],
        results['baseline'],
        results['erms'],
        results['emean']
    )
