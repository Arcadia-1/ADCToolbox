"""
Analyze phase error using raw data (high precision).

Wrapper function combining computation and optional plotting for phase error
analysis using raw (unbinned) data.
"""

from typing import Tuple
import numpy as np
from .compute_phase_error_from_raw import compute_phase_error_from_raw


def analyze_phase_error_raw_estimate(
    signal: np.ndarray,
    normalized_freq: float = None,
    show_plot: bool = True
) -> Tuple[float, float, float, float]:
    """
    Analyze phase error using raw data for high-precision estimates.

    Wrapper function that combines core computation and optional plotting
    for convenience. This method uses raw (unbinned) error data for more
    precise AM/PM parameter estimation.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    normalized_freq : float, optional
        Normalized frequency (f/fs), range 0-0.5. If None, auto-detected.
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
    error_rms : float
        Total RMS error across all samples.

    Notes
    -----
    The raw approach uses all data points without binning, providing higher
    precision but potentially more sensitive to outliers. It fits:

        error²(φ) = am_param² * cos²(φ) + pm_param² * sin²(φ) + baseline

    This method is recommended when:
    - High precision is needed
    - Data is relatively clean (few outliers)
    - Detailed analysis is required

    Examples
    --------
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000)) + 0.01*np.random.randn(1000)
    >>> am, pm, bl, rms = analyze_phase_error_raw_estimate(sig, 0.1)
    >>> print(f"Amplitude noise: {am:.3e}")
    >>> print(f"Phase noise: {pm:.3e} rad")
    >>> print(f"Total RMS error: {rms:.3e}")
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
    results = compute_phase_error_from_raw(signal, normalized_freq)

    # 2. --- Optional Plotting ---
    if show_plot:
        try:
            from .plot_error_binned_phase import plot_error_binned_phase
            # Convert raw results to binned format for plotting
            # Create binned version for visualization
            bin_count = 100
            phase_bins = np.linspace(0, 2*np.pi, bin_count, endpoint=False)
            bin_width = 2 * np.pi / bin_count

            # Bin the error data
            phase_wrapped = np.mod(results['phase'], 2*np.pi)
            bin_indices = np.floor(phase_wrapped / bin_width).astype(int)
            bin_indices = np.clip(bin_indices, 0, bin_count - 1)

            bin_counts_arr = np.zeros(bin_count)
            error_sum = np.zeros(bin_count)
            error_sq_sum = np.zeros(bin_count)

            for i in range(len(results['error'])):
                bin_idx = bin_indices[i]
                bin_counts_arr[bin_idx] += 1
                error_sum[bin_idx] += results['error'][i]
                error_sq_sum[bin_idx] += results['error'][i]**2

            with np.errstate(divide='ignore', invalid='ignore'):
                emean = np.where(bin_counts_arr > 0, error_sum / bin_counts_arr, np.nan)
                erms = np.where(
                    bin_counts_arr > 0,
                    np.sqrt(error_sq_sum / bin_counts_arr),
                    np.nan
                )

            # Create plot-compatible dict
            plot_results = {
                'erms': erms,
                'emean': emean,
                'phase_bins': phase_bins,
                'am_param': results['am_param'],
                'pm_param': results['pm_param'],
                'baseline': results['baseline'],
                'error': results['error'],
                'phase': results['phase'],
                'fundamental_amplitude': results['fundamental_amplitude'],
            }
            plot_error_binned_phase(plot_results)
        except ImportError:
            print("Warning: plot_error_binned_phase not available, skipping plot")

    # 3. --- Return Results ---
    return (
        results['am_param'],
        results['pm_param'],
        results['baseline'],
        results['error_rms']
    )
