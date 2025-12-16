"""
Analyze phase error using raw or binned approach (high precision or trend analysis).

Wrapper function combining computation and optional plotting for phase error
analysis with flexible mode selection.
"""

from typing import Tuple, Union
import numpy as np
from .rearrange_error_by_phase import rearrange_error_by_phase


def analyze_error_by_phase(
    signal: np.ndarray,
    normalized_freq: float = None,
    mode: str = "binned",
    bin_count: int = 100,
    show_plot: bool = True
) -> Union[Tuple[float, float, float, float], Tuple[float, float, float, np.ndarray, np.ndarray]]:
    """
    Analyze phase error using raw or binned approach for AM/PM decomposition.

    Wrapper function that combines core computation and optional plotting
    for convenience. Supports both high-precision raw analysis and robust
    binned trend analysis.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    normalized_freq : float, optional
        Normalized frequency (f/fs), range 0-0.5. If None, auto-detected.
    mode : str, default="binned"
        Analysis mode:
        - "raw": Direct fitting to all error samples (high precision, sensitive to outliers)
        - "binned": Fit to binned RMS values (robust to outliers, trend analysis)
    bin_count : int, default=100
        Number of phase bins (only used if mode="binned").
    show_plot : bool, default=True
        Whether to display result plot.

    Returns
    -------
    When mode="raw":
        am_param : float
            Amplitude modulation parameter (amplitude noise).
        pm_param : float
            Phase modulation parameter (phase noise in radians).
        baseline : float
            Baseline noise floor.
        error_rms : float
            Total RMS error across all samples.

    When mode="binned":
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
    Both modes fit the AM/PM model to residual error data:

        error²(φ) = am_param² * cos²(φ) + pm_param² * sin²(φ) + baseline

    **Raw Mode:**
    - Uses all error samples for fitting
    - High precision estimates
    - Sensitive to outliers
    - Lower computational cost
    - Best for clean signals

    **Binned Mode:**
    - Bins error by phase, fits RMS values
    - Robust to outliers
    - Smooth trend estimates
    - Better for trend analysis
    - Best for noisy signals with systematic variation

    Examples
    --------
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000)) + 0.01*np.random.randn(1000)

    High-precision analysis:
    >>> am, pm, bl, rms = analyze_error_by_phase(sig, 0.1, mode="raw")
    >>> print(f"Amplitude noise: {am:.3e}")
    >>> print(f"Phase noise: {pm:.3e} rad")
    >>> print(f"Total RMS error: {rms:.3e}")

    Robust trend analysis:
    >>> am, pm, bl, erms, emean = analyze_error_by_phase(sig, 0.1, mode="binned", bin_count=100)
    >>> print(f"Amplitude noise: {am:.3e}")
    >>> print(f"Phase noise: {pm:.3e} rad")
    """
    # Input validation
    if mode not in ("raw", "binned"):
        raise ValueError(f"mode must be 'raw' or 'binned', got '{mode}'")

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
    results = rearrange_error_by_phase(
        signal, normalized_freq, mode=mode, bin_count=bin_count if mode == "binned" else None
    )

    # 2. --- Optional Plotting ---
    if show_plot:
        try:
            from .plot_rearranged_error_by_phase import plot_rearranged_error_by_phase

            if mode == "raw":
                # Convert raw results to binned format for plotting
                bin_count_plot = 100
                phase_bins = np.linspace(0, 2*np.pi, bin_count_plot, endpoint=False)
                bin_width = 2 * np.pi / bin_count_plot

                # Bin the error data
                phase_wrapped = np.mod(results['phase'], 2*np.pi)
                bin_indices = np.floor(phase_wrapped / bin_width).astype(int)
                bin_indices = np.clip(bin_indices, 0, bin_count_plot - 1)

                bin_counts_arr = np.zeros(bin_count_plot)
                error_sum = np.zeros(bin_count_plot)
                error_sq_sum = np.zeros(bin_count_plot)

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
            else:
                # Binned mode: use results directly
                plot_results = results

            plot_rearranged_error_by_phase(plot_results)
        except ImportError:
            print("Warning: plot_rearranged_error_by_phase not available, skipping plot")

    # 3. --- Return Results ---
    if mode == "raw":
        return (
            results['am_param'],
            results['pm_param'],
            results['baseline'],
            results['error_rms']
        )
    else:  # mode == "binned"
        return (
            results['am_param'],
            results['pm_param'],
            results['baseline'],
            results['erms'],
            results['emean']
        )
