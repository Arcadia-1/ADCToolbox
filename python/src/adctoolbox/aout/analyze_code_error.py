"""
Analyze code-based error (for INL/DNL analysis).

Wrapper function combining computation and optional plotting for code-binned
error analysis.
"""

from typing import Tuple
import numpy as np
from .compute_error_by_code import compute_error_by_code


def analyze_code_error(
    signal: np.ndarray,
    normalized_freq: float = None,
    num_bits: int = None,
    clip_percent: float = 0.01,
    show_plot: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analyze error binned by ADC code values (for INL/DNL analysis).

    Wrapper function that combines core computation and optional plotting
    for convenience. This method bins error by code value to reveal
    code-dependent errors and static nonlinearity.

    Parameters
    ----------
    signal : np.ndarray
        Input signal, 1D numpy array.
    normalized_freq : float, optional
        Normalized frequency (f/fs), range 0-0.5. If None, auto-detected.
    num_bits : int, optional
        Number of bits for ADC resolution. If None, inferred from signal range.
    clip_percent : float, default=0.01
        Percentage of codes to clip from edges (excludes near-rail codes).
    show_plot : bool, default=True
        Whether to display result plot.

    Returns
    -------
    emean_by_code : np.ndarray
        Mean error per code bin.
    erms_by_code : np.ndarray
        RMS error per code bin.
    code_bins : np.ndarray
        Code bin centers (ADC code values).

    Notes
    -----
    This function is useful for analyzing static nonlinearity (INL/DNL)
    and code-dependent errors. It:

    1. Fits a fundamental sinewave to remove dynamic effects
    2. Computes residual error
    3. Bins error by ADC code value
    4. Computes mean and RMS error per code

    The resulting curves reveal:
    - INL-like patterns in mean error vs code
    - Code-dependent noise in RMS error vs code
    - Missing codes (bins with zero counts)

    Auto-detection logic:
    - Integer input → Treated as ADC codes
    - Float with range > 2.0 → Treated as codes
    - Float with range <= 2.0 → Treated as voltage (quantized)

    Examples
    --------
    >>> sig = np.sin(2*np.pi*0.1*np.arange(1000))
    >>> emean, erms, codes = analyze_code_error(sig, 0.1, num_bits=10)
    >>> print(f"Code range: {codes[0]} to {codes[-1]}")
    >>> print(f"Max mean error: {np.nanmax(np.abs(emean)):.3e}")
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
    results = compute_error_by_code(signal, normalized_freq, num_bits, clip_percent)

    # 2. --- Optional Plotting ---
    if show_plot:
        try:
            from .plot_error_binned_code import plot_error_binned_code
            plot_error_binned_code(results)
        except ImportError:
            print("Warning: plot_error_binned_code not available, skipping plot")

    # 3. --- Return Results ---
    return (
        results['emean_by_code'],
        results['erms_by_code'],
        results['code_bins']
    )
