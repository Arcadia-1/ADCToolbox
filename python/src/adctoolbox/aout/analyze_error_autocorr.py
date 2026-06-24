"""
Error autocorrelation function (ACF) computation and analysis.

Computes ACF of error signal to detect correlation patterns.

MATLAB counterpart: errac.m
"""

import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param
from adctoolbox.aout._fit_diagnostics import extract_fit_diagnostics

def analyze_error_autocorr(signal, frequency=None, max_lag=50, normalize=True, create_plot: bool = True,
                           ax=None, title: str = None, max_iterations: int = 1,
                           tolerance: float = 1e-9, return_fit: bool = False):
    """
    Compute and optionally plot autocorrelation function (ACF) of error signal.

    This function fits an ideal sine to the signal, computes the error,
    and analyzes its autocorrelation to detect temporal correlation patterns.

    Parameters
    ----------
    signal : np.ndarray
        ADC output signal (1D array)
    frequency : float, optional
        Normalized frequency (0-0.5). If None, auto-detected
    max_lag : int, default=50
        Maximum lag in samples
    normalize : bool, default=True
        Normalize ACF so ACF[0] = 1
    create_plot : bool, default=True
        If True, plot the autocorrelation
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes (plt.gca())
    title : str, optional
        Title for the plot. If None, uses default title
    max_iterations : int, default=1
        Frequency-refinement iterations passed to fit_sine_4param.
    tolerance : float, default=1e-9
        Frequency-refinement convergence threshold passed to fit_sine_4param.
    return_fit : bool, default=False
        If True, include scalar sine-fit diagnostics under result['fit'].

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'acf': Autocorrelation values
        - 'lags': Lag indices (-max_lag to +max_lag)
        - 'error_signal': Error signal (signal - fitted sine)
        - 'fit': Optional sine-fit diagnostics when return_fit=True

    Notes
    -----
    - Error = signal - ideal_sine (fitted using fit_sine_4param)
    - ACF reveals temporal correlation in the error signal
    - White noise shows ACF ≈ 0 for all lags except 0
    - Correlated errors show non-zero ACF at specific lags
    """
    # Fit ideal sine to extract reference
    fit_kwargs = {"max_iterations": max_iterations, "tolerance": tolerance}
    if frequency is None:
        fit_result = fit_sine_4param(signal, **fit_kwargs)
    else:
        fit_result = fit_sine_4param(signal, frequency_estimate=frequency, **fit_kwargs)

    sig_ideal = fit_result['fitted_signal']

    # Compute error
    error_signal = signal - sig_ideal

    # Ensure column data
    e = np.asarray(error_signal).flatten()
    N = len(e)

    # Subtract mean
    e = e - np.mean(e)

    # Preallocate
    lags = np.arange(-max_lag, max_lag + 1)
    acf = np.zeros_like(lags, dtype=float)

    # Compute autocorrelation manually (consistent with MATLAB implementation)
    for k in range(len(lags)):
        lag = lags[k]
        if lag >= 0:
            x1 = e[:N-lag] if lag > 0 else e
            x2 = e[lag:N] if lag > 0 else e
        else:
            lag2 = -lag
            x1 = e[lag2:N]
            x2 = e[:N-lag2]
        acf[k] = np.mean(x1 * x2)

    # Normalize if required
    if normalize:
        acf = acf / acf[lags == 0]

    # Plot if requested
    if create_plot:
        # Use provided axes or get current axes
        if ax is None:
            ax = plt.gca()

        ax.stem(lags, acf, linefmt='b-', markerfmt='b.', basefmt='k-')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Lag (samples)', fontsize=9)
        ax.set_ylabel('ACF', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-max_lag, max_lag])
        # ax.set_ylim([-0.3, 1.1])

        # Set title if provided
        if title is not None:
            ax.set_title(title, fontsize=12)

    # Return dictionary for consistency with other analyze functions
    result = {
        'acf': acf,
        'lags': lags,
        'error_signal': error_signal
    }
    if return_fit:
        result['fit'] = extract_fit_diagnostics(fit_result)

    return result
