"""
Error autocorrelation function (ACF) computation and analysis.

Computes ACF of error signal to detect correlation patterns.

MATLAB counterpart: errac.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def analyze_error_autocorr(err_data, max_lag=100, normalize=True, show_plot=False,
                           ax: Optional[plt.Axes] = None, title: str = None):
    """
    Compute and optionally plot autocorrelation function (ACF) of error signal.

    Parameters
    ----------
    err_data : array_like
        Error signal (1D array)
    max_lag : int, default=100
        Maximum lag in samples
    normalize : bool, default=True
        Normalize ACF so ACF[0] = 1
    show_plot : bool, default=False
        If True, plot the autocorrelation
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes (plt.gca())
    title : str, optional
        Title for the plot. If None, uses default title

    Returns
    -------
    acf : ndarray
        Autocorrelation values
    lags : ndarray
        Lag indices (-max_lag to +max_lag)
    """
    # Ensure column data
    e = np.asarray(err_data).flatten()
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
    if show_plot:
        # Use provided axes or get current axes
        if ax is None:
            ax = plt.gca()

        ax.stem(lags, acf, linefmt='b-', markerfmt='b.', basefmt='k-')
        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Lag (samples)', fontsize=9)
        ax.set_ylabel('ACF', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-max_lag, max_lag])
        ax.set_ylim([-0.3, 1.1])

        # Set title if provided
        if title is not None:
            ax.set_title(title, fontsize=10, fontweight='bold')

    return acf, lags
