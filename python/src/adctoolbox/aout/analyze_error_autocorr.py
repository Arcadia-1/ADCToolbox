"""
Error autocorrelation function (ACF) computation and analysis.

Computes ACF of error signal to detect correlation patterns.

MATLAB counterpart: errac.m
"""

import numpy as np
import matplotlib.pyplot as plt


def analyze_error_autocorr(err_data, max_lag=100, normalize=True, plot=False):
    """
    Compute and optionally plot autocorrelation function (ACF) of error signal.

    Parameters:
        err_data: Error signal (1D array)
        max_lag: Maximum lag in samples (default: 100)
        normalize: Normalize ACF so ACF[0] = 1 (default: True)
        plot: If True, plot the autocorrelation on current axes (default: False)

    Returns:
        acf: Autocorrelation values
        lags: Lag indices (-max_lag to +max_lag)
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
    if plot:
        plt.stem(lags, acf, linefmt='b-', markerfmt='bo', basefmt='k-', use_line_collection=True)
        plt.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        plt.xlabel('Lag (samples)', fontsize=11)
        plt.ylabel('Autocorrelation', fontsize=11)
        plt.title('Error Autocorrelation Function', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

    return acf, lags
