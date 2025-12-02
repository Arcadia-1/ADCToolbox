import numpy as np
import matplotlib.pyplot as plt


def err_auto_correlation(err_data, max_lag=100, normalize=True):
    """
    Compute autocorrelation function (ACF) of error signal.

    Parameters:
        err_data: Error signal (1D array)
        max_lag: Maximum lag in samples (default: 100)
        normalize: Normalize ACF so ACF[0] = 1 (default: True)

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

    return acf, lags
