import numpy as np
import matplotlib.pyplot as plt


def errAutoCorrelation(err_data, MaxLag=100, Normalize=True):
    """
    Compute and plot autocorrelation function (ACF) of error signal.

    Parameters:
        err_data: Error signal (1D array)
        MaxLag: Maximum lag in samples (default: 100)
        Normalize: Normalize ACF so ACF[0] = 1 (default: True)

    Returns:
        acf: Autocorrelation values
        lags: Lag indices (-MaxLag to +MaxLag)
    """
    # Ensure column data
    e = np.asarray(err_data).flatten()
    N = len(e)

    # Subtract mean
    e = e - np.mean(e)

    # Preallocate
    lags = np.arange(-MaxLag, MaxLag + 1)
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
    if Normalize:
        acf = acf / acf[lags == 0]

    # Plot with larger fonts to match MATLAB
    plt.plot(lags, acf, linewidth=2)
    plt.grid(True)
    plt.xlabel("Lag (samples)", fontsize=14)
    plt.ylabel("Autocorrelation", fontsize=14)
    plt.gca().tick_params(labelsize=14)

    return acf, lags
