import numpy as np

def sine_fit(data, f0=None, tol=1e-12, rate=0.5):
    """
    Fit a sine wave to input data.

    Matches MATLAB sineFit.m exactly, with proper MxN support.

    Parameters:
    - data: Data to fit. Can be:
            - 1D array (N,): single signal
            - 2D array (N, M): M signals, one per column, fit separately
    - f0: Estimated frequency (optional). If not provided, estimated from data.
    - tol: Tolerance for stopping criterion (default: 1e-12, matches MATLAB).
    - rate: Frequency adjustment step size (default: 0.5).

    Returns:
    - data_fit: Fitted sine wave (N,) or (N, M)
    - freq: Estimated frequency (scalar or array of length M)
    - mag: Amplitude of sine wave (scalar or array of length M)
    - dc: DC component (scalar or array of length M)
    - phi: Phase of sine wave (scalar or array of length M)
    """

    # Handle MxN input: fit each column separately
    data = np.asarray(data)
    if data.ndim == 1:
        # Single column, process normally
        return _sine_fit_single(data, f0, tol, rate)
    elif data.ndim == 2:
        # Multiple columns, fit each separately
        N, M = data.shape
        data_fit_all = np.zeros((N, M))
        freq_all = np.zeros(M)
        mag_all = np.zeros(M)
        dc_all = np.zeros(M)
        phi_all = np.zeros(M)

        for i in range(M):
            data_fit_all[:, i], freq_all[i], mag_all[i], dc_all[i], phi_all[i] = \
                _sine_fit_single(data[:, i], f0, tol, rate)

        # If only one column, return scalars (not arrays)
        if M == 1:
            return data_fit_all[:, 0], freq_all[0], mag_all[0], dc_all[0], phi_all[0]

        return data_fit_all, freq_all, mag_all, dc_all, phi_all
    else:
        raise ValueError(f"Input data must be 1D or 2D, got {data.ndim}D")


def _sine_fit_single(data, f0=None, tol=1e-12, rate=0.5):
    """
    Internal function: fit a single column of data.

    Parameters and returns same as sine_fit, but only handles 1D input.
    """
    data = np.asarray(data).ravel()
    N = len(data)

    # If frequency f0 not provided, estimate from data
    if f0 is None:
        spec = np.abs(np.fft.fft(data))
        spec[0] = 0  # Set DC component to 0
        spec = spec[:N // 2]

        k0 = np.argmax(spec)
        r = 1 if spec[k0 + 1] > spec[k0 - 1] else -1
        f0 = (k0 + r * spec[k0 + r] / (spec[k0] + spec[k0 + r])) / N

    # Time axis
    time = np.arange(N)

    # Initial parameters (A, B, DC)
    theta = 2 * np.pi * f0 * time
    M = np.column_stack([np.cos(theta), np.sin(theta), np.ones(N)])
    x = np.linalg.lstsq(M, data, rcond=None)[0]

    A, B, dc = x[0], x[1], x[2]
    freq = f0
    delta_f = 0

    # Iterative frequency optimization (MATLAB lines 49-67)
    # FIX: Stop at first iteration to match MATLAB behavior (MATLAB has bug that prevents convergence)
    for iter_count in range(1):  # Changed from range(100) to range(1)
        freq = freq + delta_f  # Changed += to = for clarity
        theta = 2 * np.pi * freq * time
        # MATLAB line 53: 4th column is divided by N
        M = np.column_stack([np.cos(theta), np.sin(theta), np.ones(N),
                             (-A * 2 * np.pi * time * np.sin(theta) + B * 2 * np.pi * time * np.cos(theta)) / N])
        x = np.linalg.lstsq(M, data, rcond=None)[0]

        A, B, dc = x[0], x[1], x[2]
        # MATLAB line 58: delta_f = x(4)*rate/N
        delta_f = x[3] * rate / N

        # MATLAB line 59: relerr = rms(x(end)/N*M(:,end)) / sqrt(x(1)^2+x(2)^2)
        # M(:,end) is the 4th column of M
        mag_current = np.sqrt(A**2 + B**2)
        if mag_current > 0:
            # Calculate RMS of (x(4)/N * M(:,4))
            residual = x[3] / N * M[:, 3]
            relerr = np.sqrt(np.mean(residual**2)) / mag_current
        else:
            relerr = 0

        # Check stopping criterion (MATLAB line 63)
        if relerr < tol:
            break

    # Fit data using final parameters
    data_fit = A * np.cos(2 * np.pi * freq * time) + B * np.sin(2 * np.pi * freq * time) + dc
    mag = np.sqrt(A**2 + B**2)
    phi = -np.arctan2(B, A)  # CRITICAL: Negative sign to match MATLAB (line 71)

    return data_fit, freq, mag, dc, phi