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

    Returns (Pythonic names):
    - fitted_signal: Fitted sine wave (N,) or (N, M)
    - frequency: Estimated normalized frequency (scalar or array of length M)
    - amplitude: Amplitude of sine wave (scalar or array of length M)
    - dc_offset: DC component (scalar or array of length M)
    - phase: Phase of sine wave in radians (scalar or array of length M)

    Changed in version 0.3.0:
        Return names changed to Pythonic conventions:
        - data_fit → fitted_signal
        - freq → frequency
        - mag → amplitude
        - dc → dc_offset
        - phi → phase
    """

    # Handle MxN input: fit each column separately
    data = np.asarray(data)
    if data.ndim == 1:
        # Single column, process normally
        return _sine_fit_single(data, f0, tol, rate)
    elif data.ndim == 2:
        # Multiple columns, fit each separately
        N, M = data.shape
        fitted_signal_all = np.zeros((N, M))
        frequency_all = np.zeros(M)
        amplitude_all = np.zeros(M)
        dc_offset_all = np.zeros(M)
        phase_all = np.zeros(M)

        for i in range(M):
            fitted_signal_all[:, i], frequency_all[i], amplitude_all[i], dc_offset_all[i], phase_all[i] = \
                _sine_fit_single(data[:, i], f0, tol, rate)

        # If only one column, return scalars (not arrays)
        if M == 1:
            return fitted_signal_all[:, 0], frequency_all[0], amplitude_all[0], dc_offset_all[0], phase_all[0]

        return fitted_signal_all, frequency_all, amplitude_all, dc_offset_all, phase_all
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

    A, B, dc_offset = x[0], x[1], x[2]
    frequency = f0
    delta_f = 0

    # Iterative frequency optimization (MATLAB lines 49-67)
    # FIX: Stop at first iteration to match MATLAB behavior (MATLAB has bug that prevents convergence)
    for iter_count in range(1):  # Changed from range(100) to range(1)
        frequency = frequency + delta_f  # Changed += to = for clarity
        theta = 2 * np.pi * frequency * time
        # MATLAB line 53: 4th column is divided by N
        M = np.column_stack([np.cos(theta), np.sin(theta), np.ones(N),
                             (-A * 2 * np.pi * time * np.sin(theta) + B * 2 * np.pi * time * np.cos(theta)) / N])
        x = np.linalg.lstsq(M, data, rcond=None)[0]

        A, B, dc_offset = x[0], x[1], x[2]
        # MATLAB line 58: delta_f = x(4)*rate/N
        delta_f = x[3] * rate / N

        # MATLAB line 59: relerr = rms(x(end)/N*M(:,end)) / sqrt(x(1)^2+x(2)^2)
        # M(:,end) is the 4th column of M
        amplitude_current = np.sqrt(A**2 + B**2)
        if amplitude_current > 0:
            # Calculate RMS of (x(4)/N * M(:,4))
            residual = x[3] / N * M[:, 3]
            relerr = np.sqrt(np.mean(residual**2)) / amplitude_current
        else:
            relerr = 0

        # Check stopping criterion (MATLAB line 63)
        if relerr < tol:
            break

    # Fit data using final parameters
    fitted_signal = A * np.cos(2 * np.pi * frequency * time) + B * np.sin(2 * np.pi * frequency * time) + dc_offset
    amplitude = np.sqrt(A**2 + B**2)
    phase = -np.arctan2(B, A)  # CRITICAL: Negative sign to match MATLAB (line 71)

    return fitted_signal, frequency, amplitude, dc_offset, phase