"""
Extract static nonlinearity coefficients from ADC transfer function.

Matches MATLAB fitstaticnl.m exactly.
"""

import numpy as np
from ..common.sine_fit import sine_fit


def fit_static_nol(sig, order):
    """
    Extract static nonlinearity coefficients from ADC transfer function.

    This function fits a polynomial to the ADC transfer function to extract
    static nonlinearity coefficients. It models the relationship between
    ideal input (fitted sine wave) and actual output (measured signal).

    Args:
        sig: ADC output signal (sinewave samples), array_like
        order: Polynomial order for fitting (positive integer, typically 2-4)
               order=1: Linear gain only (k1)
               order=2: Linear + quadratic (k1, k2)
               order=3: Linear + quadratic + cubic (k1, k2, k3)

    Returns:
        k1: Linear gain coefficient (scalar)
            For ideal ADC: k1 = 1.0
            Represents gain error in the transfer function
        k2: Quadratic nonlinearity coefficient, normalized by k1 (scalar)
            For ideal ADC: k2 = 0
            Represents pure 2nd-order distortion independent of gain
            Returns NaN if order < 2
        k3: Cubic nonlinearity coefficient, normalized by k1 (scalar)
            For ideal ADC: k3 = 0
            Represents pure 3rd-order distortion independent of gain
            Returns NaN if order < 3
        polycoeff: Full polynomial coefficients (highest to lowest order)
                   Vector [c_n, c_(n-1), ..., c_1, c_0]
                   Transfer function: y = c_n*x^n + ... + c_1*x + c_0
        fitted_sine: Fitted ideal sinewave input (reference signal)
                     Vector (N×1), same length as sig, in time order
                     This is the ideal sine wave extracted from the distorted signal
                     Used as reference for calculating measured nonlinearity
        fitted_output: Fitted output at sample points (polynomial evaluated at fitted_sine)
                       Vector (N×1), same length as sig, in time order
                       This is the fitted signal output from the polynomial model
        fitted_transfer: Fitted transfer curve for plotting, tuple (x, y)
                         x: 1000 smooth input points from min to max (sorted)
                         y: polynomial-evaluated output at those points
                         For ideal ADC, y=x (straight line with unity gain)

    Transfer Function Model:
        y = k1 * (x + k2*x^2 + k3*x^3 + ...)
        where:
          x = ideal input (zero-mean)
          y = actual output (zero-mean)
          k1 = gain (typically ≈ 1.0)
          k2, k3 = normalized distortion coefficients (independent of gain)

    Usage Examples:
        # Extract coefficients only
        sig = 0.5*np.sin(2*np.pi*0.123*np.arange(1000)) + 0.01*np.random.randn(1000)
        k1, k2, k3 = fit_static_nol(sig, 3)[:3]

        # Plot transfer function
        k1, k2, k3, polycoeff, fitted_sine, fitted_output, fitted_transfer = fit_static_nol(sig, 3)
        import matplotlib.pyplot as plt
        transfer_x, transfer_y = fitted_transfer
        plt.plot(transfer_x, transfer_y, 'b-', linewidth=2, label='Transfer Curve')
        plt.plot(transfer_x, transfer_x, 'k--', label='Ideal y=x')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.title(f'Transfer Function: k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}')

        # Plot nonlinearity (deviation from y=x)
        nonlinearity = fitted_output - fitted_sine
        plt.figure()
        sort_idx = np.argsort(fitted_sine)
        plt.plot(fitted_sine[sort_idx], nonlinearity[sort_idx], 'r-', linewidth=2)
        plt.xlabel('Input')
        plt.ylabel('Nonlinearity Error')
        plt.title('Static Nonlinearity')
        plt.grid(True)

    Notes:
        - Input signal must contain predominantly a single-tone sinewave
        - Frequency is automatically estimated from the signal
        - Coefficients are normalized (k1 ≈ 1.0 for ideal ADC)
        - Higher-order terms (k2, k3, ...) represent static distortion
        - For accurate results, signal should have good SNR (>40 dB)
        - Coefficients are denormalized to handle any amplitude range
        - DC offset is automatically removed before fitting

    Algorithm:
        1. Fit ideal sinewave to input signal using sine_fit
        2. Extract zero-mean ideal input (x) and actual output (y)
        3. Normalize x for numerical stability
        4. Fit polynomial: y = polyfit(x_normalized, order)
        5. Denormalize coefficients to get physical k1, k2, k3

    See also: sine_fit, err_hist_sine
    """
    # Input validation
    if not isinstance(sig, (list, tuple, np.ndarray)):
        raise TypeError('Signal must be array-like')

    sig = np.asarray(sig)

    if not np.isreal(sig).all():
        raise ValueError('Signal must be real-valued')

    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError('Order must be a positive integer')

    if order > 10:
        import warnings
        warnings.warn('Polynomial order > 10 may cause numerical instability',
                     UserWarning)

    # Ensure column vector orientation
    sig = sig.flatten()
    N = len(sig)

    if N < order + 2:
        raise ValueError(
            f'Signal length ({N}) must be > polynomial order ({order}) + 1')

    # Fit ideal sinewave to signal (frequency auto-detected)
    fitted_sine, _, _, _, _ = sine_fit(sig)

    # Extract transfer function components
    # x = ideal input (zero-mean)
    # y = actual output (zero-mean)
    x_ideal = fitted_sine - np.mean(fitted_sine)
    y_actual = sig - np.mean(sig)

    # Normalize for numerical stability
    # This prevents coefficient overflow for large amplitude signals
    x_max = np.max(np.abs(x_ideal))

    if x_max < 1e-10:
        raise ValueError('Signal amplitude too small for fitting (< 1e-10)')

    x_norm = x_ideal / x_max

    # Fit polynomial to transfer function
    # polycoeff: [c_n, c_(n-1), ..., c_1, c_0]
    polycoeff = np.polyfit(x_norm, y_actual, order)

    # Extract and denormalize coefficients
    # Transfer function: y = k1*x + k2*x^2 + k3*x^3 + ...
    # After normalization: y = c1*(x/x_max) + c2*(x/x_max)^2 + ...
    # Therefore: k_i = c_i / (x_max^i)

    # Linear coefficient (k1) - represents gain
    k1 = polycoeff[-2] / x_max

    # Quadratic coefficient (k2) - normalized by k1 to represent pure 2nd-order distortion
    # This makes k2 independent of gain error
    # Transfer function becomes: y = k1 * (x + k2*x^2 + k3*x^3)
    if order >= 2:
        k2_abs = polycoeff[-3] / (x_max**2)
        k2 = k2_abs / k1  # Normalize to unity gain
    else:
        k2 = np.nan

    # Cubic coefficient (k3) - normalized by k1 to represent pure 3rd-order distortion
    if order >= 3:
        k3_abs = polycoeff[-4] / (x_max**3)
        k3 = k3_abs / k1  # Normalize to unity gain
    else:
        k3 = np.nan

    # Calculate fitted output at sample points (N points in time order)
    y_fit_norm = np.polyval(polycoeff, x_norm)
    fitted_output = y_fit_norm + np.mean(sig)

    # Calculate fitted curve on a smooth grid for plotting (1000 sorted points)
    # Create smooth x-axis from min to max of fitted sine
    x_smooth = np.linspace(np.min(fitted_sine), np.max(fitted_sine), 1000)

    # Normalize smooth x values (same normalization as used in fitting)
    x_smooth_norm = (x_smooth - np.mean(fitted_sine)) / x_max

    # Evaluate polynomial at smooth points
    y_smooth_norm = np.polyval(polycoeff, x_smooth_norm)

    # Convert back to original scale
    y_smooth = y_smooth_norm + np.mean(sig)

    # Return as tuple (x, y) for easy plotting
    fitted_transfer = (x_smooth, y_smooth)

    return k1, k2, k3, polycoeff, fitted_sine, fitted_output, fitted_transfer