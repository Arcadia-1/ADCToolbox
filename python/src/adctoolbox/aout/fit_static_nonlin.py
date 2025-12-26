"""
Extract static nonlinearity coefficients (k2, k3) from distorted sinewave.

Uses least-squares fitting to extract 2nd and 3rd order nonlinearity.

MATLAB counterpart: fitstaticnl.m
"""

import numpy as np
from adctoolbox.fundamentals.fit_sine_4param import fit_sine_4param as fit_sine

def fit_static_nonlin(sig_distorted, order):
    """
    Extract static nonlinearity coefficients from distorted sinewave.

    This function extracts 2nd-order (k2) and 3rd-order (k3) static nonlinearity
    coefficients from a distorted single-tone signal. It CANNOT extract gain error
    since the sine fitting absorbs amplitude into the reference.

    Args:
        sig_distorted: Distorted sinewave signal samples, array_like
        order: Polynomial order for fitting (positive integer, typically 2-3)
               order=2: Quadratic nonlinearity only (k2)
               order=3: Quadratic + cubic nonlinearity (k2, k3)

    Returns:
        k2_extracted: Quadratic nonlinearity coefficient (scalar)
                      For ideal ADC: k2 = 0
                      Represents 2nd-order distortion
                      Returns NaN if order < 2
        k3_extracted: Cubic nonlinearity coefficient (scalar)
                      For ideal ADC: k3 = 0
                      Represents 3rd-order distortion
                      Returns NaN if order < 3
        fitted_sine: Fitted ideal sinewave input (reference signal)
                     Vector (N×1), same length as sig_distorted, in time order
                     This is the ideal sine wave extracted from the distorted signal
        fitted_transfer: Fitted transfer curve for plotting, tuple (x, y)
                         x: 1000 smooth input points from min to max (sorted)
                         y: polynomial-evaluated output at those points
                         For ideal system: y=x (straight line)

    Transfer Function Model:
        y = x + k2*x^2 + k3*x^3
        where:
          x = ideal input (zero-mean sine)
          y = actual output (zero-mean)
          k2, k3 = nonlinearity coefficients

    Usage Examples:
        # Extract coefficients only
        sig = 0.5*np.sin(2*np.pi*0.123*np.arange(1000)) + distortion
        k2, k3 = extract_static_nonlin(sig, 3)[:2]

        # Full extraction with plotting
        k2, k3, fitted_sine, fitted_transfer = extract_static_nonlin(sig, 3)
        import matplotlib.pyplot as plt

        # Plot nonlinearity curve
        transfer_x, transfer_y = fitted_transfer
        plt.plot(transfer_x, transfer_y - transfer_x, 'r-', linewidth=2)
        plt.xlabel('Input (V)')
        plt.ylabel('Nonlinearity Error (V)')
        plt.title(f'Static Nonlinearity: k2={k2:.4f}, k3={k3:.4f}')
        plt.grid(True)

    Note:
        Gain error CANNOT be extracted from a single sinewave measurement because
        the sine fitting absorbs amplitude variations. Use multi-tone or DC sweep
        methods to extract gain separately.
    """

    # Input validation
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError('Order must be an integer >= 2')

    if order > 10:
        import warnings
        warnings.warn('Polynomial order > 10 may cause numerical instability',
                     UserWarning)

    # Ensure column vector orientation
    sig_distorted = sig_distorted.flatten()
    N = len(sig_distorted)

    if N < order + 2:
        raise ValueError(
            f'Signal length ({N}) must be > polynomial order ({order}) + 1')

    # Fit ideal sinewave to signal (frequency auto-detected)
    fit_result = fit_sine(sig_distorted)
    fitted_sine = fit_result['fitted_signal']

    # Extract transfer function components
    # x = ideal input (zero-mean)
    # y = actual output (zero-mean)
    x_ideal = fitted_sine - np.mean(fitted_sine)
    y_actual = sig_distorted - np.mean(sig_distorted)

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
    # Transfer function: y = c1*x + c2*x^2 + c3*x^3 + c0
    # After normalization: y = c1*(x/x_max) + c2*(x/x_max)^2 + ...
    # Therefore: k_i = c_i / (x_max^i)

    # Linear coefficient (we don't return this, but need it for normalization)
    k1 = polycoeff[-2] / x_max

    # Quadratic coefficient (k2) - normalized by k1 to represent pure 2nd-order distortion
    if order >= 2:
        k2_abs = polycoeff[-3] / (x_max**2)
        k2_extracted = k2_abs / k1  # Normalize to unity gain
    else:
        k2_extracted = np.nan

    # Cubic coefficient (k3) - normalized by k1 to represent pure 3rd-order distortion
    if order >= 3:
        k3_abs = polycoeff[-4] / (x_max**3)
        k3_extracted = k3_abs / k1  # Normalize to unity gain
    else:
        k3_extracted = np.nan

    # Calculate fitted curve on a smooth grid for plotting (1000 sorted points)
    # Create smooth x-axis from min to max of fitted sine
    x_smooth = np.linspace(np.min(fitted_sine), np.max(fitted_sine), 1000)

    # Normalize smooth x values (same normalization as used in fitting)
    x_smooth_norm = (x_smooth - np.mean(fitted_sine)) / x_max

    # Evaluate polynomial at smooth points
    y_smooth_norm = np.polyval(polycoeff, x_smooth_norm)

    # Convert back to original scale
    y_smooth = y_smooth_norm + np.mean(sig_distorted)

    # Return as tuple (x, y) for easy plotting
    fitted_transfer = (x_smooth, y_smooth)

    return k2_extracted, k3_extracted, fitted_sine, fitted_transfer
