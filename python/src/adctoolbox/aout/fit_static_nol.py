"""
Extract static nonlinearity coefficients from ADC transfer function.

Matches MATLAB fitstaticnl.m exactly.
"""

import numpy as np


def fit_static_nol(sig, order, freq=0):
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
        freq: Normalized input frequency (0-0.5), optional
              If 0, frequency is automatically estimated (default: 0)

    Returns:
        k1: Linear gain coefficient (scalar)
            For ideal ADC: k1 = 1.0
        k2: Quadratic nonlinearity coefficient (scalar)
            For ideal ADC: k2 = 0
            Returns NaN if order < 2
        k3: Cubic nonlinearity coefficient (scalar)
            For ideal ADC: k3 = 0
            Returns NaN if order < 3
        polycoeff: Full polynomial coefficients (highest to lowest order)
                   Vector [c_n, c_(n-1), ..., c_1, c_0]
                   Transfer function: y = c_n*x^n + ... + c_1*x + c_0
        fit_curve: Fitted transfer function evaluated at signal points
                   Vector (N×1), same length as sig
                   Useful for plotting fitted curve

    Transfer Function Model:
        y = k1*x + k2*x^2 + k3*x^3 + ...
        where:
          x = ideal input (zero-mean)
          y = actual output (zero-mean)

    Examples:
        # Extract linear and quadratic coefficients (order=2)
        sig = 0.5*np.sin(2*np.pi*0.123*np.arange(1000)) + 0.01*np.random.randn(1000)
        k1, k2 = fit_static_nol(sig, 2)[:2]

        # Extract up to cubic nonlinearity with auto frequency detection
        k1, k2, k3 = fit_static_nol(sig, 3)[:3]

        # Specify frequency explicitly for faster computation
        k1, k2, k3 = fit_static_nol(sig, 3, freq=0.123)[:3]

        # Get full polynomial and plot transfer function
        k1, k2, k3, polycoeff, fit_curve = fit_static_nol(sig, 3)
        from ..common.sine_fit import sine_fit
        sig_fit, _, _, _, _ = sine_fit(sig)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(sig_fit, sig, 'b.', label='Measured')
        plt.plot(sig_fit, fit_curve, 'r-', linewidth=2, label='Fitted')
        plt.xlabel('Ideal Input')
        plt.ylabel('Actual Output')
        plt.legend()
        plt.title(f'Transfer Function: k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}')
        plt.show()

    Notes:
        - Input signal must contain predominantly a single-tone sinewave
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

    if not isinstance(freq, (int, float, np.number)) or freq < 0 or freq >= 0.5:
        raise ValueError('Frequency must be a scalar in range [0, 0.5)')

    # Ensure column vector orientation
    sig = sig.flatten()
    N = len(sig)

    if N < order + 2:
        raise ValueError(
            f'Signal length ({N}) must be > polynomial order ({order}) + 1')

    # Fit ideal sinewave to signal
    from ..common.sine_fit import sine_fit
    if freq == 0:
        sig_fit, _, _, _, _ = sine_fit(sig)
    else:
        sig_fit, _, _, _, _ = sine_fit(sig, freq)

    # Extract transfer function components
    # x = ideal input (zero-mean)
    # y = actual output (zero-mean)
    x_ideal = sig_fit - np.mean(sig_fit)
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

    # Linear coefficient (k1)
    k1 = polycoeff[-2] / x_max

    # Quadratic coefficient (k2)
    if order >= 2:
        k2 = polycoeff[-3] / (x_max**2)
    else:
        k2 = np.nan

    # Cubic coefficient (k3)
    if order >= 3:
        k3 = polycoeff[-4] / (x_max**3)
    else:
        k3 = np.nan

    # Calculate fitted curve
    # Evaluate polynomial at normalized input points
    y_fit_norm = np.polyval(polycoeff, x_norm)

    # Convert back to original scale (add mean back)
    fit_curve = y_fit_norm + np.mean(sig)

    return k1, k2, k3, polycoeff, fit_curve


if __name__ == "__main__":
    # Test with ideal sinewave
    print("[Test 1] Ideal sinewave (k1≈1, k2≈0, k3≈0)")
    N = 1000
    sig_ideal = 0.5 * np.sin(2*np.pi*0.123*np.arange(N))
    k1, k2, k3 = fit_static_nol(sig_ideal, 3)[:3]
    print(f"  k1 = {k1:.6f} (expected ≈1.0)")
    print(f"  k2 = {k2:.6e} (expected ≈0.0)")
    print(f"  k3 = {k3:.6e} (expected ≈0.0)")

    # Test with 2nd-order distortion
    print("\n[Test 2] Signal with quadratic distortion")
    sig_distorted = sig_ideal + 0.1 * sig_ideal**2
    k1, k2, k3 = fit_static_nol(sig_distorted, 3)[:3]
    print(f"  k1 = {k1:.6f}")
    print(f"  k2 = {k2:.6f} (should be non-zero)")
    print(f"  k3 = {k3:.6e}")

    # Test with lower order
    print("\n[Test 3] Order=2 (k3 should be NaN)")
    k1, k2, k3 = fit_static_nol(sig_ideal, 2)[:3]
    print(f"  k1 = {k1:.6f}")
    print(f"  k2 = {k2:.6e}")
    print(f"  k3 = {k3} (expected NaN)")
