"""
Unit Test: Verify fit_static_nonlin function for nonlinearity extraction

Purpose: Self-verify that fit_static_nonlin correctly extracts 2nd and 3rd order
         nonlinearity coefficients from distorted sine waves
"""
import numpy as np
from adctoolbox import fit_static_nonlin


def test_verify_fit_static_nonlin_ideal_signal():
    """
    Verify fit_static_nonlin on ideal (no distortion) signal.

    Test strategy:
    1. Generate ideal sine: y = sin(ωt) with no nonlinearity
    2. Fit with order=3
    3. Assert: k2 ≈ 0, k3 ≈ 0 (no distortion detected)
    """
    N = 1000
    sig_ideal = 0.5 * np.sin(2*np.pi*0.1*np.arange(N))

    k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(sig_ideal, order=3)

    print(f'\n[Verify Ideal Signal] [N={N}] [No distortion]')
    print(f'  [k2] {k2:.6e} (expected ≈ 0)')
    print(f'  [k3] {k3:.6e} (expected ≈ 0)')

    # For ideal signal, k2 and k3 should be very small
    assert abs(k2) < 0.01, f"k2 should be near zero: {k2}"
    assert abs(k3) < 0.01, f"k3 should be near zero: {k3}"
    assert fitted_sine.shape == (N,), f"fitted_sine shape mismatch: {fitted_sine.shape}"
    print(f'  [Status] PASS')


def test_verify_fit_static_nonlin_quadratic_distortion():
    """
    Verify fit_static_nonlin detects quadratic (2nd order) distortion.

    Test strategy:
    1. Generate distorted sine: y = x + k2*x^2 with known k2=0.1
    2. Fit with order=3
    3. Assert: k2 ≈ 0.1 (detects distortion)
    """
    N = 1000
    x = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))
    k2_true = 0.1
    k3_true = 0.0
    sig_distorted = x + k2_true * x**2 + k3_true * x**3

    k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(sig_distorted, order=3)

    print(f'\n[Verify Quadratic Distortion] [k2_true={k2_true}, k3_true={k3_true}]')
    print(f'  [k2_fit] {k2:.6f} (expected {k2_true})')
    print(f'  [k3_fit] {k3:.6f} (expected {k3_true})')
    print(f'  [Error] k2={(k2-k2_true):.6e}, k3={(k3-k3_true):.6e}')

    # k2 should be close to 0.1, k3 should be small
    assert abs(k2 - k2_true) < 0.05, f"k2 extraction error: {k2} vs {k2_true}"
    assert abs(k3) < 0.05, f"k3 should be near zero: {k3}"
    print(f'  [Status] PASS')


def test_verify_fit_static_nonlin_cubic_distortion():
    """
    Verify fit_static_nonlin detects cubic (3rd order) distortion.

    Test strategy:
    1. Generate distorted sine: y = x + k3*x^3 with known k3=0.05
    2. Fit with order=3
    3. Assert: k3 ≈ 0.05 (detects distortion)
    """
    N = 1000
    x = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))
    k2_true = 0.0
    k3_true = 0.05
    sig_distorted = x + k2_true * x**2 + k3_true * x**3

    k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(sig_distorted, order=3)

    print(f'\n[Verify Cubic Distortion] [k2_true={k2_true}, k3_true={k3_true}]')
    print(f'  [k2_fit] {k2:.6f} (expected {k2_true})')
    print(f'  [k3_fit] {k3:.6f} (expected {k3_true})')
    print(f'  [Error] k2={(k2-k2_true):.6e}, k3={(k3-k3_true):.6e}')

    # k3 should be close to 0.05, k2 should be small
    assert abs(k2) < 0.05, f"k2 should be near zero: {k2}"
    assert abs(k3 - k3_true) < 0.025, f"k3 extraction error: {k3} vs {k3_true}"
    print(f'  [Status] PASS')


def test_verify_fit_static_nonlin_combined_distortion():
    """
    Verify fit_static_nonlin detects combined 2nd and 3rd order distortion.

    Test strategy:
    1. Generate: y = x + k2*x^2 + k3*x^3 with k2=0.08, k3=0.03
    2. Fit with order=3
    3. Assert: Both k2 and k3 detected correctly
    """
    N = 1000
    x = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))
    k2_true = 0.08
    k3_true = 0.03
    sig_distorted = x + k2_true * x**2 + k3_true * x**3

    k2, k3, fitted_sine, fitted_transfer = fit_static_nonlin(sig_distorted, order=3)

    print(f'\n[Verify Combined Distortion] [k2={k2_true}, k3={k3_true}]')
    print(f'  [k2_fit] {k2:.6f} (expected {k2_true})')
    print(f'  [k3_fit] {k3:.6f} (expected {k3_true})')
    print(f'  [Error] k2={(k2-k2_true):.6e}, k3={(k3-k3_true):.6e}')

    # Both should be detected with reasonable accuracy
    assert abs(k2 - k2_true) < 0.05, f"k2 error: {k2} vs {k2_true}"
    assert abs(k3 - k3_true) < 0.025, f"k3 error: {k3} vs {k3_true}"
    print(f'  [Status] PASS')


def test_verify_fit_static_nonlin_order_parameter():
    """
    Verify fit_static_nonlin handles different order parameters correctly.

    Test strategy:
    1. Fit same distorted signal with order=2 and order=3
    2. Assert: order=2 returns k3=NaN, order=3 returns k3 value
    """
    N = 1000
    x = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))
    sig_distorted = x + 0.1 * x**2 + 0.05 * x**3

    # Order 2: only k2
    k2_ord2, k3_ord2, _, _ = fit_static_nonlin(sig_distorted, order=2)
    # Order 3: k2 and k3
    k2_ord3, k3_ord3, _, _ = fit_static_nonlin(sig_distorted, order=3)

    print(f'\n[Verify Order Parameter]')
    print(f'  [Order 2] k2={k2_ord2:.6f}, k3={k3_ord2} (NaN={np.isnan(k3_ord2)})')
    print(f'  [Order 3] k2={k2_ord3:.6f}, k3={k3_ord3:.6f}')

    # For order 2, k3 should be NaN
    assert np.isnan(k3_ord2), "k3 should be NaN for order=2"
    # For order 3, k3 should be a value
    assert not np.isnan(k3_ord3), "k3 should be a value for order=3"
    print(f'  [Status] PASS')


def test_verify_fit_static_nonlin_return_structure():
    """
    Verify fit_static_nonlin returns correct structure.

    Test strategy:
    1. Fit a signal
    2. Assert: Returns 4-tuple (k2, k3, fitted_sine, fitted_transfer)
    3. fitted_transfer is (x, y) tuple
    """
    sig = 0.5 * np.sin(2*np.pi*0.1*np.arange(500))
    result = fit_static_nonlin(sig, order=3)

    print(f'\n[Verify Return Structure]')
    print(f'  [Result type] {type(result)} (tuple)')
    print(f'  [Tuple length] {len(result)}')

    k2, k3, fitted_sine, fitted_transfer = result

    # Check types and shapes
    assert isinstance(k2, (float, np.floating)), f"k2 should be float, got {type(k2)}"
    assert isinstance(k3, (float, np.floating)), f"k3 should be float, got {type(k3)}"
    assert isinstance(fitted_sine, np.ndarray), f"fitted_sine should be ndarray"
    assert isinstance(fitted_transfer, tuple), f"fitted_transfer should be tuple"
    assert len(fitted_transfer) == 2, f"fitted_transfer should have 2 elements"

    transfer_x, transfer_y = fitted_transfer
    print(f'  [fitted_sine] shape={fitted_sine.shape}')
    print(f'  [fitted_transfer] (x:{transfer_x.shape}, y:{transfer_y.shape})')

    # Transfer curve should have 1000 points
    assert len(transfer_x) == 1000, f"Transfer x should have 1000 points: {len(transfer_x)}"
    assert len(transfer_y) == 1000, f"Transfer y should have 1000 points: {len(transfer_y)}"
    print(f'  [Status] PASS')


def test_verify_fit_static_nonlin_input_validation():
    """
    Verify fit_static_nonlin validates inputs correctly.

    Test strategy:
    1. Test with order < 2 (should raise ValueError)
    2. Test with signal too short (should raise ValueError)
    """
    sig = np.sin(2*np.pi*0.1*np.arange(100))

    print(f'\n[Verify Input Validation]')

    # Test order < 2
    try:
        fit_static_nonlin(sig, order=1)
        assert False, "Should raise ValueError for order < 2"
    except ValueError as e:
        print(f'  [Order < 2] Caught: {str(e)[:50]}...')

    # Test signal too short
    try:
        fit_static_nonlin(np.array([1, 2, 3]), order=3)
        assert False, "Should raise ValueError for short signal"
    except ValueError as e:
        print(f'  [Short signal] Caught: {str(e)[:50]}...')

    print(f'  [Status] PASS')


def test_verify_fit_static_nonlin_fitted_sine():
    """
    Verify fitted_sine reconstruction is reasonable.

    Test strategy:
    1. Generate distorted signal with known k2, k3
    2. Extract fitted sine
    3. Assert: fitted_sine has correct amplitude and frequency
    """
    N = 1000
    A_true = 0.5
    f_true = 0.1
    x = A_true * np.sin(2*np.pi*f_true*np.arange(N))
    k2 = 0.1
    k3 = 0.05
    sig_distorted = x + k2 * x**2 + k3 * x**3

    k2_fit, k3_fit, fitted_sine, _ = fit_static_nonlin(sig_distorted, order=3)

    # Compute amplitude and frequency of fitted sine
    fitted_mean = np.mean(fitted_sine)
    fitted_amp = np.max(fitted_sine) - np.mean(fitted_sine)

    print(f'\n[Verify Fitted Sine]')
    print(f'  [True amplitude] {A_true}')
    print(f'  [Fitted amplitude] {fitted_amp:.6f}')
    print(f'  [Fitted mean] {fitted_mean:.6e}')

    # Amplitude should be reasonable (close to true amplitude)
    assert 0.4 < fitted_amp < 0.6, f"Fitted amplitude off: {fitted_amp}"
    assert abs(fitted_mean) < 0.1, f"Fitted mean should be near zero: {fitted_mean}"
    print(f'  [Status] PASS')


def test_verify_fit_static_nonlin_transfer_curve():
    """
    Verify fitted transfer curve structure and properties.

    Test strategy:
    1. Fit a signal with known distortion
    2. Extract transfer curve (x, y)
    3. Assert: x is sorted, y shows distortion pattern
    """
    N = 1000
    x = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))
    k2 = 0.15
    sig_distorted = x + k2 * x**2

    _, _, _, fitted_transfer = fit_static_nonlin(sig_distorted, order=3)
    transfer_x, transfer_y = fitted_transfer

    print(f'\n[Verify Transfer Curve]')
    print(f'  [x range] [{transfer_x[0]:.4f}, {transfer_x[-1]:.4f}]')
    print(f'  [y range] [{transfer_y[0]:.4f}, {transfer_y[-1]:.4f}]')

    # x should be sorted (increasing)
    assert np.all(np.diff(transfer_x) > 0), "Transfer x should be sorted"

    # For quadratic distortion, y > x (positive slope on transfer curve)
    error_curve = transfer_y - transfer_x
    print(f'  [Error curve] min={np.min(error_curve):.6f}, max={np.max(error_curve):.6f}')

    # Error curve should show nonlinearity
    assert np.max(np.abs(error_curve)) > 0.001, "Transfer curve should show distortion"
    print(f'  [Status] PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING FIT_STATIC_NONLIN VERIFICATION TESTS')
    print('='*80)

    test_verify_fit_static_nonlin_ideal_signal()
    test_verify_fit_static_nonlin_quadratic_distortion()
    test_verify_fit_static_nonlin_cubic_distortion()
    test_verify_fit_static_nonlin_combined_distortion()
    test_verify_fit_static_nonlin_order_parameter()
    test_verify_fit_static_nonlin_return_structure()
    test_verify_fit_static_nonlin_input_validation()
    test_verify_fit_static_nonlin_fitted_sine()
    test_verify_fit_static_nonlin_transfer_curve()

    print('\n' + '='*80)
    print('** All fit_static_nonlin verification tests passed! **')
    print('='*80)
