"""fit_static_nonlin must return the k2/k3 of ``y = x + k2*x^2 + k3*x^3``.

The fit is made against the fundamental of the output, whose amplitude the
cubic term scales by g = 1 + 0.75*k3*A^2. Dividing every polynomial
coefficient by the linear term once (instead of once per order) returned
k2/g and k3/g^2, and centring the reference on the record mean instead of the
fitted DC biased k2 on records with a non-integer number of cycles.
"""
import numpy as np
import pytest

from adctoolbox import fit_static_nonlin

N = 8192


def _distorted(cycles, k2, k3, amplitude, dc=0.0):
    t = np.arange(N)
    x = amplitude * np.sin(2 * np.pi * cycles / N * t + 0.3)
    return x + k2 * x**2 + k3 * x**3 + dc


@pytest.mark.parametrize(
    "k2, k3, amplitude",
    [
        (0.01, 0.02, 0.5),    # weak: the old result was only 0.4 % / 0.75 % low
        (0.05, -0.10, 0.45),
        (0.0, -0.50, 0.5),    # strong compression: the old k3 was -0.609
        (0.10, -0.30, 1.0),   # old result: k2 = 0.129, k3 = -0.499
    ],
)
def test_recovers_injected_coefficients_on_coherent_record(k2, k3, amplitude):
    k2_fit, k3_fit = fit_static_nonlin(_distorted(1001, k2, k3, amplitude), 3)[:2]

    assert k2_fit == pytest.approx(k2, abs=1e-6)
    assert k3_fit == pytest.approx(k3, abs=1e-6)


@pytest.mark.parametrize("cycles", [7.41, 20.5, 100.37])
def test_recovers_injected_coefficients_on_noncoherent_record(cycles):
    # 7.41 cycles: the old k2 was 0.0112 for a true 0.01.
    k2_fit, k3_fit = fit_static_nonlin(_distorted(cycles, 0.01, 0.02, 0.5), 3)[:2]

    assert k2_fit == pytest.approx(0.01, rel=5e-3)
    assert k3_fit == pytest.approx(0.02, rel=5e-3)


def test_dc_offset_does_not_change_the_coefficients():
    without_dc = fit_static_nonlin(_distorted(20.5, 0.03, -0.05, 0.4), 3)[:2]
    with_dc = fit_static_nonlin(_distorted(20.5, 0.03, -0.05, 0.4, dc=0.7), 3)[:2]

    np.testing.assert_allclose(with_dc, without_dc, rtol=1e-9, atol=1e-12)


def test_fitted_transfer_curve_follows_the_distorted_signal():
    signal = _distorted(1001, 0.03, -0.05, 0.4, dc=0.7)
    _, _, fitted_sine, (transfer_x, transfer_y) = fit_static_nonlin(signal, 3)

    # The curve maps the fitted reference onto the measured output.
    np.testing.assert_allclose(
        np.interp(fitted_sine, transfer_x, transfer_y), signal, atol=1e-6
    )
