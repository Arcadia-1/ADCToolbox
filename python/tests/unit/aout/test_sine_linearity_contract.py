"""Independent code-width and voltage-unit invariants for sine INL/DNL."""

import numpy as np
import pytest

from adctoolbox import compute_inl_from_sine


@pytest.mark.parametrize('missing', [[], [64], list(range(16, 240, 4))])
@pytest.mark.parametrize('clip', [0, .02])
def test_interior_widths_and_code_labels(missing, clip):
    n = np.arange(2**18)
    codes = np.floor(128 + 127.9*np.sin(2*np.pi*1237*n/len(n))).astype(int)
    for code in missing:
        codes[codes == code] = code + 1
    inl, dnl, axis = compute_inl_from_sine(codes, num_bits=8, clip_percent=clip)
    counts = np.bincount(codes, minlength=256)
    transitions = -np.cos(np.pi*np.cumsum(counts)/len(codes))
    excluded = int(round(255*clip))
    expected_codes = np.arange(excluded+1, 255-excluded)
    widths = transitions[expected_codes] - transitions[expected_codes-1]
    np.testing.assert_array_equal(axis, expected_codes)
    np.testing.assert_allclose(dnl, widths/widths.mean()-1, atol=1e-13)
    np.testing.assert_array_equal(axis[dnl == -1], missing)
    assert abs(inl[-1]) < 1e-12


def test_voltage_units_do_not_change_linearity():
    n = np.arange(2**18)
    v = 1.65 + 1.60*np.sin(2*np.pi*1237*n/len(n))
    volts = compute_inl_from_sine(v, num_bits=8, full_scale=3.3, clip_percent=0)
    normalized = compute_inl_from_sine(v/3.3, num_bits=8, full_scale=1, clip_percent=0)
    for actual, expected in zip(volts, normalized):
        np.testing.assert_array_equal(actual, expected)
