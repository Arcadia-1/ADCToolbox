"""An explicit full_scale makes floating-point INL input a voltage.

The same capture expressed in different voltage units must give the INL/DNL
of its documented code mapping, round(data * 2**num_bits / full_scale),
whatever the span of the data.
"""

import numpy as np
import pytest

from adctoolbox import compute_inl_from_sine

NUM_BITS = 8
N_SAMPLES = 2**18
PHASE = 2 * np.pi * 1237 * np.arange(N_SAMPLES) / N_SAMPLES
# Coherent sine at 1.5 % .. 98.5 % of full scale.
NORMALIZED = 0.5 + 0.485 * np.sin(PHASE)
CODES = np.round(NORMALIZED * 2**NUM_BITS).astype(int)


def _inl(data, **kwargs):
    return compute_inl_from_sine(data, num_bits=NUM_BITS, clip_percent=0, **kwargs)


def _assert_same(actual, expected):
    for got, want in zip(actual, expected):  # inl, dnl, code
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("full_scale", [0.5, 1.0, 1.8, 3.3, 5.0, 12.0])
def test_explicit_full_scale_is_invariant_to_voltage_units(full_scale):
    volts = _inl(full_scale * NORMALIZED, full_scale=full_scale)

    _assert_same(volts, _inl(CODES))
    _assert_same(volts, _inl(NORMALIZED, full_scale=1.0))


def test_reported_3v3_capture_matches_its_normalized_capture():
    v = 1.65 + 1.60 * np.sin(PHASE)

    volts = _inl(v, full_scale=3.3)
    normalized = _inl(v / 3.3, full_scale=1.0)

    assert len(volts[2]) == 247
    _assert_same(volts, normalized)


def test_float_codes_without_full_scale_stay_codes():
    _assert_same(_inl(CODES.astype(float)), _inl(CODES))
    _assert_same(_inl(CODES + 0.25), _inl(CODES))


def test_integer_input_stays_codes_with_full_scale():
    _assert_same(_inl(CODES, full_scale=3.3), _inl(CODES))


@pytest.mark.parametrize("bipolar", [False, True])
def test_normalized_voltage_without_full_scale_is_unchanged(bipolar):
    data = 2 * NORMALIZED - 1 if bipolar else NORMALIZED

    _assert_same(_inl(data), _inl(CODES))
