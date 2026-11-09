"""Sine-histogram DNL/INL must report each code's own width.

The oracle is an ADC with known transition levels driven by a coherent sine.
Every internal code width is known in advance, so code labels, DNL and INL
are checked without repeating the histogram calculation.
"""

import numpy as np
import pytest

from adctoolbox import compute_inl_from_sine

N_SAMPLES = 2**20
N_CYCLES = 1237  # coprime with N_SAMPLES: the record covers N uniform phases
# This record recovers each code width to better than 1e-3 LSB.
TOL_LSB = 2e-3


def _sine_codes(code_widths):
    """Quantize a coherent sine with an 8-bit ADC of known code widths.

    ``code_widths[c - 1]`` is the width of internal code ``c`` (1..254) in
    input LSB. Codes 0 and 255 are open-ended and absorb the overdrive.
    """
    lower_edges = 1.0 + np.concatenate(([0.0], np.cumsum(code_widths)))
    phase = 2 * np.pi * N_CYCLES * np.arange(N_SAMPLES) / N_SAMPLES
    x = 128.0 + 127.9 * np.sin(phase)
    return np.searchsorted(lower_edges, x, side="right")


def _widths_with_missing(missing_codes):
    """Ideal widths with each missing code merged into the next code."""
    widths = np.ones(254)
    for code in missing_codes:
        widths[code - 1] = 0.0
        widths[code] = 2.0
    return widths


def _assert_matches_widths(code_widths, code, inl, dnl):
    """Compare with the true widths, using their mean over ``code`` as LSB."""
    widths = code_widths[code - 1]
    expected_dnl = widths / widths.mean() - 1
    np.testing.assert_allclose(dnl, expected_dnl, rtol=0, atol=TOL_LSB)
    np.testing.assert_allclose(inl, np.cumsum(expected_dnl), rtol=0, atol=TOL_LSB)
    # The DNL has zero mean and the INL ends at zero to rounding error.
    assert abs(np.mean(dnl)) < 1e-12
    assert abs(inl[-1]) < 1e-9


def test_missing_code_is_reported_at_its_own_code():
    widths = _widths_with_missing([64])

    inl, dnl, code = compute_inl_from_sine(_sine_codes(widths), num_bits=8, clip_percent=0)

    np.testing.assert_array_equal(code, np.arange(1, 255))
    np.testing.assert_array_equal(code[dnl == -1], [64])
    # Code 65 absorbs the width of code 64; the codes around them are ideal.
    neighbours = np.searchsorted(code, [63, 64, 65, 66])
    np.testing.assert_allclose(dnl[neighbours], [0, -1, 1, 0], rtol=0, atol=TOL_LSB)
    np.testing.assert_allclose(inl[neighbours], [0, -1, 0, 0], rtol=0, atol=TOL_LSB)
    _assert_matches_widths(widths, code, inl, dnl)


def test_many_missing_codes_keep_labels_and_zero_inl_endpoint():
    missing = np.arange(16, 240, 4)
    widths = _widths_with_missing(missing)

    inl, dnl, code = compute_inl_from_sine(_sine_codes(widths), num_bits=8, clip_percent=0)

    np.testing.assert_array_equal(code, np.arange(1, 255))
    np.testing.assert_array_equal(code[dnl == -1], missing)
    _assert_matches_widths(widths, code, inl, dnl)


def test_nonuniform_widths_without_missing_codes_are_not_shifted():
    # Alternating narrow and wide codes, as from a mismatched LSB element.
    widths = np.tile([0.6, 1.4], 127)

    inl, dnl, code = compute_inl_from_sine(_sine_codes(widths), num_bits=8, clip_percent=0)

    np.testing.assert_array_equal(code, np.arange(1, 255))
    assert np.min(dnl) > -1
    _assert_matches_widths(widths, code, inl, dnl)


@pytest.mark.parametrize(
    "clip_percent, first, last",
    [
        (0.01, 4, 251),  # excludes round(2.55) = 3 codes at each end
        (0.05, 14, 241),  # excludes round(12.75) = 13 codes at each end
    ],
)
def test_clip_percent_normalizes_by_the_reported_codes(clip_percent, first, last):
    # A bowed transfer curve: the excluded edge codes are wider than the
    # reported codes, so the full-range mean width is not the LSB.
    internal = np.arange(1, 255)
    widths = 1 + 0.3 * np.cos(2 * np.pi * (internal - 1) / 253)
    missing = [40, 41, 120, 200]
    widths[np.asarray(missing) - 1] = 0.0
    widths *= 254 / widths.sum()  # keep code 255 starting at 255 LSB

    inl, dnl, code = compute_inl_from_sine(
        _sine_codes(widths), num_bits=8, clip_percent=clip_percent
    )

    # The codes next to the exclusion absorb the clipped samples and are not
    # reported either.
    np.testing.assert_array_equal(code, np.arange(first, last + 1))
    np.testing.assert_array_equal(code[dnl == -1], missing)
    _assert_matches_widths(widths, code, inl, dnl)
