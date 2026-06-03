"""Test cap2weight.py - CDAC capacitor to weight conversion."""

import numpy as np

from adctoolbox.common import convert_cap_to_weight


def test_convert_cap_to_weight_4bit_binary_cdac():
    cd = [1, 2, 4, 8]
    cb = [0, 0, 0, 0]
    cp = [0, 0, 0, 0]

    weight, co = convert_cap_to_weight(cd, cb, cp)

    np.testing.assert_allclose(weight, np.array([1, 2, 4, 8]) / 15.0)
    assert co == 15.0


def test_convert_cap_to_weight_4bit_with_dummy_cap():
    cd = [1, 1, 2, 4, 8]
    cb = [0, 0, 0, 0, 0]
    cp = [0, 0, 0, 0, 0]

    weight, co = convert_cap_to_weight(cd, cb, cp)

    np.testing.assert_allclose(weight, np.array([1, 1, 2, 4, 8]) / 16.0)
    assert co == 16.0


def test_convert_cap_to_weight_with_bridge_cap_returns_positive_weights():
    cd = [1, 2, 4, 8]
    cb = [0, 0, 4, 0]
    cp = [0, 0, 0, 0]

    weight, co = convert_cap_to_weight(cd, cb, cp)

    assert co > 0
    assert np.sum(weight) > 0
