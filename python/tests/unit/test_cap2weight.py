"""Test cap2weight.py - CDAC capacitor to weight conversion."""

import numpy as np
import pytest

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


def test_convert_cap_to_weight_supports_cdacwgt_binary_order():
    cd = [8, 4, 2, 1]
    cb = [0, 0, 0, 0]
    cp = [0, 0, 0, 0]

    weight, co = convert_cap_to_weight(
        cd,
        cb,
        cp,
        input_order="msb_to_lsb",
        output_order="msb_to_lsb",
    )
    default_output_weight, default_output_co = convert_cap_to_weight(
        cd,
        cb,
        cp,
        input_order="msb_to_lsb",
    )

    np.testing.assert_allclose(weight, np.array([8, 4, 2, 1]) / 15.0)
    np.testing.assert_allclose(default_output_weight, weight)
    assert co == 15.0
    assert default_output_co == co


def test_convert_cap_to_weight_supports_cdacwgt_segmented_bridge_order():
    cd = [4, 2, 1, 4, 2, 1]
    cb = [0, 4, 0, 8 / 7, 0, 0]
    cp = [0, 0, 0, 0, 0, 1]

    weight, co = convert_cap_to_weight(
        cd,
        cb,
        cp,
        input_order="msb_to_lsb",
        output_order="msb_to_lsb",
    )

    expected = np.array([
        0.6666666666666666,
        0.1666666666666667,
        0.08333333333333333,
        0.04166666666666666,
        0.02083333333333333,
        0.01041666666666667,
    ])
    np.testing.assert_allclose(weight, expected)
    assert co == 6.0


def test_convert_cap_to_weight_output_order_can_be_selected_independently():
    cd_lsb = [1, 2, 4, 8]
    cb_lsb = [0, 0, 0, 0]
    cp_lsb = [0, 0, 0, 0]

    weight_lsb, co_lsb = convert_cap_to_weight(cd_lsb, cb_lsb, cp_lsb)
    weight_msb, co_msb = convert_cap_to_weight(
        cd_lsb,
        cb_lsb,
        cp_lsb,
        output_order="msb_to_lsb",
    )

    np.testing.assert_allclose(weight_msb, weight_lsb[::-1])
    assert co_msb == co_lsb


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_order": "bad_order"},
        {"output_order": "bad_order"},
    ],
)
def test_convert_cap_to_weight_rejects_invalid_order(kwargs):
    with pytest.raises(ValueError, match="order"):
        convert_cap_to_weight([1, 2], [0, 0], [0, 0], **kwargs)
