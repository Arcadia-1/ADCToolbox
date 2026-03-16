"""Unit tests for analyze_weight_radix."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adctoolbox import analyze_weight_radix


def test_binary_weights():
    """Binary weights [2^11, ..., 1]: radix all 2.0, effres ~ 12, wgtsca ~ 1.0."""
    weights = np.power(2.0, np.arange(11, -1, -1))
    result = analyze_weight_radix(weights, create_plot=False)

    assert isinstance(result, dict)
    assert 'radix' in result
    assert 'wgtsca' in result
    assert 'effres' in result

    radix = result['radix']
    # First element is NaN, rest should be 2.0
    assert np.isnan(radix[0])
    np.testing.assert_allclose(radix[1:], 2.0, atol=1e-10)

    # Effective resolution should be ~12
    assert abs(result['effres'] - 12.0) < 0.1

    plt.close('all')


def test_subradix_weights():
    """Sub-radix weights: effres < num_weights."""
    weights = np.array([1156, 642, 357, 198, 110, 61, 34, 18, 10, 5, 3, 2, 1, 1],
                       dtype=float)
    result = analyze_weight_radix(weights, create_plot=False)

    # effres should be less than number of weights
    assert result['effres'] < len(weights)
    assert result['effres'] > 0

    # Radix should be < 2.0 on average (sub-radix)
    radix_valid = result['radix'][~np.isnan(result['radix'])]
    assert np.mean(radix_valid) < 2.0

    plt.close('all')


def test_negative_weights():
    """Weights with negatives: wgtsca still computed."""
    weights = np.array([1024, 512, -256, 128, 64, 32, 16, 8, 4, 2, 1, 1],
                       dtype=float)
    result = analyze_weight_radix(weights, create_plot=False)

    assert result['wgtsca'] > 0
    assert result['effres'] > 0
    assert len(result['radix']) == len(weights)

    plt.close('all')


def test_significance_threshold():
    """Trailing noise weights excluded from effres."""
    # 8-bit binary + 4 tiny noise weights
    significant = np.power(2.0, np.arange(7, -1, -1))
    noise = np.array([0.01, 0.005, 0.002, 0.001])
    weights = np.concatenate([significant, noise])

    result = analyze_weight_radix(weights, create_plot=False)

    # effres should reflect ~8 bits, not 12
    assert result['effres'] < 10
    assert result['effres'] > 7

    plt.close('all')


def test_return_type_is_dict():
    """Return type should be dict, not ndarray."""
    weights = np.array([8, 4, 2, 1], dtype=float)
    result = analyze_weight_radix(weights, create_plot=False)
    assert isinstance(result, dict)
    plt.close('all')


def test_plot_creation():
    """Plot should create without error."""
    weights = np.array([1024, 512, -256, 128, 64, 32, 16, 8, 4, 2, 1, 1],
                       dtype=float)
    fig, ax = plt.subplots()
    result = analyze_weight_radix(weights, ax=ax, title='Test')
    assert isinstance(result, dict)
    plt.close('all')
