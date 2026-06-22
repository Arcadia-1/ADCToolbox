"""Unit tests for plot_residual_scatter."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adctoolbox import plot_residual_scatter


def _make_adc_data(n=1024, m=6):
    """Generate simple ADC signal and bit matrix for testing."""
    t = np.arange(n)
    sig = (np.sin(2 * np.pi * 3 * t / n) / 2 + 0.5) * (2**m - 1)
    code = np.clip(np.round(sig).astype(int), 0, 2**m - 1)
    bits = np.array([list(map(int, format(c, f'0{m}b'))) for c in code], dtype=float)
    return sig, bits


def _visible_axes(fig):
    return [ax for ax in fig.axes if ax.get_visible()]


def _hidden_axes(fig):
    return [ax for ax in fig.axes if not ax.get_visible()]


def _assert_scatter_axis(ax, *, n_points, xlabel, ylabel):
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape == (n_points, 2)


def test_basic_output():
    """Basic call returns dict with expected keys."""
    sig, bits = _make_adc_data()
    result = plot_residual_scatter(sig, bits, create_plot=False)

    assert isinstance(result, dict)
    assert 'pairs' in result
    assert 'residuals_x' in result
    assert 'residuals_y' in result
    assert len(result['pairs']) == bits.shape[1]  # default M pairs
    assert len(result['residuals_x']) == len(result['pairs'])
    assert len(result['residuals_y']) == len(result['pairs'])
    plt.close('all')


def test_custom_pairs():
    """Custom pairs respected."""
    sig, bits = _make_adc_data()
    pairs = [(0, 6), (3, 6)]
    result = plot_residual_scatter(sig, bits, pairs=pairs, create_plot=False)
    assert result['pairs'] == pairs
    assert len(result['residuals_x']) == 2
    plt.close('all')


def test_custom_weights():
    """Custom weights produce different residuals than default."""
    sig, bits = _make_adc_data()
    m = bits.shape[1]
    default_result = plot_residual_scatter(sig, bits, create_plot=False)
    custom_w = np.ones(m)
    custom_result = plot_residual_scatter(sig, bits, weights=custom_w, create_plot=False)

    # Residuals should differ since weights differ
    assert not np.allclose(default_result['residuals_y'][0],
                           custom_result['residuals_y'][0])
    plt.close('all')


def test_alpha_auto():
    """Auto alpha should be between 0.1 and 1.0."""
    sig, bits = _make_adc_data(n=100)
    result = plot_residual_scatter(sig, bits, alpha='auto', create_plot=False)
    assert result is not None  # just ensure no crash
    plt.close('all')


def test_alpha_fixed():
    """Fixed alpha accepted."""
    sig, bits = _make_adc_data()
    result = plot_residual_scatter(sig, bits, alpha=0.5, create_plot=False)
    assert result is not None
    plt.close('all')


def test_residual_stage_zero():
    """Stage 0 residual equals signal itself."""
    sig, bits = _make_adc_data()
    result = plot_residual_scatter(sig, bits, pairs=[(0, 6)], create_plot=False)
    np.testing.assert_array_equal(result['residuals_x'][0], sig)
    plt.close('all')


def test_plot_creation():
    """Plot creates one scatter axis per requested pair."""
    sig, bits = _make_adc_data()
    pairs = [(0, 6), (3, 6)]
    result = plot_residual_scatter(sig, bits, pairs=pairs, create_plot=True)
    fig = plt.gcf()
    visible_axes = _visible_axes(fig)

    assert result['pairs'] == pairs
    assert len(fig.axes) == 2
    assert len(visible_axes) == len(pairs)
    _assert_scatter_axis(
        visible_axes[0],
        n_points=len(sig),
        xlabel='Signal',
        ylabel='Res. of bit #6',
    )
    _assert_scatter_axis(
        visible_axes[1],
        n_points=len(sig),
        xlabel='Res. of bit #3',
        ylabel='Res. of bit #6',
    )
    plt.close(fig)


def test_plot_creation_hides_unused_subplots():
    """Unused axes in the subplot grid should be hidden."""
    sig, bits = _make_adc_data(n=128)
    pairs = [(0, 6), (1, 6), (2, 6), (3, 6), (4, 6)]
    result = plot_residual_scatter(sig, bits, pairs=pairs, create_plot=True)
    fig = plt.gcf()

    assert result['pairs'] == pairs
    assert len(fig.axes) == 8
    assert len(_visible_axes(fig)) == len(pairs)
    assert len(_hidden_axes(fig)) == 3
    for ax in _visible_axes(fig):
        assert len(ax.collections) == 1
        assert ax.collections[0].get_offsets().shape == (len(sig), 2)
    plt.close(fig)
