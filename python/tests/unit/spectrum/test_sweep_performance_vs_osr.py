"""Unit tests for sweep_performance_vs_osr."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adctoolbox import sweep_performance_vs_osr


def test_clean_sine_high_sndr():
    """Clean sine: SNDR should be very high at all OSR."""
    N = 1024
    t = np.arange(N)
    freq = 0.1
    sig = 0.5 * np.sin(2 * np.pi * freq * t)

    osr_values = np.array([2, 4, 8, 16, 32])
    result = sweep_performance_vs_osr(sig, osr=osr_values, create_plot=False)

    assert isinstance(result, dict)
    assert 'osr' in result
    assert 'sndr' in result
    assert 'sfdr' in result
    assert 'enob' in result
    assert len(result['sndr']) == len(osr_values)

    # Clean sine should have very high SNDR (> 80 dB)
    assert np.all(result['sndr'] > 80)
    plt.close('all')


def test_noisy_sine_sndr_increases_with_osr():
    """Sine + white noise: SNDR should increase with OSR (~3 dB per doubling)."""
    np.random.seed(42)
    N = 4096
    t = np.arange(N)
    freq = 0.05
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 0.01, N)
    data = sig + noise

    osr_values = np.array([2, 4, 8, 16, 32, 64])
    result = sweep_performance_vs_osr(data, osr=osr_values, create_plot=False)

    # SNDR should generally increase with OSR
    # Check that highest OSR has higher SNDR than lowest
    assert result['sndr'][-1] > result['sndr'][0]

    # ENOB should be positive
    assert np.all(result['enob'] > 0)
    plt.close('all')


def test_default_osr():
    """Default OSR (no osr argument) should work."""
    N = 256
    t = np.arange(N)
    sig = 0.5 * np.sin(2 * np.pi * 0.1 * t)

    result = sweep_performance_vs_osr(sig, create_plot=False)
    assert len(result['osr']) == N // 2
    assert len(result['sndr']) == N // 2
    plt.close('all')


def test_plot_with_ax():
    """Providing ax should produce single-axis plot without error."""
    N = 512
    t = np.arange(N)
    sig = 0.5 * np.sin(2 * np.pi * 0.1 * t)

    fig, ax = plt.subplots()
    result = sweep_performance_vs_osr(sig, osr=np.array([2, 4, 8]), ax=ax)
    assert result is not None
    plt.close('all')


def test_plot_auto_subplots():
    """No ax provided should create 2-subplot figure."""
    N = 512
    t = np.arange(N)
    sig = 0.5 * np.sin(2 * np.pi * 0.1 * t) + np.random.RandomState(0).normal(0, 0.01, N)

    result = sweep_performance_vs_osr(sig, osr=np.array([2, 4, 8, 16]))
    assert result is not None
    plt.close('all')


def test_enob_formula():
    """ENOB should follow (SNDR - 1.76) / 6.02."""
    N = 512
    t = np.arange(N)
    sig = 0.5 * np.sin(2 * np.pi * 0.1 * t)

    result = sweep_performance_vs_osr(sig, osr=np.array([4, 8]), create_plot=False)
    expected_enob = (result['sndr'] - 1.76) / 6.02
    np.testing.assert_allclose(result['enob'], expected_enob, atol=1e-10)
    plt.close('all')
