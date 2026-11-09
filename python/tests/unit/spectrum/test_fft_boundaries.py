"""Odd/even FFT boundaries must preserve aliases, indices and real amplitudes."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.fundamentals.frequency import fold_bin_to_nyquist
from adctoolbox.spectrum import analyze_spectrum, analyze_spectrum_virtuoso
from adctoolbox.spectrum._harmonics import _locate_harmonic_bins
from adctoolbox.spectrum.extract_freq_components import extract_freq_components


@pytest.mark.parametrize('n_fft', [100, 101, 102, 103])
def test_fractional_aliases_use_true_nyquist_boundary(n_fft):
    for alias in [0., .2, n_fft // 2, n_fft / 2 - .1, n_fft / 2]:
        for cycles in [-2, -1, 0, 1, 2]:
            for sign in [-1, 1]:
                assert fold_bin_to_nyquist(cycles*n_fft + sign*alias, n_fft) == pytest.approx(alias)


@pytest.mark.parametrize('n_fft', [100, 101, 102, 103])
def test_harmonics_at_fractional_nyquist_stay_in_rfft(n_fft):
    for harmonic in [2, 3, 5]:
        bins = _locate_harmonic_bins(n_fft / (2*harmonic), harmonic, n_fft)
        assert np.all((bins >= 0) & (bins <= n_fft // 2))
        assert bins[-1] == n_fft // 2


@pytest.mark.parametrize('n_fft', [100, 101, 102, 103])
@pytest.mark.parametrize('bin_kind', ['dc', 'interior', 'top'])
def test_band_extraction_preserves_amplitude(n_fft, bin_kind):
    k = {'dc': 0, 'interior': 7, 'top': n_fft // 2}[bin_kind]
    n = np.arange(n_fft)
    x = np.cos(2*np.pi*k*n/n_fft)
    y = extract_freq_components(x, [k/n_fft, k/n_fft])[:, 0]
    np.testing.assert_allclose(y, x, atol=1e-12)
    # A full-band filter must retain every positive/negative-frequency pair.
    rng = np.random.default_rng(109)
    data = rng.normal(size=(n_fft, 2))
    np.testing.assert_allclose(extract_freq_components(data, [0, .5]), data, atol=1e-12)


@pytest.mark.parametrize('analyze', [analyze_spectrum, analyze_spectrum_virtuoso])
@pytest.mark.parametrize('n_fft', [101, 103])
def test_public_plot_accepts_harmonic_near_odd_length_boundary(analyze, n_fft):
    n = np.arange(n_fft)
    k = n_fft / 4 - .15
    x = .8*np.sin(2*np.pi*k*n/n_fft) + .08*np.sin(2*np.pi*2*k*n/n_fft+.4)
    fig, ax = plt.subplots()
    try:
        result = analyze(x, max_scale_range=[-1, 1], win_type='hann',
                         side_bin=3, max_harmonic=2, ax=ax)
        fig.canvas.draw()
        assert np.isfinite(result['sndr_db'])
    finally:
        plt.close(fig)
