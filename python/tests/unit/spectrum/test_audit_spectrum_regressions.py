"""Numerical oracles for FFT edges, spur selection and averaged noise."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.fundamentals.frequency import fold_bin_to_nyquist
from adctoolbox.spectrum import compute_spectrum, analyze_spectrum
from adctoolbox.spectrum.extract_freq_components import extract_freq_components
from adctoolbox.spectrum._harmonics import _locate_harmonic_bins


@pytest.mark.parametrize('size', [100, 101, 103])
def test_highest_rfft_bin_and_fractional_folding(size):
    highest = size//2
    x = np.cos(2*np.pi*highest*np.arange(size)/size)
    y = extract_freq_components(x, [highest/size, highest/size])[:, 0]
    np.testing.assert_allclose(y, x, atol=1e-12)
    for frequency in [highest-.2, size/2, size/2+.2]:
        expected = min(frequency, size-frequency)
        assert fold_bin_to_nyquist(frequency, size) == pytest.approx(expected)
        bins = _locate_harmonic_bins(frequency/2, 2, size)
        assert 0 <= bins[0] <= highest


def test_odd_fft_public_plot():
    n = np.arange(101)
    x = .8*np.sin(2*np.pi*25.1*n/101) + .08*np.sin(2*np.pi*50.2*n/101+.4)
    fig = plt.figure()
    try:
        result = analyze_spectrum(x, max_scale_range=[-1,1], win_type='hann',
                                  side_bin=3, max_harmonic=2)
        assert np.isfinite(result['sndr_dbc'])
    finally:
        plt.close(fig)


@pytest.mark.parametrize('off_bin_amplitude', [.0009, .0007])
def test_sfdr_selects_largest_lobe(off_bin_amplitude):
    n = np.arange(8192)
    x = .8*np.sin(2*np.pi*123*n/len(n)) + .0008*np.sin(2*np.pi*1000*n/len(n)+.1)
    x += off_bin_amplitude*np.sin(2*np.pi*1700.5*n/len(n)+.7)
    r = compute_spectrum(x, max_scale_range=[-1,1], win_type='hann',
                         side_bin=3, max_harmonic=1, nf_method=3)
    assert r['metrics']['sfdr_dbc'] == pytest.approx(
        20*np.log10(.8/max(.0008, off_bin_amplitude)), abs=.01)


@pytest.mark.parametrize('runs', [1, 4, 16])
@pytest.mark.parametrize('coherent', [False, True])
def test_noise_median_matches_integrated_white_noise(runs, coherent):
    n = np.arange(32768)
    rng = np.random.default_rng(20261103)
    x = np.array([.7*np.sin(2*np.pi*379*n/len(n)) + rng.normal(0, .002, len(n))
                  for _ in range(runs)])
    r = compute_spectrum(x, max_scale_range=[-1,1], win_type='boxcar',
                         side_bin=0, nf_method=1, coherent_averaging=coherent)
    p = r['plot_data']['noise_parts']
    assert abs(10*np.log10(p['median']/p['exclude'])) < .15
    if coherent:
        padded = compute_spectrum(np.vstack([x, np.zeros((7, len(n)))]),
            max_scale_range=[-1,1], win_type='boxcar', side_bin=0,
            nf_method=1, coherent_averaging=True)
        assert padded['metrics']['snr_dbc'] == pytest.approx(r['metrics']['snr_dbc'], abs=1e-12)


def test_hz_annotation_has_correct_scale():
    n = np.arange(1024)
    fig, ax = plt.subplots()
    try:
        analyze_spectrum(.8*np.sin(2*np.pi*100*n/len(n)), fs=1024,
            max_scale_range=[-1,1], win_type='boxcar', side_bin=0, ax=ax)
        labels = [t.get_text() for t in ax.texts if t.get_text().startswith('Fin/fs')]
        assert labels == ['Fin/fs = 100.0 / 1.0K Hz']
    finally:
        plt.close(fig)
