"""Both spectrum plot styles use the same physical frequency units."""
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.spectrum import compute_spectrum
from adctoolbox.spectrum.plot_spectrum import plot_spectrum
from adctoolbox.spectrum.plot_spectrum_virtuoso import plot_spectrum_virtuoso


@pytest.mark.parametrize('fin,expected', [
    (.25, '0.250'), (1., '1.0'), (100., '100.0'), (999., '999.0'),
    (1000., '1.0K'), (999999., '1000.0K'), (1e6, '1.0M'),
    (999999999., '1000.0M'), (1e9, '1.0G'),
])
def test_fin_and_fs_annotations_match_across_plot_styles(fin, expected):
    n = np.arange(1024)
    fs = fin*1024/100
    result = compute_spectrum(.8*np.sin(2*np.pi*100*n/len(n)), fs=fs,
                              max_scale_range=[-1, 1], win_type='boxcar',
                              side_bin=0, max_harmonic=1, nf_method=3)
    labels = []
    for plot in [plot_spectrum, plot_spectrum_virtuoso]:
        fig, ax = plt.subplots()
        try:
            plot(result, ax=ax)
            labels.append(next(t.get_text() for t in ax.texts if t.get_text().startswith('Fin/fs')))
        finally:
            plt.close(fig)
    assert labels[0] == labels[1]
    assert labels[0].startswith(f'Fin/fs = {expected} / ')
