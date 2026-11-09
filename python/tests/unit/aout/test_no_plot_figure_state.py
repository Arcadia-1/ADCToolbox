"""Pure error-spectrum computations must not change existing figures."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.aout import analyze_error_spectrum, analyze_error_envelope_spectrum


@pytest.mark.parametrize('envelope', [False, True])
def test_no_plot_preserves_backend_figures_and_artists(envelope):
    n = np.arange(4096)
    x = .8*np.sin(2*np.pi*137*n/len(n)) + .0008*np.sin(2*np.pi*271*n/len(n))
    first, ax1 = plt.subplots()
    second, ax2 = plt.subplots()
    ax1.plot([1, 2], [3, 4])
    ax2.plot([2, 3], [4, 5])
    figures = plt.get_fignums()
    children = [list(ax.get_children()) for ax in (ax1, ax2)]
    backend = matplotlib.get_backend()
    try:
        fun = analyze_error_envelope_spectrum if envelope else analyze_error_spectrum
        kwargs = dict(input_kind='error') if envelope else dict(frequency=137/len(n))
        result = fun(x, create_plot=False, **kwargs)
        assert np.isfinite(result['sndr_dbc'])
        assert plt.get_fignums() == figures
        assert matplotlib.get_backend() == backend
        assert plt.gcf() is second
        assert [list(ax.get_children()) for ax in (ax1, ax2)] == children
    finally:
        plt.close(first)
        plt.close(second)
