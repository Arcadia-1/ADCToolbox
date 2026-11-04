"""Computing residual spectra must leave the caller's Matplotlib session intact."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from adctoolbox.aout import analyze_error_spectrum, analyze_error_envelope_spectrum


@pytest.mark.parametrize('analyze,kwargs', [
    (analyze_error_spectrum, {'frequency': 137/4096}),
    (analyze_error_envelope_spectrum, {'frequency': 137/4096}),
    (analyze_error_envelope_spectrum, {'input_kind': 'error'}),
])
@pytest.mark.parametrize('provide_ax', [False, True])
def test_no_plot_preserves_existing_figures_and_matches_plotted_metrics(analyze, kwargs, provide_ax, monkeypatch):
    n = np.arange(4096)
    x = .8*np.sin(2*np.pi*137*n/len(n)) + .0008*np.sin(2*np.pi*271*n/len(n))
    figures = [plt.subplots(), plt.subplots()]
    try:
        for fig, ax in figures:
            ax.plot([1, 2], [3, 4], label='user curve')
            ax.set(title='user title', xlabel='user x', ylabel='user y')
        numbers = plt.get_fignums()
        current = plt.gca()
        backend = matplotlib.get_backend()
        states = [(tuple(ax.get_children()), ax.get_xlim(), ax.get_ylim()) for _, ax in figures]

        def fail_backend_switch(*args, **kwargs):
            pytest.fail('create_plot=False must not switch the Matplotlib backend')

        with monkeypatch.context() as patch:
            patch.setattr(matplotlib, 'use', fail_backend_switch)
            patch.setattr(plt, 'switch_backend', fail_backend_switch)
            result = analyze(x, create_plot=False, ax=figures[0][1] if provide_ax else None,
                             title='ignored title', **kwargs)
        assert plt.get_fignums() == numbers
        assert plt.gca() is current
        assert matplotlib.get_backend() == backend
        for (_, ax), (children, xlim, ylim) in zip(figures, states):
            assert tuple(ax.get_children()) == children
            assert ax.get_xlim() == xlim and ax.get_ylim() == ylim
            assert (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) == ('user title', 'user x', 'user y')

        plot_fig, plot_ax = plt.subplots()
        try:
            plotted = analyze(x, create_plot=True, ax=plot_ax, **kwargs)
            assert result.keys() == plotted.keys()
            for key in result:
                if isinstance(result[key], (str, tuple)) or result[key] is None:
                    assert result[key] == plotted[key]
                else:
                    np.testing.assert_allclose(result[key], plotted[key], equal_nan=True)
        finally:
            plt.close(plot_fig)
    finally:
        for fig, _ in figures:
            plt.close(fig)
