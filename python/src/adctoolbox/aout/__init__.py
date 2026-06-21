"""
Analog output (AOUT) analysis tools.

This subpackage defines the public API of the AOUT analysis domain.
"""

# ----------------------------------------------------------------------
# Value / Phase error analysis
# ----------------------------------------------------------------------

from adctoolbox.aout.analyze_error_by_value import analyze_error_by_value
from adctoolbox.aout.analyze_error_by_phase import analyze_error_by_phase
from adctoolbox.aout.rearrange_error_by_value import rearrange_error_by_value
from adctoolbox.aout.rearrange_error_by_phase import rearrange_error_by_phase
from adctoolbox.aout.plot_rearranged_error_by_value import plot_rearranged_error_by_value
from adctoolbox.aout.plot_rearranged_error_by_phase import plot_rearranged_error_by_phase

# Additional error analysis
from adctoolbox.aout.analyze_error_pdf import analyze_error_pdf
from adctoolbox.aout.analyze_error_autocorr import analyze_error_autocorr
from adctoolbox.aout.analyze_error_spectrum import analyze_error_spectrum
from adctoolbox.aout.analyze_error_envelope_spectrum import analyze_error_envelope_spectrum
from adctoolbox.aout.analyze_phase_plane import analyze_phase_plane
from adctoolbox.aout.analyze_error_phase_plane import analyze_error_phase_plane

# ----------------------------------------------------------------------
# Harmonic decomposition
# ----------------------------------------------------------------------

from adctoolbox.aout.analyze_decomposition_time import analyze_decomposition_time
from adctoolbox.aout.analyze_decomposition_polar import analyze_decomposition_polar
from adctoolbox.aout.decompose_harmonic_error import decompose_harmonic_error
from adctoolbox.aout.plot_decomposition_time import plot_decomposition_time
from adctoolbox.aout.plot_decomposition_polar import plot_decomposition_polar

# ----------------------------------------------------------------------
# INL / DNL from sine
# ----------------------------------------------------------------------

from adctoolbox.aout.analyze_inl_from_sine import analyze_inl_from_sine
from adctoolbox.aout.compute_inl_from_sine import compute_inl_from_sine
from adctoolbox.aout.plot_dnl_inl import plot_dnl_inl

# ----------------------------------------------------------------------
# Static nonlinearity fitting
# ----------------------------------------------------------------------

from adctoolbox.aout.fit_static_nonlin import fit_static_nonlin
from adctoolbox.spectrum import analyze_spectrum as _analyze_spectrum


def _with_legacy_spectrum_keys(result):
    """Add pre-0.8 metric names while preserving current keys."""
    aliases = {
        'sndr_db': 'sndr_dbc',
        'sfdr_db': 'sfdr_dbc',
        'snr_db': 'snr_dbc',
        'thd_db': 'thd_dbc',
        'noise_floor_db': 'noise_floor_dbfs',
    }
    for old, new in aliases.items():
        if new in result and old not in result:
            result[old] = result[new]
    return result


def analyze_spectrum(*args, n_thd=None, **kwargs):
    """Compatibility wrapper for the spectrum-domain analyzer.

    Older tests imported ``analyze_spectrum`` from ``adctoolbox.aout`` and used
    ``n_thd`` plus ``*_db`` metric keys.  The canonical function is now
    ``adctoolbox.spectrum.analyze_spectrum`` with ``max_harmonic`` and
    ``*_dbc``/``*_dbfs`` names.
    """
    if n_thd is not None and "max_harmonic" not in kwargs:
        kwargs["max_harmonic"] = n_thd
    return _with_legacy_spectrum_keys(_analyze_spectrum(*args, **kwargs))


def decompose_harmonics(signal, freq=None, harmonic=5, disp=1):
    """Compatibility wrapper returning the historical decomposition tuple."""
    result = analyze_decomposition_time(
        signal,
        harmonic=harmonic,
        create_plot=bool(disp),
    )
    return (
        result['fundamental_signal'],
        result['harmonic_signal'] + result['noise_residual'],
        result['harmonic_signal'],
        result['noise_residual'],
    )


def plot_error_pdf(*args, **kwargs):
    """Compatibility wrapper returning the historical error-PDF tuple."""
    result = analyze_error_pdf(*args, **kwargs)
    return (
        result['err_lsb'],
        result['mu'],
        result['sigma'],
        result['kl_divergence'],
        result['x'],
        result['pdf'],
        result['gauss_pdf'],
    )


def plot_error_autocorr(error_signal, max_lag=50, normalize=True, **kwargs):
    """Compatibility wrapper for autocorrelation of a precomputed error signal."""
    result = analyze_error_autocorr(
        error_signal,
        max_lag=max_lag,
        normalize=normalize,
        **kwargs,
    )
    return result['acf'], result['lags']


def plot_envelope_spectrum(*args, **kwargs):
    """Compatibility alias for envelope spectrum analysis."""
    return _with_legacy_spectrum_keys(analyze_error_envelope_spectrum(*args, **kwargs))


def plot_error_hist_phase(signal, bins=100, freq=None, disp=1, **kwargs):
    """Compatibility tuple wrapper for phase-binned error analysis."""
    result = analyze_error_by_phase(
        signal,
        norm_freq=freq,
        n_bins=bins,
        create_plot=bool(disp),
        **kwargs,
    )
    return (
        result.get('bin_error_mean_v'),
        result.get('bin_error_rms_v'),
        result.get('phase_bin_centers_rad'),
        result.get('am_noise_rms_v'),
        result.get('pm_noise_rms_v'),
        result.get('error'),
        result.get('phase'),
    )


def plot_error_hist_code(signal, bins=100, freq=None, disp=1, **kwargs):
    """Compatibility tuple wrapper for value-binned error analysis."""
    result = analyze_error_by_value(
        signal,
        norm_freq=freq,
        n_bins=bins,
        create_plot=bool(disp),
        **kwargs,
    )
    return (
        result.get('error_mean'),
        result.get('error_rms'),
        result.get('bin_centers'),
        result.get('error'),
        result.get('value'),
    )

# ----------------------------------------------------------------------
# Public API of aout subpackage
# ----------------------------------------------------------------------

__all__ = [
    # Error analysis
    'analyze_error_by_value',
    'analyze_error_by_phase',
    'rearrange_error_by_value',
    'rearrange_error_by_phase',

    'analyze_error_pdf',
    'analyze_error_autocorr',
    'analyze_error_spectrum',
    'analyze_phase_plane',
    'analyze_error_phase_plane',
    'analyze_error_envelope_spectrum',

    # Harmonic decomposition
    'analyze_decomposition_time',
    'analyze_decomposition_polar',
    'decompose_harmonic_error',

    # INL / DNL
    'analyze_inl_from_sine',
    'compute_inl_from_sine',

    # Static nonlinearity
    'fit_static_nonlin',

    # Plotting (AOUT domain only)
    'analyze_spectrum',
    'decompose_harmonics',
    'plot_error_pdf',
    'plot_error_autocorr',
    'plot_envelope_spectrum',
    'plot_error_hist_phase',
    'plot_error_hist_code',
    'plot_rearranged_error_by_value',
    'plot_rearranged_error_by_phase',
    'plot_decomposition_time',
    'plot_decomposition_polar',
    'plot_dnl_inl',
]
