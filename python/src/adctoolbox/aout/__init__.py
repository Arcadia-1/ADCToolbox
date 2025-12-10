"""Analog output (time-domain) analysis tools - legacy re-exports from spectrum module."""

# Import spectrum analysis functions from the new spectrum module
from adctoolbox.spectrum import (
    analyze_spectrum,
    analyze_spectrum_polar,
    analyze_two_tone_spectrum,
    compute_spectrum,
    plot_spectrum,
    plot_spectrum_polar,
)

# Harmonic decomposition functions
from .compute_harmonic_decomposition import compute_harmonic_decomposition
from .plot_harmonic_decomposition_time import plot_harmonic_decomposition_time
from .plot_harmonic_decomposition_polar import plot_harmonic_decomposition_polar
from .analyze_harmonic_decomposition import analyze_harmonic_decomposition
from .compute_inl_from_sine import compute_inl_from_sine
from .analyze_inl_from_sine import analyze_inl_from_sine
from .fit_static_nonlin import fit_static_nonlin
from .plot_dnl_inl import plot_dnl_inl

# Placeholder functions for functions not yet moved
try:
    from .calculate_lms_decomposition import calculate_lms_decomposition
    from .analyze_decomposition_time import analyze_decomposition_time
    from .analyze_decomposition_polar import analyze_decomposition_polar
    from .plot_decomposition_time import plot_decomposition_time
    from .plot_decomposition_polar import plot_decomposition_polar
except ImportError:
    # These functions may not exist yet
    pass

try:
    from .plot_envelope_spectrum import plot_envelope_spectrum
    from .plot_error_autocorr import plot_error_autocorr
    from .plot_error_hist_code import plot_error_hist_code
    from .plot_error_hist_phase import plot_error_hist_phase
    from .plot_error_pdf import plot_error_pdf
except ImportError:
    # These functions may not exist yet
    pass

__all__ = [
    # Spectrum functions (from spectrum module)
    'analyze_spectrum',
    'analyze_spectrum_polar',
    'analyze_two_tone_spectrum',
    # Calculation engines
    'compute_spectrum',
    'compute_inl_from_sine',
    'compute_harmonic_decomposition',
    # Plotting functions
    'plot_spectrum',
    'plot_spectrum_polar',
    'plot_dnl_inl',
    'plot_harmonic_decomposition_time',
    'plot_harmonic_decomposition_polar',
    # Analysis wrappers
    'analyze_inl_from_sine',
    'analyze_harmonic_decomposition',
    'fit_static_nonlin',
]

