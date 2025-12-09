"""Analog output (time-domain) analysis tools - legacy re-exports from spectrum module."""

# Import spectrum analysis functions from the new spectrum module
from adctoolbox.spectrum import (
    analyze_spectrum,
    analyze_spectrum_coherent_averaging,
    analyze_spectrum_polar,
    analyze_two_tone_spectrum,
    calculate_spectrum_data,
    calculate_coherent_spectrum,
    plot_spectrum,
    plot_spectrum_polar,
)

# Legacy functions (backward compatibility)
from .decompose_harmonics import decompose_harmonics
from .calc_inl_sine import calc_inl_sine
from .fit_static_nonlin import fit_static_nonlin

# Placeholder functions for functions not yet moved
try:
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
    'analyze_spectrum_coherent_averaging',
    'analyze_spectrum_polar',
    'analyze_two_tone_spectrum',
    # Calculation engines
    'calculate_spectrum_data',
    'calculate_coherent_spectrum',
    # Plotting functions
    'plot_spectrum',
    'plot_spectrum_polar',
    # Legacy functions
    'decompose_harmonics',
    'calc_inl_sine',
    'fit_static_nonlin',
]

