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

# Core mathematical kernels (Layer 1: Math Core)
from .decompose_harmonics import fit_sinewave_components
from .fit_sine_4param import fit_sine_4param

# Harmonic decomposition functions
from .compute_harmonic_decomposition import compute_harmonic_decomposition
from .plot_harmonic_decomposition_time import plot_harmonic_decomposition_time
from .plot_harmonic_decomposition_polar import plot_harmonic_decomposition_polar
from .analyze_harmonic_decomposition import analyze_harmonic_decomposition
from .compute_inl_from_sine import compute_inl_from_sine
from .analyze_inl_from_sine import analyze_inl_from_sine
from .fit_static_nonlin import fit_static_nonlin
from .plot_dnl_inl import plot_dnl_inl

# Phase error analysis functions (Layer 2: Analysis Kernels)
from .compute_phase_error_from_binned import compute_phase_error_from_binned
from .compute_phase_error_from_raw import compute_phase_error_from_raw
from .compute_error_by_code import compute_error_by_code

# Phase error analysis wrappers (Layer 3: User Interface)
from .analyze_phase_error_trend import analyze_phase_error_trend
from .analyze_phase_error_raw_estimate import analyze_phase_error_raw_estimate
from .analyze_code_error import analyze_code_error

# Plotting functions for phase and code error
from .plot_error_binned_phase import plot_error_binned_phase
from .plot_error_binned_code import plot_error_binned_code

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
    # Layer 1: Math Core
    'fit_sinewave_components',
    'fit_sine_4param',
    # Layer 2: Calculation Engines
    'compute_spectrum',
    'compute_inl_from_sine',
    'compute_harmonic_decomposition',
    'compute_phase_error_from_binned',
    'compute_phase_error_from_raw',
    'compute_error_by_code',
    # Layer 3: Analysis Wrappers
    'analyze_inl_from_sine',
    'analyze_harmonic_decomposition',
    'analyze_phase_error_trend',
    'analyze_phase_error_raw_estimate',
    'analyze_code_error',
    # Plotting functions
    'plot_spectrum',
    'plot_spectrum_polar',
    'plot_dnl_inl',
    'plot_harmonic_decomposition_time',
    'plot_harmonic_decomposition_polar',
    'plot_error_binned_phase',
    'plot_error_binned_code',
    # Other
    'fit_static_nonlin',
]

