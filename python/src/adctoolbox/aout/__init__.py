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
from .fit_sine_harmonics import fit_sine_harmonics
from .fit_sine_4param import fit_sine_4param

# Harmonic decomposition functions
from .compute_harmonic_decomposition import compute_harmonic_decomposition
from .analyze_harmonic_decomposition import analyze_harmonic_decomposition

try:
    from .plot_harmonic_decomposition_time import plot_harmonic_decomposition_time
    from .plot_harmonic_decomposition_polar import plot_harmonic_decomposition_polar
except ImportError:
    # These functions may not exist yet
    pass
from .compute_inl_from_sine import compute_inl_from_sine
from .analyze_inl_from_sine import analyze_inl_from_sine
from .fit_static_nonlin import fit_static_nonlin
from .plot_dnl_inl import plot_dnl_inl

# Phase error analysis functions (Layer 2: Computation Engines)
from .rearrange_error_by_phase import rearrange_error_by_phase
from .rearrange_error_by_code import rearrange_error_by_code

# Phase error analysis wrappers (Layer 3: User Interface)
from .analyze_error_by_phase import analyze_error_by_phase
from .analyze_error_by_code import analyze_error_by_code

# Plotting functions for phase and code error
from .plot_rearranged_error_by_phase import plot_rearranged_error_by_phase
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
    from .plot_rearranged_error_by_code import plot_rearranged_error_by_code
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
    'fit_sine_harmonics',
    'fit_sine_4param',
    # Layer 2: Computation Engines
    'compute_spectrum',
    'compute_inl_from_sine',
    'compute_harmonic_decomposition',
    'rearrange_error_by_phase',
    'rearrange_error_by_code',
    # Layer 3: Analysis Wrappers
    'analyze_inl_from_sine',
    'analyze_harmonic_decomposition',
    'analyze_error_by_phase',
    'analyze_error_by_code',
    # Plotting functions
    'plot_spectrum',
    'plot_spectrum_polar',
    'plot_dnl_inl',
    'plot_rearranged_error_by_phase',
    'plot_error_binned_code',
    # Other
    'fit_static_nonlin',
]

