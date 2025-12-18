"""
Analog output (AOUT) analysis tools.

This subpackage defines the public API of the AOUT analysis domain.
"""

# ----------------------------------------------------------------------
# Math core
# ----------------------------------------------------------------------

from .fit_sine_4param import fit_sine_4param
from .fit_sine_harmonics import fit_sine_harmonics

# ----------------------------------------------------------------------
# Value / Phase error analysis
# ----------------------------------------------------------------------

from .analyze_error_by_value import analyze_error_by_value
from .analyze_error_by_phase import analyze_error_by_phase
from .rearrange_error_by_value import rearrange_error_by_value
from .rearrange_error_by_phase import rearrange_error_by_phase

from .plot_rearranged_error_by_value import plot_rearranged_error_by_value
from .plot_rearranged_error_by_phase import plot_rearranged_error_by_phase

# ----------------------------------------------------------------------
# Harmonic decomposition
# ----------------------------------------------------------------------

from .analyze_harmonic_decomposition import analyze_harmonic_decomposition
from .analyze_decomposition_time import analyze_decomposition_time
from .decompose_harmonic_error import decompose_harmonic_error
from .plot_decomposition_time import plot_decomposition_time
from .plot_decomposition_polar import plot_decomposition_polar

# ----------------------------------------------------------------------
# INL / DNL from sine
# ----------------------------------------------------------------------

from .analyze_inl_from_sine import analyze_inl_from_sine
from .compute_inl_from_sine import compute_inl_from_sine
from .plot_dnl_inl import plot_dnl_inl

# ----------------------------------------------------------------------
# Static nonlinearity fitting
# ----------------------------------------------------------------------

from .fit_static_nonlin import fit_static_nonlin


# ----------------------------------------------------------------------
# Public API of aout subpackage
# ----------------------------------------------------------------------

__all__ = [
    # Math core
    'fit_sine_4param',
    'fit_sine_harmonics',

    # Error analysis
    'analyze_error_by_value',
    'analyze_error_by_phase',
    'rearrange_error_by_value',
    'rearrange_error_by_phase',

    # Harmonic decomposition
    'analyze_harmonic_decomposition',
    'analyze_decomposition_time',
    'decompose_harmonic_error',

    # INL / DNL
    'analyze_inl_from_sine',
    'compute_inl_from_sine',

    # Static nonlinearity
    'fit_static_nonlin',

    # Plotting (AOUT domain only)
    'plot_rearranged_error_by_value',
    'plot_rearranged_error_by_phase',
    'plot_decomposition_time',
    'plot_decomposition_polar',
    'plot_dnl_inl',
]
