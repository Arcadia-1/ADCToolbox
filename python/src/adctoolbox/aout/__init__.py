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
    'plot_rearranged_error_by_value',
    'plot_rearranged_error_by_phase',
    'plot_decomposition_time',
    'plot_decomposition_polar',
    'plot_dnl_inl',
]
