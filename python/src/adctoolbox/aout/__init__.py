"""Analog output (time-domain) analysis tools."""

# High-level wrappers (user-facing)
from .analyze_spectrum import analyze_spectrum
from .analyze_spectrum_coherent_averaging import analyze_spectrum_coherent_averaging
from .analyze_spectrum_polar import analyze_spectrum_polar
from .analyze_phase_spectrum import analyze_phase_spectrum
from .analyze_two_tone_spectrum import analyze_two_tone_spectrum
from .analyze_decomposition_time import analyze_decomposition_time
from .analyze_decomposition_polar import analyze_decomposition_polar

# Calculation engines (modular core)
from .calculate_spectrum_data import calculate_spectrum_data
from .calculate_coherent_spectrum import calculate_coherent_spectrum
from .calculate_lms_decomposition import calculate_lms_decomposition

# Plotting functions (visualization)
from .plot_spectrum import plot_spectrum
from .plot_polar_phase import plot_polar_phase
from .plot_spectrum_polar import plot_spectrum_polar
from .plot_decomposition_time import plot_decomposition_time
from .plot_decomposition_polar import plot_decomposition_polar
from .plot_envelope_spectrum import plot_envelope_spectrum
from .plot_error_autocorr import plot_error_autocorr
from .plot_error_hist_code import plot_error_hist_code
from .plot_error_hist_phase import plot_error_hist_phase
from .plot_error_pdf import plot_error_pdf

# Legacy functions (backward compatibility)
from .decompose_harmonics import decompose_harmonics
from .calc_inl_sine import calc_inl_sine
from .fit_static_nonlin import fit_static_nonlin

__all__ = [
    # High-level wrappers
    'analyze_spectrum',
    'analyze_spectrum_coherent_averaging',
    'analyze_spectrum_polar',
    'analyze_phase_spectrum',
    'analyze_two_tone_spectrum',
    'analyze_decomposition_time',
    'analyze_decomposition_polar',
    # Calculation engines
    'calculate_spectrum_data',
    'calculate_coherent_spectrum',
    'calculate_lms_decomposition',
    # Plotting functions
    'plot_spectrum',
    'plot_polar_phase',
    'plot_spectrum_polar',
    'plot_decomposition_time',
    'plot_decomposition_polar',
    'plot_envelope_spectrum',
    'plot_error_autocorr',
    'plot_error_hist_code',
    'plot_error_hist_phase',
    'plot_error_pdf',
    # Legacy functions
    'decompose_harmonics',
    'calc_inl_sine',
    'fit_static_nonlin',
]
