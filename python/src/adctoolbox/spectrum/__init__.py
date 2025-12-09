"""Spectrum analysis tools for ADC characterization."""

# High-level wrappers (user-facing)
from .analyze_spectrum import analyze_spectrum
from .analyze_spectrum_coherent_averaging import analyze_spectrum_coherent_averaging
from .analyze_spectrum_polar import analyze_spectrum_polar
from .analyze_two_tone_spectrum import analyze_two_tone_spectrum

# Calculation engines (modular core)
from .calculate_spectrum_data import calculate_spectrum_data
from .calculate_coherent_spectrum import calculate_coherent_spectrum

# Plotting functions (visualization)
from .plot_spectrum import plot_spectrum
from .plot_spectrum_polar import plot_spectrum_polar

# Helper functions (internal)
from ._prepare_fft_input import _prepare_fft_input
from ._find_fundamental import _find_fundamental
from ._find_harmonic_bins import _find_harmonic_bins
from ._align_spectrum_phase import _align_spectrum_phase

__all__ = [
    # High-level wrappers
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
    # Helper functions
    '_prepare_fft_input',
    '_find_fundamental',
    '_find_harmonic_bins',
    '_align_spectrum_phase',
]
