"""
Spectrum analysis tools for ADC characterization.
"""

# ----------------------------------------------------------------------
# High-level wrappers (user-facing)
# ----------------------------------------------------------------------

from .analyze_spectrum import analyze_spectrum, analyze_spectrum_virtuoso
from .analyze_spectrum_polar import analyze_spectrum_polar

# ----------------------------------------------------------------------
# Calculation engines (core computation)
# ----------------------------------------------------------------------

from .compute_spectrum import compute_spectrum
from .quick_sndr import quick_sndr
from .extract_freq_components import extract_freq_components

# ----------------------------------------------------------------------
# Plotting functions (visualization)
# ----------------------------------------------------------------------

from .plot_spectrum import plot_spectrum
from .plot_spectrum_polar import plot_spectrum_polar
from .plot_spectrum_virtuoso import plot_spectrum_virtuoso

# ----------------------------------------------------------------------
# Sweep / parametric analysis
# ----------------------------------------------------------------------

from .sweep_performance_vs_osr import sweep_performance_vs_osr

# ----------------------------------------------------------------------
# Internal helpers (NOT part of public API)
# ----------------------------------------------------------------------

from ._prepare_fft_input import _prepare_fft_input
from ._locate_fundamental import _locate_fundamental
from ._harmonics import _locate_harmonic_bins
from ._align_spectrum_phase import _align_spectrum_phase

# ----------------------------------------------------------------------
# Public API of spectrum subpackage
# ----------------------------------------------------------------------

__all__ = [
    # High-level analysis
    'analyze_spectrum',
    'analyze_spectrum_polar',
    'analyze_spectrum_virtuoso',

    # Core computation
    'compute_spectrum',
    'quick_sndr',
    'extract_freq_components',

    # Visualization
    'plot_spectrum',
    'plot_spectrum_polar',
    'plot_spectrum_virtuoso',

    # Sweep / parametric
    'sweep_performance_vs_osr',
]
