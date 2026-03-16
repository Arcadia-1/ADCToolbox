"""
Spectrum analysis tools for ADC characterization.
"""

# ----------------------------------------------------------------------
# High-level wrappers (user-facing)
# ----------------------------------------------------------------------

from .analyze_spectrum import analyze_spectrum
from .analyze_spectrum_polar import analyze_spectrum_polar

# ----------------------------------------------------------------------
# Calculation engines (core computation)
# ----------------------------------------------------------------------

from .compute_spectrum import compute_spectrum

# ----------------------------------------------------------------------
# Plotting functions (visualization)
# ----------------------------------------------------------------------

from .plot_spectrum import plot_spectrum
from .plot_spectrum_polar import plot_spectrum_polar

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

    # Core computation
    'compute_spectrum',

    # Visualization
    'plot_spectrum',
    'plot_spectrum_polar',

    # Sweep / parametric
    'sweep_performance_vs_osr',
]
