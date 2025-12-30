"""
Spectrum analysis tools for ADC characterization.
"""

# ----------------------------------------------------------------------
# High-level wrappers (user-facing)
# ----------------------------------------------------------------------

from .analyze_spectrum import analyze_spectrum
from .analyze_spectrum_polar import analyze_spectrum_polar
from .analyze_two_tone_spectrum import analyze_two_tone_spectrum

# ----------------------------------------------------------------------
# Calculation engines (core computation)
# ----------------------------------------------------------------------

from .compute_spectrum import compute_spectrum
from .compute_two_tone_spectrum import compute_two_tone_spectrum

# ----------------------------------------------------------------------
# Plotting functions (visualization)
# ----------------------------------------------------------------------

from .plot_spectrum import plot_spectrum
from .plot_spectrum_polar import plot_spectrum_polar

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
    'analyze_two_tone_spectrum',

    # Core computation
    'compute_spectrum',
    'compute_two_tone_spectrum',

    # Visualization
    'plot_spectrum',
    'plot_spectrum_polar',
]
