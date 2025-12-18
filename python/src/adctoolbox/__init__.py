"""
ADCToolbox: A comprehensive toolbox for ADC testing and characterization.

This package provides tools for analyzing both analog and digital aspects of
Analog-to-Digital Converters, including spectrum analysis, error characterization,
calibration algorithms, and more.

Modules:
--------
- fundamentals: Core utilities (sine fitting, frequency utils, unit conversions, FOM metrics)
- spectrum: FFT-based analysis (single-tone, two-tone, polar visualization)
- aout: Analog output error analysis (decomposition, PDF, autocorrelation, etc.)
- dout: Digital output calibration (foreground calibration, weight estimation)
- siggen: Signal generation with non-idealities
- oversampling: Noise transfer function analysis

Usage:
------
>>> from adctoolbox import analyze_spectrum, fit_sine_4param, calibrate_weight_sine
>>> from adctoolbox import find_coherent_frequency, analyze_error_by_phase
"""

__version__ = '0.4.0'

# ======================================================================
# Public API Registry
# ======================================================================

__all__ = []


def _export(name, obj):
    """
    Register a public API symbol.

    Guarantees:
    1. The symbol exists in the module namespace
    2. The symbol is listed in __all__
    """
    globals()[name] = obj
    __all__.append(name)


# ======================================================================
# Core Fundamental Functions (Essential Utilities)
# ======================================================================

from .fundamentals import (
    find_coherent_frequency,
    estimate_frequency,
    fold_bin_to_nyquist,
    fold_frequency_to_nyquist,
    amplitudes_to_snr,
    db_to_mag,
    mag_to_db,
    snr_to_enob,
    enob_to_snr,
    snr_to_nsd,
    fit_sine_4param,
)

_export('find_coherent_frequency', find_coherent_frequency)
_export('estimate_frequency', estimate_frequency)
_export('fold_bin_to_nyquist', fold_bin_to_nyquist)
_export('fold_frequency_to_nyquist', fold_frequency_to_nyquist)
_export('amplitudes_to_snr', amplitudes_to_snr)
_export('db_to_mag', db_to_mag)
_export('mag_to_db', mag_to_db)
_export('snr_to_enob', snr_to_enob)
_export('enob_to_snr', enob_to_snr)
_export('snr_to_nsd', snr_to_nsd)
_export('fit_sine_4param', fit_sine_4param)

# ======================================================================
# Spectrum Analysis Functions
# ======================================================================

from .spectrum import (
    analyze_spectrum,
    analyze_two_tone_spectrum,
    analyze_spectrum_polar,
)

_export('analyze_spectrum', analyze_spectrum)
_export('analyze_two_tone_spectrum', analyze_two_tone_spectrum)
_export('analyze_spectrum_polar', analyze_spectrum_polar)


# ======================================================================
# Analog Output (AOUT) Analysis Functions
# ======================================================================

from .aout import (
    analyze_inl_from_sine,
    analyze_decomposition_time,
    analyze_decomposition_polar,
    analyze_error_by_value,
    analyze_error_by_phase,
    analyze_error_pdf,
    analyze_error_spectrum,
    analyze_error_autocorr,
    analyze_error_envelope_spectrum,
    fit_static_nonlin
)

_export('analyze_inl_from_sine', analyze_inl_from_sine)
_export('analyze_decomposition_time', analyze_decomposition_time)
_export('analyze_decomposition_polar', analyze_decomposition_polar)
_export('analyze_error_by_value', analyze_error_by_value)
_export('analyze_error_by_phase', analyze_error_by_phase)
_export('analyze_error_pdf', analyze_error_pdf)
_export('analyze_error_spectrum', analyze_error_spectrum)
_export('analyze_error_autocorr', analyze_error_autocorr)
_export('analyze_error_envelope_spectrum', analyze_error_envelope_spectrum)
_export('fit_static_nonlin', fit_static_nonlin)


# ======================================================================
# Digital Output (DOUT) Analysis Functions
# ======================================================================

from .dout import (
    calibrate_weight_sine,
    calibrate_weight_sine_osr,
    calibrate_weight_two_tone,
)

_export('calibrate_weight_sine', calibrate_weight_sine)
_export('calibrate_weight_sine_osr', calibrate_weight_sine_osr)
_export('calibrate_weight_two_tone', calibrate_weight_two_tone)


# ======================================================================
# Oversampling Analysis Functions
# ======================================================================

from .oversampling import (
    ntf_analyzer,
)

_export('ntf_analyzer', ntf_analyzer)


# ======================================================================
# Submodules (for explicit imports like: from adctoolbox.aout import ...)
# ======================================================================

from . import fundamentals
from . import aout
from . import dout
from . import oversampling
from . import spectrum

_export('fundamentals', fundamentals)
_export('aout', aout)
_export('dout', dout)
_export('oversampling', oversampling)
_export('spectrum', spectrum)

