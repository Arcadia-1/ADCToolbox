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

__version__ = '0.5.0'

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
    bin_to_freq,
    freq_to_bin,
    amplitudes_to_snr,
    db_to_mag,
    mag_to_db,
    db_to_power,
    power_to_db,
    snr_to_enob,
    enob_to_snr,
    snr_to_nsd,
    nsd_to_snr,
    lsb_to_volts,
    volts_to_lsb,
    dbm_to_vrms,
    vrms_to_dbm,
    dbm_to_mw,
    mw_to_dbm,
    sine_amplitude_to_power,
    fit_sine_4param,
    calculate_walden_fom,
    calculate_schreier_fom,
    calculate_thermal_noise_limit,
    calculate_jitter_limit,
)

_export('find_coherent_frequency', find_coherent_frequency)
_export('estimate_frequency', estimate_frequency)
_export('fold_bin_to_nyquist', fold_bin_to_nyquist)
_export('fold_frequency_to_nyquist', fold_frequency_to_nyquist)
_export('bin_to_freq', bin_to_freq)
_export('freq_to_bin', freq_to_bin)
_export('amplitudes_to_snr', amplitudes_to_snr)
_export('db_to_mag', db_to_mag)
_export('mag_to_db', mag_to_db)
_export('db_to_power', db_to_power)
_export('power_to_db', power_to_db)
_export('snr_to_enob', snr_to_enob)
_export('enob_to_snr', enob_to_snr)
_export('snr_to_nsd', snr_to_nsd)
_export('nsd_to_snr', nsd_to_snr)
_export('fit_sine_4param', fit_sine_4param)
_export('lsb_to_volts', lsb_to_volts)
_export('volts_to_lsb', volts_to_lsb)
_export('dbm_to_vrms', dbm_to_vrms)
_export('vrms_to_dbm', vrms_to_dbm)
_export('dbm_to_mw', dbm_to_mw)
_export('mw_to_dbm', mw_to_dbm)
_export('sine_amplitude_to_power', sine_amplitude_to_power)
_export('calculate_walden_fom', calculate_walden_fom)
_export('calculate_schreier_fom', calculate_schreier_fom)
_export('calculate_thermal_noise_limit', calculate_thermal_noise_limit)
_export('calculate_jitter_limit', calculate_jitter_limit)

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
# Calibration Functions
# ======================================================================

from .calibration import (
    calibrate_weight_sine,
)

_export('calibrate_weight_sine', calibrate_weight_sine)

# ======================================================================
# Digital Output (DOUT) Analysis Functions
# ======================================================================

from .dout import (
    analyze_bit_activity,
    analyze_overflow,
    analyze_weight_radix,
    analyze_enob_sweep,
)

_export('analyze_bit_activity', analyze_bit_activity)
_export('analyze_overflow', analyze_overflow)
_export('analyze_weight_radix', analyze_weight_radix)
_export('analyze_enob_sweep', analyze_enob_sweep)

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
from . import calibration
from . import dout
from . import oversampling
from . import spectrum

_export('fundamentals', fundamentals)
_export('aout', aout)
_export('calibration', calibration)
_export('dout', dout)
_export('oversampling', oversampling)
_export('spectrum', spectrum)

