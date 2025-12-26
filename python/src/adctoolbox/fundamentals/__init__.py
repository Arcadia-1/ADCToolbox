"""Fundamental utility functions for ADC analysis."""

# Frequency utilities (consolidated in frequency.py)
from .frequency import (
    fold_bin_to_nyquist,
    fold_frequency_to_nyquist,
    find_coherent_frequency,
    estimate_frequency,
)

# SNR/NSD utilities (consolidated in snr_nsd.py)
from .snr_nsd import amplitudes_to_snr, snr_to_nsd, nsd_to_snr

# Sine fitting
from .fit_sine_4param import fit_sine_4param

# Circuit calculations
from .convert_cap_to_weight import convert_cap_to_weight

# Validation utilities
from .validate import validate_aout_data, validate_dout_data

# Unit conversions
from .units import (
    db_to_mag, mag_to_db, db_to_power, power_to_db,
    lsb_to_volts, volts_to_lsb,
    bin_to_freq, freq_to_bin,
    snr_to_enob, enob_to_snr,
    dbm_to_vrms, vrms_to_dbm,
    dbm_to_mw, mw_to_dbm,
    sine_amplitude_to_power
)

# FOM/Performance metrics
from .metrics import (
    calculate_walden_fom, calculate_schreier_fom,
    calculate_thermal_noise_limit, calculate_jitter_limit
)

__all__ = [
    # Sine fitting
    'fit_sine_4param',
    # Circuit calculations
    'convert_cap_to_weight',
    # Frequency utilities
    'fold_bin_to_nyquist',
    'fold_frequency_to_nyquist',
    'find_coherent_frequency',
    'estimate_frequency',
    # SNR/NSD utilities
    'amplitudes_to_snr',
    'snr_to_nsd',
    'nsd_to_snr',
    # Validation utilities
    'validate_aout_data',
    'validate_dout_data',
    # FOM/Performance metrics
    'calculate_walden_fom',
    'calculate_schreier_fom',
    'calculate_thermal_noise_limit',
    'calculate_jitter_limit',
    # Unit conversions
    'db_to_mag',
    'mag_to_db',
    'db_to_power',
    'power_to_db',
    'lsb_to_volts',
    'volts_to_lsb',
    'bin_to_freq',
    'freq_to_bin',
    'snr_to_enob',
    'enob_to_snr',
    'dbm_to_vrms',
    'vrms_to_dbm',
    'dbm_to_mw',
    'mw_to_dbm',
    'sine_amplitude_to_power',
]
