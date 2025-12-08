"""Common utility functions for ADC analysis."""

from .calculate_aliased_freq import calculate_aliased_freq
from .calculate_coherent_freq import calculate_coherent_freq
from .calculate_snr_from_amplitude import calculate_snr_from_amplitude
from .estimate_frequency import estimate_frequency
from .fit_sine import fit_sine
from .extract_freq_components import extract_freq_components
from .convert_cap_to_weight import convert_cap_to_weight
from .calculate_target_vpp import calculate_target_vpp
from .validate import validate_aout_data, validate_dout_data
from .unit_conversions import (
    db_to_mag, mag_to_db, db_to_power, power_to_db,
    lsb_to_volts, volts_to_lsb,
    bin_to_freq, freq_to_bin,
    snr_to_enob, enob_to_snr,
    snr_to_nsd, nsd_to_snr,
    dbm_to_vrms, vrms_to_dbm,
    dbm_to_mw, mw_to_dbm,
    sine_amplitude_to_power
)
from .calculate_fom import (
    calculate_fom_walden, calculate_fom_schreier,
    calculate_thermal_noise_limit, calculate_jitter_limit
)

__all__ = [
    # Function names
    'calculate_aliased_freq',
    'calculate_coherent_freq',
    'calculate_snr_from_amplitude',
    'calculate_target_vpp',
    'calculate_fom_walden',
    'calculate_fom_schreier',
    'calculate_thermal_noise_limit',
    'calculate_jitter_limit',
    # Other functions
    'estimate_frequency',
    'fit_sine',
    'extract_freq_components',
    'convert_cap_to_weight',
    'validate_aout_data',
    'validate_dout_data',
    # Conversions
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
    'snr_to_nsd',
    'nsd_to_snr',
    'dbm_to_vrms',
    'vrms_to_dbm',
    'dbm_to_mw',
    'mw_to_dbm',
    'sine_amplitude_to_power',
]
