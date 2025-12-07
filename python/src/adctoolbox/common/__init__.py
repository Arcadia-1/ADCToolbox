"""Common utility functions for ADC analysis."""

from .calc_aliased_freq import calc_aliased_freq
from .calc_coherent_freq import calc_coherent_freq
from .estimate_frequency import estimate_frequency
from .fit_sine import fit_sine
from .extract_freq_components import extract_freq_components
from .convert_cap_to_weight import convert_cap_to_weight
from .calc_target_vpp import calc_target_vpp
from .validate import validate_aout_data, validate_dout_data
from .conversions import (
    db_to_mag, mag_to_db, db_to_power, power_to_db,
    lsb_to_volts, volts_to_lsb,
    bin_to_freq, freq_to_bin,
    snr_to_enob, enob_to_snr,
    snr_to_nsd, nsd_to_snr,
    dbm_to_vrms, vrms_to_dbm,
    dbm_to_mw, mw_to_dbm,
    sine_amplitude_to_power
)
from .calc_fom import (
    calc_fom_walden, calc_fom_schreier,
    calc_thermal_noise_limit, calc_jitter_limit
)

__all__ = [
    'calc_aliased_freq',
    'calc_coherent_freq',
    'estimate_frequency',
    'fit_sine',
    'extract_freq_components',
    'convert_cap_to_weight',
    'calc_target_vpp',
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
    # FoM calculations
    'calc_fom_walden',
    'calc_fom_schreier',
    'calc_thermal_noise_limit',
    'calc_jitter_limit',
]
