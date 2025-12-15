"""Common utility functions for ADC analysis."""

from .fold_bin_to_nyquist import fold_bin_to_nyquist
from .fold_frequency_to_nyquist import fold_frequency_to_nyquist
from .find_coherent_frequency import find_coherent_frequency
from .amplitudes_to_snr import amplitudes_to_snr
from .estimate_frequency import estimate_frequency
from .extract_freq_components import extract_freq_components
from .convert_cap_to_weight import convert_cap_to_weight
from .vpp_for_target_dbfs import vpp_for_target_dbfs
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
    calculate_walden_fom, calculate_schreier_fom,
    calculate_thermal_noise_limit, calculate_jitter_limit
)

__all__ = [
    # Function names
    'fold_bin_to_nyquist',
    'fold_frequency_to_nyquist',
    'find_coherent_frequency',
    'amplitudes_to_snr',
    'vpp_for_target_dbfs',
    'calculate_walden_fom',
    'calculate_schreier_fom',
    'calculate_thermal_noise_limit',
    'calculate_jitter_limit',
    # Other functions
    'estimate_frequency',
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
