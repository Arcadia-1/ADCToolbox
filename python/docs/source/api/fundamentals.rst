Fundamental Utilities (fundamentals)
=====================================

The ``fundamentals`` module provides core utility functions and signal processing tools used across the toolbox.

.. currentmodule:: adctoolbox

Sine Fitting
------------

.. autofunction:: fit_sine_4param

Frequency Utilities
-------------------

.. autofunction:: find_coherent_frequency
.. autofunction:: estimate_frequency
.. autofunction:: fold_frequency_to_nyquist
.. autofunction:: fold_bin_to_nyquist

Unit Conversions
----------------

.. autofunction:: db_to_mag
.. autofunction:: mag_to_db
.. autofunction:: db_to_power
.. autofunction:: power_to_db
.. autofunction:: snr_to_enob
.. autofunction:: enob_to_snr
.. autofunction:: adctoolbox.fundamentals.lsb_to_volts
.. autofunction:: adctoolbox.fundamentals.volts_to_lsb
.. autofunction:: adctoolbox.fundamentals.bin_to_freq
.. autofunction:: adctoolbox.fundamentals.freq_to_bin
.. autofunction:: adctoolbox.fundamentals.dbm_to_vrms
.. autofunction:: adctoolbox.fundamentals.vrms_to_dbm
.. autofunction:: adctoolbox.fundamentals.dbm_to_mw
.. autofunction:: adctoolbox.fundamentals.mw_to_dbm
.. autofunction:: adctoolbox.fundamentals.sine_amplitude_to_power

SNR/NSD Conversion
------------------

.. autofunction:: amplitudes_to_snr
.. autofunction:: snr_to_nsd
.. autofunction:: adctoolbox.fundamentals.nsd_to_snr

Validation
----------

.. autofunction:: adctoolbox.fundamentals.validate_aout_data
.. autofunction:: adctoolbox.fundamentals.validate_dout_data

Figures of Merit
----------------

.. autofunction:: adctoolbox.fundamentals.calculate_walden_fom
.. autofunction:: adctoolbox.fundamentals.calculate_schreier_fom
.. autofunction:: adctoolbox.fundamentals.calculate_thermal_noise_limit
.. autofunction:: adctoolbox.fundamentals.calculate_jitter_limit
