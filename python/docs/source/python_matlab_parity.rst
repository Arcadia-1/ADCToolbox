Python vs MATLAB Parity
=======================

This page tracks the functional parity between the Python ``adctoolbox`` package
and the original MATLAB implementation. The goal is to keep both versions
feature-equivalent.

Last updated: 2026-03-10

Function Mapping
----------------

The table below maps every MATLAB core function to its Python equivalent.
Legacy/deprecated MATLAB wrappers (e.g., ``sineFit`` → ``sinfit``) are omitted;
only the current MATLAB name is listed.

.. list-table::
   :header-rows: 1
   :widths: 20 30 10

   * - MATLAB
     - Python
     - Status
   * - ``plotspec``
     - ``analyze_spectrum``
     - Matched
   * - ``plotphase``
     - ``analyze_spectrum_polar``
     - Matched
   * - ``sinfit``
     - ``fit_sine_4param``
     - Matched
   * - ``findbin``
     - ``find_coherent_frequency``
     - Matched
   * - ``findfreq``
     - ``estimate_frequency``
     - Matched
   * - ``alias``
     - ``fold_frequency_to_nyquist``, ``fold_bin_to_nyquist``
     - Matched
   * - ``tomdec``
     - ``analyze_decomposition_time``, ``analyze_decomposition_polar``
     - Matched
   * - ``errsin``
     - ``analyze_error_by_phase``, ``analyze_error_by_value``
     - Matched
   * - ``inlsin``
     - ``analyze_inl_from_sine``
     - Matched
   * - ``wcalsin``
     - ``calibrate_weight_sine``
     - Matched
   * - ``bitchk``
     - ``analyze_overflow``
     - Matched
   * - ``plotwgt``
     - ``analyze_weight_radix``
     - Matched
   * - ``ntfperf``
     - ``ntf_analyzer``
     - Matched
   * - ``adcpanel``
     - ``toolset/generate_aout_dashboard``, ``toolset/generate_dout_dashboard``
     - Matched
   * - ``cdacwgt``
     - *not implemented*
     - **MATLAB only**
   * - ``ifilter``
     - *not implemented*
     - **MATLAB only**
   * - ``perfosr``
     - *not implemented*
     - **MATLAB only**
   * - ``plotres``
     - *not implemented*
     - **MATLAB only**

MATLAB-Only Functions
---------------------

The following four MATLAB functions have **no Python equivalent** yet:

``cdacwgt(cd, cb, cp)``
   Calculate bit weights for a multi-segment capacitive DAC (CDAC) with bridge
   capacitors and parasitic capacitances. Returns normalized weights and total
   capacitance.

``ifilter(sigin, passband)``
   Ideal FFT-based brickwall filter. Retains only the specified frequency bands
   from an input signal. Operates column-wise on matrices.

``perfosr(sig, ...)``
   Sweep ADC performance (SNDR, SFDR, ENOB) versus oversampling ratio (OSR).
   Separates ideal signal from error via sine fitting, then re-evaluates metrics
   at narrowing bandwidths.

``plotres(sig, bits, ...)``
   Plot partial-sum residuals of an ADC bit matrix. Scatter-plots residuals
   between bit stages to reveal correlations, nonlinearity patterns, and
   redundancy.

Python-Only Functions
---------------------

The following functions exist **only in the Python package** and have no
MATLAB counterpart.

Spectrum
^^^^^^^^

- ``analyze_two_tone_spectrum`` — Two-tone FFT analysis with IMD2/IMD3 metrics.

Analog Error Analysis
^^^^^^^^^^^^^^^^^^^^^

- ``analyze_error_pdf`` — Error probability density function via KDE.
- ``analyze_error_spectrum`` — Error spectrum computed from fitting residual.
- ``analyze_error_autocorr`` — Autocorrelation function (ACF) of error signal.
- ``analyze_error_envelope_spectrum`` — Envelope spectrum via Hilbert transform
  to reveal AM modulation patterns.
- ``fit_static_nonlin`` — Extract static nonlinearity coefficients (k2, k3) from
  a distorted sinewave.

Digital Output Analysis
^^^^^^^^^^^^^^^^^^^^^^^

- ``analyze_bit_activity`` — Percentage of 1's per bit (DC offset / clipping
  detection).
- ``analyze_enob_sweep`` — ENOB vs. number of calibration bits.

Unit Conversions & Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

These are small utilities that MATLAB users typically write inline:

- ``db_to_mag``, ``mag_to_db``, ``db_to_power``, ``power_to_db``
- ``snr_to_enob``, ``enob_to_snr``, ``snr_to_nsd``, ``nsd_to_snr``
- ``lsb_to_volts``, ``volts_to_lsb``
- ``dbm_to_vrms``, ``vrms_to_dbm``, ``dbm_to_mw``, ``mw_to_dbm``
- ``sine_amplitude_to_power``, ``amplitudes_to_snr``
- ``bin_to_freq``, ``freq_to_bin``
- ``calculate_walden_fom``, ``calculate_schreier_fom``
- ``calculate_thermal_noise_limit``, ``calculate_jitter_limit``

Signal Generation
^^^^^^^^^^^^^^^^^

The Python ``siggen`` module provides a ``Nonidealities`` class for building
realistic ADC test signals with chainable impairments. This has no MATLAB
equivalent:

- Thermal noise, clock jitter, quantization noise
- Static nonlinearity (polynomial or harmonic-dB specification)
- Memory effect, incomplete sampling / settling
- Residue amplifier gain error (static and dynamic)
- Reference error (settling + droop)
- AM noise, AM tone, clipping, drift, glitch injection
- First-order noise shaping

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 30 10 10

   * - Category
     - MATLAB
     - Python
   * - Core analysis functions
     - 18
     - 43
   * - Matched across both
     - 14
     - 14
   * - MATLAB-only
     - 4
     - —
   * - Python-only
     - —
     - 29
   * - Signal generation module
     - No
     - Yes (``siggen``)
   * - Unit conversion utilities
     - No (inline)
     - Yes (20 functions)
   * - Two-tone analysis
     - No
     - Yes
