Python vs MATLAB Parity
=======================

This page tracks the functional parity between the Python ``adctoolbox`` package
and the original MATLAB implementation. The goal is to keep both versions
feature-equivalent.

Last updated: 2026-06-07

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
     - ``ntfperf`` / ``ntf_analyzer``
     - Matched
   * - ``adcpanel``
     - ``toolset/generate_aout_dashboard``, ``toolset/generate_dout_dashboard``
     - Matched
   * - ``cdacwgt``
     - *not implemented*
     - **MATLAB only**
   * - ``ifilter``
     - ``ifilter`` / ``extract_freq_components``
     - Matched
   * - ``perfosr``
     - ``perfosr`` / ``sweep_performance_vs_osr``
     - Matched
   * - ``plotres``
     - *not implemented*
     - **MATLAB only**

MATLAB-Only Functions
---------------------

The following two MATLAB functions have **no Python equivalent** yet:

``cdacwgt(cd, cb, cp)``
   Calculate bit weights for a multi-segment capacitive DAC (CDAC) with bridge
   capacitors and parasitic capacitances. Returns normalized weights and total
   capacitance.

``plotres(sig, bits, ...)``
   Plot partial-sum residuals of an ADC bit matrix. Scatter-plots residuals
   between bit stages to reveal correlations, nonlinearity patterns, and
   redundancy.

Oversampling Compatibility
--------------------------

The MATLAB oversampling analysis helpers are available from
``adctoolbox.oversampling``:

``ifilter(sigin, passband)``
   Ideal FFT-based brickwall filter. Retains only the specified normalized
   frequency bands from an input signal. Operates column-wise on matrices and
   follows the MATLAB orientation rule for wide inputs.

``perfosr(sig, ...)``
   Sweep ADC performance (SNDR, SFDR, ENOB) versus oversampling ratio (OSR).
   Returns MATLAB-style outputs ``osr, sndr, sfdr, enob`` while sharing the
   same calculation engine as ``sweep_performance_vs_osr``.

``ntfperf(ntf, fl, fh, disp=False)``
   MATLAB-style wrapper around ``ntf_analyzer`` for theoretical NTF in-band
   noise suppression.

Python-Only Functions
---------------------

The following functions exist **only in the Python package** and have no
MATLAB counterpart.

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
- ``plot_residual_scatter`` — Partial-sum residual visualization.

ADC Behavioral Models
^^^^^^^^^^^^^^^^^^^^^

- ``sar_convert`` / ``sar_reconstruct`` — SAR forward conversion and analog
  reconstruction.
- ``sar_ideal_weights`` — Ideal binary or redundant SAR weights.
- ``sar_apply_cap_mismatch`` — Unit-cap/Pelgrom-style CDAC mismatch model.
- ``sar_apply_mismatch`` — Legacy per-weight gaussian perturbation helper.

Time-Interleaved ADC Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``deinterleave`` / ``interleave`` — Split and recombine interleaved samples.
- ``extract_mismatch_sine`` — Estimate channel gain, offset, and timing skew
  from a sine capture.
- ``predict_spurs`` — Predict mismatch spur locations and amplitudes.
- ``fractional_delay_fft`` / ``fractional_delay_farrow`` — Fractional-delay
  correction primitives.
- ``calibrate_foreground`` — Foreground offset/gain/skew correction.

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
     - 41
   * - Signal generation module
     - No
     - Yes (``siggen``)
   * - Unit conversion utilities
     - No (inline)
     - Yes (20 functions)
   * - SAR behavioral models
     - No
     - Yes
   * - Time-interleaved ADC tools
     - No
     - Yes
