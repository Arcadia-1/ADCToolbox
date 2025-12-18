Analog Output Analysis (aout)
==============================

The ``aout`` module provides comprehensive tools for analyzing analog ADC outputs.

.. currentmodule:: adctoolbox

Error Analysis by Value
-----------------------

.. autofunction:: analyze_error_by_value
.. autofunction:: adctoolbox.aout.rearrange_error_by_value
.. autofunction:: adctoolbox.aout.plot_rearranged_error_by_value

Error Analysis by Phase
-----------------------

.. autofunction:: analyze_error_by_phase
.. autofunction:: adctoolbox.aout.rearrange_error_by_phase
.. autofunction:: adctoolbox.aout.plot_rearranged_error_by_phase

Statistical Error Analysis
--------------------------

.. autofunction:: analyze_error_pdf

Spectrum-Based Error Analysis
------------------------------

.. autofunction:: analyze_error_spectrum
.. autofunction:: analyze_error_envelope_spectrum
.. autofunction:: analyze_error_autocorr

Harmonic Decomposition
-----------------------

.. autofunction:: analyze_decomposition_time
.. autofunction:: analyze_decomposition_polar
.. autofunction:: adctoolbox.aout.decompose_harmonic_error
.. autofunction:: adctoolbox.aout.plot_decomposition_time
.. autofunction:: adctoolbox.aout.plot_decomposition_polar

INL/DNL Analysis
----------------

.. autofunction:: analyze_inl_from_sine
.. autofunction:: adctoolbox.aout.compute_inl_from_sine
.. autofunction:: adctoolbox.aout.plot_dnl_inl

Static Nonlinearity Fitting
----------------------------

.. autofunction:: fit_static_nonlin
