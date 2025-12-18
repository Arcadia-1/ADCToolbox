Analog Output Analysis (aout)
==============================

The ``aout`` module provides comprehensive tools for analyzing analog ADC outputs.

.. currentmodule:: adctoolbox

Error Analysis by Value
-----------------------

.. autofunction:: analyze_error_by_value
.. autofunction:: plot_error_by_value

Error Analysis by Phase
-----------------------

.. autofunction:: analyze_error_by_phase
.. autofunction:: plot_error_by_phase

Statistical Error Analysis
--------------------------

.. autofunction:: analyze_error_pdf
.. autofunction:: plot_error_pdf

Spectrum-Based Error Analysis
------------------------------

.. autofunction:: analyze_error_spectrum
.. autofunction:: plot_error_spectrum

.. autofunction:: analyze_error_envelope_spectrum
.. autofunction:: plot_error_envelope_spectrum

.. autofunction:: analyze_error_autocorr
.. autofunction:: plot_error_autocorr

Harmonic Decomposition
-----------------------

.. autofunction:: decompose_harmonic_error
.. autofunction:: analyze_harmonic_decomposition
.. autofunction:: plot_harmonic_decomposition

INL/DNL Analysis
----------------

.. autofunction:: analyze_inl_from_sine
.. autofunction:: compute_inl_from_sine
.. autofunction:: plot_dnl_inl

Toolset
-------

.. autofunction:: adctoolbox.aout.toolset_aout
