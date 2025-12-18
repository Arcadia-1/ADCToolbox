Digital Output Analysis (dout)
===============================

The ``dout`` module provides tools for analyzing digital ADC outputs and bit-weighted architectures.

.. currentmodule:: adctoolbox

Weight Calibration
------------------

.. autofunction:: calibrate_weight_sine
.. autofunction:: calibrate_weight_sine_osr
.. autofunction:: calibrate_weight_two_tone

Overflow Detection
------------------

.. autofunction:: check_overflow

Bit Activity
------------

.. autofunction:: check_bit_activity

ENOB Analysis
-------------

.. autofunction:: analyze_enob_sweep

Visualization
-------------

.. autofunction:: plot_weight_radix

Dashboard
---------

.. autofunction:: adctoolbox.dout.generate_dout_dashboard

Toolset
-------

.. autofunction:: adctoolbox.dout.toolset_dout
