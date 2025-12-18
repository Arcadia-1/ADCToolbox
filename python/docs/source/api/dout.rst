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

.. autofunction:: adctoolbox.dout.check_overflow

Bit Activity
------------

.. autofunction:: adctoolbox.dout.check_bit_activity

ENOB Analysis
-------------

.. autofunction:: adctoolbox.dout.analyze_enob_sweep

Visualization
-------------

.. autofunction:: adctoolbox.dout.plot_weight_radix
