Digital Output Analysis (dout)
===============================

The ``dout`` module provides tools for analyzing digital ADC outputs and bit-weighted architectures.

.. currentmodule:: adctoolbox

Weight Calibration
------------------

.. autofunction:: calibrate_weight_sine
.. autofunction:: adctoolbox.calibration.calibrate_weight_sine_lite

Bit And Weight Analysis
-----------------------

``analyze_weight_radix`` returns ``radix``, ``wgtsca``, and ``effres``.
``effres`` is computed from the significant absolute weights as
``log2(sum(abs_w_sig) / min(abs_w_sig) + 1)``. It is a theoretical
weight-list span, not a missing-code, DNL, INL, or SAR reachability proof.

.. autofunction:: analyze_bit_activity
.. autofunction:: analyze_overflow
.. autofunction:: analyze_weight_radix

ENOB Analysis
-------------

.. autofunction:: analyze_enob_sweep

Visualization
-------------

.. autofunction:: plot_residual_scatter
