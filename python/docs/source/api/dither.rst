Dither Calibration
==================

The public interfaces consume aligned observables; probe injection and device
simulation stay with the caller. Use ``DitherGainTracker`` for a single-probe,
gain-only stage and ``DitherHDCTracker`` for the three-PN, nominal-gain-2 HDC
stage. Their input conventions differ. The class notes below state each
tracker's observability requirements and accuracy trade-offs; theory,
references and complete experiments are in :doc:`../dither`.

The HDC tracker runs a per-sample loop. For long captures, install the optional
Numba acceleration with ``pip install "adctoolbox[accel]"``.

.. autoclass:: adctoolbox.DitherGainTracker
   :members: process, apply, update, reset

.. autofunction:: adctoolbox.apply_dither_gain

.. autoclass:: adctoolbox.DitherHDCTracker
   :members: process, coefficients, reset
