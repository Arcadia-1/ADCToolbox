Dither Calibration
==================

The public interfaces consume aligned observables; probe injection and device
simulation stay with the caller. Use ``DitherGainTracker`` for the gain-only
Pipeline model, and ``DitherHDCTracker`` for the three-PN nominal-gain-2 model.
Their physical input conventions differ. Theory, references and complete
experiments are in :doc:`../dither`; no workflow orchestrator is required.

.. autoclass:: adctoolbox.DitherGainTracker
   :members: apply, update, reset

.. autofunction:: adctoolbox.apply_dither_gain

.. autoclass:: adctoolbox.DitherHDCTracker
   :members: process, coefficients, reset
   :special-members: __init__
