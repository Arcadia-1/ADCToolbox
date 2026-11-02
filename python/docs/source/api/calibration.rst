Calibration (calibration)
=========================

Start with :doc:`../calibration_workflows` for supervised foreground correction
and :doc:`../foreground_theory` for mathematical definitions and scope.

Foreground bit-weight identification
------------------------------------

These functions identify affine reconstruction weights from a dedicated sine
capture. The full and lite variants belong to the same calibration family.

.. autofunction:: adctoolbox.calibrate_weight_sine
.. autofunction:: adctoolbox.calibration.calibrate_weight_sine_lite

Scaling and identification diagnostics
--------------------------------------

These helpers support calibration; they are not separate identification methods.

.. autofunction:: adctoolbox.scale_calibration_output
.. autofunction:: adctoolbox.diagnose_calibration_matrix

Other calibration workflows
---------------------------

TI sine mismatch extraction and correction are documented in
:doc:`timeinterleave`. See :doc:`../calibration_workflows` for foreground benchmarks.

Supervised foreground correction
--------------------------------

References must be supplied explicitly. These functions fit a direct correction
mapping, distinct from forward distortion characterization with
``fit_static_nonlin``. Models are JSON-compatible dictionaries.
Learned weights describe a digital correction, not individual ADC component
parameters. Normal operation evaluates a frozen model without a reference or
adaptation. See :doc:`../foreground_theory` for equations and applicability.

.. autofunction:: adctoolbox.build_calibration_features
.. autofunction:: adctoolbox.fit_calibration_polynomial
.. autofunction:: adctoolbox.fit_calibration_mlp
.. autofunction:: adctoolbox.apply_calibration
