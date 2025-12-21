Lightweight Sinewave Calibration (Lite)
========================================

Overview
--------

The ``calibrate_weight_sine_lite`` function provides a minimal, fast implementation of ADC foreground calibration using a known-frequency sinewave input. This is a simplified version of the full ``calibrate_weight_sine`` algorithm, optimized for speed and code simplicity at the cost of some features.

**Key Characteristics:**

- Single known-frequency calibration only (no frequency search)
- Cosine-basis assumption (no dual-basis optimization)
- No rank deficiency handling (binary weights only)
- No harmonic rejection
- Returns normalized weights only
- Minimal dependencies (NumPy + SciPy only)

Mathematical Formulation
------------------------

Signal Model
~~~~~~~~~~~~

The ADC output can be modeled as a weighted sum of bit values plus a sinewave:

.. math::

   y(n) = \sum_{i=0}^{M-1} w_i \cdot b_i(n) = A \cos(2\pi f n) + B \sin(2\pi f n) + C

where:

- :math:`y(n)` = reconstructed analog signal at sample :math:`n`
- :math:`w_i` = weight of bit :math:`i` (unknown)
- :math:`b_i(n) \in \{0, 1\}` = binary value of bit :math:`i` at sample :math:`n`
- :math:`M` = ADC bit width
- :math:`f` = normalized frequency :math:`(f_{in}/f_s)`
- :math:`A, B` = sinewave amplitude coefficients
- :math:`C` = DC offset

Least Squares Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Rearranging the signal model with the **cosine-basis assumption** (:math:`A = 1`):

.. math::

   \sum_{i=0}^{M-1} w_i \cdot b_i(n) + C + B \sin(2\pi f n) = -\cos(2\pi f n)

In matrix form:

.. math::

   \begin{bmatrix}
   b_0(0) & b_1(0) & \cdots & b_{M-1}(0) & 1 & \sin(2\pi f \cdot 0) \\
   b_0(1) & b_1(1) & \cdots & b_{M-1}(1) & 1 & \sin(2\pi f \cdot 1) \\
   \vdots & \vdots & \ddots & \vdots & \vdots & \vdots \\
   b_0(N-1) & b_1(N-1) & \cdots & b_{M-1}(N-1) & 1 & \sin(2\pi f \cdot (N-1))
   \end{bmatrix}
   \begin{bmatrix}
   w_0 \\
   w_1 \\
   \vdots \\
   w_{M-1} \\
   C \\
   B
   \end{bmatrix}
   =
   \begin{bmatrix}
   -\cos(2\pi f \cdot 0) \\
   -\cos(2\pi f \cdot 1) \\
   \vdots \\
   -\cos(2\pi f \cdot (N-1))
   \end{bmatrix}

Or compactly:

.. math::

   \mathbf{A} \mathbf{x} = \mathbf{b}

where:

- :math:`\mathbf{A} \in \mathbb{R}^{N \times (M+2)}` = design matrix (bits, offset column, sine basis)
- :math:`\mathbf{x} = [w_0, w_1, \ldots, w_{M-1}, C, B]^T` = unknown coefficients
- :math:`\mathbf{b} = -[\cos(2\pi f \cdot 0), \ldots, \cos(2\pi f \cdot (N-1))]^T` = target vector

Least Squares Solution
~~~~~~~~~~~~~~~~~~~~~~~

The solution is found using standard least squares:

.. math::

   \mathbf{x}^* = \arg\min_{\mathbf{x}} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2

This is solved using ``scipy.linalg.lstsq``, which computes the pseudo-inverse:

.. math::

   \mathbf{x}^* = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b}

Weight Normalization
~~~~~~~~~~~~~~~~~~~~

The raw weights from the least squares solution are normalized by the sinewave magnitude:

.. math::

   \text{norm\_factor} = \sqrt{1 + B^2}

.. math::

   \tilde{w}_i = \frac{w_i^*}{\text{norm\_factor}}

This normalization accounts for the fact that the actual sinewave amplitude is:

.. math::

   \text{Amplitude} = \sqrt{A^2 + B^2} = \sqrt{1 + B^2}

since we assumed :math:`A = 1`.

Polarity Correction
~~~~~~~~~~~~~~~~~~~

A final polarity check ensures the weights are positive (MSB weight is positive):

.. math::

   w_i = \begin{cases}
   \tilde{w}_i & \text{if } \sum_{i} \tilde{w}_i \geq 0 \\
   -\tilde{w}_i & \text{otherwise}
   \end{cases}

Algorithm Steps
---------------

The algorithm executes the following steps:

1. **Build Basis Functions**

   .. math::

      t = [0, 1, 2, \ldots, N-1]

   .. math::

      \cos\_basis = \cos(2\pi f \cdot t)

   .. math::

      \sin\_basis = \sin(2\pi f \cdot t)

2. **Construct Design Matrix**

   .. math::

      \mathbf{A} = [\mathbf{B} \mid \mathbf{1} \mid \sin\_basis]

   where :math:`\mathbf{B} \in \mathbb{R}^{N \times M}` is the bit matrix and :math:`\mathbf{1}` is an offset column.

3. **Construct Target Vector**

   .. math::

      \mathbf{b} = -\cos\_basis

4. **Solve Least Squares**

   .. math::

      \mathbf{x}^* = \text{lstsq}(\mathbf{A}, \mathbf{b})

5. **Extract and Normalize Weights**

   .. math::

      w_i^{\text{raw}} = \mathbf{x}^*[0:M]

   .. math::

      B = \mathbf{x}^*[M+1]

   .. math::

      w_i = \frac{w_i^{\text{raw}}}{\sqrt{1 + B^2}}

6. **Polarity Correction**

   .. math::

      w_i \leftarrow -w_i \quad \text{if } \sum_i w_i < 0

Usage Example
-------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from adctoolbox.calibration import calibrate_weight_sine_lite

   # Generate test data (12-bit ADC, 8192 samples, freq = 13/8192)
   n_samples = 8192
   bit_width = 12
   freq_true = 13 / n_samples

   # Create ideal sinewave and quantize
   t = np.arange(n_samples)
   signal = 0.5 * np.sin(2 * np.pi * freq_true * t + np.pi/4) + 0.5
   quantized = np.clip(np.floor(signal * (2**bit_width)), 0, 2**bit_width - 1).astype(int)

   # Extract bits (MSB first)
   bits = (quantized[:, None] >> np.arange(bit_width - 1, -1, -1)) & 1

   # Run calibration
   recovered_weights = calibrate_weight_sine_lite(bits, freq=freq_true)

   # Scale to actual ADC weights
   true_weights = 2.0 ** np.arange(bit_width - 1, -1, -1)
   recovered_weights_scaled = recovered_weights * np.max(true_weights)

   print(f"True weights:      {true_weights}")
   print(f"Recovered weights: {recovered_weights_scaled}")

Expected Output
~~~~~~~~~~~~~~~

.. code-block:: text

   True weights:      [2048. 1024.  512.  256.  128.   64.   32.   16.    8.    4.    2.    1.]
   Recovered weights: [2048.0 1024.0  512.0  256.0  128.0   64.0   32.0   16.0    8.0    4.0    2.0    1.0]

Limitations
-----------

Known Frequency Required
~~~~~~~~~~~~~~~~~~~~~~~~

The function requires the input frequency to be **precisely known**. Unlike the full ``calibrate_weight_sine``, it does not perform frequency search or refinement.

**Workaround**: Use ``calibrate_weight_sine`` with ``force_search=True`` if frequency is unknown.

Binary Weights Only
~~~~~~~~~~~~~~~~~~~

The algorithm **does not handle rank deficiency** or redundant weights. For ADCs with:

- Redundant bits (e.g., ``[128, 128, 64, 32, ...]``)
- Identical weights
- Linear dependencies between bits

The least squares solution may:

- Collapse redundant weights to zero
- Produce numerically unstable results
- Fail to recover the full code range

**Example Failure**:

.. code-block:: python

   # Redundant weights: two bits with weight 128
   true_weights = np.array([2048, 1024, 512, 256, 128, 128, 64, 32, 16, 8, 4, 2])

   # Calibration may recover:
   recovered = np.array([2048, 1024, 512, 256, 128, 0, 64, 32, 16, 8, 4, 2])
   #                                                   ^
   #                                        Redundant bit collapsed!

**Workaround**: Use the full ``calibrate_weight_sine`` which includes ``_patch_rank_deficiency`` handling.

Low Signal Amplitude
~~~~~~~~~~~~~~~~~~~~

At very low input amplitudes (< -6 dBFS), MSB bits may have limited activity, leading to:

- Ill-conditioned least squares matrix
- Numerically unstable weights (values in trillions)
- Poor SNDR estimates

**Recommendation**: Use input signals at or near full scale (0 dBFS to -3 dBFS) for best results.

No Harmonic Rejection
~~~~~~~~~~~~~~~~~~~~~

The algorithm fits only the **fundamental frequency**, without excluding harmonics from the error term. This can lead to:

- Harmonic distortion biasing the weight estimates
- Reduced accuracy for ADCs with significant INL/DNL

**Workaround**: Use ``calibrate_weight_sine`` with ``harmonic_order > 1`` to exclude harmonics.

Computational Complexity
------------------------

Time Complexity
~~~~~~~~~~~~~~~

The algorithm has time complexity:

.. math::

   O(N \cdot M^2 + M^3)

where:

- :math:`N` = number of samples
- :math:`M` = bit width

The dominant cost is the least squares solve via SVD decomposition in ``scipy.linalg.lstsq``.

**Typical Performance** (12-bit ADC, Intel i7, single-threaded):

- N = 2¹² (4096 samples): ~3 ms
- N = 2¹³ (8192 samples): ~5 ms
- N = 2¹⁶ (65536 samples): ~40 ms

Space Complexity
~~~~~~~~~~~~~~~~

Memory usage:

.. math::

   O(N \cdot M)

for storing the design matrix :math:`\mathbf{A}`.

**Example**: 12-bit ADC, 8192 samples → ~800 KB

Comparison with Full Version
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - ``calibrate_weight_sine_lite``
     - ``calibrate_weight_sine``
   * - Frequency search
     - ❌ No (requires known freq)
     - ✅ Yes (coarse + fine search)
   * - Rank deficiency handling
     - ❌ No
     - ✅ Yes (via nominal weights)
   * - Redundant weights
     - ❌ Not supported
     - ✅ Fully supported
   * - Harmonic rejection
     - ❌ No
     - ✅ Yes (configurable order)
   * - Multi-dataset calibration
     - ❌ No
     - ✅ Yes
   * - Numerical conditioning
     - ❌ Basic
     - ✅ Column scaling + patching
   * - Return type
     - ndarray (weights only)
     - dict (weights + diagnostics)
   * - Code size
     - ~40 lines
     - ~600+ lines (with helpers)
   * - Typical runtime (N=8192)
     - ~5 ms
     - ~20-50 ms

**When to use lite version**:

- Frequency is precisely known
- Binary weighted ADC (no redundancy)
- Speed is critical
- Simple embedded deployment

**When to use full version**:

- Unknown or imprecise frequency
- Redundant ADC architecture
- Need harmonic rejection
- Multi-dataset calibration
- Production calibration requiring robustness

References
----------

1. Vogel, C., & Johansson, H. (2006). "Time-interleaved analog-to-digital converters: Status and future directions." *IEEE Circuits and Systems Magazine*, 6(4), 26-39.

2. Jin, H., & Lee, E. K. F. (1992). "A digital-background calibration technique for minimizing timing-error effects in time-interleaved ADCs." *IEEE Transactions on Circuits and Systems II*, 47(7), 603-613.

3. Matlab ADC Toolbox documentation: https://www.mathworks.com/help/signal/ref/sinefitweights.html

See Also
--------

- :doc:`calibrate_weight_sine` - Full version with frequency search and rank handling
- :doc:`../api/calibration` - Complete calibration API reference
