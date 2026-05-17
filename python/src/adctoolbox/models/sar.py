"""SAR ADC forward model — binary or sub-radix-2, with optional non-idealities.

Function-based, no internal state. Suitable for stimulus generation,
calibration-algorithm development, ENoB Monte Carlo, and RTL backend
verification.

Convention
----------
``vin``     : normalized unipolar input, ``vin ∈ [0, 1]`` (mid-rail = 0.5)
``weights`` : normalized by ``sum(bit_weights) + 1 LSB``. For a non-redundant
              4-bit ADC this is ``[8, 4, 2, 1] / (8+4+2+1+1)``.
``codes``   : array of {0, 1}, MSB at column 0
``recon``   : ``codes @ digital_weights``. The maximum all-ones code
              reconstructs to ``sum(digital_weights)``.

For a fully-differential SAR with physical inputs VIP/VIN ∈ [0, VDD], map
into the normalized model with ``vin = (VIP - VIN + VDD) / (2 * VDD)``.
The algorithm is invariant under this affine transformation.

References
----------
Ported from Arcadia-1/SpecMind ``dataset_generation/simulation_engines/sar_adc.py``
and ``matlab_reference/sar_model_simplest.m`` (icdesign lab, 2025-03-25).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def sar_ideal_weights(num_bits: int, redundant_bit: Optional[int] = None) -> np.ndarray:
    """Generate ideal binary CDAC weights, optionally with one duplicated bit.

    The normalization base is the sum of all bit weights plus one LSB:
    ``denom = sum(raw_bit_weights) + raw_lsb``. In other words, a 4-bit
    ideal ADC uses ``[8, 4, 2, 1] / (8+4+2+1+1) = /16``, not
    endpoint-normalized ``/ 15`` weights. With one duplicated redundant bit,
    ``[8, 4, 4, 2, 1]`` normalizes by ``20``.

    Parameters
    ----------
    num_bits : int
        Architectural resolution (number of distinct decision steps).
    redundant_bit : int, optional
        If given, the cap at index ``redundant_bit`` is duplicated, yielding
        an output array of length ``num_bits + 1``. Use to model sub-radix-2
        SAR with one bit of redundancy. Architectural resolution stays at
        ``num_bits`` — the redundancy adds error-correction margin, not bits.

    Returns
    -------
    weights : ndarray of shape (B,)
        Cap weights, MSB at index 0. The denominator is
        ``sum(raw_bit_weights) + raw_lsb``. If ``redundant_bit`` is used, the
        duplicated cap participates in both the sum and the resulting
        overrange / correction margin.

    Examples
    --------
    >>> import numpy as np
    >>> w = sar_ideal_weights(4)
    >>> np.allclose(w, [8/16, 4/16, 2/16, 1/16])
    True
    >>> w = sar_ideal_weights(4, redundant_bit=1)
    >>> len(w) == 5 and np.allclose(w, [8/20, 4/20, 4/20, 2/20, 1/20])
    True
    """
    w = [2 ** (num_bits - 1 - i) for i in range(num_bits)]
    if redundant_bit is not None:
        w.insert(redundant_bit + 1, w[redundant_bit])
    w = np.asarray(w, dtype=float)
    return w / (w.sum() + w[-1])


def sar_apply_mismatch(
    weights: np.ndarray,
    sigma: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Realize one chip's per-cap mismatch as gaussian perturbation.

    Each cap is perturbed independently by ``N(1, sigma)``. Result is NOT
    re-normalized. Use the perturbed array as the analog CDAC weights passed
    to ``sar_encode``. For reconstruction, pass the digital weights actually
    used by the backend: nominal weights for an uncalibrated path, calibrated
    weights after estimation, or the same actual weights for an ideal
    observer.

    Parameters
    ----------
    weights : ndarray of shape (B,)
        Nominal analog CDAC weights (typically from
        :func:`sar_ideal_weights`).
    sigma : float
        RMS relative mismatch per cap. e.g. ``0.01`` = 1% σ. Typical values
        in 28 nm small-cap matrix: 0.01-0.08 depending on cap area.
    rng : np.random.Generator, optional
        Numpy random generator. Use a deterministic seed to lock the
        mismatch realization (same chip across train/test).

    Returns
    -------
    perturbed_weights : ndarray of shape (B,)
        Cap weights with per-element gaussian mismatch applied.
    """
    if rng is None:
        rng = np.random.default_rng()
    return weights * (1.0 + sigma * rng.standard_normal(len(weights)))


def sar_encode(
    vin: np.ndarray,
    weights: np.ndarray,
    noise_rms: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Batch SAR conversion, successive-approximation in normalized space.

    Per-sample algorithm::

        v_dac = 0
        for j in range(B):
            v_test = v_dac + weights[j]
            bit[j] = 1 if (vin + noise > v_test) else 0
            if bit[j]: v_dac = v_test

    Vectorized over samples so the whole batch runs in one Python loop of
    length ``B`` (typically 4-20 iterations), not ``N`` (typically 10⁴-10⁶).

    Parameters
    ----------
    vin : array-like of shape (N,)
        Normalized input voltage trace, ``vin ∈ [0, 1]`` recommended.
        Values outside this range produce saturated (all-1 or all-0) codes
        without hard-clipping artifacts.
    weights : array-like of shape (B,)
        Actual analog CDAC weights (MSB first). For ideal weights from
        :func:`sar_ideal_weights`, the normalization base is
        ``sum(raw_bit_weights) + raw_lsb``.
    noise_rms : float, default 0.0
        Comparator input-referred noise RMS, normalized to the same scale
        as ``vin``. Typical realistic value for a strongARM comparator in
        28 nm: ``0.5 / 2**num_bits`` (= half-LSB).
    rng : np.random.Generator, optional
        RNG for the comparator-noise stream. Pass **independent** RNGs
        across train/test captures on the same chip for honest held-out
        evaluation.

    Returns
    -------
    codes : ndarray of shape (N, B), dtype int8
        Raw bit decisions, MSB at column 0. Pass to :func:`sar_reconstruct`
        for the analog estimate, or to a calibration routine (e.g.
        :func:`adctoolbox.calibration.calibrate_weight_sine`) for weight
        estimation.

    Notes
    -----
    The model does NOT include kT/C sampling noise, DAC settling errors,
    metastability delay, charge-injection artifacts, or PVT drift. For
    those, supplement with a Spectre / Verilog-A simulation.
    """
    vin = np.atleast_1d(np.asarray(vin, dtype=float))
    weights = np.asarray(weights, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    N, B = len(vin), len(weights)
    codes = np.zeros((N, B), dtype=np.int8)
    v_dac = np.zeros(N)
    for j in range(B):
        v_test = v_dac + weights[j]
        noise = noise_rms * rng.standard_normal(N) if noise_rms > 0 else 0.0
        bit = (vin + noise >= v_test).astype(np.int8)
        codes[:, j] = bit
        v_dac = np.where(bit, v_test, v_dac)
    return codes


def sar_reconstruct(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Linear weighted-sum reconstruction: ``codes @ digital_weights``.

    ``weights`` are the digital reconstruction weights, which usually match
    the analog CDAC weights in an ideal model. With mismatch, keep the two
    roles explicit: encode with actual analog weights, then reconstruct with
    the nominal, calibrated, or actual digital weights you want to evaluate.
    Subtract the sample mean before spectrum analysis to remove the DC offset
    introduced by the unipolar encoding.

    Parameters
    ----------
    codes : ndarray of shape (N, B)
        Raw bit decisions from :func:`sar_encode`.
    weights : ndarray of shape (B,)
        Digital reconstruction weights. Use the nominal weights for an
        "uncalibrated" output; use cal-estimated or actual weights to assess
        calibration quality.

    Returns
    -------
    aout : ndarray of shape (N,)
        Reconstructed analog estimate, range ``[0, sum(weights)]``.
    """
    return codes.astype(float) @ np.asarray(weights, dtype=float)
