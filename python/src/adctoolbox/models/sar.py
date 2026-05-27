"""SAR ADC forward model — binary or sub-radix-2, with optional non-idealities.

Function-based, no internal state. Suitable for stimulus generation,
calibration-algorithm development, ENoB Monte Carlo, and RTL backend
verification.

Convention
----------
``vin``     : input voltage in ``quant_range``. Default ``quant_range`` is
              ``(0, 1)``, so the model remains normalized by default.
``weights`` : normalized by ``sum(bit_weights) + 1 LSB``. For a non-redundant
              4-bit ADC this is ``[8, 4, 2, 1] / (8+4+2+1+1)``.
``codes``   : array of {0, 1}, MSB at column 0
``aout``    : reconstructed voltage in ``quant_range``.

For a fully-differential SAR, pass the differential signal ``VIP - VIN`` with
the corresponding differential range, for example ``quant_range=(-VDD, VDD)``.

References
----------
Ported from Arcadia-1/SpecMind ``dataset_generation/simulation_engines/sar_adc.py``
and ``matlab_reference/sar_model_simplest.m`` (icdesign lab, 2025-03-25).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _parse_quant_range(quant_range: tuple[float, float]) -> tuple[float, float]:
    if len(quant_range) != 2:
        raise ValueError(f"quant_range must contain exactly 2 values, got {len(quant_range)}")
    v_min, v_max = float(quant_range[0]), float(quant_range[1])
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        raise ValueError("quant_range values must be finite")
    if v_max <= v_min:
        raise ValueError(f"quant_range must be increasing, got {quant_range}")
    return v_min, v_max


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


def sar_apply_cap_mismatch(
    weights: np.ndarray,
    sigma: float,
    rng: Optional[np.random.Generator] = None,
    cap_units: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Realize unit-cap/Pelgrom-style CDAC mismatch.

    ``sigma`` is the RMS relative mismatch of a single unit capacitor. A bit
    made from ``n`` unit capacitors gets relative RMS mismatch
    ``sigma / sqrt(n)``. This captures the usual ``sigma(C) / C ∝ 1/sqrt(C)``
    trend: MSB capacitors are larger and therefore have smaller relative
    mismatch than LSB capacitors.

    By default, unit counts are inferred from the normalized weights by
    dividing by the smallest positive weight. For ideal binary weights this
    gives ``[8, 4, 2, 1]`` for a 4-bit ADC and ``[8, 4, 4, 2, 1]`` for a
    redundant array. Pass ``cap_units`` explicitly for custom capacitor
    arrays.

    Parameters
    ----------
    weights : ndarray of shape (B,)
        Nominal analog CDAC weights, typically from
        :func:`sar_ideal_weights`.
    sigma : float
        RMS relative mismatch of one unit capacitor. For example, ``0.01`` is
        1% RMS for a 1-Cu element; an 8-Cu MSB then has ``0.01/sqrt(8)``
        relative RMS.
    rng : np.random.Generator, optional
        Numpy random generator. Use a deterministic seed to lock one chip.
    cap_units : ndarray of shape (B,), optional
        Unit capacitor counts or relative capacitor sizes for each weight.
        Values must be positive. If omitted, they are inferred from
        ``weights / min(weights)``.

    Returns
    -------
    perturbed_weights : ndarray of shape (B,)
        Cap weights with unit-cap-scaled gaussian mismatch applied. The
        result is NOT re-normalized.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights.shape}")
    if len(weights) == 0:
        raise ValueError("weights must not be empty")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights <= 0):
        raise ValueError("weights must be positive to infer capacitor sizes")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError(f"sigma must be finite and non-negative, got {sigma}")

    if cap_units is None:
        cap_units = weights / np.min(weights)
    else:
        cap_units = np.asarray(cap_units, dtype=float)
        if cap_units.shape != weights.shape:
            raise ValueError(
                f"cap_units shape must match weights shape {weights.shape}, got {cap_units.shape}"
            )
        if not np.all(np.isfinite(cap_units)):
            raise ValueError("cap_units must be finite")
        if np.any(cap_units <= 0):
            raise ValueError("cap_units must be positive")

    if rng is None:
        rng = np.random.default_rng()

    relative_sigma = sigma / np.sqrt(cap_units)
    return weights * (1.0 + relative_sigma * rng.standard_normal(len(weights)))


def sar_apply_mismatch(
    weights: np.ndarray,
    sigma: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Legacy per-weight gaussian mismatch helper.

    This preserves the pre-0.8.2 behavior where each supplied weight receives
    the same relative RMS perturbation. For Pelgrom/unit-cap-scaled mismatch,
    use :func:`sar_apply_cap_mismatch`.
    """
    if rng is None:
        rng = np.random.default_rng()
    return np.asarray(weights, dtype=float) * (1.0 + sigma * rng.standard_normal(len(weights)))


def sar_convert(
    vin: np.ndarray,
    weights: np.ndarray,
    quant_range: tuple[float, float] = (0.0, 1.0),
    sampling_noise_rms: float = 0.0,
    comparator_noise_rms: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Batch SAR conversion.

    Per-sample algorithm::

        vin_sampled = vin + sampling_noise
        v_dac = 0
        for j in range(B):
            v_test = v_dac + weights[j]
            bit[j] = 1 if (vin_sampled + comparator_noise > v_test) else 0
            if bit[j]: v_dac = v_test

    Vectorized over samples so the whole batch runs in one Python loop of
    length ``B`` (typically 4-20 iterations), not ``N`` (typically 10⁴-10⁶).

    Parameters
    ----------
    vin : array-like of shape (N,)
        Input voltage trace. Interpreted relative to ``quant_range``.
    weights : array-like of shape (B,)
        Actual analog CDAC weights (MSB first). Resolution and redundancy are
        inferred from this vector. Use :func:`sar_ideal_weights` for an ideal
        binary or redundant array, :func:`sar_apply_cap_mismatch` for
        unit-cap/Pelgrom-style capacitor mismatch.
    quant_range : tuple(float, float), default (0.0, 1.0)
        ADC input and output range ``(v_min, v_max)``. This mirrors
        ``ADC_Signal_Generator.apply_quantization_noise(..., quant_range=...)``.
    sampling_noise_rms : float, default 0.0
        Sampled input noise RMS, in the same unit as ``vin``. One random
        value is added per sample before the SAR bit trials.
    comparator_noise_rms : float, default 0.0
        Comparator input-referred thermal noise RMS, in the same unit as
        ``vin``. A fresh random value is used for each bit decision.
    rng : np.random.Generator, optional
        RNG for sampling and comparator noise.

    Returns
    -------
    codes : ndarray of shape (N, B), dtype int8
        Raw bit decisions, MSB at column 0. Pass to :func:`sar_reconstruct`
        for the analog estimate, or to a calibration routine (e.g.
        :func:`adctoolbox.calibration.calibrate_weight_sine`) for weight
        estimation.

    Notes
    -----
    The model does NOT include DAC settling errors, metastability delay,
    charge-injection artifacts, or PVT drift. For those, supplement with a
    Spectre / Verilog-A simulation.
    """
    vin = np.atleast_1d(np.asarray(vin, dtype=float))
    weights = np.asarray(weights, dtype=float)
    v_min, v_max = _parse_quant_range(quant_range)
    full_scale = v_max - v_min
    if rng is None:
        rng = np.random.default_rng()

    vin_sampled = vin.copy()
    if sampling_noise_rms > 0:
        vin_sampled = vin_sampled + sampling_noise_rms * rng.standard_normal(len(vin_sampled))

    vin_norm = (vin_sampled - v_min) / full_scale
    comparator_noise_norm = comparator_noise_rms / full_scale
    N, B = len(vin_norm), len(weights)
    codes = np.zeros((N, B), dtype=np.int8)
    v_dac = np.zeros(N)
    for j in range(B):
        v_test = v_dac + weights[j]
        noise = comparator_noise_norm * rng.standard_normal(N) if comparator_noise_rms > 0 else 0.0
        bit = (vin_norm + noise >= v_test).astype(np.int8)
        codes[:, j] = bit
        v_dac = np.where(bit, v_test, v_dac)
    return codes


def sar_reconstruct(
    codes: np.ndarray,
    weights: np.ndarray,
    quant_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
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
        Raw bit decisions from :func:`sar_convert`.
    weights : ndarray of shape (B,)
        Digital reconstruction weights. Use the nominal weights for an
        "uncalibrated" output; use cal-estimated or actual weights to assess
        calibration quality.
    quant_range : tuple(float, float), default (0.0, 1.0)
        Output voltage range ``(v_min, v_max)``. The default preserves the
        normalized reconstruction convention.

    Returns
    -------
    aout : ndarray of shape (N,)
        Reconstructed analog estimate, range
        ``[v_min, v_min + (v_max - v_min) * sum(weights)]``.
    """
    v_min, v_max = _parse_quant_range(quant_range)
    return (codes.astype(float) @ np.asarray(weights, dtype=float)) * (v_max - v_min) + v_min
