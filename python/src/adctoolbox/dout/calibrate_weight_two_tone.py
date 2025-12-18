"""
Two-Frequency Foreground Calibration

Calibration using two different input frequencies to exclude
frequency-dependent errors.

Ported from MATLAB: FGCalSine_2freq.m
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from adctoolbox.fundamentals.frequency import fold_frequency_to_nyquist


def calibrate_weight_two_tone(
    bits1: np.ndarray,
    bits2: np.ndarray,
    rel_freq1: float,
    rel_freq2: float,
    order: int = 1,
    nom_weight: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Two-frequency foreground calibration.

    Calibrates ADC using two different input frequencies to
    exclude frequency-dependent errors (e.g., settling errors).

    Args:
        bits1: Raw bit data at frequency 1, shape (N1, M)
        bits2: Raw bit data at frequency 2, shape (N2, M)
        rel_freq1: Relative frequency 1 (Fin1/Fs)
        rel_freq2: Relative frequency 2 (Fin2/Fs)
        order: Order of distortion excluded (default 1)
        nom_weight: Nominal weights for dithering (optional)

    Returns:
        Tuple of:
            - weight: Calibrated weights (normalized to MSB=1)
            - offset: Calibrated offset
            - post_cal1: Calibrated output for frequency 1
            - post_cal2: Calibrated output for frequency 2

    Note:
        Data format: MSB left - LSB right
        Weight is normalized by MSB, i.e., weight[MSB] = 1
    """
    bits1 = np.asarray(bits1, dtype=float)
    bits2 = np.asarray(bits2, dtype=float)

    # Ensure correct shape
    if bits1.shape[1] > bits1.shape[0]:
        bits1 = bits1.T
    if bits2.shape[1] > bits2.shape[0]:
        bits2 = bits2.T

    n1, m1 = bits1.shape
    n2, m2 = bits2.shape

    if m1 != m2:
        raise ValueError(f"Data bit width mismatch: {m1} vs {m2}")

    m = m1
    order = max(order, 1)

    # Apply dithering if nominal weights provided
    if nom_weight is not None:
        nom_weight = np.asarray(nom_weight, dtype=float)
        dgain = 1.0 / np.sqrt(nom_weight)

        dither1 = np.random.rand(n1, m - 1) * dgain[1:]
        bits_patch1 = bits1.copy()
        bits_patch1[:, 0] -= dither1 @ nom_weight[1:] / nom_weight[0]
        bits_patch1[:, 1:] += dither1

        dither2 = np.random.rand(n2, m - 1) * dgain[1:]
        bits_patch2 = bits2.copy()
        bits_patch2[:, 0] -= dither2 @ nom_weight[1:] / nom_weight[0]
        bits_patch2[:, 1:] += dither2
    else:
        bits_patch1 = bits1.copy()
        bits_patch2 = bits2.copy()

    # Build basis functions for both frequencies
    # Cosine and sine components for harmonics
    xc1 = np.zeros((n1, order))
    xs1 = np.zeros((n1, order))
    xc2 = np.zeros((n2, order))
    xs2 = np.zeros((n2, order))

    for k in range(order):
        freq_k = rel_freq1 * (k + 1)
        aliased_freq = fold_frequency_to_nyquist(freq_k, 1)
        xc1[:, k] = np.cos(2 * np.pi * aliased_freq * np.arange(n1))
        xs1[:, k] = np.sin(2 * np.pi * aliased_freq * np.arange(n1))

        freq_k = rel_freq2 * (k + 1)
        aliased_freq = fold_frequency_to_nyquist(freq_k, 1)
        xc2[:, k] = np.cos(2 * np.pi * aliased_freq * np.arange(n2))
        xs2[:, k] = np.sin(2 * np.pi * aliased_freq * np.arange(n2))

    # Build combined matrix
    # A * x = b
    # x = [w2, w3, ..., wM, offset, cos_coeffs, sin_coeffs]
    a1 = np.hstack([
        bits_patch1[:, 1:m],  # Bits 2 to M
        np.ones((n1, 1)),      # Offset
        xc1,                    # Cosine terms
        xs1                     # Sine terms
    ])

    a2 = np.hstack([
        bits_patch2[:, 1:m],
        np.ones((n2, 1)),
        xc2,
        xs2
    ])

    a = np.vstack([a1, a2])
    b = np.concatenate([-bits_patch1[:, 0], -bits_patch2[:, 0]])

    # Solve least squares
    x, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

    # Extract weights (normalized to MSB=1)
    weight = np.zeros(m)
    weight[0] = 1.0
    weight[1:] = x[:m - 1]

    # Extract offset
    offset = -x[m - 1]

    # Calculate calibrated outputs
    post_cal1 = weight @ bits1.T - offset
    post_cal2 = weight @ bits2.T - offset

    return weight, offset, post_cal1, post_cal2


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FGCalSine_2freq.py")
    print("=" * 60)

    # Generate test data
    np.random.seed(42)

    n = 4096
    m = 10  # 10-bit ADC

    # True weights (with some mismatch)
    true_weight = np.array([1.0, 0.51, 0.24, 0.13, 0.062, 0.031, 0.016, 0.008, 0.004, 0.002])
    true_offset = 0.01

    # Two frequencies
    rel_freq1 = 0.0123
    rel_freq2 = 0.0567

    # Generate signals
    t1 = np.arange(n)
    t2 = np.arange(n)

    signal1 = 0.45 * np.sin(2 * np.pi * rel_freq1 * t1) + 0.5
    signal2 = 0.45 * np.sin(2 * np.pi * rel_freq2 * t2) + 0.5

    # Ideal quantization to bits
    def signal_to_bits(signal, num_bits):
        levels = 2 ** num_bits
        codes = np.clip(np.floor(signal * levels), 0, levels - 1).astype(int)
        bits = np.zeros((len(signal), num_bits))
        for i in range(num_bits):
            bits[:, i] = (codes >> (num_bits - 1 - i)) & 1
        return bits

    bits1 = signal_to_bits(signal1, m)
    bits2 = signal_to_bits(signal2, m)

    # Add noise
    bits1 = bits1 + np.random.randn(*bits1.shape) * 0.01
    bits2 = bits2 + np.random.randn(*bits2.shape) * 0.01

    print(f"\n[Config]")
    print(f"  N samples: {n}")
    print(f"  Bits: {m}")
    print(f"  Freq 1: {rel_freq1}")
    print(f"  Freq 2: {rel_freq2}")

    weight, offset, post_cal1, post_cal2 = fg_cal_sine_2freq(
        bits1, bits2, rel_freq1, rel_freq2, order=1
    )

    print(f"\n[Results]")
    print(f"  Weights: {weight}")
    print(f"  Offset: {offset:.6f}")
    print(f"  Post-cal 1 range: [{post_cal1.min():.4f}, {post_cal1.max():.4f}]")
    print(f"  Post-cal 2 range: [{post_cal2.min():.4f}, {post_cal2.max():.4f}]")

    print("\n" + "=" * 60)
