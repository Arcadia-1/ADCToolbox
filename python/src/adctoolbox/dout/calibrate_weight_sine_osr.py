"""
Oversampled Foreground Calibration

Calibration for oversampled ADCs (e.g., Delta-Sigma).
Only considers in-band errors; ignores out-of-band noise.

Ported from MATLAB: FGCalSineOS.m
"""

import numpy as np
from typing import Optional, Tuple


def calibrate_weight_sine_osr(
    bits: np.ndarray,
    rel_freq: float,
    osr: int,
    order: int = 1,
    nom_weight: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Oversampled foreground calibration.

    Calibrates oversampled ADC using sine wave input.
    Only considers in-band portion of the spectrum.

    Args:
        bits: Raw bit data, shape (N, M) where N=samples, M=bits
        rel_freq: Relative input frequency (Fin/Fs)
        osr: Oversampling ratio
        order: Order of distortion excluded (default 1)
        nom_weight: Nominal weights for enhanced robustness (optional)

    Returns:
        Tuple of:
            - weight: Calibrated weights (normalized to MSB=1)
            - offset: Calibrated offset
            - post_cal: Calibrated output
            - ideal: Ideal sine signal
            - error: Residue errors after calibration

    Note:
        Data format: MSB left - LSB right
        Weight is normalized by MSB
    """
    bits = np.asarray(bits, dtype=float)

    # Ensure correct shape (N x M)
    if bits.shape[1] > bits.shape[0]:
        bits = bits.T

    n, m = bits.shape
    n2 = n // (2 * osr)

    order = max(order, 1)

    # FFT of bits
    bit_spec = np.fft.fft(bits, axis=0) / n
    bit_spec = bit_spec[:n2, :]  # Keep in-band portion

    # Apply dithering if nominal weights provided
    if nom_weight is not None:
        nom_weight = np.asarray(nom_weight, dtype=float)
        dgain = 1.0 / np.sqrt(nom_weight)
        dither = np.random.rand(n2, m - 1) * dgain[1:]
        bits_patch = bit_spec.copy()
        bits_patch[:, 0] -= dither @ nom_weight[1:] / nom_weight[0]
        bits_patch[:, 1:] += dither
    else:
        bits_patch = bit_spec.copy()

    # Build tone matrix (delta functions at harmonic bins)
    tone = np.zeros((n2, order), dtype=complex)
    for k in range(order):
        b = round((k + 1) * rel_freq * n)
        if b < n2:
            tone[b, k] = 1.0

    # Build system matrix
    # [bits_2:M, DC, tones] * x = -bits_1
    dc_col = np.zeros((n2, 1), dtype=complex)
    dc_col[0] = 1.0

    a = np.hstack([bits_patch[:, 1:m], dc_col, tone])
    b = -bits_patch[:, 0]

    # Solve least squares
    x, _, _, _ = np.linalg.lstsq(a, b, rcond=None)

    # Extract weights (real part, normalized to MSB=1)
    weight = np.zeros(m)
    weight[0] = 1.0
    weight[1:] = np.real(x[:m - 1])

    # Extract offset
    offset = -np.real(x[m - 1])

    # Calculate calibrated output (time domain)
    post_cal = weight @ bits.T - offset

    # Reconstruct ideal sine (time domain)
    t = np.arange(n)
    xct = np.cos(2 * np.pi * rel_freq * t)
    xst = np.sin(2 * np.pi * rel_freq * t)

    # Ideal signal from first harmonic coefficient
    ideal = (-xct * np.real(x[m]) + xst * np.imag(x[m])) * 2

    # Error
    error = post_cal - ideal

    return weight, offset, post_cal, ideal, error


if __name__ == "__main__":
    print("=" * 60)
    print("Testing FGCalSineOS.py")
    print("=" * 60)

    # Generate test data for oversampled ADC
    np.random.seed(42)

    n = 8192
    m = 8  # 8-bit ADC
    osr = 16  # Oversampling ratio

    rel_freq = 0.01  # Input at 1% of Nyquist

    # Generate signal
    t = np.arange(n)
    signal = 0.45 * np.sin(2 * np.pi * rel_freq * t) + 0.5

    # Add out-of-band noise (simulating noise shaping)
    noise = np.random.randn(n) * 0.1
    # High-pass filter the noise (simple differentiation)
    noise = np.diff(np.concatenate([[0], noise]))
    signal_noisy = signal + noise * 0.01

    # Quantize to bits
    def signal_to_bits(sig, num_bits):
        levels = 2 ** num_bits
        codes = np.clip(np.floor(sig * levels), 0, levels - 1).astype(int)
        bits = np.zeros((len(sig), num_bits))
        for i in range(num_bits):
            bits[:, i] = (codes >> (num_bits - 1 - i)) & 1
        return bits

    bits = signal_to_bits(signal_noisy, m)

    print(f"\n[Config]")
    print(f"  N samples: {n}")
    print(f"  Bits: {m}")
    print(f"  OSR: {osr}")
    print(f"  Rel freq: {rel_freq}")

    weight, offset, post_cal, ideal, error = fg_cal_sine_os(
        bits, rel_freq, osr, order=1
    )

    print(f"\n[Results]")
    print(f"  Weights: {weight}")
    print(f"  Offset: {offset:.6f}")
    print(f"  Post-cal range: [{post_cal.min():.4f}, {post_cal.max():.4f}]")
    print(f"  Error RMS: {np.std(error):.6f}")

    print("\n" + "=" * 60)
