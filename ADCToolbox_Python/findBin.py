"""
Find Coherent Sampling Bin

Find the nearest prime FFT bin for coherent sampling.
Prime bins avoid spectral leakage in coherent sampling.

Ported from MATLAB: findBin.m
"""

import numpy as np


def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def find_bin(fs: float, fin: float, n: int) -> int:
    """
    Find the nearest prime FFT bin for coherent sampling.

    For coherent sampling, the number of signal cycles (M) should be
    prime and coprime with N to avoid spectral leakage.

    Args:
        fs: Sampling frequency (Hz)
        fin: Desired input frequency (Hz)
        n: Number of FFT points (samples)

    Returns:
        bin: Prime bin number (number of cycles in N samples)

    Example:
        # For 1 MHz sampling, ~100 kHz input, 4096 samples
        bin = find_bin(1e6, 100e3, 4096)
        # Returns nearest prime to floor(100e3/1e6 * 4096) = 409
        # Actual fin = bin * fs / n
    """
    # Calculate initial bin
    bin_val = int(np.floor(fin / fs * n))

    # Find next prime
    while not is_prime(bin_val):
        bin_val += 1

    return bin_val


def find_fin_coherent(fs: float, fin_target: float, n: int) -> tuple:
    """
    Find coherent sampling frequency near target.

    Args:
        fs: Sampling frequency (Hz)
        fin_target: Target input frequency (Hz)
        n: Number of FFT points

    Returns:
        tuple: (actual_fin, bin_number)
    """
    bin_val = find_bin(fs, fin_target, n)
    actual_fin = bin_val * fs / n
    return actual_fin, bin_val


if __name__ == "__main__":
    # Test
    print("=" * 60)
    print("Testing findBin.py")
    print("=" * 60)

    # Test case 1: Standard coherent sampling
    fs = 1e6
    fin_target = 100e3
    n = 4096

    bin_val = find_bin(fs, fin_target, n)
    actual_fin = bin_val * fs / n

    print(f"[Test 1] fs={fs/1e6:.1f}MHz, fin_target={fin_target/1e3:.1f}kHz, N={n}")
    print(f"  [Result] bin={bin_val}, actual_fin={actual_fin/1e3:.3f}kHz")
    print(f"  [Prime check] {bin_val} is prime: {is_prime(bin_val)}")

    # Test case 2
    fs = 10e6
    fin_target = 1.23e6
    n = 8192

    bin_val = find_bin(fs, fin_target, n)
    actual_fin = bin_val * fs / n

    print(f"\n[Test 2] fs={fs/1e6:.1f}MHz, fin_target={fin_target/1e6:.2f}MHz, N={n}")
    print(f"  [Result] bin={bin_val}, actual_fin={actual_fin/1e6:.6f}MHz")
    print(f"  [Prime check] {bin_val} is prime: {is_prime(bin_val)}")

    print("\n" + "=" * 60)
