"""
Find Coherent Sampling Bin

Find the nearest prime FFT bin for coherent sampling.
Prime bins avoid spectral leakage in coherent sampling.

Ported from MATLAB: findBin.m
"""

import numpy as np


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


def find_bin(fs: float, fin: float, n: int) -> int:
    """
    Find the nearest coherent FFT bin.

    For coherent sampling, the number of signal cycles (M) should be
    coprime with N (gcd(M,N) = 1) to avoid spectral leakage.

    MATLAB reference: findBin.m lines 2-5

    Args:
        fs: Sampling frequency (Hz)
        fin: Desired input frequency (Hz)
        n: Number of FFT points (samples)

    Returns:
        bin: Bin number coprime with N (number of cycles in N samples)

    Example:
        # For 10 GHz sampling, ~400 MHz input, 16384 samples
        bin = find_bin(10e9, 400e6, 16384)
        # Returns first bin >= floor(400e6/10e9 * 16384) where gcd(bin,16384)==1
        # Actual fin = bin * fs / n
    """
    # Calculate initial bin (MATLAB line 2)
    bin_val = int(np.floor(fin / fs * n))

    # Find next bin coprime with N (MATLAB lines 3-5)
    while gcd(bin_val, n) > 1:
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
    print(f"  [Coprime check] gcd({bin_val}, {n}) = {gcd(bin_val, n)}")

    # Test case 2: Matches MATLAB test_jitter.m
    fs = 10e9
    fin_target = 400e6
    n = 16384

    bin_val = find_bin(fs, fin_target, n)
    actual_fin = bin_val * fs / n

    print(f"\n[Test 2] fs={fs/1e9:.1f}GHz, fin_target={fin_target/1e6:.0f}MHz, N={n}")
    print(f"  [Result] bin={bin_val}, actual_fin={actual_fin/1e6:.6f}MHz")
    print(f"  [Coprime check] gcd({bin_val}, {n}) = {gcd(bin_val, n)}")

    print("\n" + "=" * 60)
