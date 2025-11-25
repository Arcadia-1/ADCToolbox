"""
Find Input Vpp Calculator

Calculate required Vpp to achieve target signal level.

Ported from MATLAB: findVinpp.m
"""

import numpy as np


def find_vinpp(
    vpp_old: float,
    signal_db: float,
    signal_target_db: float = -0.5
) -> float:
    """
    Calculate new Vpp to achieve target signal level.

    Given current Vpp and measured signal level, calculate the Vpp
    needed to achieve a target signal level (in dBFS).

    Args:
        vpp_old: Current input Vpp (V)
        signal_db: Measured signal level (dBFS)
        signal_target_db: Target signal level (dBFS), default -0.5 dBFS

    Returns:
        vpp_new: Required Vpp to achieve target level (V)

    Example:
        # Current Vpp=1.0V gives -3dB signal, want -0.5dB
        vpp_new = find_vinpp(1.0, -3.0, -0.5)
        # Returns ~1.41V
    """
    # Linear scaling based on dB difference
    # signal_target = signal * (vpp_new / vpp_old)
    # 20*log10(signal_target) = 20*log10(signal) + 20*log10(vpp_new/vpp_old)
    # signal_target_db = signal_db + 20*log10(vpp_new/vpp_old)
    # vpp_new/vpp_old = 10^((signal_target_db - signal_db)/20)

    vpp_new = vpp_old / 10**(signal_db / 20) * 10**(signal_target_db / 20)
    return vpp_new


def db_to_linear(db: float) -> float:
    """Convert dB to linear scale (voltage ratio)."""
    return 10**(db / 20)


def linear_to_db(linear: float) -> float:
    """Convert linear scale to dB."""
    return 20 * np.log10(linear)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing findVinpp.py")
    print("=" * 60)

    # Test 1: Increase signal level
    vpp_old = 1.0
    signal_db = -3.0
    target_db = -0.5

    vpp_new = find_vinpp(vpp_old, signal_db, target_db)
    print(f"\n[Test 1] Increase signal level")
    print(f"  Current: Vpp={vpp_old}V, signal={signal_db}dBFS")
    print(f"  Target: {target_db}dBFS")
    print(f"  New Vpp: {vpp_new:.4f}V")

    # Test 2: Decrease signal level
    vpp_old = 2.0
    signal_db = -0.5
    target_db = -6.0

    vpp_new = find_vinpp(vpp_old, signal_db, target_db)
    print(f"\n[Test 2] Decrease signal level")
    print(f"  Current: Vpp={vpp_old}V, signal={signal_db}dBFS")
    print(f"  Target: {target_db}dBFS")
    print(f"  New Vpp: {vpp_new:.4f}V")

    # Test 3: Already at target
    vpp_old = 1.5
    signal_db = -0.5
    target_db = -0.5

    vpp_new = find_vinpp(vpp_old, signal_db, target_db)
    print(f"\n[Test 3] Already at target")
    print(f"  Current: Vpp={vpp_old}V, signal={signal_db}dBFS")
    print(f"  Target: {target_db}dBFS")
    print(f"  New Vpp: {vpp_new:.4f}V (unchanged)")

    print("\n" + "=" * 60)
