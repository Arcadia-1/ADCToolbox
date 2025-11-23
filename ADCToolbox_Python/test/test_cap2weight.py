"""Test cap2weight.py - CDAC capacitor to weight conversion."""

import numpy as np
import sys
import os

from ADC_Toolbox_Python.cap2weight import cap2weight


def run_tests():
    print("=" * 50)
    print("Test: cap2weight")
    print("=" * 50)

    results = []

    # Test 1: 4-bit binary CDAC (no bridge, no parasitics)
    cd = [1, 2, 4, 8]
    cb = [0, 0, 0, 0]
    cp = [0, 0, 0, 0]
    weight, co = cap2weight(cd, cb, cp)
    expected = np.array([1, 2, 4, 8]) / 15.0
    ok = np.allclose(weight, expected) and co == 15.0
    results.append(("4-bit binary", ok, f"sum={np.sum(weight):.4f}"))
    print(f"\n[4-bit binary CDAC]")
    print(f"  weights: {weight}")
    print(f"  co: {co}")

    # Test 2: 4-bit with dummy cap (SAR style)
    cd = [1, 1, 2, 4, 8]
    cb = [0, 0, 0, 0, 0]
    cp = [0, 0, 0, 0, 0]
    weight, co = cap2weight(cd, cb, cp)
    expected = np.array([1, 1, 2, 4, 8]) / 16.0
    ok = np.allclose(weight, expected) and co == 16.0
    results.append(("4-bit with dummy", ok, f"sum={np.sum(weight):.4f}"))
    print(f"\n[4-bit with dummy]")
    print(f"  weights: {weight}")
    print(f"  co: {co}")

    # Test 3: With bridge cap
    cd = [1, 2, 4, 8]
    cb = [0, 0, 4, 0]  # Bridge at 3rd bit
    cp = [0, 0, 0, 0]
    weight, co = cap2weight(cd, cb, cp)
    ok = co > 0 and np.sum(weight) > 0
    results.append(("With bridge cap", ok, f"co={co:.2f}"))
    print(f"\n[With bridge cap]")
    print(f"  weights: {weight}")
    print(f"  co: {co}")

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, info in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {info}")
    print(f"Total: {passed}/{len(results)} passed")

    return passed == len(results)


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
