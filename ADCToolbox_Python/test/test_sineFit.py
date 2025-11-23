"""Test sineFit.py - sine wave fitting validation."""

import numpy as np
import sys
import os

from ADC_Toolbox_Python.sineFit import sine_fit, find_relative_freq


def run_tests():
    print("Testing sineFit.py")
    print("=" * 50)

    N, fs = 4096, 1e6
    results = []

    # Test 1: Basic sine
    fin = 101 * fs / N
    t = np.arange(N) / fs
    signal = 0.9 * np.sin(2 * np.pi * fin * t + np.pi/4) + 0.5
    _, freq, mag, dc, _ = sine_fit(signal)

    err_f = abs(freq - fin/fs)
    err_m = abs(mag - 0.9)
    err_d = abs(dc - 0.5)
    ok = err_f < 1e-8 and err_m < 1e-4 and err_d < 1e-4
    results.append(("Basic sine", ok, f"freq_err={err_f:.1e}, mag_err={err_m:.1e}"))

    # Test 2: With noise
    np.random.seed(42)
    signal = np.sin(2 * np.pi * fin * t) * 1000 + 2048 + np.random.randn(N) * 10
    _, freq, mag, _, _ = sine_fit(signal)

    err_f = abs(freq - fin/fs)
    err_m = abs(mag - 1000) / 1000
    ok = err_f < 1e-6 and err_m < 0.01
    results.append(("With noise", ok, f"freq_err={err_f:.1e}, mag_err={err_m*100:.2f}%"))

    # Test 3: Different frequency bins
    all_ok = True
    for bin_num in [11, 101, 503, 1021]:
        fin = bin_num * fs / N
        signal = np.sin(2 * np.pi * fin * t)
        _, freq, _, _, _ = sine_fit(signal)
        if abs(freq - fin/fs) >= 1e-8:
            all_ok = False
    results.append(("Multi-freq bins", all_ok, f"bins=[11,101,503,1021]"))

    # Test 4: 2D batch data
    data = np.zeros((N, 8))
    fin = 101 * fs / N
    for i in range(8):
        data[:, i] = np.sin(2 * np.pi * fin * t + np.random.rand() * 2 * np.pi)
    _, freq, _, _, _ = sine_fit(data)
    ok = abs(freq - fin/fs) < 1e-8
    results.append(("Batch 2D input", ok, f"shape={data.shape}"))

    # Test 5: find_relative_freq
    signal = 0.5 * np.sin(2 * np.pi * 101 * fs / N * t) + 0.5
    rel_freq = find_relative_freq(signal)
    ok = abs(rel_freq - 101/N) < 1e-8
    results.append(("find_relative_freq", ok, f"err={abs(rel_freq - 101/N):.1e}"))

    # Print results
    passed = 0
    for name, ok, info in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] [{name}]: {info}")
        if ok:
            passed += 1

    print("-" * 50)
    print(f"  Total: {passed}/{len(results)} passed")
    return passed == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
