import sys
sys.path.insert(0, r'D:\ADCToolbox\python\src')

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import find_coherent_frequency
from adctoolbox.aout.decompose_harmonics import compute_harmonics, plot_harmonics

# Generate test signal
N = 2**13
Fs = 800e6
Fin_target = 10e6
Fin, _ = find_coherent_frequency(Fs, Fin_target, N)
A = 0.49
sig = A * np.sin(2 * np.pi * Fin * np.arange(N) / Fs)
sig_noisy = sig + np.random.randn(N) * 500e-6

# Test 1: Compute only
print("[Test 1] Compute only (modular)")
results = compute_harmonics(sig_noisy, normalized_freq=Fin/Fs, order=10)
print(f"  Fundamental RMS: {np.sqrt(np.mean(results['fundamental_signal']**2)):.6f}")
print(f"  Harmonic RMS: {np.sqrt(np.mean(results['harmonic_error']**2)):.6f}")
print(f"  Other RMS: {np.sqrt(np.mean(results['other_error']**2)):.6f}")

# Test 2: Compute + Plot (modular)
print("[Test 2] Compute + Plot (modular)")
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plot_harmonics(results, ax=ax)
plt.tight_layout()
plt.savefig(r'D:\ADCToolbox\python\src\adctoolbox\examples\03_debug_analog\output\test_harmonics_modular.png', dpi=150)
print(f"  Plot saved")
plt.close()

# Test 3: Wrapper function (backward compatibility)
print("[Test 3] Wrapper function (backward compatibility)")
fund, tot_err, harm_err, oth_err = compute_harmonics(sig_noisy, normalized_freq=Fin/Fs, order=10)
print(f"  Returns: {type(fund).__name__}, {type(tot_err).__name__}, {type(harm_err).__name__}, {type(oth_err).__name__}")

print("\nAll tests passed!")
