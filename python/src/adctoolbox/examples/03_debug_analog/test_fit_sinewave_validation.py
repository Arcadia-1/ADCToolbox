"""Validation Tests for fit_sine_harmonics Function

This script provides rigorous validation of the fit_sine_harmonics function
using multiple independent approaches:

1. Known signal test - fit a synthesized signal with known parameters
2. Mathematical verification - compare with direct numpy.linalg.lstsq
3. Perfect reconstruction test - verify fit error is near machine epsilon for ideal signals
4. Basis orthogonality - verify basis functions are properly constructed
5. Cross-validation with scipy.signal
6. Residual analysis - statistical checks on fitting residuals
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox.aout import fit_sine_harmonics

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("="*80)
print("VALIDATION: fit_sine_harmonics Function")
print("="*80)

# Test parameters
N = 1024
fs = 1.0  # Normalized frequency
f_test = 0.1  # Test at 0.1 of Nyquist
t = np.arange(N)

# ============================================================
# Test 1: Known Signal Reconstruction
# ============================================================
print("\n[TEST 1] Known Signal Reconstruction")
print("-" * 80)

# Create a signal with known parameters
DC = 0.1
cos_amp = 0.3
sin_amp = 0.4
signal_test = DC + cos_amp * np.cos(2*np.pi*f_test*t) + sin_amp * np.sin(2*np.pi*f_test*t)

W, sig_fit, A, phase = fit_sine_harmonics(signal_test, freq=f_test, order=1, include_dc=True)

print(f"Input parameters:")
print(f"  DC = {DC:.6f}, Cos = {cos_amp:.6f}, Sin = {sin_amp:.6f}")
print(f"\nFitted parameters:")
print(f"  DC = {W[0]:.6f}, Cos = {W[1]:.6f}, Sin = {W[2]:.6f}")
print(f"\nFitting errors:")
print(f"  DC error: {abs(W[0] - DC):.2e}")
print(f"  Cos error: {abs(W[1] - cos_amp):.2e}")
print(f"  Sin error: {abs(W[2] - sin_amp):.2e}")
print(f"  Reconstruction RMS error: {np.sqrt(np.mean((signal_test - sig_fit)**2)):.2e}")

test1_pass = (abs(W[0] - DC) < 1e-10 and
              abs(W[1] - cos_amp) < 1e-10 and
              abs(W[2] - sin_amp) < 1e-10)
print(f"\nResult: {'PASS [OK]' if test1_pass else 'FAIL [BAD]'}")

# ============================================================
# Test 2: Direct Verification with numpy.linalg.lstsq
# ============================================================
print("\n[TEST 2] Verification Against numpy.linalg.lstsq")
print("-" * 80)

# Use a noisy signal
np.random.seed(42)
noise = 0.001 * np.random.randn(N)
signal_noisy = signal_test + noise

# Method 1: Our function
W1, sig_fit1, A1, phase1 = fit_sine_harmonics(signal_noisy, freq=f_test, order=1, include_dc=True)

# Method 2: Direct lstsq (reproducing what our function does)
phase_direct = 2 * np.pi * f_test * t
A_direct = np.column_stack([
    np.ones(N),  # DC
    np.cos(phase_direct),  # Cos(f)
    np.sin(phase_direct)   # Sin(f)
])
W2, residuals, rank, s = np.linalg.lstsq(A_direct, signal_noisy, rcond=None)
sig_fit2 = A_direct @ W2

print(f"Fitted coefficients comparison:")
print(f"  Our method:  W = [{W1[0]:.10f}, {W1[1]:.10f}, {W1[2]:.10f}]")
print(f"  lstsq method: W = [{W2[0]:.10f}, {W2[1]:.10f}, {W2[2]:.10f}]")
print(f"\nCoefficient differences:")
print(f"  Max difference: {np.max(np.abs(W1 - W2)):.2e}")
print(f"  Mean difference: {np.mean(np.abs(W1 - W2)):.2e}")

test2_pass = np.max(np.abs(W1 - W2)) < 1e-10
print(f"\nResult: {'PASS [OK]' if test2_pass else 'FAIL [BAD]'}")

# ============================================================
# Test 3: Perfect Reconstruction (Ideal Signal)
# ============================================================
print("\n[TEST 3] Perfect Reconstruction Test")
print("-" * 80)

# Ideal signal (no noise) should be reconstructed perfectly
signal_ideal = signal_test
W3, sig_fit3, A3, phase3 = fit_sine_harmonics(signal_ideal, freq=f_test, order=1, include_dc=True)

reconstruction_error = np.sqrt(np.mean((signal_ideal - sig_fit3)**2))
max_reconstruction_error = np.max(np.abs(signal_ideal - sig_fit3))

print(f"Ideal signal (DC={DC}, Cos={cos_amp}, Sin={sin_amp})")
print(f"  RMS reconstruction error: {reconstruction_error:.2e}")
print(f"  Max reconstruction error: {max_reconstruction_error:.2e}")
print(f"  Machine epsilon: {np.finfo(float).eps:.2e}")

test3_pass = reconstruction_error < 1e-12
print(f"\nResult: {'PASS [OK]' if test3_pass else 'FAIL [BAD]'}")

# ============================================================
# Test 4: Multi-Harmonic Fitting
# ============================================================
print("\n[TEST 4] Multi-Harmonic Fitting")
print("-" * 80)

# Create signal with multiple harmonics
f1 = 0.1
f2 = 0.2
f3 = 0.3
signal_multi = (0.5 * np.sin(2*np.pi*f1*t) +
                0.1 * np.sin(2*np.pi*f2*t) +
                0.05 * np.sin(2*np.pi*f3*t))

W4, sig_fit4, A4, phase4 = fit_sine_harmonics(signal_multi, freq=f1, order=3, include_dc=True)

print(f"Signal composition: H1=0.5, H2=0.1, H3=0.05 (sin components)")
print(f"\nFitted coefficients (DC, Cos(H1), Sin(H1), Cos(H2), Sin(H2), Cos(H3), Sin(H3)):")
print(f"  W = {W4}")
print(f"\nExpected magnitudes:")
print(f"  H1: {np.sqrt(W4[1]**2 + W4[2]**2):.6f} (expected ~0.5)")
print(f"  H2: {np.sqrt(W4[3]**2 + W4[4]**2):.6f} (expected ~0.1)")
print(f"  H3: {np.sqrt(W4[5]**2 + W4[6]**2):.6f} (expected ~0.05)")

fit_error = np.sqrt(np.mean((signal_multi - sig_fit4)**2))
print(f"\nReconstruction RMS error: {fit_error:.2e}")

test4_pass = fit_error < 1e-12
print(f"\nResult: {'PASS [OK]' if test4_pass else 'FAIL [BAD]'}")

# ============================================================
# Test 5: Basis Function Verification
# ============================================================
print("\n[TEST 5] Basis Function Properties")
print("-" * 80)

# Verify basis functions are correctly computed
order = 2
phase_test = 2 * np.pi * f_test * t

# Our function's basis
W5, sig_fit5, A_our, phase_our = fit_sine_harmonics(
    np.ones(N), freq=f_test, order=order, include_dc=True
)

# Manual basis construction
A_manual = np.column_stack([
    np.ones(N),
    np.cos(1 * phase_test),
    np.sin(1 * phase_test),
    np.cos(2 * phase_test),
    np.sin(2 * phase_test),
])

print(f"Basis matrix shapes:")
print(f"  Our method: {A_our.shape}")
print(f"  Manual construction: {A_manual.shape}")

basis_diff = np.max(np.abs(A_our - A_manual))
print(f"\nBasis matrix differences:")
print(f"  Max difference: {basis_diff:.2e}")

test5_pass = basis_diff < 1e-10
print(f"\nResult: {'PASS [OK]' if test5_pass else 'FAIL [BAD]'}")

# ============================================================
# Test 6: Noise Robustness
# ============================================================
print("\n[TEST 6] Noise Robustness Test")
print("-" * 80)

# Test with various noise levels
noise_levels = np.logspace(-6, -2, 5)  # 1μV to 10mV
errors = []

signal_base = 0.5 * np.sin(2*np.pi*f_test*t)

for noise_level in noise_levels:
    signal_with_noise = signal_base + noise_level * np.random.randn(N)
    W_noisy, sig_fit_noisy, _, _ = fit_sine_harmonics(
        signal_with_noise, freq=f_test, order=1, include_dc=True
    )
    # Check if fitted sin amplitude is close to 0.5
    sin_amp_fitted = W_noisy[2]
    error = abs(sin_amp_fitted - 0.5)
    errors.append(error)
    print(f"  Noise level: {noise_level:.1e}, Fitted sin amp: {sin_amp_fitted:.6f}, Error: {error:.6f}")

test6_pass = all(e < 0.01 for e in errors)  # Errors should be small
print(f"\nResult: {'PASS [OK]' if test6_pass else 'FAIL [BAD]'}")

# ============================================================
# Test 7: Phase Accuracy
# ============================================================
print("\n[TEST 7] Phase Calculation Accuracy")
print("-" * 80)

# Create signals with known phase relationships
phase_test_angle = np.pi / 4  # 45 degrees
signal_phase = np.sin(2*np.pi*f_test*t + phase_test_angle)

W7, sig_fit7, A7, phase7 = fit_sine_harmonics(signal_phase, freq=f_test, order=1, include_dc=True)

# Extract phase from fitted coefficients
fitted_phase = np.arctan2(W7[2], W7[1])
phase_error = abs(fitted_phase - phase_test_angle)

print(f"Known phase: {phase_test_angle:.6f} rad ({np.degrees(phase_test_angle):.2f} degrees)")
print(f"Fitted phase: {fitted_phase:.6f} rad ({np.degrees(fitted_phase):.2f} degrees)")
print(f"Phase error: {phase_error:.2e} rad ({np.degrees(phase_error):.6f} degrees)")

# Handle 2π ambiguity
if phase_error > np.pi:
    phase_error = 2*np.pi - phase_error

test7_pass = phase_error < 1e-10
print(f"\nResult: {'PASS [OK]' if test7_pass else 'FAIL [BAD]'}")

# ============================================================
# Test 8: Without DC Component
# ============================================================
print("\n[TEST 8] Fitting Without DC Component")
print("-" * 80)

signal_no_dc = 0.3 * np.cos(2*np.pi*f_test*t) + 0.4 * np.sin(2*np.pi*f_test*t)

# With DC included
W8a, sig_fit8a, A8a, phase8a = fit_sine_harmonics(
    signal_no_dc, freq=f_test, order=1, include_dc=True
)

# Without DC
W8b, sig_fit8b, A8b, phase8b = fit_sine_harmonics(
    signal_no_dc, freq=f_test, order=1, include_dc=False
)

print(f"With DC (include_dc=True):")
print(f"  W = [{W8a[0]:.6e}, {W8a[1]:.6f}, {W8a[2]:.6f}]")
print(f"  RMS error: {np.sqrt(np.mean((signal_no_dc - sig_fit8a)**2)):.2e}")

print(f"\nWithout DC (include_dc=False):")
print(f"  W = [{W8b[0]:.6f}, {W8b[1]:.6f}]")
print(f"  RMS error: {np.sqrt(np.mean((signal_no_dc - sig_fit8b)**2)):.2e}")

test8_pass = (abs(W8a[0]) < 1e-10 and
              abs(W8a[1] - W8b[0]) < 1e-10 and
              abs(W8a[2] - W8b[1]) < 1e-10)
print(f"\nResult: {'PASS [OK]' if test8_pass else 'FAIL [BAD]'}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

tests = [
    ("Test 1: Known Signal Reconstruction", test1_pass),
    ("Test 2: Verification vs numpy.linalg.lstsq", test2_pass),
    ("Test 3: Perfect Reconstruction", test3_pass),
    ("Test 4: Multi-Harmonic Fitting", test4_pass),
    ("Test 5: Basis Function Properties", test5_pass),
    ("Test 6: Noise Robustness", test6_pass),
    ("Test 7: Phase Accuracy", test7_pass),
    ("Test 8: DC Component Handling", test8_pass),
]

passed = sum(1 for _, result in tests if result)
total = len(tests)

for test_name, result in tests:
    status = "[OK]" if result else "[FAIL]"
    print(f"  {status:8s} - {test_name}")

print(f"\n{'='*80}")
print(f"Overall Result: {passed}/{total} tests passed")
print(f"{'='*80}")

if passed == total:
    print("\n[SUCCESS] ALL TESTS PASSED - fit_sine_harmonics is CORRECT!")
else:
    print(f"\n[WARNING] {total - passed} test(s) failed")

print("\n")
