"""Test phase error analysis - binned vs raw approaches

This example demonstrates both phase error computation methods:
1. compute_phase_error_from_binned - trend analysis with binning
2. compute_phase_error_from_raw - high-precision raw data analysis

Tests:
1. Pure Gaussian noise - both methods should give similar results
2. Phase jitter - shows PM modulation
3. Amplitude noise - shows AM modulation
4. Comparison of binned vs raw accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import (
    compute_phase_error_from_binned,
    compute_phase_error_from_raw
)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Setup parameters
N = 2**13
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49

print(f"[Phase Error Analysis] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, A={A:.3f} V, N={N}")

# ============================================================
# Test 1: Pure Gaussian Noise
# ============================================================
print("\n" + "="*70)
print("Test 1: Pure Gaussian Noise")
print("="*70)

noise_rms = 0.001
sig_noise = A * np.sin(2 * np.pi * Fin * t) + np.random.randn(N) * noise_rms

snr_noise = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_noise = snr_to_nsd(snr_noise, fs=Fs, osr=1)
print(f"[Signal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_noise:.2f} dB], Theoretical NSD=[{nsd_noise:.2f} dBFS/Hz]\n")

result_binned = compute_phase_error_from_binned(sig_noise, normalized_freq, bin_count=100)
result_raw = compute_phase_error_from_raw(sig_noise, normalized_freq)

print(f"Expected noise RMS: {noise_rms:.6e}")
print(f"\nBinned method:")
print(f"  AM param: {result_binned['am_param']:.6e}")
print(f"  PM param: {result_binned['pm_param']:.6e} rad")
print(f"  Baseline: {result_binned['baseline']:.6e}")
print(f"\nRaw method:")
print(f"  AM param: {result_raw['am_param']:.6e}")
print(f"  PM param: {result_raw['pm_param']:.6e} rad")
print(f"  Baseline: {result_raw['baseline']:.6e}")
print(f"  Error RMS: {result_raw['error_rms']:.6e}")

# ============================================================
# Test 2: Phase Jitter (PM Dominant)
# ============================================================
print("\n" + "="*70)
print("Test 2: Phase Jitter (PM Dominant)")
print("="*70)

# Add timing jitter to the signal
phase_jitter = 0.05 * np.random.randn(N)  # 0.05 rad RMS
jitter_phase = 2 * np.pi * Fin * t + phase_jitter
sig_jitter = A * np.sin(jitter_phase) + np.random.randn(N) * 10e-6

result_binned_jit = compute_phase_error_from_binned(sig_jitter, normalized_freq, bin_count=100)
result_raw_jit = compute_phase_error_from_raw(sig_jitter, normalized_freq)

print(f"Added phase jitter RMS: 0.05 rad")
print(f"\nBinned method:")
print(f"  AM param: {result_binned_jit['am_param']:.6e}")
print(f"  PM param: {result_binned_jit['pm_param']:.6e} rad")
print(f"\nRaw method:")
print(f"  AM param: {result_raw_jit['am_param']:.6e}")
print(f"  PM param: {result_raw_jit['pm_param']:.6e} rad")

# ============================================================
# Test 3: Amplitude Noise (AM Dominant)
# ============================================================
print("\n" + "="*70)
print("Test 3: Amplitude Noise (AM Dominant)")
print("="*70)

# Add gain variation
gain_noise = 1.0 + 0.001 * np.sin(2 * np.pi * 100e3 * t)  # ~0.1% gain variation
sig_gain = gain_noise * A * np.sin(2 * np.pi * Fin * t) + np.random.randn(N) * 10e-6

result_binned_am = compute_phase_error_from_binned(sig_gain, normalized_freq, bin_count=100)
result_raw_am = compute_phase_error_from_raw(sig_gain, normalized_freq)

print(f"Added amplitude modulation: ~0.1% gain variation")
print(f"\nBinned method:")
print(f"  AM param: {result_binned_am['am_param']:.6e}")
print(f"  PM param: {result_binned_am['pm_param']:.6e} rad")
print(f"\nRaw method:")
print(f"  AM param: {result_raw_am['am_param']:.6e}")
print(f"  PM param: {result_raw_am['pm_param']:.6e} rad")

# ============================================================
# Visualization
# ============================================================
print("\n[Generating plots...]")

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Phase Error Analysis - Binned vs Raw', fontsize=16, fontweight='bold')

sample_range = slice(0, 2000)

# Row 1: Pure Gaussian Noise
ax = axes[0, 0]
ax.plot(t[sample_range], sig_noise[sample_range], 'b-', label='Input signal', linewidth=0.8)
ax.plot(t[sample_range], result_binned['fitted_signal'][sample_range], 'r--', label='Fitted', linewidth=1.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Test 1: Pure Gaussian Noise - Time Domain')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
phase_wrapped = np.mod(result_binned['phase'], 2*np.pi)
ax.scatter(phase_wrapped, result_binned['error'], alpha=0.3, s=1, label='Error vs phase')
ax.plot(result_binned['phase_bins'], result_binned['emean'], 'r-', linewidth=2, label='Mean error (binned)')
ax.set_xlabel('Phase (rad)')
ax.set_ylabel('Error')
ax.set_title('Test 1: Error Distribution vs Phase')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 2: Phase Jitter
ax = axes[1, 0]
ax.plot(t[sample_range], sig_jitter[sample_range], 'b-', label='Input signal (with jitter)', linewidth=0.8)
ax.plot(t[sample_range], result_binned_jit['fitted_signal'][sample_range], 'r--', label='Fitted', linewidth=1.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Test 2: Phase Jitter - Time Domain')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
phase_jit = np.mod(result_binned_jit['phase'], 2*np.pi)
ax.scatter(phase_jit, result_binned_jit['error']**2, alpha=0.3, s=1, label='Error² vs phase')
ax.plot(result_binned_jit['phase_bins'], result_binned_jit['erms']**2, 'r-', linewidth=2.5, label='RMS² (binned)')
ax.set_xlabel('Phase (rad)')
ax.set_ylabel('Error²')
ax.set_title('Test 2: Error² Distribution vs Phase (PM Dominant)')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 3: Amplitude Noise
ax = axes[2, 0]
ax.plot(t[sample_range], sig_gain[sample_range], 'b-', label='Input signal (with AM)', linewidth=0.8)
ax.plot(t[sample_range], result_binned_am['fitted_signal'][sample_range], 'r--', label='Fitted', linewidth=1.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Test 3: Amplitude Noise - Time Domain')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
phase_am = np.mod(result_binned_am['phase'], 2*np.pi)
ax.scatter(phase_am, result_binned_am['error']**2, alpha=0.3, s=1, label='Error² vs phase')
ax.plot(result_binned_am['phase_bins'], result_binned_am['erms']**2, 'r-', linewidth=2.5, label='RMS² (binned)')
ax.set_xlabel('Phase (rad)')
ax.set_ylabel('Error²')
ax.set_title('Test 3: Error² Distribution vs Phase (AM Dominant)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = output_dir / 'exp_a53_phase_error_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

# Comparison figure: Binned vs Raw
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Binned vs Raw Method Comparison', fontsize=14, fontweight='bold')

# Test 1: Noise case
ax = axes[0]
methods = ['Binned', 'Raw']
am_values = [result_binned['am_param'], result_raw['am_param']]
pm_values = [result_binned['pm_param'], result_raw['pm_param']]

x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, am_values, width, label='AM param', alpha=0.8)
ax.bar(x + width/2, pm_values, width, label='PM param', alpha=0.8)
ax.set_ylabel('Parameter Value')
ax.set_title('Test 1: Pure Noise')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Test 2: Jitter case
ax = axes[1]
am_jit = [result_binned_jit['am_param'], result_raw_jit['am_param']]
pm_jit = [result_binned_jit['pm_param'], result_raw_jit['pm_param']]
ax.bar(x - width/2, am_jit, width, label='AM param', alpha=0.8)
ax.bar(x + width/2, pm_jit, width, label='PM param', alpha=0.8)
ax.set_ylabel('Parameter Value')
ax.set_title('Test 2: Phase Jitter (PM Dominant)')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Test 3: AM case
ax = axes[2]
am_amp = [result_binned_am['am_param'], result_raw_am['am_param']]
pm_amp = [result_binned_am['pm_param'], result_raw_am['pm_param']]
ax.bar(x - width/2, am_amp, width, label='AM param', alpha=0.8)
ax.bar(x + width/2, pm_amp, width, label='PM param', alpha=0.8)
ax.set_ylabel('Parameter Value')
ax.set_title('Test 3: Amplitude Noise (AM Dominant)')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = output_dir / 'exp_a53_binned_vs_raw_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

print("\n[Complete!]")
