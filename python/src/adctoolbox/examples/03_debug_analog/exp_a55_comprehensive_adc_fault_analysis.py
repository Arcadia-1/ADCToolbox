"""Comprehensive ADC Fault Analysis - Using All New Functions

This example demonstrates a complete ADC diagnostic workflow using all the
new analysis functions:
1. fit_sinewave_components - LS fit kernel
2. compute_harmonic_decomposition - Signal decomposition
3. compute_phase_error_from_binned - Phase error (trend)
4. compute_phase_error_from_raw - Phase error (precision)
5. compute_error_by_code - Code-based error (INL/DNL)

Scenario: Analyze a realistic ADC with combined faults:
- Static nonlinearity (creates harmonics)
- Random thermal noise
- Timing jitter (phase modulation)
- Some gain variation (amplitude modulation)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import (
    fit_sinewave_components,
    compute_harmonic_decomposition,
    compute_phase_error_from_binned,
    compute_phase_error_from_raw,
    compute_error_by_code
)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# ============================================================
# Setup Parameters
# ============================================================
N = 2**14
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
A = 0.49
num_bits = 10
LSB = 1.0 / (2**num_bits)

print("="*80)
print("COMPREHENSIVE ADC FAULT ANALYSIS")
print("="*80)
print(f"Parameters: Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, A={A:.3f} V, Samples={N}")
print(f"ADC Resolution: {num_bits}-bit, LSB={LSB:.6f}\n")

# ============================================================
# Generate Test Signal with Multiple Faults
# ============================================================
print("="*80)
print("SIGNAL GENERATION - Multiple Faults")
print("="*80)

# Base signal
sig_ideal = A * np.sin(2 * np.pi * Fin * t)

# Fault 1: Static nonlinearity (HD2=-80dB, HD3=-70dB)
print("[OK] Fault 1: Static nonlinearity (HD2=-80dB, HD3=-70dB)")
hd2_dB, hd3_dB = -80, -70
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)
sig_nonlin = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3

# Fault 2: Thermal noise (500 uV RMS)
print("[OK] Fault 2: Thermal noise (500 uV RMS)")
noise_rms = 500e-6
sig_noise = sig_nonlin + np.random.randn(N) * noise_rms

# Fault 3: Timing jitter (0.02 rad RMS phase modulation)
print("[OK] Fault 3: Timing jitter (0.02 rad RMS)")
phase_jitter = 0.02 * np.random.randn(N)
# Apply jitter to fundamental signal
jitter_phase = 2 * np.pi * Fin * t + phase_jitter
sig_with_jitter = A * np.sin(jitter_phase)
# Combine jitter effect with nonlinear signal: replace fundamental with jittered version
sig_jitter_nonlin = sig_with_jitter + k2 * sig_with_jitter**2 + k3 * sig_with_jitter**3 + np.random.randn(N) * noise_rms

# Fault 4: Gain variation (1% at 500 kHz modulation frequency)
print("[OK] Fault 4: Gain variation (1% amplitude modulation)")
gain_variation = 1.0 + 0.01 * np.sin(2 * np.pi * 500e3 * t)
sig_measured = gain_variation * sig_jitter_nonlin  # Final measured signal

# Theoretical SNR/NSD for the combined signal
snr_combined = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_combined = snr_to_nsd(snr_combined, fs=Fs, osr=1)
print(f"\n[Combined Signal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], HD2={hd2_dB}dB, HD3={hd3_dB}dB")
print(f"                  Jitter=0.02rad, Gain Mod=1%, Theoretical SNR=[{snr_combined:.2f} dB], Theoretical NSD=[{nsd_combined:.2f} dBFS/Hz]\n")

# ============================================================
# Analysis 1: Fit Sinewave Components
# ============================================================
print("="*80)
print("ANALYSIS 1: FIT SINEWAVE COMPONENTS (LS Fit Kernel)")
print("="*80)

W, sig_fit, A_matrix, phase = fit_sinewave_components(
    sig_measured,
    freq=normalized_freq,
    order=1,
    include_dc=True
)

print(f"Fitted DC offset: {W[0]:.6e}")
print(f"Fitted Cos amplitude: {W[1]:.6e}")
print(f"Fitted Sin amplitude: {W[2]:.6e}")
fitted_amplitude = np.sqrt(W[1]**2 + W[2]**2)
fitted_phase = np.arctan2(W[2], W[1]) * 180 / np.pi
print(f"Fitted fundamental magnitude: {fitted_amplitude:.6e} (expected: {A:.6e})")
print(f"Fitted phase: {fitted_phase:.2f}°")

# ============================================================
# Analysis 2: Harmonic Decomposition
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 2: HARMONIC DECOMPOSITION")
print("="*80)

decomp = compute_harmonic_decomposition(sig_measured, normalized_freq=normalized_freq, order=10)

fundamental_power = np.mean(decomp['fundamental_signal']**2)
harmonic_power = np.mean(decomp['harmonic_error']**2)
other_power = np.mean(decomp['other_error']**2)
total_power = np.mean(sig_measured**2)

print(f"Fundamental power: {fundamental_power:.6e} ({fundamental_power/total_power*100:.2f}%)")
print(f"Harmonic power: {harmonic_power:.6e} ({harmonic_power/total_power*100:.2f}%)")
print(f"Other (noise) power: {other_power:.6e} ({other_power/total_power*100:.2f}%)")
print(f"Total power: {total_power:.6e}")

thd = np.sqrt(harmonic_power) / np.sqrt(fundamental_power) * 100 if fundamental_power > 0 else 0
sinad = 10 * np.log10(fundamental_power / (harmonic_power + other_power)) if (harmonic_power + other_power) > 0 else np.inf
enob = (sinad - 1.76) / 6.02 if sinad != np.inf else 0

print(f"THD: {thd:.2f}%")
print(f"SINAD: {sinad:.2f} dB")
print(f"ENOB: {enob:.2f} bits")

# ============================================================
# Analysis 3: Phase Error (Binned)
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 3: PHASE ERROR - BINNED APPROACH (Trend Analysis)")
print("="*80)

result_binned = compute_phase_error_from_binned(
    sig_measured,
    normalized_freq,
    bin_count=100
)

print(f"AM parameter (amplitude modulation): {result_binned['am_param']:.6e}")
print(f"PM parameter (phase modulation): {result_binned['pm_param']:.6e} rad")
print(f"Baseline noise: {result_binned['baseline']:.6e}")
print(f"Fundamental amplitude: {result_binned['fundamental_amplitude']:.6e}")
print(f"Valid phase bins: {np.sum(~np.isnan(result_binned['erms']))}/{len(result_binned['erms'])}")

# ============================================================
# Analysis 4: Phase Error (Raw)
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 4: PHASE ERROR - RAW APPROACH (High Precision)")
print("="*80)

result_raw = compute_phase_error_from_raw(sig_measured, normalized_freq)

print(f"AM parameter (amplitude modulation): {result_raw['am_param']:.6e}")
print(f"PM parameter (phase modulation): {result_raw['pm_param']:.6e} rad")
print(f"Baseline noise: {result_raw['baseline']:.6e}")
print(f"Error RMS: {result_raw['error_rms']:.6e}")
print(f"Fundamental amplitude: {result_raw['fundamental_amplitude']:.6e}")

print("\nComparison (Binned vs Raw):")
print(f"  AM diff: {abs(result_binned['am_param'] - result_raw['am_param']):.6e}")
print(f"  PM diff: {abs(result_binned['pm_param'] - result_raw['pm_param']):.6e}")

# ============================================================
# Analysis 5: Code-Based Error
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 5: CODE-BASED ERROR (INL/DNL Analysis)")
print("="*80)

result_code = compute_error_by_code(
    sig_measured,
    normalized_freq,
    num_bits=num_bits,
    clip_percent=0.01
)

emean = result_code['emean_by_code']
erms = result_code['erms_by_code']
codes = result_code['code_bins']

valid_mask = ~np.isnan(emean)
print(f"Codes used: {np.sum(valid_mask)}/{2**num_bits}")
print(f"Max mean error: {np.nanmax(np.abs(emean)):.6e} ({np.nanmax(np.abs(emean))/LSB:.2f} LSB)")
print(f"Max RMS error: {np.nanmax(erms):.6e} ({np.nanmax(erms)/LSB:.2f} LSB)")
print(f"Mean RMS error: {np.nanmean(erms[valid_mask]):.6e} ({np.nanmean(erms[valid_mask])/LSB:.2f} LSB)")

# ============================================================
# Generate Comprehensive Report Figures
# ============================================================
print("\n" + "="*80)
print("GENERATING COMPREHENSIVE REPORT")
print("="*80)

# Figure 1: Time domain overview
print("[Figure 1] Time domain signals...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('ADC Fault Analysis - Time Domain Overview', fontsize=14, fontweight='bold')

sample_range = slice(0, 3000)

ax = axes[0, 0]
ax.plot(t[sample_range], sig_ideal[sample_range], 'k--', label='Ideal', linewidth=1, alpha=0.7)
ax.plot(t[sample_range], sig_measured[sample_range], 'b-', label='Measured', linewidth=0.7, alpha=0.8)
ax.plot(t[sample_range], sig_fit[sample_range], 'r--', label='Fitted', linewidth=1.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Input Signal vs Fitted Fundamental')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(t[sample_range], sig_measured[sample_range], 'b-', label='Measured', linewidth=0.8)
ax.plot(t[sample_range], decomp['fundamental_signal'][sample_range], 'r-', label='Fundamental', linewidth=1.5)
ax.plot(t[sample_range], decomp['harmonic_error'][sample_range], 'orange', label='Harmonics', linewidth=1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Harmonic Decomposition')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
residual = sig_measured - sig_fit
ax.plot(t[sample_range], residual[sample_range], 'g-', linewidth=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Residual Error')
ax.set_title('Fitting Residual (Error)')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
phase_wrapped = np.mod(result_binned['phase'], 2*np.pi)
ax.scatter(phase_wrapped[::10], residual[::10]**2, alpha=0.3, s=2, label='Error vs phase')
ax.plot(result_binned['phase_bins'], result_binned['erms']**2, 'r-', linewidth=2.5, label='RMS (binned)')
ax.set_xlabel('Phase (rad)')
ax.set_ylabel('Error squared')
ax.set_title('Error Distribution vs Phase')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = output_dir / 'exp_a55_time_domain_overview.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  [Save] {fig_path.name}")
plt.close(fig)

# Figure 2: Harmonic analysis and power distribution
print("[Figure 2] Harmonic analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ADC Fault Analysis - Harmonic Content', fontsize=14, fontweight='bold')

ax = axes[0]
categories = ['Fundamental', 'Harmonics', 'Noise']
powers = [fundamental_power, harmonic_power, other_power]
colors = ['red', 'orange', 'green']
bars = ax.bar(categories, powers, color=colors, alpha=0.7)
ax.set_ylabel('Power (V²)')
ax.set_title('Power Distribution')
ax.set_yscale('log')
for bar, pwr in zip(bars, powers):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{pwr:.2e}', ha='center', va='bottom', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
metrics = ['THD (%)', 'SINAD (dB)', 'ENOB (bits)']
values = [thd, sinad if sinad != np.inf else 0, enob]
colors_metrics = ['blue', 'green', 'red']
bars = ax.bar(metrics, values, color=colors_metrics, alpha=0.7)
ax.set_ylabel('Value')
ax.set_title('ADC Performance Metrics')
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = output_dir / 'exp_a55_harmonic_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  [Save] {fig_path.name}")
plt.close(fig)

# Figure 3: Phase error analysis
print("[Figure 3] Phase error analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ADC Fault Analysis - Phase Error (AM/PM)', fontsize=14, fontweight='bold')

ax = axes[0]
methods = ['Binned', 'Raw']
am_values = [result_binned['am_param'], result_raw['am_param']]
pm_values = [result_binned['pm_param'], result_raw['pm_param']]
x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, am_values, width, label='AM param', alpha=0.8, color='blue')
ax.bar(x + width/2, pm_values, width, label='PM param', alpha=0.8, color='red')
ax.set_ylabel('Parameter Value')
ax.set_title('AM/PM Modulation Parameters')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
valid_mask_phase = ~np.isnan(result_binned['erms'])
phase_bin_centers = result_binned['phase_bins'][valid_mask_phase]
erms_values = result_binned['erms'][valid_mask_phase]
ax.plot(phase_bin_centers, erms_values, 'b-', linewidth=2, marker='o', markersize=4, label='Binned RMS')
ax.set_xlabel('Phase (rad)')
ax.set_ylabel('RMS Error')
ax.set_title('RMS Error vs Phase (Binned)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
fig_path = output_dir / 'exp_a55_phase_error_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  [Save] {fig_path.name}")
plt.close(fig)

# Figure 4: Code-based error analysis
print("[Figure 4] Code-based error analysis...")
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('ADC Fault Analysis - Code-Based Error (INL/DNL)', fontsize=14, fontweight='bold')

valid_codes = codes[valid_mask]
valid_emean = emean[valid_mask]
valid_erms = erms[valid_mask]

ax.fill_between(valid_codes, -valid_erms, valid_erms, alpha=0.3, color='blue', label='RMS band')
ax.plot(valid_codes, valid_emean, 'b-', linewidth=2, marker='o', markersize=3, label='Mean error')
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Code Value')
ax.set_ylabel('Error (Normalized)')
ax.set_title('Code-Based Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics
textstr = f'Max error: {np.nanmax(np.abs(emean)):.2e}\nRMS error: {np.nanmean(valid_erms):.2e}'
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
fig_path = output_dir / 'exp_a55_code_error_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  [Save] {fig_path.name}")
plt.close(fig)

# ============================================================
# Summary Report
# ============================================================
print("\n" + "="*80)
print("ANALYSIS SUMMARY")
print("="*80)
print(f"""
Faults Detected:
  1. Static nonlinearity:  THD = {thd:.2f}% (detected via harmonic decomposition)
  2. Thermal noise:        sigma ~= {np.sqrt(other_power):.2e} (other_power)
  3. Phase jitter:         PM = {result_binned['pm_param']:.2e} rad
  4. Gain variation:       AM = {result_binned['am_param']:.2e}

ADC Performance Metrics:
  SINAD: {sinad:.2f} dB
  ENOB:  {enob:.2f} bits (effective bits)
  THD:   {thd:.2f}%

Diagnostic Recommendations:
  - High THD indicates static nonlinearity (check transistor mismatch)
  - PM parameter shows timing jitter issues (check clock distribution)
  - AM parameter indicates gain variation (check bias networks)
  - Code error analysis useful for INL/DNL characterization
""")

print("="*80)
print("[COMPLETE!]")
print("="*80)
