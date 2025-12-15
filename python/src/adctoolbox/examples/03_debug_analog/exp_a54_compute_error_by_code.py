"""Test code-based error analysis - INL/DNL detection

This example demonstrates code-based error analysis, which is used to detect
static nonlinearity errors (INL/DNL) and missing codes.

Tests:
1. Ideal ADC (no nonlinearity)
2. ADC with INL error
3. ADC with missing codes (catastrophic failure)
4. Impact of input signal amplitude on error resolution
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout import compute_error_by_code

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Setup parameters
N = 2**14
Fs = 800e6
Fin_target = 10e6
Fin, Fin_bin = find_coherent_frequency(Fs, Fin_target, N)
normalized_freq = Fin / Fs
t = np.arange(N) / Fs
num_bits = 10
LSB = 1.0 / (2**num_bits)

print(f"[Code Error Analysis] Fs={Fs/1e6:.1f} MHz, Fin={Fin/1e6:.6f} MHz, num_bits={num_bits}")
print(f"LSB = {LSB:.6f}\n")

# ============================================================
# Test 1: Ideal ADC (no nonlinearity)
# ============================================================
print("="*70)
print("Test 1: Ideal ADC")
print("="*70)

A_ideal = 0.49
base_noise = 10e-6
sig_ideal = A_ideal * np.sin(2 * np.pi * Fin * t)

snr_ideal = amplitudes_to_snr(sig_amplitude=A_ideal, noise_amplitude=base_noise)
nsd_ideal = snr_to_nsd(snr_ideal, fs=Fs, osr=1)
print(f"[Signal] A={A_ideal:.3f} V, Noise RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_ideal:.2f} dB], Theoretical NSD=[{nsd_ideal:.2f} dBFS/Hz]\n")

result_ideal = compute_error_by_code(sig_ideal, normalized_freq, num_bits=num_bits)
emean_ideal = result_ideal['emean_by_code']
erms_ideal = result_ideal['erms_by_code']
codes_ideal = result_ideal['code_bins']

print(f"Number of codes used: {np.sum(~np.isnan(emean_ideal))}")
print(f"Max mean error: {np.nanmax(np.abs(emean_ideal)):.6e}")
print(f"Max RMS error: {np.nanmax(erms_ideal):.6e}")
print(f"Mean RMS across codes: {np.nanmean(erms_ideal[~np.isnan(erms_ideal)]):.6e}")

# ============================================================
# Test 2: ADC with INL error
# ============================================================
print("\n" + "="*70)
print("Test 2: ADC with INL Error")
print("="*70)

# Create INL error: parabolic shape + some local deviations
inl_error = np.zeros(2**num_bits)
codes_arr = np.arange(2**num_bits)
normalized_codes = (codes_arr - 2**(num_bits-1)) / 2**(num_bits-1)
inl_error = 2 * LSB * normalized_codes**2  # Parabolic INL
inl_error[300:320] += 3 * LSB * np.sin(np.linspace(0, 2*np.pi, 20))  # Add some ripple

# Apply INL to signal
codes_input = np.round((sig_ideal + 0.5) * (2**num_bits - 1)).astype(int)
codes_input = np.clip(codes_input, 0, 2**num_bits - 1)
sig_with_inl = sig_ideal + inl_error[codes_input]

result_inl = compute_error_by_code(sig_with_inl, normalized_freq, num_bits=num_bits)
emean_inl = result_inl['emean_by_code']
erms_inl = result_inl['erms_by_code']

print(f"Number of codes used: {np.sum(~np.isnan(emean_inl))}")
print(f"Max mean error: {np.nanmax(np.abs(emean_inl)):.6e}")
print(f"Max RMS error: {np.nanmax(erms_inl):.6e}")
print(f"INL peak-to-peak: {np.max(inl_error) - np.min(inl_error):.6e}")

# ============================================================
# Test 3: ADC with missing codes
# ============================================================
print("\n" + "="*70)
print("Test 3: ADC with Missing Codes (Catastrophic Failure)")
print("="*70)

# Create code dropouts
dropout_codes = np.arange(200, 220)
codes_with_dropout = np.delete(np.arange(2**num_bits), dropout_codes)

# Map signal to remaining codes only
sig_dropout = np.copy(sig_ideal)
for i, code in enumerate(codes_with_dropout):
    if i < len(codes_with_dropout) - 1:
        # Map range of analog values to available codes
        pass

# Simpler approach: add large error only at dropped codes
codes_input_dropout = np.round((sig_ideal + 0.5) * (2**num_bits - 1)).astype(int)
codes_input_dropout = np.clip(codes_input_dropout, 0, 2**num_bits - 1)

# Add error to codes that are "missing" by substituting with nearest valid code
for i in range(len(sig_dropout)):
    if codes_input_dropout[i] in dropout_codes:
        # Replace with nearest valid code
        nearest_valid = codes_with_dropout[np.argmin(np.abs(codes_with_dropout - codes_input_dropout[i]))]
        sig_dropout[i] = sig_ideal[i] + (nearest_valid - codes_input_dropout[i]) * LSB

result_dropout = compute_error_by_code(sig_dropout, normalized_freq, num_bits=num_bits)
emean_dropout = result_dropout['emean_by_code']
erms_dropout = result_dropout['erms_by_code']

missing_count = np.sum(np.isnan(emean_dropout))
print(f"Missing codes: {missing_count}")
print(f"Dropout code range: {dropout_codes[0]}-{dropout_codes[-1]}")
print(f"Number of valid codes: {np.sum(~np.isnan(emean_dropout))}")

# ============================================================
# Visualization
# ============================================================
print("\n[Generating plots...]")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Code-Based Error Analysis', fontsize=16, fontweight='bold')

# Plot 1: Ideal ADC - mean error
ax = axes[0, 0]
valid_mask = ~np.isnan(emean_ideal)
ax.plot(codes_ideal[valid_mask], emean_ideal[valid_mask], 'b-', linewidth=1.5, label='Mean error')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Code Value')
ax.set_ylabel('Mean Error (LSB)')
ax.set_title('Test 1: Ideal ADC - Mean Error')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Ideal ADC - RMS error
ax = axes[0, 1]
valid_mask = ~np.isnan(erms_ideal)
ax.plot(codes_ideal[valid_mask], erms_ideal[valid_mask], 'r-', linewidth=1.5, label='RMS error')
ax.set_xlabel('Code Value')
ax.set_ylabel('RMS Error (LSB)')
ax.set_title('Test 1: Ideal ADC - RMS Error')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: ADC with INL - mean error
ax = axes[1, 0]
valid_mask = ~np.isnan(emean_inl)
ax.plot(codes_ideal[valid_mask], emean_inl[valid_mask], 'b-', linewidth=1.5, label='Mean error (INL)')
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Code Value')
ax.set_ylabel('Mean Error (LSB)')
ax.set_title('Test 2: ADC with INL - Mean Error')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: ADC with missing codes - mean error
ax = axes[1, 1]
valid_mask = ~np.isnan(emean_dropout)
ax.plot(codes_ideal[valid_mask], emean_dropout[valid_mask], 'g-', linewidth=1.5, label='Mean error')
# Highlight missing codes
missing_mask = np.isnan(emean_dropout)
ax.scatter(codes_ideal[missing_mask], np.zeros(np.sum(missing_mask)),
           color='red', marker='x', s=100, label='Missing codes', linewidth=2)
ax.axhline(0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Code Value')
ax.set_ylabel('Mean Error (LSB)')
ax.set_title('Test 3: ADC with Missing Codes')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = output_dir / 'exp_a54_code_error_analysis.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

# Additional figure: Detailed comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Code Error Analysis - Detailed Comparison', fontsize=14, fontweight='bold')

# Ideal ADC
ax = axes[0]
valid_mask = ~np.isnan(emean_ideal)
ax.fill_between(codes_ideal[valid_mask],
                 -erms_ideal[valid_mask], erms_ideal[valid_mask],
                 alpha=0.3, color='blue', label='RMS band')
ax.plot(codes_ideal[valid_mask], emean_ideal[valid_mask], 'b-', linewidth=2, label='Mean error')
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Code Value')
ax.set_ylabel('Error (LSB)')
ax.set_title('Test 1: Ideal ADC')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.1, 0.1])

# With INL
ax = axes[1]
valid_mask = ~np.isnan(emean_inl)
ax.fill_between(codes_ideal[valid_mask],
                 -erms_inl[valid_mask], erms_inl[valid_mask],
                 alpha=0.3, color='green', label='RMS band')
ax.plot(codes_ideal[valid_mask], emean_inl[valid_mask], 'g-', linewidth=2, label='Mean error')
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Code Value')
ax.set_ylabel('Error (LSB)')
ax.set_title('Test 2: With INL Error')
ax.legend()
ax.grid(True, alpha=0.3)

# With missing codes
ax = axes[2]
valid_mask = ~np.isnan(emean_dropout)
valid_codes = codes_ideal[valid_mask]
valid_errors = emean_dropout[valid_mask]
valid_erms = erms_dropout[valid_mask]
ax.fill_between(valid_codes, -valid_erms, valid_erms,
                 alpha=0.3, color='red', label='RMS band')
ax.plot(valid_codes, valid_errors, 'r-', linewidth=2, label='Mean error')

# Highlight missing regions
missing_mask = np.isnan(emean_dropout)
missing_codes = codes_ideal[missing_mask]
for i in range(len(missing_codes)):
    if i == 0 or missing_codes[i] - missing_codes[i-1] > 1.5:
        start = missing_codes[i]
    if i == len(missing_codes) - 1 or missing_codes[i+1] - missing_codes[i] > 1.5:
        end = missing_codes[i]
        if 'start' in locals():
            ax.axvspan(start, end, alpha=0.2, color='gray', label='Missing codes' if i == 0 else '')

ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Code Value')
ax.set_ylabel('Error (LSB)')
ax.set_title('Test 3: With Missing Codes')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = output_dir / 'exp_a54_code_error_comparison.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)

print("\n[Complete!]")
