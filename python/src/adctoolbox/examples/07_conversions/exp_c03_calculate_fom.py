"""
Demonstrates Figure of Merit (FOM) calculations and performance limits for ADCs.
Shows Walden FOM, Schreier FOM, jitter-limited SNR, and thermal noise-limited SNR.
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.fundamentals import (
    calculate_walden_fom, calculate_schreier_fom,
    calculate_thermal_noise_limit, calculate_jitter_limit
)
from adctoolbox import snr_to_enob
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Create 2x2 subplot figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# ============================================================================
# Plot 1: ENOB vs Sampling Frequency (for constant Walden FOM values)
# ============================================================================
fs_sweep = np.logspace(6, 11, 100)  # 1 MHz to 100 GHz
power_fixed = 10e-3  # 10 mW fixed power
walden_fom_values = [10, 100, 1000]  # fJ/conv-step
colors1 = ['green', 'blue', 'orange', 'red', 'brown']

for fom_fj, color in zip(walden_fom_values, colors1):
    fom_j = fom_fj * 1e-15  # Convert fJ to J
    # Solve for ENOB: FOM = Power / (2^ENOB * Fs) => ENOB = log2(Power / (FOM * Fs))
    enob_sweep = np.log2(power_fixed / (fom_j * fs_sweep))
    ax1.semilogx(fs_sweep / 1e6, enob_sweep, color=color, linewidth=2,
                 label=f'FOM = {fom_fj} fJ/step')

# Add horizontal ENOB reference lines (no legend)
enob_refs_plot1 = [6, 8, 10, 12, 14]
for enob in enob_refs_plot1:
    ax1.axhline(enob, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(1.2, enob + 0.2, f'{enob}-bit', fontsize=8, color='gray', alpha=0.7)

ax1.grid(True, which='both', alpha=0.3)
ax1.set_xlabel('Sampling Frequency (MHz)', fontsize=11)
ax1.set_ylabel('ENOB (bits)', fontsize=11)
ax1.set_title('Walden FOM: Achievable ENOB vs Sampling Frequency\n(Power = 10 mW)',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.set_ylim([4, 16])

# ============================================================================
# Plot 2: SNDR vs Bandwidth (for constant Schreier FOM values)
# ============================================================================
bw_sweep2 = np.logspace(5, 10, 100)  # 100 kHz to 10 GHz
power_fixed2 = 10e-3  # 10 mW fixed power
schreier_fom_values = [160, 170, 180, 190]  # dB
colors2 = ['blue', 'green', 'orange', 'red']

for fom_db, color in zip(schreier_fom_values, colors2):
    # Solve for SNDR: FOM = SNDR + 10*log10(BW / Power) => SNDR = FOM - 10*log10(BW / Power)
    sndr_sweep = fom_db - 10 * np.log10(bw_sweep2 / power_fixed2)
    ax2.semilogx(bw_sweep2 / 1e6, sndr_sweep, color=color, linewidth=2,
                 label=f'FOM = {fom_db} dB')

# Add horizontal ENOB reference markers (no legend)
enob_refs_plot2 = [8, 10, 12, 14]
for enob in enob_refs_plot2:
    sndr_ref = 6.02 * enob + 1.76
    ax2.axhline(sndr_ref, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(0.12, sndr_ref + 1, f'{enob}-bit', fontsize=8, color='gray', alpha=0.7)

ax2.grid(True, which='both', alpha=0.3)
ax2.set_xlabel('Signal Bandwidth (MHz)', fontsize=11)
ax2.set_ylabel('SNDR (dB)', fontsize=11)
ax2.set_title('Schreier FOM: Achievable SNDR vs Signal Bandwidth\n(Power = 10 mW)',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.set_ylim([50, 110])

# ============================================================================
# Plot 3: Jitter-Limited SNR vs Input Frequency (sweep Fin, different jitter values)
# ============================================================================
fin_sweep = np.logspace(7, 10, 100)  # 10 MHz to 10 GHz
jitter_values_fs = [10, 50, 100, 500]  # femtoseconds
colors3 = ['green', 'blue', 'orange', 'red']

for jitter_fs, color in zip(jitter_values_fs, colors3):
    jitter_sec = jitter_fs * 1e-15
    snr_jitter = calculate_jitter_limit(fin_sweep, jitter_sec)
    ax3.semilogx(fin_sweep / 1e6, snr_jitter, color=color, linewidth=2,
                 label=f'jitter = {jitter_fs:.0f} fs rms')

# Add vertical reference markers (no legend)
ax3.axvline(100, color='darkblue', linestyle=':', linewidth=2, alpha=0.7)
ax3.axvline(1000, color='darkred', linestyle=':', linewidth=2, alpha=0.7)
ax3.text(100, 115, '100 MHz', fontsize=8, color='darkblue', ha='center')
ax3.text(1000, 115, '1 GHz', fontsize=8, color='darkred', ha='center')

# Add ENOB reference lines (no legend)
enob_refs = [8, 10, 12, 14]
for enob in enob_refs:
    snr_ref = 6.02 * enob + 1.76
    ax3.axhline(snr_ref, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.text(12, snr_ref + 1, f'{enob}-bit', fontsize=8, color='gray', alpha=0.7)

ax3.grid(True, which='both', alpha=0.3)
ax3.set_xlabel('Input Frequency (MHz)', fontsize=11)
ax3.set_ylabel('Maximum SNR (dB)', fontsize=11)
ax3.set_title('Jitter-Limited SNR vs Input Frequency',
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='upper right')
ax3.set_ylim([50, 120])
ax3.set_xlim([10, 1e4])

# ============================================================================
# Plot 4: Thermal Noise-Limited SNR vs Sampling Capacitance (sweep Cap)
# ============================================================================
cap_sweep = np.logspace(-2, 2, 100)  # 0.01 pF (10 fF) to 100 pF
v_fs_values = [0.5, 1.0, 2.0]  # Different full-scale voltages
colors4 = ['blue', 'green', 'red']
linestyles = ['-', '--', '-.']

for v_fs, color, ls in zip(v_fs_values, colors4, linestyles):
    snr_thermal = calculate_thermal_noise_limit(cap_sweep, v_fs)
    ax4.semilogx(cap_sweep, snr_thermal, color=color, linestyle=ls, linewidth=2,
                 label=f'VFS = {v_fs} V')

# Add vertical reference markers (no legend)
ax4.axvline(0.01, color='purple', linestyle=':', linewidth=2, alpha=0.7)
ax4.axvline(0.1, color='darkblue', linestyle=':', linewidth=2, alpha=0.7)
ax4.axvline(1.0, color='darkred', linestyle=':', linewidth=2, alpha=0.7)
ax4.axvline(10.0, color='darkgreen', linestyle=':', linewidth=2, alpha=0.7)
ax4.text(0.01, 105, '10 fF', fontsize=8, color='purple', ha='center')
ax4.text(0.1, 105, '100 fF', fontsize=8, color='darkblue', ha='center')
ax4.text(1.0, 105, '1 pF', fontsize=8, color='darkred', ha='center')
ax4.text(10.0, 105, '10 pF', fontsize=8, color='darkgreen', ha='center')

# Add ENOB reference lines (no legend)
for enob in enob_refs:
    snr_ref = 6.02 * enob + 1.76
    ax4.axhline(snr_ref, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.text(0.012, snr_ref + 1, f'{enob}-bit', fontsize=8, color='gray', alpha=0.7)

ax4.grid(True, which='both', alpha=0.3)
ax4.set_xlabel('Sampling Capacitance (pF)', fontsize=11)
ax4.set_ylabel('Maximum SNR (dB)', fontsize=11)
ax4.set_title('Thermal Noise (kT/C) Limited SNR vs Capacitance',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='lower right')
ax4.set_ylim([50, 110])
ax4.set_xlim([0.01, 100])

# Save figure
plt.tight_layout()
fig_path = output_dir / 'exp_c03_calculate_fom.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Figure saved] -> {fig_path.resolve()}")
plt.close()

# Print summary information
print("\n" + "="*70)
print("ADC Figure of Merit (FOM) Summary")
print("="*70)

print("\n[1. Walden FOM] (Lower is better)")
print("    Formula: Power / (2^ENOB * Fs)")
print(f"    Example: {power_fixed*1e3:.1f} mW, 100 MHz, 10-bit")
walden_example = calculate_walden_fom(power_fixed, 100e6, 10)
print(f"    FOM = {walden_example*1e15:.2f} fJ/conv-step")

print("\n[2. Schreier FOM] (Higher is better)")
print("    Formula: SNDR + 10*log10(BW / Power)")
print(f"    Example: {power_fixed*1e3:.1f} mW, 1 MHz BW, 80 dB SNDR")
schreier_example = calculate_schreier_fom(power_fixed, 80, 1e6)
print(f"    FOM = {schreier_example:.1f} dB")

print("\n[3. Jitter-Limited SNR]")
print("    Formula: SNR = -20*log10(2*pi*fin*tj)")
print(f"    Example: 100 MHz input, 1 fs RMS jitter")
jitter_example = calculate_jitter_limit(100e6, 1e-15)
enob_jitter = snr_to_enob(jitter_example)
print(f"    Max SNR = {jitter_example:.1f} dB ({enob_jitter:.2f}-bit ENOB)")

print("\n[4. Thermal Noise (kT/C) Limited SNR]")
print("    Formula: SNR = 10*log10(P_signal / P_noise)")
print(f"    Example: 1 pF cap, 1 V full-scale")
thermal_example = calculate_thermal_noise_limit(1.0, 1.0)
enob_thermal = snr_to_enob(thermal_example)
print(f"    Max SNR = {thermal_example:.1f} dB ({enob_thermal:.2f}-bit ENOB)")

print("="*70)
