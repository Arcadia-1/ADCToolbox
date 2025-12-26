"""
Calculates theoretical SNR from signal amplitude and noise level. Sweeps noise from 1µV to 10mV
and compares against ADC quantization noise for 6-14 bit converters, visualizing SNR vs noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import amplitudes_to_snr, snr_to_enob
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
A_fixed = 0.5  # Signal amplitude (V)
FSR = 1.0  # Full scale range (V)

# Create plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

noise_sweep = np.logspace(-6, -2, 100)

# Ideal case
snr_sweep = amplitudes_to_snr(A_fixed, noise_sweep)
ax.semilogx(noise_sweep * 1e6, snr_sweep, 'b-', linewidth=2.5, label='Ideal (no quantization noise)')

# ADC quantization noise for 6, 8, 10, 12, 14-bit
adc_bits = [6, 8, 10, 12, 14]
colors = ['gray', 'orange', 'red', 'purple', 'brown']
linestyles = ['--', '--', '-.', ':', '--']

for bits, color, ls in zip(adc_bits, colors, linestyles):
    lsb = FSR / (2 ** bits)
    quant_noise_rms = lsb / np.sqrt(12)
    noise_total = np.sqrt(noise_sweep**2 + quant_noise_rms**2)
    snr_with_quant = amplitudes_to_snr(A_fixed, noise_total)

    ax.semilogx(noise_sweep * 1e6, snr_with_quant, color=color, linestyle=ls, linewidth=2,
                label=f'{bits}-bit ADC (Q-noise={quant_noise_rms*1e6:.1f} uV)')
    ax.axvline(quant_noise_rms * 1e6, color=color, linestyle=':', linewidth=1, alpha=0.5)

ax.grid(True, which='both', alpha=0.3)
ax.set_xlabel('Noise RMS (uV)', fontsize=12)
ax.set_ylabel('SNR (dB)', fontsize=12)
ax.set_title(f'SNR vs Noise Level (A = {A_fixed} V, FSR = {FSR} V)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.set_ylim([30, 120])

plt.tight_layout()
fig_path = output_dir / 'exp_c04_snr_calculations.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Figure saved] -> {fig_path.resolve()}")
plt.close()

# Print summary
print("\n" + "="*70)
print("Summary: SNR = 20*log10(A_RMS / noise_RMS) = 20*log10(A/sqrt(2) / sigma)")
print("="*70)
print(f"Signal Amplitude: A = {A_fixed} V, FSR = {FSR} V")
print(f"\nADC Quantization Noise:")
for bits in adc_bits:
    lsb = FSR / (2 ** bits)
    quant_noise_rms = lsb / np.sqrt(12)
    max_snr = amplitudes_to_snr(A_fixed, quant_noise_rms)
    theory_snr = 6.02 * bits + 1.76
    print(f"  {bits:2d}-bit: Q-noise={quant_noise_rms*1e6:7.1f} uV, SNR={max_snr:5.2f} dB (Theory={theory_snr:.2f} dB)")
print("="*70)

