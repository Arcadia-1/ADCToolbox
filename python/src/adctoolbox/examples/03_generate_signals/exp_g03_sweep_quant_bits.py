"""Sweep quantization bits to analyze how noise floor changes with ADC resolution.

Demonstrates quantization noise scaling across different bit depths.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum
from adctoolbox.siggen import ADC_Signal_Generator

# Setup
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**16
Fs = 1000e6
Fin_target = 80e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
A, DC = 0.5, 0.5

print(f"[Setup] Fs={Fs/1e6:.0f}MHz, Fin={Fin/1e6:.2f}MHz")
print(f"[Setup] Sweeping Quantization Bits: 2, 4, 6, 8, 10, 12, 14, 16")

# Initialize Generator
gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

# Define the Sweep List
bits_sweep = [2, 4, 6, 8, 10, 12, 14, 16]

# Prepare Figure (2 rows x 4 columns)
n_cols = 4
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6))
axes = axes.flatten()

print("=" * 60)
print(f"{'Bits':<6} | {'ENOB':<8} | {'SNR (dB)':<10} | {'Theory SNR':<10}")
print("-" * 60)

# Run Sweep
for idx, n_bits in enumerate(bits_sweep):
    # Generate Signal
    signal = gen.apply_quantization_noise(n_bits=n_bits, quant_range=(0.0, 1.0))
    
    # Plot on specific subplot
    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs)
    
    # Custom Title & Formatting
    theory_snr = 6.02 * n_bits + 1.76
    title = f"{n_bits}-Bit Quantization"
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
            
    # Print Metrics to Console
    print(f"{n_bits:<6d} | {result['enob']:<8.2f} | {result['snr_dbc']:<10.2f} | {theory_snr:<10.2f}")

# Finalize and Save
plt.suptitle(f'Quantization Noise Sweep: 2-bit to 16-bit\n(Theoretical SNR = 6.02N + 1.76 dB)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.90)

fig_path = output_dir / "exp_g03_sweep_quant_bits.png"
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
