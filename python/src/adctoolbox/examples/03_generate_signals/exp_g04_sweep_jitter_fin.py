"""Sweep input frequency to analyze sampling jitter impact on SNR.

Demonstrates jitter-induced SNR degradation at higher frequencies.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum
from adctoolbox.siggen import ADC_Signal_Generator

# Setup
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**13
Fs = 128e9
A, DC = 0.5, 0.5
jitter_rms = 50e-15

print(f"[Setup] Fs={Fs/1e9:.0f}GHz, N={N}")
print(f"[Setup] Fixed Jitter RMS = {jitter_rms*1e15:.1f} fs")

# Define the Sweep List (Input Frequencies in GHz)
fin_sweep_ghz = [0.25 * (2**i) for i in range(8)]

# Prepare Figure (2 rows x 4 columns)
n_cols = 4
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes = axes.flatten()

print("=" * 75)
print(f"{'Fin (GHz)':<10} | {'Meas SNR':<10} | {'Theory SNR':<12} | {'Meas ENOB':<10}")
print("-" * 75)

# Run Sweep
for idx, fin_val_ghz in enumerate(fin_sweep_ghz):
    fin_target = fin_val_ghz * 1e9
    
    # Calculate Coherent Frequency for this step
    Fin, _ = find_coherent_frequency(Fs, fin_target, N)
    
    # Re-initialize Generator with new Fin
    gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)
    
    # Generate Signal with Jitter
    signal = gen.apply_jitter(input_signal=None, jitter_rms=jitter_rms)
    
    # Plot on specific subplot
    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs)
    
    # Calculate Theoretical SNR limited by Jitter
    theory_snr = -20 * np.log10(2 * np.pi * Fin * jitter_rms)
    
    # Custom Title & Formatting
    title = f"Fin = {fin_val_ghz} GHz (Jitter=50fs)"
    axes[idx].set_title(title, fontsize=12, fontweight='bold')    
    axes[idx].set_ylim([-140, 0])
        
    # Print Metrics
    print(f"{fin_val_ghz:<10.1f} | {result['snr_dbc']:<10.2f} | {theory_snr:<12.2f} | {result['enob']:<10.2f}")

# Finalize and Save
plt.suptitle(f'Jitter Sensitivity Sweep: Fin 1GHz-8GHz (Jitter=50fs)\n(Theoretical SNR drops ~6dB per octave)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.90)

fig_path = output_dir / "exp_g04_sweep_jitter_fin.png"
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
