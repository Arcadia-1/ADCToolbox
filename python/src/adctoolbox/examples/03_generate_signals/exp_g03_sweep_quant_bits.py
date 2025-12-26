"""Sweep quantization bits to analyze how noise floor changes with ADC resolution.

Demonstrates quantization noise scaling across different bit depths.
"""

import time
t_start = time.perf_counter()

import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum
from adctoolbox.siggen import ADC_Signal_Generator

t_import = time.perf_counter() - t_start
t_prep_start = time.perf_counter()

# Setup
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**16
Fs = 1000e6
Fin_target = 80e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
A, DC = 0.5, 0.5   # Full Scale range approx 0V-1V

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

t_prep = time.perf_counter() - t_prep_start
t_loop_start = time.perf_counter()

t_tool_total = 0.0
# Run Sweep
for idx, n_bits in enumerate(bits_sweep):
    # Generate Signal
    signal = gen.apply_quantization_noise(n_bits=n_bits, quant_range=(0.0, 1.0))
    
    # Plot on specific subplot
    plt.sca(axes[idx])
    
    t_tool_start = time.perf_counter()
    result = analyze_spectrum(signal, fs=Fs)
    t_tool_total += time.perf_counter() - t_tool_start
    
    # Custom Title & Formatting
    theory_snr = 6.02 * n_bits + 1.76
    title = f"{n_bits}-Bit Quantization"
    axes[idx].set_title(title, fontsize=12, fontweight='bold')
            
    # Print Metrics to Console
    print(f"{n_bits:<6d} | {result['enob']:<8.2f} | {result['snr_db']:<10.2f} | {theory_snr:<10.2f}")

t_loop = time.perf_counter() - t_loop_start
t_fig_start = time.perf_counter()

# Finalize and Save
plt.suptitle(f'Quantization Noise Sweep: 2-bit to 16-bit\n(Theoretical SNR = 6.02N + 1.76 dB)', 
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.90)

fig_path = output_dir / "exp_g03_sweep_quant_bits.png"
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

t_fig = time.perf_counter() - t_fig_start
t_total = time.perf_counter() - t_start

print(f"\n{'='*60}")
print(f"Timing Report:")
print(f"{'='*60}")
print(f"  Import time:      {t_import*1000:7.2f} ms")
print(f"  Preparation time: {t_prep*1000:7.2f} ms")
print(f"  Core tool time:   {t_tool_total*1000:7.2f} ms  (analyze_spectrum x8)")
print(f"  Main loop time:   {t_loop*1000:7.2f} ms")
print(f"  Figure time:      {t_fig*1000:7.2f} ms")
print(f"  Total runtime:    {t_total*1000:7.2f} ms")
print(f"{'='*60}\n")
