"""Sweep different interference types to show effects on ADC spectrum.

Demonstrates harmonic, IMD, spur, and DC offset interference.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum
from adctoolbox.siggen import ADC_Signal_Generator

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 1000e6
Fin_target = 80e6
Fin, _ = find_coherent_frequency(Fs, Fin_target, N)

A, DC = 0.5, 0
base_noise = 10e-6

gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

print(f"[Setup] Fs={Fs/1e6:.0f} MHz | N={N} | Fin={Fin/1e6:.2f} MHz (Coherent)")
print(f"[Setup] A={A:.2f}V | DC={DC:.2f}V | base_noise={base_noise*1e6:.2f}uV\n")

# Generate clean coherent signal baseline
t = np.arange(N) / Fs
clean_signal = A * np.sin(2 * np.pi * Fin * t)

# Define 8 Interference Cases using lambdas
INTERFERENCES = [
    {
        'title': 'Clean Signal (reference)',
        'method': lambda sig: sig.copy(),
    },
    {
        'title': 'Glitch (prob=0.05%, amp=0.1)',
        'method': lambda sig: gen.apply_glitch(input_signal=sig, glitch_prob=0.0005, glitch_amplitude=0.1),
    },
    {
        'title': 'AM Tone (500 kHz, 0.05%)',
        'method': lambda sig: gen.apply_am_tone(input_signal=sig, am_tone_freq=500e3, am_tone_depth=0.0005),
    },
    {
        'title': 'AM Noise (strength=0.1%)',
        'method': lambda sig: gen.apply_am_noise(input_signal=sig, strength=0.001),
    },
    {
        'title': 'Clipping (level=1%)',
        'method': lambda sig: gen.apply_clipping(input_signal=sig, percentile_clip=1),
    },
    {
        'title': 'Clipping (level=2%)',
        'method': lambda sig: gen.apply_clipping(input_signal=sig, percentile_clip=2),
    },
    {
        'title': 'Drift (scale=2e-5)',
        'method': lambda sig: gen.apply_drift(input_signal=sig, drift_scale=2e-5),
    },
    {
        'title': 'Reference Error (tau=2.0, droop=0.01)',
        'method': lambda sig: gen.apply_reference_error(input_signal=sig, settling_tau=2.0, droop_strength=0.01),
    },
]

# Prepare Figure (2 rows x 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(24, 10))
axes = axes.flatten()

print("=" * 100)
print(f"{'#':<3} | {'Interference Type':<35} | {'SFDR (dB)':<10} | {'THD (dB)':<10} | {'SNR (dB)':<10}")
print("-" * 100)

# Run Sweep
for idx, config in enumerate(INTERFERENCES):
    
    # Apply interference to clean coherent signal
    signal = config['method'](clean_signal)
    signal = gen.apply_thermal_noise(signal, noise_rms=base_noise)
    
    # Spectrum analysis
    plt.sca(axes[idx])
    result = analyze_spectrum(signal, fs=Fs)
    
    axes[idx].set_title(config['title'], fontsize=11, fontweight='bold')
    axes[idx].set_ylim([-140, 0])
    
    print(f"{idx+1:<3} | {config['title']:<35} | {result['sfdr_dbc']:<10.2f} | {result['thd_dbc']:<10.2f} | {result['snr_dbc']:<10.2f}")

# Finalize
plt.suptitle('Interference Effects on Coherent Sampled Signal\n(Each type applied independently to Fin=80MHz)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

fig_path = output_dir / "exp_g07_sweep_interferences.png"
print(f"\n[Save figure] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
