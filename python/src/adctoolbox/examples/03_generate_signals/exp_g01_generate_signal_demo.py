"""Demonstrate thermal noise effect on signal spectrum.

This example shows how thermal noise affects ADC signal quality and spectrum.
Generates clean signal and three thermal noise levels for comparison.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd, analyze_spectrum
from adctoolbox.siggen import ADC_Signal_Generator

# Setup output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True, parents=True)

# Parameters
N = 2**13
Fs = 100e6
Fin_target = 12e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
A, DC = 0.5, 0.5

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.2f} MHz], Bin/N=[{J}/{N}], A=[{A:.3f} Vpeak]")
print(f"\nThermal Noise Demo - Single Signal Comparison")
print("=" * 80)

# Initialize signal generator
gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC)

# Define signals (clean + 3 thermal noise levels)
signals = [
    ('Clean Sinewave (No Noise)', gen.get_clean_signal()),
    ('Thermal Noise: RMS=50 uV', gen.apply_thermal_noise(noise_rms=50e-6)),
    ('Thermal Noise: RMS=100 uV', gen.apply_thermal_noise(noise_rms=100e-6)),
    ('Thermal Noise: RMS=200 uV', gen.apply_thermal_noise(noise_rms=200e-6)),
]

# Calculate theoretical values for reference
for noise_rms in [0, 50e-6, 100e-6, 200e-6]:
    if noise_rms == 0:
        print(f"\nClean signal (no noise): Ideal performance")
    else:
        snr_theory = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
        nsd_theory = snr_to_nsd(snr_theory, fs=Fs, osr=1)
        print(f"Noise RMS={noise_rms*1e6:6.0f} uV: SNR_theory={snr_theory:6.2f} dB, NSD_theory={nsd_theory:7.2f} dBFS/Hz")

print("\n" + "=" * 80)
print("Measured Spectrum Analysis:")
print("=" * 80)

# Create single figure with 4 subplots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

results = []
for idx, (title, signal) in enumerate(signals):
    plt.sca(axes[idx])

    result = analyze_spectrum(signal, fs=Fs, create_plot=True, show_title=False,
                             show_label=True, ax=axes[idx])
    results.append(result)

    axes[idx].set_title(title, fontsize=11, fontweight='bold')
    axes[idx].set_ylim([-140, 0])

    # Print metrics
    print(f"{idx+1}. {title:35s} | "
          f"ENOB={result['enob']:5.2f}b | "
          f"SNR={result['snr_dbc']:6.2f}dB | "
          f"SNDR={result['sndr_dbc']:6.2f}dB | "
          f"NSD={result['nsd_dbfs_hz']:7.2f}dBFS/Hz")

plt.suptitle('ADC Signal Quality with Thermal Noise', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()

fig_path = output_dir / 'exp_g01_generate_signal_demo_thermal_noise.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[Save figure] -> [{fig_path}]")
