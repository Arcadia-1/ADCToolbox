"""
Two-tone IMD comparison: demonstrates how nonlinearity strength affects IMD products.
Left subplot shows weak nonlinearity (low IMD), right shows strong nonlinearity (high IMD).
IMD2 and IMD3 levels increase with higher k2/k3 coefficients in static nonlinearity model.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_two_tone_spectrum, find_coherent_frequency, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**16
Fs = 1000e6
A1 = 0.5
A2 = 0.5
noise_rms = 50e-6

F1, bin_F1 = find_coherent_frequency(fs=Fs, fin_target=110e6, n_fft=N_fft)
F2, bin_F2 = find_coherent_frequency(fs=Fs, fin_target=100e6, n_fft=N_fft)

# Calculate combined signal amplitude for two tones
sig_amplitude = np.sqrt(A1**2 + A2**2)
snr_ref = amplitudes_to_snr(sig_amplitude=sig_amplitude, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs)

print(f"[Sinewave] Fs=[{Fs/1e6:.1f} MHz], F1=[{F1/1e6:.2f} MHz] (Bin/N={bin_F1}/{N_fft}), F2=[{F2/1e6:.2f} MHz] (Bin/N={bin_F2}/{N_fft})")
print(f"[Sinewave] A1=[{A1:.3f} Vpeak], A2=[{A2:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Two different nonlinearity strengths
nonlinearity_configs = [
    {'name': 'Weak Nonlinearity', 'k2': 0.00001, 'k3': 0.00003},
    {'name': 'Strong Nonlinearity', 'k2': 0.0001, 'k3': 0.0003}
]

t = np.arange(N_fft) / Fs
sig_ideal = A1 * np.sin(2*np.pi*F1*t) + A2 * np.sin(2*np.pi*F2*t)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for idx, config in enumerate(nonlinearity_configs):
    k2 = config['k2']
    k3 = config['k3']

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    result = analyze_two_tone_spectrum(signal, fs=Fs, ax=axes[idx])

    axes[idx].set_title(f"{config['name']} (k2={k2:.5f}, k3={k3:.5f})",
                       fontsize=12, fontweight='bold', pad=10)

    print(f"[{config['name']:20s}] IMD2=[{result['imd2_dbc']:6.2f} dB], IMD3=[{result['imd3_dbc']:6.2f} dB], SFDR=[{result['sfdr_dbc']:6.2f} dB], SNDR=[{result['sndr_dbc']:6.2f} dB]")

plt.suptitle('Two-Tone IMD Comparison: Effect of Nonlinearity Strength',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

fig_path = (output_dir / 'exp_s22_two_tone_imd_comparison.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
