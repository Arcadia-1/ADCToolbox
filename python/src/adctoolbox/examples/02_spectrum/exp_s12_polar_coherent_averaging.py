"""
Polar spectrum with coherent averaging: aligns phases across runs before averaging complex FFT.
Preserves phase relationships between fundamental and harmonics on polar plot. Noise floor
improves with more runs while harmonic phases remain stable. Superior to power averaging.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd, analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**10
Fs = 100e6
A = 0.499
noise_rms = 100e-6
hd2_dB = -80
hd3_dB = -73
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)

k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=5e6, n_fft=N_fft)

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.6f} MHz] (coherent, Bin {Fin_bin}), N=[{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]")
print(f"[Nonlinearity] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB]\n")

# Number of runs to test
N_runs = [1, 10, 100]

# Generate signals for all runs
t = np.arange(N_fft) / Fs
N_max = max(N_runs)
signal_matrix = np.zeros((N_max, N_fft))  # M x N: (runs, samples)

for run_idx in range(N_max):
    phase_random = np.random.uniform(0, 2 * np.pi)
    sig_ideal = A * np.sin(2 * np.pi * Fin * t + phase_random)

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    sig_distorted = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    signal_matrix[run_idx, :] = sig_distorted

print(f"[Generated] {N_max} runs with random phase\n")

fig, axes = plt.subplots(1, len(N_runs), figsize=(len(N_runs)*6, 6), subplot_kw={'projection': 'polar'})

# Store axes limits for restoration after tight_layout
axes_info = []

for idx, N_run in enumerate(N_runs):
    signal_data = signal_matrix[:N_run, :]

    result = analyze_spectrum_polar(
        signal_data,
        fs=Fs,
        harmonic=5,
        ax=axes[idx],
        fixed_radial_range=120
    )

    axes[idx].set_title(f'N_run = {N_run}', pad=20, fontsize=14, fontweight='bold')

    # Store axis and its ylim for later restoration
    axes_info.append((axes[idx], axes[idx].get_ylim()))

    print(f"[{N_run:3d} Run(s)] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_dbc']:6.2f} dB], SNR=[{result['snr_dbc']:6.2f} dB]")

plt.suptitle(f'Coherent Spectrum Averaging: Phase Alignment Across Runs (N_fft = {N_fft})',
             fontsize=16, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Restore ylim after tight_layout (which resets polar axis limits)
for ax, ylim in axes_info:
    ax.set_ylim(ylim)

fig_path = output_dir / 'exp_s12_polar_coherent_averaging.png'
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
