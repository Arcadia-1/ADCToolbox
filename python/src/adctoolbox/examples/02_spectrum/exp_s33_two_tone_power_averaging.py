"""
Two-tone power spectrum averaging: reduces noise floor by averaging FFT magnitudes across runs.
1 run: noisy spectrum. 8 runs: ~9dB noise floor improvement. 64 runs: ~18dB improvement.
Power averaging is magnitude-only (|FFT|²) - phase information is discarded.
Demonstrates IMD2/IMD3 measurement improvement with multiple runs. Includes weak nonlinearity (k2=0.00001, k3=0.00003).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_two_tone_spectrum, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 1000e6
A1 = 0.5
A2 = 0.5
noise_rms = 100e-6

F1, bin_F1 = find_coherent_frequency(fs=Fs, fin_target=410e6, n_fft=N_fft)
F2, bin_F2 = find_coherent_frequency(fs=Fs, fin_target=400e6, n_fft=N_fft)

# Calculate combined signal amplitude for two tones
sig_amplitude = np.sqrt(A1**2 + A2**2)
snr_ref = amplitudes_to_snr(sig_amplitude=sig_amplitude, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs)

print(f"[Sinewave] Fs=[{Fs/1e6:.1f} MHz], F1=[{F1/1e6:.2f} MHz] (Bin/N={bin_F1}/{N_fft}), F2=[{F2/1e6:.2f} MHz] (Bin/N={bin_F2}/{N_fft})")
print(f"[Sinewave] A1=[{A1:.3f} Vpeak], A2=[{A2:.3f} Vpeak]")
print(f"[Noise] RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]")

# Add weak nonlinearity to generate IMD products
k2 = 0.0001
k3 = 0.0003
print(f"[Nonlinearity] k2={k2:.5f}, k3={k3:.5f} (Strong IMD)\n")

# === Generate Multiple Runs (Random Phase + Noise) ===
N_runs = [1, 8, 64]
t = np.arange(N_fft) / Fs

# Generate the maximum number of runs needed (64)
N_max = max(N_runs)
signal_matrix = np.zeros((N_max, N_fft))  # M x N: (runs, samples)

for run_idx in range(N_max):
    phase_random_1 = np.random.uniform(0, 2 * np.pi)
    phase_random_2 = np.random.uniform(0, 2 * np.pi)

    # Generate two-tone signal with random phases
    sig1 = A1 * np.sin(2 * np.pi * F1 * t + phase_random_1)
    sig2 = A2 * np.sin(2 * np.pi * F2 * t + phase_random_2)
    sig_ideal = sig1 + sig2

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms
    signal_matrix[run_idx, :] = signal

print(f"[Generated] {N_max} runs with random phase\n")

fig, axes = plt.subplots(1, len(N_runs), figsize=(len(N_runs)*6, 6))

for idx, N_run in enumerate(N_runs):
    signal_data = signal_matrix[:N_run, :]

    plt.sca(axes[idx])
    result = analyze_two_tone_spectrum(signal_data, fs=Fs)
    axes[idx].set_ylim([-140, 0])

    print(f"[{N_run:2d} Run(s)] SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], IMD2=[{result['imd2_db']:6.2f} dB], IMD3=[{result['imd3_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

plt.suptitle(f"Two-Tone Power Spectrum Averaging: Noise Reduction over Multiple Runs (N_fft = {N_fft})",
             fontsize=16, fontweight='bold')

plt.tight_layout()
fig_path = (output_dir / 'exp_s33_two_tone_power_averaging.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()
