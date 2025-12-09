"""
Power spectrum averaging: reduces noise floor by averaging FFT magnitudes across runs.
1 run: noisy spectrum. 8 runs: ~9dB noise floor improvement. 64 runs: ~18dB improvement.
Power averaging is magnitude-only (|FFT|²) - phase information is discarded.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calculate_coherent_freq, analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**10
Fs = 100e6
A = 0.499
noise_rms = 100e-6
hd2_dB = -100
hd3_dB = -90

Fin, Fin_bin = calculate_coherent_freq(fs=Fs, fin_target=5e6, n_fft=N_fft)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.6f} MHz] (coherent, Bin {Fin_bin}), N=[{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB], Noise RMS=[{noise_rms*1e6:.2f} uVrms]\n")
hd2_amp = 10**(hd2_dB/20)  # Harmonic amplitude / Fundamental amplitude
hd3_amp = 10**(hd3_dB/20)

# Compute nonlinearity coefficients to achieve target HD levels
# HD2: k2 * A^2 / 2 = hd2_amp * A  →  k2 = hd2_amp / (A/2)
# HD3: k3 * A^3 / 4 = hd3_amp * A  →  k3 = hd3_amp / (A^2/4)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# === Generate Multiple Runs (Random Phase + Noise + Nonlinearity) ===
N_runs = [1, 8, 64]
t = np.arange(N_fft) / Fs

# Generate the maximum number of runs needed (64)
N_max = max(N_runs)
signal_matrix = np.zeros((N_fft, N_max))

for run_idx in range(N_max):
    phase_random = np.random.uniform(0, 2 * np.pi)

    # Generate sine with random phase
    sig_ideal = A * np.sin(2 * np.pi * Fin * t + phase_random)

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    sig_distorted = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    # Store in matrix
    signal_matrix[:, run_idx] = sig_distorted

print(f"[Generated] {N_max} runs with random phase")

fig, axes = plt.subplots(1, len(N_runs), figsize=(len(N_runs)*6, 6))

for idx, N_run in enumerate(N_runs):
    if N_run == 1:
        # Single run - no averaging
        signal_data = signal_matrix[:, 0]
    else:
        # Multiple runs - average
        signal_data = signal_matrix[:, :N_run].T

    plt.sca(axes[idx])
    result = analyze_spectrum(signal_data, fs=Fs)

    axes[idx].set_ylim([-120, 0])
    axes[idx].set_title(f'N_run = {N_run}', fontsize=12, fontweight='bold')

    print(f"[{N_run:2d} Run(s)] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SFDR=[{result['sfdr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

fig.suptitle(f'Spectral Averaging (N_fft = {N_fft})', fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = (output_dir / 'exp_s10_spectrum_power_averaging.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close()