"""
Coherent spectrum averaging with OSR: aligns phases across runs before averaging complex FFT values.
Demonstrates coherent integration gain with oversampling ratio (OSR=4) and varying number of runs.
Compare power vs coherent averaging results with OSR.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N_fft = 2**13
Fs = 100e6
A = 0.499
noise_rms = 100e-6
hd2_dB = -100
hd3_dB = -90
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)
osr = 4

Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=1e6, n_fft=N_fft)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.6f} MHz] (coherent, Bin {Fin_bin}), N=[{N_fft}], A=[{A:.3f} Vpeak], OSR=[{osr}]")
print(f"[Nonideal] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB], Noise RMS=[{noise_rms*1e6:.2f} uVrms]\n")

# Number of runs to test
N_runs = [1, 4, 16]

# Generate signals for all runs - same method as exp_s07
t = np.arange(N_fft) / Fs
N_max = max(N_runs)
signal_matrix = np.zeros((N_fft, N_max))


for run_idx in range(N_max):
    phase_random = np.random.uniform(0, 2 * np.pi)
    sig_ideal = A * np.sin(2 * np.pi * Fin * t + phase_random)

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    sig_distorted = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    signal_matrix[:, run_idx] = sig_distorted

print(f"[Generated] {N_max} runs with random phase\n")

# Create comparison plots
# Each subplot is 6x5 inches
subplot_width = 6
subplot_height = 5
fig_width = subplot_width * len(N_runs)
fig_height = subplot_height * 2  # 2 rows
fig, axes = plt.subplots(2, len(N_runs), figsize=(fig_width, fig_height))


for idx, N_run in enumerate(N_runs):
    # Prepare signal data
    if N_run == 1:
        signal_single = signal_matrix[:, 0]
        signal_data = signal_single
    else:
        signal_data = signal_matrix[:, :N_run].T

    # Traditional power averaging (analyze_spectrum, coherent_averaging=False by default)
    plt.sca(axes[0, idx])
    result_trad = analyze_spectrum(signal_data, fs=Fs, osr=osr)
    axes[0, idx].set_ylim([-140, 0])

    # Coherent averaging (coherent_averaging=True)
    plt.sca(axes[1, idx])
    result_coh = analyze_spectrum(signal_data, fs=Fs, osr=osr, coherent_averaging=True)
    axes[1, idx].set_ylim([-140, 0])

    print(f"[{N_run:3d} Run(s)] Power Avg: ENoB=[{result_trad['enob']:5.2f} b], SNR=[{result_trad['snr_db']:6.2f} dB] | Coherent Avg: ENoB=[{result_coh['enob']:5.2f} b], SNR=[{result_coh['snr_db']:6.2f} dB]")

# Add overall title
fig.suptitle(f'Power Spectrum Averaging vs Complex Spectrum Coherent Averaging (N_fft = {N_fft}, OSR = {osr})',
             fontsize=16, fontweight='bold')

plt.tight_layout()
fig_path = output_dir / 'exp_s12_spectrum_coherent_averaging.png'
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
