"""
Two-tone coherent spectrum averaging: aligns phases across runs before averaging complex FFT values.
Preserves phase relationships between tones and IMD products. More effective than power averaging
for reducing noise while maintaining IMD structure. Compare power vs coherent averaging results.
Includes weak nonlinearity (k2=0.00001, k3=0.00003).

KNOWN BUG: Coherent averaging shows incorrect signal amplitude in 10/100 run plots.
The second tone appears with reduced amplitude after phase alignment.
Noise floor improvement (+10/20 dB) is correct, but visual spectrum amplitude is wrong.
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

print(f"[Sinewave] Fs=[{Fs/1e6:.1f} MHz], F1=[{F1/1e6:.6f} MHz] (coherent, Bin {bin_F1}), F2=[{F2/1e6:.6f} MHz] (coherent, Bin {bin_F2}), N=[{N_fft}]")
print(f"[Sinewave] A1=[{A1:.3f} Vpeak], A2=[{A2:.3f} Vpeak]")
print(f"[Noise] RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]")

# Add weak nonlinearity to generate IMD products
k2 = 0.0001
k3 = 0.0003
print(f"[Nonlinearity] k2={k2:.5f}, k3={k3:.5f} (Strong IMD)\n")

# Number of runs to test
N_runs = [1, 10, 100]

# Generate signals for all runs
t = np.arange(N_fft) / Fs
N_max = max(N_runs)
signal_matrix = np.zeros((N_max, N_fft))  # M x N: (runs, samples)

for run_idx in range(N_max):
    # For coherent averaging, both tones need the same phase offset
    # This maintains their fixed phase relationship across runs
    phase_offset = np.random.uniform(0, 2 * np.pi)

    # Generate two-tone signal with same phase offset (coherent tones)
    sig1 = A1 * np.sin(2 * np.pi * F1 * t + phase_offset)
    sig2 = A2 * np.sin(2 * np.pi * F2 * t + phase_offset)
    sig_ideal = sig1 + sig2

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms
    signal_matrix[run_idx, :] = signal

print(f"[Generated] {N_max} runs with coherent phase (same offset for both tones)\n")

# Create comparison plots
# Each subplot is 6x5 inches
subplot_width = 6
subplot_height = 5
fig_width = subplot_width * len(N_runs)
fig_height = subplot_height * 2  # 2 rows
fig, axes = plt.subplots(2, len(N_runs), figsize=(fig_width, fig_height))

for idx, N_run in enumerate(N_runs):
    signal_data = signal_matrix[:N_run, :]

    # Power averaging (default)
    plt.sca(axes[0, idx])
    result_trad = analyze_two_tone_spectrum(signal_data, fs=Fs, coherent_averaging=False)
    axes[0, idx].set_ylim([-140, 0])

    # Coherent averaging with phase alignment
    plt.sca(axes[1, idx])
    result_coh = analyze_two_tone_spectrum(signal_data, fs=Fs, coherent_averaging=True)
    axes[1, idx].set_ylim([-140, 0])

    print(f"[{N_run:3d} Run(s)] Power Avg:    ENoB=[{result_trad['enob']:5.2f} b], SNR=[{result_trad['snr_db']:6.2f} dB], IMD2=[{result_trad['imd2_db']:6.2f} dB], IMD3=[{result_trad['imd3_db']:6.2f} dB]")
    print(f"[{N_run:3d} Run(s)] Coherent Avg: ENoB=[{result_coh['enob']:5.2f} b], SNR=[{result_coh['snr_db']:6.2f} dB], IMD2=[{result_coh['imd2_db']:6.2f} dB], IMD3=[{result_coh['imd3_db']:6.2f} dB]")

# Add overall title
fig.suptitle(f'Two-Tone: Power vs Coherent Averaging Comparison (N_fft = {N_fft})',
             fontsize=16, fontweight='bold')

plt.tight_layout()
fig_path = output_dir / 'exp_s34_two_tone_coherent_averaging.png'
print(f"\n[Save fig] -> [{fig_path}]")
plt.savefig(fig_path, dpi=150)
plt.close(fig)
