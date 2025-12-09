"""
Polar spectrum with coherent averaging: aligns phases across runs before averaging complex FFT.
Preserves phase relationships between fundamental and harmonics on polar plot. Noise floor
improves with more runs while harmonic phases remain stable. Superior to power averaging.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calculate_coherent_freq, calculate_snr_from_amplitude, snr_to_nsd, analyze_spectrum_polar

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

Fin, Fin_bin = calculate_coherent_freq(fs=Fs, fin_target=5e6, n_fft=N_fft)

snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.6f} MHz] (coherent, Bin {Fin_bin}), N=[{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB], Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Number of runs to test
N_runs = [1, 10, 100]

# Generate signals for all runs
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

# Create polar plots with coherent averaging
# Save each plot individually to avoid MemoryError with subplots
for idx, N_run in enumerate(N_runs):
    # Prepare signal data
    if N_run == 1:
        signal_data = signal_matrix[:, 0]
    else:
        # Multiple runs - stack as 2D array for coherent averaging
        signal_data = signal_matrix[:, :N_run].T

    # Save individual polar plot
    fig_path = output_dir / f'exp_s22_polar_coherent_avg_n{N_run}.png'
    coherent_result = analyze_spectrum_polar(
        signal_data,
        fs=Fs,
        harmonic=5,
        win_type='boxcar',
        save_path=fig_path,
        show_plot=False,
        fixed_radial_range=120
    )

    # Extract metrics
    snr_db = coherent_result['metrics']['snr_db']
    sndr_db = coherent_result['metrics']['sndr_db']
    enob = coherent_result['metrics']['enob']

    print(f"[{N_run:3d} Run(s)] ENoB=[{enob:5.2f} b], SNDR=[{sndr_db:6.2f} dB], SNR=[{snr_db:6.2f} dB], HD2 phase=[{coherent_result['hd2_phase_deg']:6.1f}°], HD3 phase=[{coherent_result['hd3_phase_deg']:6.1f}°]")

print(f"\n[Saved {len(N_runs)} individual plots] -> [{output_dir}]")
