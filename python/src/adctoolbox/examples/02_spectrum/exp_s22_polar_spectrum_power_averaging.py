"""
Polar spectrum with power averaging: shows limitation - power averaging discards phase information.
Displays only 1st run's phase on polar plot since averaged magnitude has no coherent phase.
Compare with exp_s23 coherent averaging which preserves phase relationships across runs.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calculate_coherent_freq, analyze_spectrum
from adctoolbox.spectrum import analyze_spectrum_polar

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
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.6f} MHz] (coherent, Bin {Fin_bin}), N=[{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB], Noise RMS=[{noise_rms*1e6:.2f} uVrms]\n")

# === Generate Multiple Runs (Random Phase + Noise + Nonlinearity) ===
N_runs = [1, 8, 64]
t = np.arange(N_fft) / Fs

# Generate the maximum number of runs needed
N_max = max(N_runs)
signal_matrix = np.zeros((N_fft, N_max))

for run_idx in range(N_max):
    phase_random = np.random.uniform(0, 2 * np.pi)

    # Generate sine with random phase
    sig_ideal = A * np.sin(2 * np.pi * Fin * t + phase_random)

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    sig_distorted = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    signal_matrix[:, run_idx] = sig_distorted

print(f"[Generated] {N_max} runs with random phase\n")

# Create polar plots showing power averaging (single run per plot, no coherent averaging)
# Save each plot individually to avoid MemoryError with subplots
for idx, N_run in enumerate(N_runs):
    if N_run == 1:
        # Single run - no averaging
        signal_data = signal_matrix[:, 0]
    else:
        # Multiple runs - use power averaging (analyze_spectrum)
        signal_data = signal_matrix[:, :N_run].T

    # For power averaging, we just show the first run's phase on polar plot
    # This demonstrates the limitation: power averaging loses phase coherence
    single_signal = signal_matrix[:, 0]

    # Save individual polar plot
    fig_path = output_dir / f'exp_s22_polar_power_avg_n{N_run}.png'
    analyze_spectrum_polar(
        single_signal,
        fs=Fs,
        harmonic=5,
        win_type='boxcar',
        title=f'Power Avg (N_run={N_run})\nShowing 1st run phase only',
        save_path=fig_path,
        show_plot=False
    )

    # Also run analyze_spectrum to get metrics from power averaging
    result = analyze_spectrum(signal_data, fs=Fs, win_type='boxcar')

    print(f"[{N_run:2d} Run(s)] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_db']:6.2f} dB], SNR=[{result['snr_db']:6.2f} dB]")

print(f"\n[Saved {len(N_runs)} individual plots] -> [{output_dir}]")
