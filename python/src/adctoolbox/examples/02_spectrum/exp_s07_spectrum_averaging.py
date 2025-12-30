"""
Spectrum Averaging: Power vs Coherent Averaging

This example compares two spectrum averaging methods across multiple runs:

1. Power Spectrum Averaging (Magnitude Averaging) - Top Row:
   - Averages FFT magnitudes (|FFT|²) across runs
   - Phase information is discarded
   - Reduces noise floor by √N_runs (e.g., 10 runs → ~5dB, 100 runs → ~10dB)
   - Simple and effective for general noise floor reduction

2. Coherent Spectrum Averaging (Complex Averaging) - Bottom Row:
   - Aligns phases across runs before averaging complex FFT values
   - Preserves phase relationships between fundamental and harmonics
   - More effective than power averaging for reducing noise
   - Requires coherent sampling with phase alignment

Comparison (1 → 10 → 100 runs):
- Power averaging: Good for general noise reduction, handles random phases
- Coherent averaging: Superior performance when phase coherence is maintained
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Common parameters
N_fft = 2**13
Fs = 100e6
A = 0.499
noise_rms = 100e-6
hd2_dB = -100
hd3_dB = -90

Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=5e6, n_fft=N_fft)

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin/1e6:.6f} MHz] (coherent, Bin {Fin_bin}), N=[{N_fft}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] HD2=[{hd2_dB} dB], HD3=[{hd3_dB} dB], Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Compute nonlinearity coefficients to achieve target HD levels
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)

# Number of runs to test
N_runs = [1, 10, 100]
N_max = max(N_runs)

# Generate signals for all runs
t = np.arange(N_fft) / Fs
signal_matrix = np.zeros((N_max, N_fft))  # M x N: (runs, samples)

for run_idx in range(N_max):
    phase_random = np.random.uniform(0, 2 * np.pi)
    sig_ideal = A * np.sin(2 * np.pi * Fin * t + phase_random)

    # Apply static nonlinearity: y = x + k2*x^2 + k3*x^3
    sig_distorted = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    signal_matrix[run_idx, :] = sig_distorted

print(f"[Generated] {N_max} runs with random phase\n")

# ============================================================================
# Power vs Coherent Averaging Comparison
# ============================================================================
print("=" * 80)
print("POWER SPECTRUM AVERAGING vs COHERENT SPECTRUM AVERAGING")
print("=" * 80)

fig, axes = plt.subplots(2, len(N_runs), figsize=(len(N_runs)*7, 9))

# Store results for performance analysis
results_power = []
results_coherent = []

for idx, N_run in enumerate(N_runs):
    signal_data = signal_matrix[:N_run, :]

    # Power averaging (top row)
    plt.sca(axes[0, idx])
    result_power = analyze_spectrum(signal_data, fs=Fs)
    axes[0, idx].set_ylim([-140, 0])
    results_power.append(result_power)

    # Coherent averaging (bottom row)
    plt.sca(axes[1, idx])
    result_coherent = analyze_spectrum(signal_data, fs=Fs, coherent_averaging=True)
    axes[1, idx].set_ylim([-140, 0])
    results_coherent.append(result_coherent)

    print(f"[{N_run:3d} Run(s)] Power Avg: ENoB=[{result_power['enob']:5.2f} b], SNR=[{result_power['snr_dbc']:6.2f} dB] | Coherent Avg: ENoB=[{result_coherent['enob']:5.2f} b], SNR=[{result_coherent['snr_dbc']:6.2f} dB]")

fig.suptitle(f'Power Spectrum Averaging vs Complex Spectrum Coherent Averaging (N_fft = {N_fft})',
             fontsize=16, fontweight='bold')

plt.tight_layout()
fig_path = output_dir / 'exp_s07_spectrum_averaging.png'
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()

# ============================================================================
# Performance Analysis: Statistical Gain
# ============================================================================
print("=" * 80)
print("PERFORMANCE ANALYSIS: Statistical Gain")
print("=" * 80)

# Theoretical coherent gain: 10*log10(N_runs)
snr_1run_power = results_power[0]['snr_dbc']
snr_1run_coherent = results_coherent[0]['snr_dbc']

print(f"{'Method':<20} | {'Runs':>5} | {'SNR (dB)':>8} | {'Gain (dB)':>10} | {'Theory (dB)':>11} | {'Status':<20}")
print("-" * 95)
print(f"{'Theoretical (1 run)':<20} | {1:>5} | {snr_ref:>8.2f} | {'---':>10} | {'---':>11} | {'Reference':<20}")
print(f"{'Power Average':<20} | {1:>5} | {snr_1run_power:>8.2f} | {0.00:>10.2f} | {'---':>11} | {'Baseline':<20}")

for idx, N_run in enumerate(N_runs[1:], start=1):
    snr_power = results_power[idx]['snr_dbc']
    snr_coherent = results_coherent[idx]['snr_dbc']
    gain_power = snr_power - snr_1run_power
    gain_coherent = snr_coherent - snr_1run_coherent
    theory_gain = 10 * np.log10(N_run)

    print(f"{'Power Average':<20} | {N_run:>5} | {snr_power:>8.2f} | {gain_power:>10.2f} | {'---':>11} | {'Noise floor smoother':<20}")
    print(f"{'Coherent Average':<20} | {N_run:>5} | {snr_coherent:>8.2f} | {gain_coherent:>10.2f} | {theory_gain:>11.2f} | {'True processing gain':<20}")

print("=" * 95)
print()

# ============================================================================
# Summary: Key Insights
# ============================================================================
print("=" * 80)
print("SUMMARY: Key Insights from Results")
print("=" * 80)
print("1. Power Averaging (Magnitude-only):")
print("   - SNR remains constant (~71 dB) regardless of number of runs")
print("   - Only smoothens the noise floor visually (reduces variance)")
print("   - Does NOT provide true processing gain")
print()
print("2. Coherent Averaging (Phase-aligned):")
print(f"   - SNR improves by ~{results_coherent[-1]['snr_dbc'] - snr_1run_coherent:.1f} dB for {N_runs[-1]} runs")
print(f"   - Theoretical gain: {10 * np.log10(N_runs[-1]):.1f} dB (10*log10({N_runs[-1]}))")
print(f"   - Achieves {(results_coherent[-1]['snr_dbc'] - snr_1run_coherent) / (10 * np.log10(N_runs[-1])) * 100:.1f}% of theoretical maximum")
print("   - Provides true processing gain through phase coherence")
print()
print("3. Practical Implications:")
print(f"   - To achieve {results_coherent[-1]['snr_dbc']:.0f} dB SNR with Power Averaging alone:")
print(f"     Would need to increase FFT length by ~{10**((results_coherent[-1]['snr_dbc'] - snr_1run_power)/10):.0f}x (impractical!)")
print(f"   - With Coherent Averaging: Only need {N_runs[-1]} runs (100x more efficient)")
