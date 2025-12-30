"""
Sweeping FFT Length and OSR: Comprehensive Parameter Study

This example demonstrates two key spectrum analysis parameters:

1. FFT Length (N) - Top Row:
   - Controls frequency resolution (bin width = Fs/N)
   - Affects noise floor per bin (NSD decreases with √N)
   - Longer FFT → finer resolution, lower noise floor per bin
   - Trade-off: Computation time and memory

2. Oversampling Ratio (OSR) - Bottom Row:
   - Focuses analysis on narrow bandwidth around signal
   - SNR improvement: +10*log10(OSR) dB
   - Filters out-of-band noise
   - Essential for Delta-Sigma ADCs and narrowband applications

Key Insights:
- FFT length: Improves frequency resolution without changing SNR
- OSR: Directly improves SNR by narrowing analysis bandwidth
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_spectrum, find_coherent_frequency, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Common parameters
Fs = 100e6
A = 0.5
noise_rms = 150e-6

# Theoretical SNR and NSD
snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# ============================================================================
# Scenario 1: FFT Length Sweep
# ============================================================================
print("=" * 80)
print("SCENARIO 1: FFT LENGTH SWEEP (N = 2^7 to 2^16)")
print("=" * 80)

Fin_target_fft = 12e6
N_values = [2**7, 2**10, 2**13, 2**16]

# Create 2x4 figure
n_cols = 4
fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 6, 10))

for idx, N_fft in enumerate(N_values):
    # Calculate coherent frequency for this N
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target_fft, n_fft=N_fft)

    # Generate signal
    t = np.arange(N_fft) / Fs
    signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N_fft) * noise_rms

    # Analyze spectrum
    plt.sca(axes[0, idx])
    result = analyze_spectrum(signal, fs=Fs)
    axes[0, idx].set_ylim([-140, 0])

    bin_width = Fs / N_fft
    axes[0, idx].set_title(f"N = {N_fft} (Bin: {bin_width/1e3:.2f} kHz)", fontsize=12, fontweight='bold')

    print(f"[N={N_fft:8d} (2^{int(np.log2(N_fft)):2d})] [Bin = {bin_width/1e3:8.3f} kHz] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_dbc']:6.2f} dB], SNR=[{result['snr_dbc']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz]")

print()

# ============================================================================
# Scenario 2: OSR Sweep
# ============================================================================
print("=" * 80)
print("SCENARIO 2: OVERSAMPLING RATIO SWEEP (OSR = 1, 2, 4, 10)")
print("=" * 80)

# Use large FFT for OSR demo
N_fft_osr = 2**16
Fin_target_osr = 0.1e6
Fin_osr, Fin_bin_osr = find_coherent_frequency(fs=Fs, fin_target=Fin_target_osr, n_fft=N_fft_osr)

print(f"[Sinewave] Fin=[{Fin_osr/1e6:.6f} MHz] (coherent, Bin {Fin_bin_osr}), N=[{N_fft_osr}]")

# Generate signal once for OSR sweep
t_osr = np.arange(N_fft_osr) / Fs
signal_osr = A * np.sin(2*np.pi*Fin_osr*t_osr) + np.random.randn(N_fft_osr) * noise_rms

osr_values = [1, 2, 4, 10]
snr_baseline = None

for idx, osr in enumerate(osr_values):
    plt.sca(axes[1, idx])
    result = analyze_spectrum(signal_osr, fs=Fs, osr=osr)
    if idx == 0:
        snr_baseline = result['snr_dbc']

    axes[1, idx].set_ylim([-140, 0])

    snr_improvement = result['snr_dbc'] - snr_baseline
    theory_improvement = 10 * np.log10(osr)

    axes[1, idx].set_title(f"OSR = {osr} (SNR +{snr_improvement:.1f} dB)", fontsize=12, fontweight='bold')

    print(f"[OSR={osr:3d}] ENoB=[{result['enob']:5.2f} b], SNDR=[{result['sndr_dbc']:6.2f} dB], SNR=[{result['snr_dbc']:6.2f} dB], NSD=[{result['nsd_dbfs_hz']:7.2f} dBFS/Hz], Gain=[+{snr_improvement:.1f} dB] (Theory: +{theory_improvement:.1f} dB)")

# Add row labels
fig.text(0.01, 0.75, 'FFT Length Sweep', va='center', rotation='vertical',
         fontsize=14, fontweight='bold')
fig.text(0.01, 0.25, 'OSR Sweep', va='center', rotation='vertical',
         fontsize=14, fontweight='bold')

fig.suptitle('Sweeping FFT Length and Oversampling Ratio (OSR)', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0.02, 0, 1, 0.98])

fig_path = output_dir / 'exp_s06_sweeping_fft_and_osr.png'
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()

# ============================================================================
# Summary: Performance Analysis
# ============================================================================
print("=" * 80)
print("SUMMARY: FFT Length vs OSR")
print("=" * 80)
print("1. FFT Length (N):")
print("   - Increases frequency resolution (bin width = Fs/N)")
print("   - Lowers noise floor per bin (NSD improves)")
print("   - Does NOT improve SNR (noise bandwidth unchanged)")
print("   - Use case: Resolve closely spaced frequency components")
print()
print("2. Oversampling Ratio (OSR):")
print(f"   - Directly improves SNR: Measured gain = +{snr_improvement:.1f} dB for OSR={osr_values[-1]}")
print(f"   - Theoretical gain: +{10*np.log10(osr_values[-1]):.1f} dB (10*log10({osr_values[-1]}))")
print("   - Narrows analysis bandwidth to Fs/(2*OSR)")
print("   - Use case: Delta-Sigma ADCs, narrowband signal analysis")
print()
print("3. Practical Guidelines:")
print("   - For resolving harmonics: Increase FFT length")
print("   - For improving SNR: Increase OSR (if signal is narrowband)")
print("   - For best results: Combine both (large N + appropriate OSR)")
