"""
Comprehensive Window Function Deep Dive: Three Critical Scenarios

This example demonstrates how window functions behave in three different scenarios:

1. Non-Coherent Sampling (Spectral Leakage):
   - Rectangular: ~2b ENOB (severe leakage with wide skirts)
   - Hann/Hamming: ~6b ENOB (moderate suppression)
   - Blackman: ~9.5b ENOB (good)
   - Blackman-Harris/Flat-top/Kaiser/Chebyshev: ~12b ENOB (excellent)
   Rule: Use Kaiser/Blackman-Harris for best leakage suppression

2. Coherent Sampling (No Leakage):
   - Most windows achieve ~12.5b ENOB, SFDR >103 dB
   - Signal sits perfectly in one bin - no spectral leakage
   - Rectangular/Hann/Hamming/Blackman/Blackman-Harris all perform excellently
   Rule: For coherent sampling, simpler windows (Rectangular/Hann/Hamming) work equally well

3. Short FFT (Coarse Resolution):
   - N=128, bin width 781 kHz - coarse resolution limits performance
   - Rectangular through Chebyshev: ~12.7b ENOB
   - Kaiser: SEVERE degradation (5.35b ENOB!) - 8 side bins consume excessive noise
   - Wide main lobes spread signal power across many bins
   Rule: For short FFT, avoid very wide windows (Kaiser). Use Rectangular/Hann/Hamming instead
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, amplitudes_to_snr, snr_to_nsd

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Common parameters
Fs = 100e6
A = 0.5
noise_rms = 50e-6
Fin_target = 10e6

# Window configurations with consistent side bins
WINDOW_CONFIGS = {
    'rectangular': {'description': 'Rectangular (no window)', 'side_bins': 2},
    'hann': {'description': 'Hann (raised cosine)', 'side_bins': 3},
    'hamming': {'description': 'Hamming', 'side_bins': 3},
    'blackman': {'description': 'Blackman', 'side_bins': 4},
    'blackmanharris': {'description': 'Blackman-Harris', 'side_bins': 5},
    'flattop': {'description': 'Flat-top', 'side_bins': 4},
    'kaiser': {'description': 'Kaiser (beta=38)', 'side_bins': 8},
    'chebwin': {'description': 'Chebyshev (100 dB)', 'side_bins': 4}
}

snr_ref = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)

# ============================================================================
# Scenario 1: Non-Coherent Sampling (Spectral Leakage)
# ============================================================================
print("=" * 80)
print("SCENARIO 1: NON-COHERENT SAMPLING (SPECTRAL LEAKAGE)")
print("=" * 80)

N_fft_1 = 2**13
Fin_1 = 10e6  # Non-coherent

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin_1/1e6:.2f} MHz] (non-coherent), N=[{N_fft_1}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t_1 = np.arange(N_fft_1) / Fs
signal_1 = A * np.sin(2*np.pi*Fin_1*t_1) + np.random.randn(N_fft_1) * noise_rms

n_cols = 4
n_rows = 2
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes1 = axes1.flatten()

results_1 = []
for idx, win_type in enumerate(WINDOW_CONFIGS.keys()):
    plt.sca(axes1[idx])
    result = analyze_spectrum(signal_1, fs=Fs, win_type=win_type, side_bin=WINDOW_CONFIGS[win_type]['side_bins'])
    axes1[idx].set_ylim([-140, 0])
    axes1[idx].set_title(f'{WINDOW_CONFIGS[win_type]["description"]} Window', fontsize=12, fontweight='bold')

    results_1.append({
        'window': win_type,
        'description': WINDOW_CONFIGS[win_type]['description'],
        'enob': result['enob'],
        'sndr_db': result['sndr_db'],
        'sfdr_db': result['sfdr_db'],
        'snr_db': result['snr_db'],
        'nsd_dbfs_hz': result['nsd_dbfs_hz']
    })

# Sort by ENoB (descending) and print table
results_1.sort(key=lambda x: x['enob'], reverse=True)
print(f"{'Window':<25} {'ENoB (b)':>9} {'SNDR (dB)':>10} {'SFDR (dB)':>10} {'SNR (dB)':>9} {'NSD (dBFS/Hz)':>14}")
print("-" * 78)
for r in results_1:
    print(f"{r['description']:<25} {r['enob']:>9.2f} {r['sndr_db']:>10.2f} {r['sfdr_db']:>10.2f} {r['snr_db']:>9.2f} {r['nsd_dbfs_hz']:>14.2f}")

fig1.suptitle(f'Scenario 1: Spectral Leakage - Window Comparison (Fin={Fin_1/1e6:.1f} MHz, N={N_fft_1})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig1_path = output_dir / 'exp_s08_windowing_1_leakage.png'
print(f"\n[Save fig 1/3] -> [{fig1_path}]\n")
plt.savefig(fig1_path, dpi=150)
plt.close()

# ============================================================================
# Scenario 2: Coherent Sampling (No Leakage)
# ============================================================================
print("=" * 80)
print("SCENARIO 2: COHERENT SAMPLING (NO LEAKAGE)")
print("=" * 80)

N_fft_2 = 2**13
Fin_2, Fin_bin_2 = find_coherent_frequency(Fs, Fin_target, N_fft_2)

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin_2/1e6:.6f} MHz] (coherent, Bin {Fin_bin_2}), N=[{N_fft_2}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t_2 = np.arange(N_fft_2) / Fs
signal_2 = A * np.sin(2*np.pi*Fin_2*t_2) + np.random.randn(N_fft_2) * noise_rms

fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes2 = axes2.flatten()

results_2 = []
for idx, win_type in enumerate(WINDOW_CONFIGS.keys()):
    plt.sca(axes2[idx])
    result = analyze_spectrum(signal_2, fs=Fs, win_type=win_type, side_bin=WINDOW_CONFIGS[win_type]['side_bins'])
    axes2[idx].set_ylim([-140, 0])
    axes2[idx].set_title(f'{WINDOW_CONFIGS[win_type]["description"]} Window', fontsize=12, fontweight='bold')

    results_2.append({
        'window': win_type,
        'description': WINDOW_CONFIGS[win_type]['description'],
        'enob': result['enob'],
        'sndr_db': result['sndr_db'],
        'sfdr_db': result['sfdr_db'],
        'snr_db': result['snr_db'],
        'nsd_dbfs_hz': result['nsd_dbfs_hz']
    })

# Sort by ENoB (descending) and print table
results_2.sort(key=lambda x: x['enob'], reverse=True)
print(f"{'Window':<25} {'ENoB (b)':>9} {'SNDR (dB)':>10} {'SFDR (dB)':>10} {'SNR (dB)':>9} {'NSD (dBFS/Hz)':>14}")
print("-" * 78)
for r in results_2:
    print(f"{r['description']:<25} {r['enob']:>9.2f} {r['sndr_db']:>10.2f} {r['sfdr_db']:>10.2f} {r['snr_db']:>9.2f} {r['nsd_dbfs_hz']:>14.2f}")

fig2.suptitle(f'Scenario 2: Coherent Sampling - Window Comparison (Fin={Fin_2/1e6:.6f} MHz, Bin {Fin_bin_2}, N={N_fft_2})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig2_path = output_dir / 'exp_s08_windowing_2_coherent.png'
print(f"\n[Save fig 2/3] -> [{fig2_path}]\n")
plt.savefig(fig2_path, dpi=150)
plt.close()

# ============================================================================
# Scenario 3: Short FFT (Coarse Resolution)
# ============================================================================
print("=" * 80)
print("SCENARIO 3: SHORT FFT (COARSE RESOLUTION)")
print("=" * 80)

N_fft_3 = 128
Fin_3, Fin_bin_3 = find_coherent_frequency(Fs, Fin_target, N_fft_3)

print(f"[Sinewave] Fs=[{Fs/1e6:.2f} MHz], Fin=[{Fin_3/1e6:.6f} MHz] (coherent, Bin {Fin_bin_3}), N=[{N_fft_3}], A=[{A:.3f} Vpeak]")
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]")
print(f"[Bin width] = {Fs/N_fft_3/1e3:.1f} kHz (coarse resolution)\n")

t_3 = np.arange(N_fft_3) / Fs
signal_3 = A * np.sin(2*np.pi*Fin_3*t_3) + np.random.randn(N_fft_3) * noise_rms

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes3 = axes3.flatten()

results_3 = []
for idx, win_type in enumerate(WINDOW_CONFIGS.keys()):
    plt.sca(axes3[idx])
    result = analyze_spectrum(signal_3, fs=Fs, win_type=win_type, side_bin=WINDOW_CONFIGS[win_type]['side_bins'])
    axes3[idx].set_ylim([-140, 0])
    axes3[idx].set_title(f'{WINDOW_CONFIGS[win_type]["description"]} Window', fontsize=12, fontweight='bold')

    results_3.append({
        'window': win_type,
        'description': WINDOW_CONFIGS[win_type]['description'],
        'enob': result['enob'],
        'sndr_db': result['sndr_db'],
        'sfdr_db': result['sfdr_db'],
        'snr_db': result['snr_db'],
        'nsd_dbfs_hz': result['nsd_dbfs_hz']
    })

# Sort by ENoB (descending) and print table
results_3.sort(key=lambda x: x['enob'], reverse=True)
print(f"{'Window':<25} {'ENoB (b)':>9} {'SNDR (dB)':>10} {'SFDR (dB)':>10} {'SNR (dB)':>9} {'NSD (dBFS/Hz)':>14}")
print("-" * 78)
for r in results_3:
    print(f"{r['description']:<25} {r['enob']:>9.2f} {r['sndr_db']:>10.2f} {r['sfdr_db']:>10.2f} {r['snr_db']:>9.2f} {r['nsd_dbfs_hz']:>14.2f}")

fig3.suptitle(f'Scenario 3: Short FFT - Window Comparison (Fin={Fin_3/1e6:.6f} MHz, Bin {Fin_bin_3}, N={N_fft_3})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig3_path = output_dir / 'exp_s08_windowing_3_short_fft.png'
print(f"\n[Save fig 3/3] -> [{fig3_path}]\n")
plt.savefig(fig3_path, dpi=150)
plt.close()

print("=" * 80)
print("SUMMARY: Window Function Selection Rules")
print("=" * 80)
print("1. Non-coherent sampling: Use Kaiser/Blackman-Harris for best leakage suppression")
print("2. Coherent sampling: Simpler windows (Rectangular/Hann/Hamming) work equally well")
print("3. Short FFT: Avoid very wide windows (Kaiser). Use Rectangular/Hann/Hamming")
print("=" * 80)
