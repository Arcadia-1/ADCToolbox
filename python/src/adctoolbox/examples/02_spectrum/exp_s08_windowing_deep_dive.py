"""Window function comparison across three scenarios: non-coherent sampling (spectral leakage),
coherent sampling (no leakage), and short FFT (coarse resolution). Demonstrates automatic
side_bin selection and window performance trade-offs.
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
DC = 0.5
noise_rms = 50e-6
Fin_target = 10e6

# Harmonic distortion parameters
hd2_dB = -100
hd3_dB = -100
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = -hd3_amp / (A**2 / 4)  # Negative k3

# Window configurations (side_bin automatically selected based on window type and coherence)
WINDOW_CONFIGS = {
    'rectangular': {'description': 'Rectangular (no window)'},
    'hann': {'description': 'Hann (raised cosine)'},
    'hamming': {'description': 'Hamming'},
    'blackman': {'description': 'Blackman'},
    'blackmanharris': {'description': 'Blackman-Harris'},
    'flattop': {'description': 'Flat-top'},
    'kaiser': {'description': 'Kaiser (beta=38)'},
    'chebwin': {'description': 'Chebyshev (100 dB)'}
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
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], HD2={hd2_dB}dB, HD3={hd3_dB}dB")
print(f"[Theoretical] SNR=[{snr_ref:.2f} dB], NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t_1 = np.arange(N_fft_1) / Fs
sig_ideal_1 = A * np.sin(2*np.pi*Fin_1*t_1)
signal_1 = sig_ideal_1 + k2 * sig_ideal_1**2 + k3 * sig_ideal_1**3 + DC + np.random.randn(N_fft_1) * noise_rms

n_cols = 4
n_rows = 2
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes1 = axes1.flatten()

results_1 = []
for idx, win_type in enumerate(WINDOW_CONFIGS.keys()):
    plt.sca(axes1[idx])
    result = analyze_spectrum(signal_1, fs=Fs, win_type=win_type)
    axes1[idx].set_ylim([-140, 0])
    axes1[idx].set_title(f'{WINDOW_CONFIGS[win_type]["description"]} Window', fontsize=12, fontweight='bold')

    results_1.append({
        'window': win_type,
        'description': WINDOW_CONFIGS[win_type]['description'],
        'enob': result['enob'],
        'sndr_dbc': result['sndr_dbc'],
        'sfdr_dbc': result['sfdr_dbc'],
        'snr_dbc': result['snr_dbc'],
        'nsd_dbfs_hz': result['nsd_dbfs_hz']
    })

# Sort by ENoB (descending) and print table
results_1.sort(key=lambda x: x['enob'], reverse=True)
print(f"{'Window':<25} {'ENoB (b)':>9} {'SNDR (dB)':>10} {'SFDR (dB)':>10} {'SNR (dB)':>9} {'NSD (dBFS/Hz)':>14}")
print("-" * 78)
for r in results_1:
    print(f"{r['description']:<25} {r['enob']:>9.2f} {r['sndr_dbc']:>10.2f} {r['sfdr_dbc']:>10.2f} {r['snr_dbc']:>9.2f} {r['nsd_dbfs_hz']:>14.2f}")

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
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], HD2={hd2_dB}dB, HD3={hd3_dB}dB")
print(f"[Theoretical] SNR=[{snr_ref:.2f} dB], NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

t_2 = np.arange(N_fft_2) / Fs
sig_ideal_2 = A * np.sin(2*np.pi*Fin_2*t_2)
signal_2 = sig_ideal_2 + k2 * sig_ideal_2**2 + k3 * sig_ideal_2**3 + DC + np.random.randn(N_fft_2) * noise_rms

fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes2 = axes2.flatten()

results_2 = []
for idx, win_type in enumerate(WINDOW_CONFIGS.keys()):
    plt.sca(axes2[idx])
    result = analyze_spectrum(signal_2, fs=Fs, win_type=win_type)
    axes2[idx].set_ylim([-140, 0])
    axes2[idx].set_title(f'{WINDOW_CONFIGS[win_type]["description"]} Window', fontsize=12, fontweight='bold')

    results_2.append({
        'window': win_type,
        'description': WINDOW_CONFIGS[win_type]['description'],
        'enob': result['enob'],
        'sndr_dbc': result['sndr_dbc'],
        'sfdr_dbc': result['sfdr_dbc'],
        'snr_dbc': result['snr_dbc'],
        'nsd_dbfs_hz': result['nsd_dbfs_hz']
    })

# Sort by ENoB (descending) and print table
results_2.sort(key=lambda x: x['enob'], reverse=True)
print(f"{'Window':<25} {'ENoB (b)':>9} {'SNDR (dB)':>10} {'SFDR (dB)':>10} {'SNR (dB)':>9} {'NSD (dBFS/Hz)':>14}")
print("-" * 78)
for r in results_2:
    print(f"{r['description']:<25} {r['enob']:>9.2f} {r['sndr_dbc']:>10.2f} {r['sfdr_dbc']:>10.2f} {r['snr_dbc']:>9.2f} {r['nsd_dbfs_hz']:>14.2f}")

fig2.suptitle(f'Scenario 2: Coherent Sampling - Window Comparison (Fin={Fin_2/1e6:.6f} MHz, Bin {Fin_bin_2}, N={N_fft_2})', fontsize=14, fontweight='bold')
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
print(f"[Nonideal] Noise RMS=[{noise_rms*1e6:.2f} uVrms], HD2={hd2_dB}dB, HD3={hd3_dB}dB")
print(f"[Theoretical] SNR=[{snr_ref:.2f} dB], NSD=[{nsd_ref:.2f} dBFS/Hz]")
print(f"[Bin width] = {Fs/N_fft_3/1e3:.1f} kHz (coarse resolution)\n")

t_3 = np.arange(N_fft_3) / Fs
sig_ideal_3 = A * np.sin(2*np.pi*Fin_3*t_3)
signal_3 = sig_ideal_3 + k2 * sig_ideal_3**2 + k3 * sig_ideal_3**3 + DC + np.random.randn(N_fft_3) * noise_rms

fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
axes3 = axes3.flatten()

results_3 = []
for idx, win_type in enumerate(WINDOW_CONFIGS.keys()):
    plt.sca(axes3[idx])
    result = analyze_spectrum(signal_3, fs=Fs, win_type=win_type)
    axes3[idx].set_ylim([-140, 0])
    axes3[idx].set_title(f'{WINDOW_CONFIGS[win_type]["description"]} Window', fontsize=12, fontweight='bold')

    results_3.append({
        'window': win_type,
        'description': WINDOW_CONFIGS[win_type]['description'],
        'enob': result['enob'],
        'sndr_dbc': result['sndr_dbc'],
        'sfdr_dbc': result['sfdr_dbc'],
        'snr_dbc': result['snr_dbc'],
        'nsd_dbfs_hz': result['nsd_dbfs_hz']
    })

# Sort by ENoB (descending) and print table
results_3.sort(key=lambda x: x['enob'], reverse=True)
print(f"{'Window':<25} {'ENoB (b)':>9} {'SNDR (dB)':>10} {'SFDR (dB)':>10} {'SNR (dB)':>9} {'NSD (dBFS/Hz)':>14}")
print("-" * 78)
for r in results_3:
    print(f"{r['description']:<25} {r['enob']:>9.2f} {r['sndr_dbc']:>10.2f} {r['sfdr_dbc']:>10.2f} {r['snr_dbc']:>9.2f} {r['nsd_dbfs_hz']:>14.2f}")

fig3.suptitle(f'Scenario 3: Short FFT - Window Comparison (Fin={Fin_3/1e6:.6f} MHz, Bin {Fin_bin_3}, N={N_fft_3})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig3_path = output_dir / 'exp_s08_windowing_3_short_fft.png'
print(f"\n[Save fig 3/3] -> [{fig3_path}]\n")
plt.savefig(fig3_path, dpi=150)
plt.close()
