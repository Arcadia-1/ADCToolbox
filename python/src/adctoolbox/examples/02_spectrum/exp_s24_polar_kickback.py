"""
Polar phase spectrum analysis: MSB-dependent kickback distortion.
Demonstrates kickback effect where MSB transition from previous sample affects current sample,
creating a characteristic distortion pattern visible in the polar phase plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, calculate_snr_from_amplitude, snr_to_nsd, analyze_spectrum_polar

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
A, DC = 0.49, 0.5
base_noise = 50e-6

snr_ref = calculate_snr_from_amplitude(sig_amplitude=A, noise_amplitude=base_noise)
nsd_ref = snr_to_nsd(snr_ref, fs=Fs, osr=1)
print(f"[Sinewave] Fs=[{Fs/1e6:.0f} MHz], Fin=[{Fin/1e6:.1f} MHz] (coherent, Bin {J}), N=[{N}], A=[{A:.3f} Vpeak]")
print(f"[Base Noise] RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_ref:.2f} dB], Theoretical NSD=[{nsd_ref:.2f} dBFS/Hz]\n")

# Generate kickback signals with different strengths
t_ext = np.arange(N+1) / Fs
sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**12) / 2**12
msb_shifted = msb_ext[:-1]
msb = msb_ext[1:]
lsb = lsb_ext[1:]

kickback_strengths = [0.005, 0.009]
signals = []
titles = []
params = []

for kb_strength in kickback_strengths:
    signal_kb = msb + lsb + kb_strength * msb_shifted
    signals.append(signal_kb)
    titles.append(f'Kickback (strength={kb_strength})')
    params.append(f'strength = {kb_strength}')

# Create 1x2 figure for comparison
fig = plt.figure(figsize=(14, 6))

# Store axes and their limits for restoration after tight_layout
axes_info = []

for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    # Create individual subplot with polar projection
    ax = fig.add_subplot(1, 2, i+1, projection='polar')

    # Analyze spectrum with polar phase visualization
    coherent_result = analyze_spectrum_polar(
        signal,
        fs=Fs,
        harmonic=5,
        win_type='boxcar',
        show_plot=False,
        ax=ax,
        fixed_radial_range=120
    )

    # Set title outside
    ax.set_title(f'{title}\n{param}', pad=20, fontsize=14, fontweight='bold')

    # Store axis and its ylim for later restoration
    axes_info.append((ax, ax.get_ylim()))

    metrics = coherent_result.get('metrics', {})
    snr_measured = metrics.get('snr_db', 0)
    print(f"[{title:30s}] SNR=[{snr_measured:6.2f} dB], Bin=[{coherent_result['bin_idx']}], Noise floor=[{coherent_result['minR_dB']:7.2f} dB]")

plt.tight_layout()

# Restore ylim after tight_layout (which resets polar axis limits)
for ax, ylim in axes_info:
    ax.set_ylim(ylim)

fig_path = output_dir / 'exp_s24_polar_kickback.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")
