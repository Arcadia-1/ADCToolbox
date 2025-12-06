"""Phase spectrum analysis: noise, jitter, harmonic distortion, kickback"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, analyze_spectrum_phase

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
J = find_bin(Fs, Fin_target, N)
Fin = J * Fs / N
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

# Signal 1: Noise
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms

# Signal 2: Jitter
jitter_rms = 1.3e-12
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise

# Signal 3: Harmonic distortion
hd2_dB, hd3_dB = -80, -73
hd2, hd3 = 10**(hd2_dB/20), 10**(hd3_dB/20)
signal_harmonic = A * np.sin(2*np.pi*Fin*t) + DC + hd2 * np.sin(2*2*np.pi*Fin*t) + hd3 * np.sin(3*2*np.pi*Fin*t) + np.random.randn(N) * base_noise

# Signal 4: Kickback
t_ext = np.arange(N+1) / Fs
sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**12) / 2**12
msb_shifted = msb_ext[:-1]
msb = msb_ext[1:]
lsb = lsb_ext[1:]
kickback_strength = 0.009
signal_kickback = msb + lsb + kickback_strength * msb_shifted

signals = [signal_noise, signal_jitter, signal_harmonic, signal_kickback]
titles = ['Noise', 'Jitter', 'Harmonic Distortion', 'Kickback']
params = [f'RMS = {noise_rms*1e3:.2f} mV',
          f'{jitter_rms*1e12:.1f} ps',
          f'HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB',
          f'strength = {kickback_strength}']

print(f"[Phase Spectrum Analysis] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]")

# Create 2x2 figure for comparison
fig = plt.figure(figsize=(14, 10))

for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    # Create individual subplot with polar projection
    ax = fig.add_subplot(2, 2, i+1, projection='polar')

    # Generate phase spectrum plot
    save_path_individual = output_dir / f'exp_a09_phase_{title.lower().replace(" ", "_")}.png'
    result = analyze_phase_spectrum(signal, harmonic=5, mode='FFT', save_path=str(save_path_individual), show_plot=False)

    # Get spectrum and bin
    spec = result['spec']
    bin_idx = result['bin']

    # Plot polar spectrum
    freq_bins = np.arange(len(spec))
    mag = np.abs(spec)
    phase = np.angle(spec)

    # Plot all points
    ax.scatter(phase, 20*np.log10(mag + 1e-10), s=1, c='gray', alpha=0.3)

    # Highlight harmonics
    for h in range(1, 6):
        h_bin = (bin_idx * h) % len(spec)
        if h_bin < len(spec):
            ax.scatter(phase[h_bin], 20*np.log10(mag[h_bin] + 1e-10),
                      s=100, marker='o', edgecolors='red', linewidths=2, facecolors='none',
                      label=f'H{h}' if h <= 3 else '')

    ax.set_title(f'{title}\n{param}', fontsize=11, fontweight='bold', pad=20)
    ax.set_ylim([-120, 0])
    if i == 0:
        ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    print(f"  {title:20s} - Bin: {bin_idx}, Harmonics marked: 5")

plt.tight_layout()
fig_path = output_dir / f'exp_a09_spec_plot_phase_fin_{int(Fin/1e6)}M.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")
