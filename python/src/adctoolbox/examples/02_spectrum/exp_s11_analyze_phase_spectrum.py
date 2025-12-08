"""Phase spectrum analysis: noise, jitter, harmonic distortion, kickback

Demonstrates different ADC impairments using modular phase spectrum analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import calc_coherent_freq
from adctoolbox.aout.calculate_coherent_spectrum import calculate_coherent_spectrum
from adctoolbox.aout.plot_polar_phase import plot_polar_phase

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = calc_coherent_freq(Fs, Fin_target, N)
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
print("\n[Modular Structure Used]")
print("  1. calculate_coherent_spectrum() - Compute phase-aligned spectrum")
print("  2. plot_polar_phase() - Visualize the results")

# Create 2x2 figure for comparison
fig = plt.figure(figsize=(14, 10))

for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    # Create individual subplot with polar projection
    ax = fig.add_subplot(2, 2, i+1, projection='polar')

    # Step 1: Calculate coherent spectrum (pure computation)
    coherent_result = calculate_coherent_spectrum(
        signal,
        fs=Fs,
        osr=1,
        win_type='boxcar'
    )

    # Step 2: Prepare plot data
    plot_data = {
        'complex_spec_coherent': coherent_result['complex_spec_coherent'],
        'minR_dB': coherent_result['minR_dB'],
        'bin_idx': coherent_result['bin_idx'],
        'N_fft': coherent_result['n_fft']
    }

    # Step 3: Plot using pure visualization function
    plot_polar_phase(plot_data, harmonic=5, ax=ax)

    # Customize title and parameters
    ax.set_title(f'{title}\n{param}', fontsize=11, fontweight='bold', pad=20)

    # Print progress
    print(f"  {title:20s} - Bin: {coherent_result['bin_idx']}, Noise floor: {coherent_result['minR_dB']:.1f} dB")

# Adjust legend position
handles, labels = ax.get_legend_handles_labels()
if handles:
    fig.legend(handles[:3], labels[:3], loc='upper right', fontsize=9, bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
fig_path = output_dir / 'exp_s11_analyze_phase_spectrum.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")

# Also save individual plots for better visibility
for i, (signal, title) in enumerate(zip(signals, titles)):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    # Calculate and plot
    coherent_result = calculate_coherent_spectrum(signal, fs=Fs, win_type='boxcar')
    plot_data = {
        'complex_spec_coherent': coherent_result['complex_spec_coherent'],
        'minR_dB': coherent_result['minR_dB'],
        'bin_idx': coherent_result['bin_idx'],
        'N_fft': coherent_result['n_fft']
    }
    plot_polar_phase(plot_data, harmonic=5, ax=ax)

    ax.set_title(f'Phase Spectrum - {title}', fontsize=12, fontweight='bold', pad=20)

    # Save individual plot
    individual_path = output_dir / f'exp_s11_phase_{title.lower().replace(" ", "_")}.png'
    plt.savefig(individual_path, dpi=150, bbox_inches='tight')
    plt.close()