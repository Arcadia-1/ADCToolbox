"""Error histogram vs sine phase: noise, jitter, harmonic distortion, kickback"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, err_hist_sine

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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    row, col = i // 2, i % 2

    emean, erms, phase_deg, anoi, pnoi, err, xx = err_hist_sine(signal, bin=100, mode=0, disp=0)

    # Plot RMS error as bar chart (matching MATLAB style)
    axes[row, col].bar(phase_deg, erms, width=360/100*0.8, color='skyblue', alpha=0.7)
    axes[row, col].set_xlabel('Phase (deg)')
    axes[row, col].set_ylabel('RMS Error (V)')
    axes[row, col].set_title(f'{title}: {param}')
    axes[row, col].set_xlim([0, 360])
    axes[row, col].set_ylim([0, np.max(erms)*1.2])
    axes[row, col].grid(True, alpha=0.3)

    # Add annotations for amplitude and phase noise
    axes[row, col].text(0.05, 0.95, f'Amp Noise = {anoi/A:.2e}\nPhase Noise = {pnoi:.2e} rad',
                        transform=axes[row, col].transAxes, fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle(f'Error vs Sine Phase: 4 Non-idealities (Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = output_dir / 'exp_a04_err_hist_sine_phase.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"[Error vs Sine Phase] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]")
for title, param in zip(titles, params):
    print(f"  [{title:20s}] {param}")
print(f"[Save fig] -> [{fig_path}]")

# Generate individual full plots for each non-ideality (matching MATLAB style)
for i, (signal, title, param) in enumerate(zip(signals, titles, params)):
    emean, erms, phase_deg, anoi, pnoi, err, xx = err_hist_sine(signal, bin=100, mode=0, disp=0)

    # Create figure with 2 subplots
    fig_full = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    # Top subplot: data and error vs phase (dual y-axis)
    ax1_left = ax1
    ax1_left.plot(xx, signal, 'k.', markersize=2, label='data')
    ax1_left.set_xlim([0, 360])
    ax1_left.set_ylim([np.min(signal), np.max(signal)])
    ax1_left.set_ylabel('data', color='k')
    ax1_left.tick_params(axis='y', labelcolor='k')

    ax1_right = ax1.twinx()
    ax1_right.plot(xx, err, 'r.', markersize=2, alpha=0.5)
    ax1_right.plot(phase_deg, emean, 'b-', linewidth=2, label='error')
    ax1_right.set_xlim([0, 360])
    ax1_right.set_ylim([np.min(err), np.max(err)])
    ax1_right.set_ylabel('error', color='r')
    ax1_right.tick_params(axis='y', labelcolor='r')

    ax1.legend(['data', 'error'], loc='upper right')
    ax1.set_xlabel('phase(deg)')
    ax1.set_title('Error - Phase')
    ax1.grid(True, alpha=0.3)

    # Bottom subplot: RMS error bars with fitted curves
    bin_width = 360/100
    ax2.bar(phase_deg, erms, width=bin_width*0.8, color='skyblue', alpha=0.7)

    # Compute amplitude and phase sensitivity curves
    asen = np.abs(np.cos(phase_deg/360*2*np.pi))**2
    psen = np.abs(np.sin(phase_deg/360*2*np.pi))**2

    # Compute baseline from mean of erms^2
    erms_squared = erms**2
    valid_mask = ~np.isnan(erms)
    ermsbl = np.mean(erms_squared[valid_mask])

    # Compute fitted curves
    amp_fit = anoi**2 * asen + ermsbl
    phase_fit = pnoi**2 * psen * A**2 + ermsbl

    # Plot fitted curves
    ax2.plot(phase_deg, np.sqrt(np.maximum(amp_fit, 0)), 'b-', linewidth=2)
    ax2.plot(phase_deg, np.sqrt(np.maximum(phase_fit, 0)), 'r-', linewidth=2)
    ax2.set_xlim([0, 360])
    ax2.set_ylim([0, np.max(erms)*1.2])

    # Add text annotations
    ax2.text(10, np.max(erms)*1.15,
            f'Normalized Amplitude Noise RMS = {anoi/A:.2e}',
            color='b', fontsize=10)
    ax2.text(10, np.max(erms)*1.05,
            f'Phase Noise RMS = {pnoi:.2e} rad',
            color='r', fontsize=10)

    ax2.set_xlabel('phase(deg)')
    ax2.set_ylabel('RMS error')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save individual figure
    fig_name = title.lower().replace(' ', '_')
    fig_path_full = output_dir / f'exp_a04_err_hist_sine_phase_{fig_name}.png'
    plt.savefig(fig_path_full, dpi=150)
    plt.close()
    print(f"[Save full plot] -> [{fig_path_full}]")
