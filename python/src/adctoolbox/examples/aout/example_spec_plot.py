"""
Example: spec_plot - Frequency Spectrum Analysis

This example demonstrates how to use spec_plot to analyze the frequency spectrum
of an ADC output signal and calculate key performance metrics.

Metrics calculated:
- ENoB (Effective Number of Bits)
- SNDR (Signal-to-Noise and Distortion Ratio)
- SFDR (Spurious-Free Dynamic Range)
- SNR (Signal-to-Noise Ratio)
- THD (Total Harmonic Distortion)
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox.aout import spec_plot
from adctoolbox.common import find_bin

# Create output directory
import os
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("Example: spec_plot - Frequency Spectrum Analysis")
print("=" * 60)

#%% Generate Test Signal
# Create a sinewave with some harmonic distortion and noise

N = 2**13  # Number of samples (8192)
Fs = 1.0   # Normalized sampling frequency
Fin_norm = 0.0789  # Normalized input frequency (Fin/Fs)

# Find coherent frequency bin
J = find_bin(Fs, Fin_norm, N)
Fin = J / N * Fs  # Actual coherent frequency

print(f"\nTest Signal Parameters:")
print(f"  Samples (N): {N}")
print(f"  Frequency bin (J): {J}")
print(f"  Normalized frequency: {Fin:.6f}")

# Generate ideal sinewave
t = np.arange(N)
A = 0.49  # Amplitude (avoiding full scale)
DC = 0.5  # DC offset

ideal_signal = A * np.sin(2 * np.pi * Fin * t) + DC

# Add harmonic distortion (HD2, HD3)
HD2_dB = -60  # 2nd harmonic at -60dB
HD3_dB = -70  # 3rd harmonic at -70dB

HD2_amp = 10**(HD2_dB / 20) * A
HD3_amp = 10**(HD3_dB / 20) * A

signal_with_distortion = (ideal_signal +
                          HD2_amp * np.sin(2 * 2 * np.pi * Fin * t) +
                          HD3_amp * np.sin(3 * 2 * np.pi * Fin * t))

# Add white noise
noise_level = 1e-4  # Noise RMS
signal = signal_with_distortion + np.random.randn(N) * noise_level

print(f"  HD2: {HD2_dB} dB")
print(f"  HD3: {HD3_dB} dB")
print(f"  Noise RMS: {noise_level}")

#%% Example 1: Basic Spectrum Plot with Labels
print("\n" + "=" * 60)
print("Example 1: Basic Spectrum Plot with Labels")
print("=" * 60)

fig1 = plt.figure(figsize=(12, 8))
enob, sndr, sfdr, snr, thd, pwr, nf, h = spec_plot(
    signal,
    label=True,          # Show metric labels on plot
    harmonic=5,          # Show first 5 harmonics
    osr=1,               # Oversampling ratio (1 = Nyquist)
    win_type='hamming'   # Window type
)

plt.title('Spectrum Plot with Labels')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spec_plot_basic.png'), dpi=150)
plt.close()

print(f"\nPerformance Metrics:")
print(f"  ENoB:  {enob:.2f} bits")
print(f"  SNDR:  {sndr:.2f} dB")
print(f"  SFDR:  {sfdr:.2f} dB")
print(f"  SNR:   {snr:.2f} dB")
print(f"  THD:   {thd:.2f} dB")
print(f"  Signal Power: {pwr:.2f} dB")
print(f"  Noise Floor: {nf:.2f} dB")

#%% Example 2: Spectrum Plot without Labels
print("\n" + "=" * 60)
print("Example 2: Spectrum Plot without Labels (Clean View)")
print("=" * 60)

fig2 = plt.figure(figsize=(12, 8))
enob2, sndr2, sfdr2, snr2, thd2, _, _, _ = spec_plot(
    signal,
    label=False,         # No labels on plot
    harmonic=10,         # Show first 10 harmonics
    osr=1,
    win_type='hamming'
)

plt.title('Spectrum Plot - Clean View')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spec_plot_clean.png'), dpi=150)
plt.close()

print(f"Saved figure: spec_plot_clean.png")

#%% Example 3: Different Window Functions
print("\n" + "=" * 60)
print("Example 3: Comparing Window Functions")
print("=" * 60)

window_types = ['hamming', 'hann', 'blackman', 'flattop']

fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, win_type in enumerate(window_types):
    plt.sca(axes[idx])
    enob_w, sndr_w, sfdr_w, _, _, _, _, _ = spec_plot(
        signal,
        label=True,
        harmonic=5,
        osr=1,
        win_type=win_type
    )
    axes[idx].set_title(f'{win_type.capitalize()} Window\nENoB: {enob_w:.2f}, SNDR: {sndr_w:.2f} dB')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spec_plot_windows.png'), dpi=150)
plt.close()

print(f"\nWindow Function Comparison:")
for win_type in window_types:
    enob_w, sndr_w, sfdr_w, _, _, _, _, _ = spec_plot(signal, label=False,
                                                       harmonic=5, osr=1,
                                                       win_type=win_type)
    print(f"  {win_type:10s}: ENoB={enob_w:.2f}, SNDR={sndr_w:.2f} dB, SFDR={sfdr_w:.2f} dB")

#%% Example 4: Oversampling Analysis
print("\n" + "=" * 60)
print("Example 4: Oversampling Analysis")
print("=" * 60)

osr_values = [1, 2, 4, 8]

fig4, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, osr in enumerate(osr_values):
    plt.sca(axes[idx])
    enob_osr, sndr_osr, _, _, _, _, _, _ = spec_plot(
        signal,
        label=True,
        harmonic=5,
        osr=osr,
        win_type='hamming'
    )
    axes[idx].set_title(f'OSR = {osr}\nENoB: {enob_osr:.2f}, SNDR: {sndr_osr:.2f} dB')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'spec_plot_osr.png'), dpi=150)
plt.close()

print(f"\nOversampling Ratio Comparison:")
for osr in osr_values:
    enob_osr, sndr_osr, _, _, _, _, _, _ = spec_plot(signal, label=False,
                                                      harmonic=5, osr=osr,
                                                      win_type='hamming')
    print(f"  OSR={osr:2d}: ENoB={enob_osr:.2f}, SNDR={sndr_osr:.2f} dB (SNR gain: {sndr_osr - sndr:.2f} dB)")

#%% Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"\nGenerated figures saved to: {output_dir}/")
print("  - spec_plot_basic.png        (Basic spectrum with labels)")
print("  - spec_plot_clean.png        (Clean view without labels)")
print("  - spec_plot_windows.png      (Window function comparison)")
print("  - spec_plot_osr.png          (Oversampling analysis)")

print("\nKey Takeaways:")
print("  1. spec_plot calculates ENoB, SNDR, SFDR, SNR, THD automatically")
print("  2. Use label=True for annotated plots, label=False for clean view")
print("  3. Window functions affect spectral leakage (flattop is most accurate)")
print("  4. Oversampling (OSR>1) improves SNR by processing only in-band noise")

print("\n" + "=" * 60)
print("Example completed successfully!")
print("=" * 60)
