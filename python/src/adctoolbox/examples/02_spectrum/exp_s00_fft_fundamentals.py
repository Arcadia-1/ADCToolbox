"""Demonstrate the raw principles of the Discrete Fourier Transform (FFT) without windows.
Specifically addresses the mathematical differences between EVEN (16-point) and ODD (15-point) FFTs.

Concepts covered:
1. Two-sided FFT magnitude and symmetry.
2. Why sine waves split their amplitudes across positive and negative frequencies.
3. Why DC (Bin 0) and exactly-Nyquist (Bin N/2, even only) do NOT split, and thus are NOT multiplied by 2.
4. How an odd N (15-point) FFT has no exact Nyquist bin and thus ALL internal AC bins are multiplied by 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

def run_fft_analysis(N, axes_row):
    Fs = float(N)  # Set Fs = N so that bin k corresponds exactly to k Hz. (1 Hz resolution)
    t = np.arange(N) / Fs

    # Define signal components
    A_dc = 1.0
    A_mid = 2.0
    A_high = 1.5

    k_mid = 2
    
    if N % 2 == 0:
        k_high = N // 2  # Exact Nyquist bin
        high_desc = f"Nyquist({k_high}Hz)"
    else:
        k_high = N // 2  # Highest available positive integer bin (closest to Nyquist)
        high_desc = f"HighFreq({k_high}Hz)"
        
    # Generate the time-domain signal
    # A cosine at exact Nyquist (if N is even) becomes exactly: [1, -1, 1, -1, ...]
    signal = A_dc + A_mid * np.sin(2 * np.pi * k_mid * t) + A_high * np.cos(2 * np.pi * k_high * t)

    # Compute raw FFT
    fft_raw = np.fft.fft(signal)
    fft_mag = np.abs(fft_raw)

    print("=" * 70)
    print(f"--- ANALYZING N={N} POINTS ---")
    print("=" * 70)
    print(f"Signal components: DC={A_dc}V, Mid({k_mid}Hz)={A_mid}V, {high_desc}={A_high}V")
    print(f"{'Bin':<5} | {'Freq (Hz)':<10} | {'Raw Mag':<10}")
    print("-" * 50)
    for k in range(N):
        freq = k * (Fs / N)
        print(f"{k:<5} | {freq:<10.1f} | {fft_mag[k]:<10.2f}")

    # Form Single-Sided Amplitude Spectrum (0 to Fs/2)
    # 1. Divide all raw bins by N
    # 2. Multiply AC bins by 2 (because their energy was split between positive and negative frequencies)
    # 3. NEVER multiply Bin 0 (DC) by 2.
    # 4. NEVER multiply Bin N/2 (Nyquist) by 2 (if N is even, it exists and doesn't split).
    
    num_single_sided_bins = N // 2 + 1
    single_sided_mag = np.zeros(num_single_sided_bins)
    
    for k in range(num_single_sided_bins):
        mag_normalized = fft_mag[k] / N
        
        # Check if this bin is DC or EXACT Nyquist
        is_dc = (k == 0)
        is_nyquist = (N % 2 == 0 and k == N // 2)
        
        if is_dc or is_nyquist:
            single_sided_mag[k] = mag_normalized      # Energy did not split
        else:
            single_sided_mag[k] = mag_normalized * 2  # Energy split, multiply by 2 to recover
            
    print(f"\n{'-'*50}\nRECONSTRUCTED SINGLE-SIDED AMPLITUDE (0 to Fs/2)\n{'-'*50}")
    print("Rules applied:")
    print("- Bin 0 (DC): (Raw / N)")
    if N % 2 == 0:
        print(f"- Bin {N//2} (Nyquist): (Raw / N)  <-- No split!")
        print(f"- Bin 1 to {N//2 - 1}: (Raw / N) * 2")
    else:
        print("- NO exact Nyquist bin exists.")
        print(f"- Bin 1 to {N//2}: (Raw / N) * 2  <-- ALL AC bins multiplied by 2!")

    print(f"\n{'Bin':<5} | {'Freq (Hz)':<10} | {'Reconstructed Amplitude (V)'}")
    print("-" * 50)
    for k in range(num_single_sided_bins):
        freq = k * (Fs / N)
        print(f"{k:<5} | {freq:<10.1f} | {single_sided_mag[k]:.3f} V")
    print("\n")

    # Plots
    # Subplot 1: Time domain
    axes_row[0].plot(t, signal, 'ko-', label=f'N={N}')
    axes_row[0].set_title(f'Time Domain (N={N})')
    axes_row[0].set_xlabel('Time (s)')
    axes_row[0].set_ylabel('Amplitude (V)')
    axes_row[0].grid(True)
    axes_row[0].legend()

    # Subplot 2: Two-sided Fast Fourier Transform
    k_all = np.arange(N)
    freq_all = k_all * (Fs / N)
    m, s, b = axes_row[1].stem(freq_all, fft_mag, basefmt=" ")
    plt.setp(s, 'linewidth', 2)
    plt.setp(m, 'markersize', 6)
    axes_row[1].set_title(f"Raw 2-Sided FFT (N={N})\nNotice symmetries")
    axes_row[1].set_xlabel('Frequency (Hz)')
    axes_row[1].set_ylabel('Raw Magnitude')
    axes_row[1].grid(True)
    axes_row[1].set_xticks(freq_all)

    # Subplot 3: Single-sided True Amplitude
    k_half = np.arange(num_single_sided_bins)
    freq_half = k_half * (Fs / N)
    m, s, b = axes_row[2].stem(freq_half, single_sided_mag, basefmt=" ")
    plt.setp(s, 'linewidth', 2)
    plt.setp(m, 'markersize', 6)
    
    if N % 2 == 0:
        desc = "DC and Nyquist perfectly preserved (NO factor of 2)"
    else:
        desc = "NO Nyquist bin! ALL AC bins (k>0) multiplied by 2"
        
    axes_row[2].set_title(f"Single-Sided True Amplitude (N={N})\n{desc}")
    axes_row[2].set_xlabel('Frequency (Hz)')
    axes_row[2].set_ylabel('True Amplitude (V)')
    axes_row[2].grid(True)
    axes_row[2].set_xticks(freq_half)

# Run for both Even and Odd
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

run_fft_analysis(16, axes[0, :])
run_fft_analysis(15, axes[1, :])

plt.tight_layout()
out_path = output_dir / "exp_s00_fft_fundamentals.png"
plt.savefig(out_path, dpi=120)
print(f"[Plot saved] -> {out_path}")
plt.close()
