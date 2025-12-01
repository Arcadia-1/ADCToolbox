"""
example_00_basic.py - Basic Sine Wave Generation and Plotting

This is a simple, self-contained example that demonstrates:
- Generating a sine wave using numpy
- Creating plots with matplotlib
- Saving figures and data to files
- Creating different data types (vector, matrix, scalar)

This example is designed for educational purposes and can be run immediately
after installing ADCToolbox via pip.

Usage:
    python example_00_basic.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib font size for better readability
plt.rcParams['font.size'] = 14


def main():
    """Generate a basic sine wave, plot it, and save outputs."""

    # 1. Setup Output Directory
    # Using relative path so it creates 'output' folder right next to this script
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================
    # Step 1: Configure sine wave parameters
    # ========================================
    N = 1024        # Number of samples
    Fs = 1e3        # Sampling frequency (Hz)
    Fin = 99        # Input signal frequency (Hz)
    A = 0.49        # Amplitude
    DC = 0.5        # DC offset

    print(f"\n[Configuration] N={N}, Fs={Fs:.0f} Hz, Fin={Fin} Hz, A={A}, DC={DC}")

    # ========================================
    # Step 2: Generate sine wave
    # ========================================
    # Create time vector (in seconds)
    t = np.arange(N) / Fs

    # Generate clean sine wave (no noise for clarity)
    sinewave = A * np.sin(2 * np.pi * Fin * t) + DC

    print(f"[Generated] Sine wave with {N} samples, Range=[{sinewave.min():.4f}, {sinewave.max():.4f}]")

    # ========================================
    # Step 3: Prepare zoomed view (first 3 periods)
    # ========================================
    period_samples = round(Fs / Fin)  # Samples per period
    n_periods = 3
    n_zoom = min(period_samples * n_periods, N)
    t_zoom = t[:n_zoom]
    sinewave_zoom = sinewave[:n_zoom]

    # ========================================
    # Step 4: Create two-panel plot
    # ========================================
    plt.figure(figsize=(10, 8))

    # Panel 1: Full waveform
    plt.subplot(2, 1, 1)
    plt.plot(t * 1e3, sinewave, 'b-', linewidth=2)
    plt.xlim([0, max(t) * 1e3])
    plt.ylim([min(sinewave) - 0.1, max(sinewave) + 0.1])
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title(f'Full Sine Wave (Fin={Fin} Hz, Fs={int(Fs)} Hz, N={N})')
    plt.grid(True)

    # Panel 2: Zoomed view (first 3 periods)
    plt.subplot(2, 1, 2)
    plt.plot(t_zoom * 1e3, sinewave_zoom, '-o', linewidth=2, markersize=4)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title(f'Zoomed View (First {n_periods} Periods)')
    plt.ylim([min(sinewave_zoom) - 0.1, max(sinewave_zoom) + 0.1])
    plt.grid(True)

    plt.tight_layout()

    # ========================================
    # Step 5: Save figure
    # ========================================
    fig_path = output_dir / "exp00_sinewave_basic_python.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [save figure] -> [{fig_path.name}]")

    # ========================================
    # Step 6: Save data to CSV files
    # ========================================
    
    # 1. Save Vector (The sine wave itself)
    csv_vec = output_dir / "exp00_sinewave_python.csv"
    np.savetxt(csv_vec, sinewave, delimiter=',', fmt='%.16f')
    print(f"  [save vector] -> [{csv_vec.name}]")

    # 2. Save Matrix (Reshape to 4x256, Fortran order to match MATLAB)
    # Using order='F' ensures the data layout matches MATLAB's column-major reshape
    test_matrix = sinewave.reshape((4, int(N/4)), order='F')
    csv_mat = output_dir / "exp00_test_matrix_python.csv"
    np.savetxt(csv_mat, test_matrix, delimiter=',', fmt='%.16f')
    print(f"  [save matrix] -> [{csv_mat.name}]")

    # 3. Save Scalar (Mean value)
    # Note: scalars must be wrapped in [] for savetxt
    test_scalar = np.mean(sinewave)
    csv_scalar = output_dir / "exp00_test_scalar_python.csv"
    np.savetxt(csv_scalar, [test_scalar], delimiter=',', fmt='%.16f')
    print(f"  [save scalar] -> [{csv_scalar.name}]")

    print("\n[DONE] Example 00 completed successfully! Please check the figures.\n")

if __name__ == "__main__":
    main()