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
    python -m adctoolbox.examples.basic.example_00_basic
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib font size for better readability
plt.rcParams['font.size'] = 14


def save_csv(output_dir, data, filename):
    """Save data to CSV file."""
    filepath = output_dir / filename
    data = np.atleast_1d(data)

    # Truncate to max 1000 elements (matching MATLAB convention)
    if data.ndim == 1:
        data = data[:1000]
    elif data.shape[0] > data.shape[1]:
        data = data[:1000, :]
    else:
        data = data[:, :1000]

    fmt = '%d' if np.issubdtype(data.dtype, np.integer) else '%.16f'
    np.savetxt(filepath, data, delimiter=',', fmt=fmt)

    print(f"  [save]->[{filepath.absolute()}]")
    return filepath


def main():
    """Generate a basic sine wave, plot it, and save outputs."""

    print("\n" + "="*60)
    print("ADCToolbox - Basic Sine Wave Example")
    print("="*60)

    # Create output directory for this example
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
    fig_path = output_dir / "sinewave_basic_python.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  [save]->[{fig_path.absolute()}]")

    # ========================================
    # Step 6: Save data to CSV files
    # ========================================
    # Save vector (1D array - the sine wave)
    save_csv(output_dir, sinewave, 'sinewave_python.csv')

    # Save matrix (2D array - reshape sine wave into 4x256 matrix)
    # Using Fortran order ('F') to match MATLAB column-major ordering
    test_matrix = sinewave.reshape((4, int(N/4)), order='F')
    save_csv(output_dir, test_matrix, 'test_matrix_python.csv')

    # Save scalar (single value - mean of sine wave)
    test_scalar = np.mean(sinewave)
    save_csv(output_dir, test_scalar, 'test_scalar_python.csv')

    print("\n[PASS] Basic sine wave example completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
