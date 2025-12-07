"""
Python port of tomdec.m
Decompose Harmonics - Decompose signal into fundamental and harmonic errors

Decomposes ADC output into:
- fundamental_signal: Ideal fundamental (DC + fundamental)
- total_error: Total error (sig - fundamental)
- harmonic_error: Harmonic distortions (2nd through nth harmonics)
- other_error: All other errors (not captured by harmonics)

Original MATLAB code: matlab/src/tomdec.m (Thompson decomposition algorithm)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add findFin module path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def decompose_harmonics(data, re_fin=None, order=10, disp=1):
    """
    Decompose Harmonics - Decompose signal into fundamental and harmonic errors

    Parameters:
        data: ADC output data, 1D numpy array
        re_fin: Relative input frequency (normalized frequency f_in/f_sample), auto-detect if None
        order: Harmonic order for fitting (default 10, fits fundamental + harmonics 2 through order)
        disp: Whether to display result plot (0/1), default 1

    Returns:
        fundamental_signal: Fundamental sinewave component (including DC)
        total_error: Total error (sig - fundamental_signal)
        harmonic_error: Harmonic distortions (2nd through nth harmonics)
        other_error: All other errors (sig - all harmonics)

    Notes:
        Matches MATLAB tomdec.m outputs (Thompson decomposition algorithm):
        - fundamental_signal → MATLAB 'sine'
        - total_error → MATLAB 'err'
        - harmonic_error → MATLAB 'har'
        - other_error → MATLAB 'oth'

    Principle:
        fundamental_signal = DC + WI*cos(ωt) + WQ*sin(ωt)  # Fundamental only
        signal_all = DC + Σ[WI_k*cos(kωt) + WQ_k*sin(kωt)]  # Fundamental + harmonics
        total_error = data - fundamental_signal
        harmonic_error = signal_all - fundamental_signal  # Harmonic components (2nd to nth)
        other_error = data - signal_all  # Residual after removing fundamental and harmonics
    """

    # Ensure data is a column vector
    fig = None
    data = np.asarray(data).flatten()
    N = len(data)

    # If no normalized frequency provided, auto-detect
    if re_fin is None or np.isnan(re_fin):
        try:
            from findFin import findFin
            re_fin = findFin(data)
        except ImportError:
            # Simple FFT frequency detection
            spec = np.abs(np.fft.fft(data))
            spec[0] = 0
            bin_max = np.argmax(spec[:N//2])
            re_fin = bin_max / N
            print(f"Warning: findFin not found, using simple FFT detection: re_fin = {re_fin:.6f}")

    # Time axis
    t = np.arange(N)

    # Calculate fundamental I/Q components
    SI = np.cos(t * re_fin * 2 * np.pi)
    SQ = np.sin(t * re_fin * 2 * np.pi)

    # Estimate fundamental weights and DC
    WI = np.mean(SI * data) * 2
    WQ = np.mean(SQ * data) * 2
    DC = np.mean(data)

    # Reconstruct fundamental_signal (fundamental only)
    fundamental_signal = DC + SI * WI + SQ * WQ

    # Fundamental phase
    phi = -np.arctan2(WQ, WI)

    # Build multi-order harmonic matrix
    SI_matrix = np.zeros((N, order))
    SQ_matrix = np.zeros((N, order))

    for ii in range(order):
        SI_matrix[:, ii] = np.cos(t * re_fin * (ii + 1) * 2 * np.pi)
        SQ_matrix[:, ii] = np.sin(t * re_fin * (ii + 1) * 2 * np.pi)

    # Merge I/Q matrices
    A = np.column_stack([SI_matrix, SQ_matrix])

    # Least squares solution for harmonic weights
    W, residuals, rank, s = np.linalg.lstsq(A, data, rcond=None)

    # Reconstruct signal with all harmonics (DC + fundamental + harmonics)
    signal_all = DC + A @ W

    # Error decomposition (matches MATLAB tomdec.m)
    total_error = data - fundamental_signal
    harmonic_error = signal_all - fundamental_signal  # Harmonic distortion (2nd through nth)
    other_error = data - signal_all  # Other errors (not captured by harmonics)

    # Visualization
    if disp:
        # Only create new figure if one doesn't exist
        if plt.get_fignums() == []:
            fig = plt.figure(figsize=(12, 6))

        # Left Y-axis: Signal
        ax1 = plt.gca()
        ax1.plot(data, 'kx', label='data', markersize=3, alpha=0.5)
        ax1.plot(fundamental_signal, '-', color=[0.5, 0.5, 0.5], label='fundamental_signal', linewidth=1.5)

        # Limit display range (show first 3 periods or at least 100 points)
        xlim_max = min(max(int(3 / re_fin), 100), N)
        ax1.set_xlim([0, xlim_max])

        data_min, data_max = np.min(data), np.max(data)
        ax1.set_ylim([data_min * 1.1, data_max * 1.1])
        ax1.set_ylabel('Signal', color='k')
        ax1.tick_params(axis='y', labelcolor='k')

        # Right Y-axis: Error
        ax2 = ax1.twinx()

        # Calculate RMS for legend labels
        rms_harmonic = np.sqrt(np.mean(harmonic_error**2))
        rms_other = np.sqrt(np.mean(other_error**2))
        rms_total = np.sqrt(np.mean(total_error**2))

        # Determine appropriate unit (uV, mV, or V)
        if rms_total < 1e-3:
            unit = 'uV'
            scale = 1e6
        elif rms_total < 1:
            unit = 'mV'
            scale = 1e3
        else:
            unit = 'V'
            scale = 1

        # Calculate power percentages (RMS^2 / Total^2)
        harmonic_pct = (rms_harmonic / rms_total)**2 * 100
        other_pct = (rms_other / rms_total)**2 * 100

        ax2.plot(harmonic_error, 'r-',
                label=f'harmonics ({rms_harmonic*scale:.1f}{unit}, {harmonic_pct:.1f}%)',
                linewidth=1.5)
        ax2.plot(other_error, 'b-',
                label=f'other errors ({rms_other*scale:.1f}{unit}, {other_pct:.1f}%)',
                linewidth=1)

        error_min, error_max = np.min(total_error), np.max(total_error)
        ax2.set_ylim([error_min * 1.1, error_max * 1.1])
        ax2.set_ylabel('Error', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax1.set_xlabel('Samples')
        ax1.set_title(f'Decompose Harmonics (freq={re_fin:.6f}, order={order})')

        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

    # Note: Figure is left open for caller to save/close
    # (tests need to save the figure before closing)

    return fundamental_signal, total_error, harmonic_error, other_error


if __name__ == "__main__":
    print("=" * 70)
    print("decompose_harmonics.py - Decompose Harmonics Test")
    print("=" * 70)

    # Test case: Generate signal with harmonic distortion and noise
    N = 4096
    fs = 1e6
    fin = 28320.3125  # Coherent sampling frequency
    re_fin = fin / fs

    t = np.arange(N) / fs

    # Ideal sine wave
    signal_ideal = np.sin(2 * np.pi * fin * t) * 1000 + 2048

    # Add harmonic distortion (3rd harmonic, 5th harmonic)
    harmonic_3rd = 0.05 * np.sin(3 * 2 * np.pi * fin * t) * 1000
    harmonic_5th = 0.02 * np.sin(5 * 2 * np.pi * fin * t) * 1000

    # Add random noise
    noise = 10 * np.random.randn(N)

    # Synthesize ADC output
    adc_output = signal_ideal + harmonic_3rd + harmonic_5th + noise

    print(f"\nTest parameters:")
    print(f"  Sample count: {N}")
    print(f"  Sampling frequency: {fs/1e6:.2f} MHz")
    print(f"  Input frequency: {fin/1e3:.2f} kHz")
    print(f"  Normalized frequency: {re_fin:.10f}")
    print(f"  3rd harmonic amplitude: 5%")
    print(f"  5th harmonic amplitude: 2%")
    print(f"  Noise RMS: 10 LSB")

    # Execute harmonic decomposition
    print(f"\nExecuting harmonic decomposition...")
    fundamental_signal, total_error, harmonic_error, other_error = decompose_harmonics(adc_output, re_fin=re_fin, order=10, disp=1)

    # Analyze results
    print(f"\nDecomposition results:")
    print(f"  Signal RMS: {np.sqrt(np.mean(fundamental_signal**2)):.2f}")
    print(f"  Total error RMS: {np.sqrt(np.mean(total_error**2)):.2f}")
    print(f"  Harmonic error RMS: {np.sqrt(np.mean(harmonic_error**2)):.2f}")
    print(f"  Other error RMS: {np.sqrt(np.mean(other_error**2)):.2f}")

    # Theoretical verification
    theoretical_harmonic_rms = np.sqrt(np.mean((harmonic_3rd + harmonic_5th)**2))
    theoretical_other_rms = np.sqrt(np.mean(noise**2))

    print(f"\nTheoretical comparison:")
    print(f"  Theoretical harmonic error RMS: {theoretical_harmonic_rms:.2f}")
    print(f"  Actual harmonic error RMS: {np.sqrt(np.mean(harmonic_error**2)):.2f}")
    print(f"  Error: {abs(theoretical_harmonic_rms - np.sqrt(np.mean(harmonic_error**2))):.2f}")
    print(f"")
    print(f"  Theoretical other error RMS: {theoretical_other_rms:.2f}")
    print(f"  Actual other error RMS: {np.sqrt(np.mean(other_error**2)):.2f}")
    print(f"  Error: {abs(theoretical_other_rms - np.sqrt(np.mean(other_error**2))):.2f}")

    print(f"\nHarmonic decomposition test complete!")
    print("=" * 70)
