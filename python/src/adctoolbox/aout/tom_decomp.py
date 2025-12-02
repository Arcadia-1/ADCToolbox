"""
Python port of tomDecomp.m
Thompson Decomposition - Thompson error decomposition algorithm

Decomposes ADC output into:
- signal: Ideal signal (DC + fundamental + specified order harmonics)
- error: Total error
- indep: Independent error (random noise)
- dep: Dependent error (phase-correlated error)
- phi: Fundamental phase

Original MATLAB code: matlab_reference/tomDecomp.m
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add findFin module path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def tom_decomp(data, re_fin=None, order=10, disp=1):
    """
    Thompson Decomposition - Thompson error decomposition

    Parameters:
        data: ADC output data, 1D numpy array
        re_fin: Relative input frequency (normalized frequency f_in/f_sample), auto-detect if None
        order: Harmonic order for dependent error calculation (default 10, means fundamental + first 10 harmonics are viewed as dependent error)
        disp: Whether to display result plot (0/1), default 1

    Returns:
        signal: Ideal signal (DC + fundamental)
        error: Total error (data - signal)
        indep: Independent error (random noise portion)
        dep: Dependent error (harmonic distortion portion)
        phi: Fundamental phase (radians)

    Principle:
        signal = DC + WI*cos(ωt) + WQ*sin(ωt)  # Fundamental only
        signal_all = DC + Σ[WI_k*cos(kωt) + WQ_k*sin(kωt)]  # Fundamental + harmonics
        error = data - signal
        indep = data - signal_all  # Residual after removing fundamental and harmonics
        dep = signal_all - signal  # Harmonic components
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

    # Reconstruct signal (fundamental only)
    signal = DC + SI * WI + SQ * WQ

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

    # Reconstruct signal (DC + fundamental + harmonics)
    signal_all = DC + A @ W

    # Error decomposition
    error = data - signal
    indep = data - signal_all
    dep = signal - signal_all  # Fixed: was signal_all - signal

    # Visualization
    if disp:
        # Only create new figure if one doesn't exist
        if plt.get_fignums() == []:
            fig = plt.figure(figsize=(12, 6))

        # Left Y-axis: Signal
        ax1 = plt.gca()
        ax1.plot(data, 'kx', label='data', markersize=3, alpha=0.5)
        ax1.plot(signal, '-', color=[0.5, 0.5, 0.5], label='signal', linewidth=1.5)

        # Limit display range (show at most 1.5 periods or 100 points)
        xlim_max = min(max(int(1.5 / re_fin), 100), N)
        ax1.set_xlim([0, xlim_max])

        data_min, data_max = np.min(data), np.max(data)
        ax1.set_ylim([data_min * 1.1, data_max * 1.1])
        ax1.set_ylabel('Signal', color='k')
        ax1.tick_params(axis='y', labelcolor='k')

        # Right Y-axis: Error
        ax2 = ax1.twinx()
        ax2.plot(dep, 'r-', label='dependent err', linewidth=1.5)
        ax2.plot(indep, 'b-', label='independent err', linewidth=1)

        error_min, error_max = np.min(error), np.max(error)
        ax2.set_ylim([error_min * 1.1, error_max * 1.1])
        ax2.set_ylabel('Error', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax1.set_xlabel('Samples')
        ax1.set_title(f'Thompson Decomposition (freq={re_fin:.6f}, order={order})')

        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

    # Note: Figure is left open for caller to save/close
    # (tests need to save the figure before closing)

    return signal, error, indep, dep, phi


if __name__ == "__main__":
    print("=" * 70)
    print("tomDecomp.py - Thompson Decomposition Test")
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

    # Execute Thompson decomposition
    print(f"\nExecuting Thompson decomposition...")
    signal, error, indep, dep, phi = tomDecomp(adc_output, re_fin=re_fin, order=10, disp=1)

    # Analyze results
    print(f"\nDecomposition results:")
    print(f"  Signal RMS: {np.sqrt(np.mean(signal**2)):.2f}")
    print(f"  Total error RMS: {np.sqrt(np.mean(error**2)):.2f}")
    print(f"  Dependent error RMS: {np.sqrt(np.mean(dep**2)):.2f}")
    print(f"  Independent error RMS: {np.sqrt(np.mean(indep**2)):.2f}")
    print(f"  Fundamental phase: {np.rad2deg(phi):.2f} degrees")

    # Theoretical verification
    theoretical_dep_rms = np.sqrt(np.mean((harmonic_3rd + harmonic_5th)**2))
    theoretical_indep_rms = np.sqrt(np.mean(noise**2))

    print(f"\nTheoretical comparison:")
    print(f"  Theoretical dependent error RMS: {theoretical_dep_rms:.2f}")
    print(f"  Actual dependent error RMS: {np.sqrt(np.mean(dep**2)):.2f}")
    print(f"  Error: {abs(theoretical_dep_rms - np.sqrt(np.mean(dep**2))):.2f}")
    print(f"")
    print(f"  Theoretical independent error RMS: {theoretical_indep_rms:.2f}")
    print(f"  Actual independent error RMS: {np.sqrt(np.mean(indep**2)):.2f}")
    print(f"  Error: {abs(theoretical_indep_rms - np.sqrt(np.mean(indep**2))):.2f}")

    print(f"\nThompson decomposition test complete!")
    print("=" * 70)
