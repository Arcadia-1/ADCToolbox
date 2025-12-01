"""
test_basic.py - Unit test for basic infrastructure

Generates a simple sine wave to verify:
- numpy array operations work
- matplotlib plotting works (if enabled)
- File I/O works

Output structure:
    test_output/test_basic/
        sinewave_python.csv - Generated sine wave (first 1000 samples)
"""

import numpy as np
import matplotlib.pyplot as plt
from tests._utils import save_variable, save_fig

plt.rcParams['font.size'] = 14

def test_basic(project_root):
    """Generate basic sine wave, plot it, and save to CSV."""
    test_output_dir = project_root / "test_output" / "test_basic"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Test output directory: [{test_output_dir}]")

    # Configuration (must match MATLAB test_basic.m)
    N = 1024
    Fs = 1e3
    Fin = 99
    A = 0.49
    DC = 0.5

    # Generate sine wave
    t = np.arange(N) / Fs
    sinewave = A * np.sin(2*np.pi*Fin*t) + DC

    period_samples = round(Fs/Fin)
    n_periods = 3
    n_zoom = min(period_samples*n_periods, N)
    t_zoom = t[:n_zoom]
    sinewave_zoom = sinewave[:n_zoom]

    # Plot sine wave (matches MATLAB figure with 2 subplots)
    plt.figure(figsize=(10, 8))

    # Full waveform in subplot 1
    plt.subplot(2, 1, 1)
    plt.plot(t*1e3, sinewave, 'b-', linewidth=2)
    plt.xlim([0, max(t) * 1e3])
    plt.ylim([min(sinewave) - 0.1, max(sinewave) + 0.1])
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title(f'Full Sine Wave (Fin={Fin} Hz, Fs={int(Fs)} Hz, N={N})')
    plt.grid(True)

    # Zoomed (first 3 periods) in subplot 2
    plt.subplot(2, 1, 2)    
    plt.plot(t_zoom*1e3, sinewave_zoom, '-o', linewidth=2, markersize=4)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title(f'Zoomed View (First {n_periods} Periods)')
    plt.ylim([min(sinewave_zoom) - 0.1, max(sinewave_zoom) + 0.1])
    plt.grid(True)
    
    plt.tight_layout()

    # Save figure and variable
    save_fig(test_output_dir, "sinewave_basic_python.png")
    save_variable(test_output_dir, sinewave, 'sinewave')
    
    test_matrix = sinewave.reshape((4, int(N/4)), order='F')
    test_scalar = np.mean(sinewave)
    save_variable(test_output_dir, test_matrix, 'test_matrix')
    save_variable(test_output_dir, test_scalar, 'test_scalar')