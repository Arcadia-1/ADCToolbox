"""
Verification Testbench for spec_plot_phase FFT and LMS modes

This script generates synthetic data with known characteristics
and verifies that both modes correctly identify harmonics.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adctoolbox.aout import spec_plot_phase

print('=' * 60)
print('spec_plot_phase Verification Testbench')
print('=' * 60)
print()

# Generate Synthetic Test Signal
N = 8192           # Number of samples
Fs = 1e9           # Sampling frequency
J = 323            # Fundamental bin (prime number for coherent)
Fin = J * Fs / N   # Fundamental frequency

print('Test Signal Parameters:')
print(f'  N = {N} samples')
print(f'  Fs = {Fs:.0f} Hz')
print(f'  Bin = {J} (coherent)')
print(f'  Fin = {Fin:.6f} Hz (normalized: {Fin/Fs:.6f})')
print()

# Time vector
t = np.arange(N) / Fs

# Generate signal with known harmonics
A_fundamental = 0.45
A_HD2 = 0.05   # 2nd harmonic at -19 dB
A_HD3 = 0.02   # 3rd harmonic at -27 dB
A_HD4 = 0.01   # 4th harmonic at -33 dB

# Known phases (radians)
phi_fundamental = 0.5
phi_HD2 = 1.2
phi_HD3 = -0.8
phi_HD4 = 0.3

# Build signal
signal = (A_fundamental * np.sin(2*np.pi*Fin*t + phi_fundamental) +
          A_HD2 * np.sin(2*np.pi*2*Fin*t + phi_HD2) +
          A_HD3 * np.sin(2*np.pi*3*Fin*t + phi_HD3) +
          A_HD4 * np.sin(2*np.pi*4*Fin*t + phi_HD4))

# Add DC offset and small noise
signal = signal + 0.5 + np.random.randn(N) * 1e-5

print('Expected Values:')
print(f'  Fundamental: A={A_fundamental:.4f}, phase={phi_fundamental:.4f} rad')
print(f'  HD2: A={A_HD2:.4f} ({20*np.log10(A_HD2/A_fundamental):.1f} dB), phase={phi_HD2:.4f} rad')
print(f'  HD3: A={A_HD3:.4f} ({20*np.log10(A_HD3/A_fundamental):.1f} dB), phase={phi_HD3:.4f} rad')
print(f'  HD4: A={A_HD4:.4f} ({20*np.log10(A_HD4/A_fundamental):.1f} dB), phase={phi_HD4:.4f} rad')
print()

# Test FFT Mode
print('=' * 60)
print('Testing FFT Mode')
print('=' * 60)
result_fft = spec_plot_phase(signal, harmonic=5, mode='FFT',
                              save_path='test_plots/verify_plotphase_fft_python.png')

print('FFT Mode Outputs:')
print(f'  harm_phase: {result_fft["harm_phase"]} (should be empty)')
print(f'  harm_mag: {result_fft["harm_mag"]} (should be empty)')
print(f'  freq: {result_fft["freq"]} (should be NaN)')
print(f'  noise_dB: {result_fft["noise_dB"]} (should be NaN)')
print()

# Test LMS Mode
print('=' * 60)
print('Testing LMS Mode')
print('=' * 60)
result_lms = spec_plot_phase(signal, harmonic=5, mode='LMS',
                              save_path='test_plots/verify_plotphase_lms_python.png')

harm_phase_lms = result_lms['harm_phase']
harm_mag_lms = result_lms['harm_mag']
freq_lms = result_lms['freq']
noise_dB_lms = result_lms['noise_dB']

print('LMS Mode Outputs:')
print(f'  Detected frequency: {freq_lms:.6f} (normalized)')
print(f'  Noise floor: {noise_dB_lms:.2f} dB')
print(f'  Number of harmonics: {len(harm_phase_lms)}')
print()

print('Harmonic Analysis (LMS):')
print('  H# |  Magnitude  | Mag (dB) |  Phase (rad) | Phase (deg)')
print('  ---|-------------|----------|--------------|------------')
for ii in range(min(5, len(harm_phase_lms))):
    mag_dB = 20*np.log10(harm_mag_lms[ii]/harm_mag_lms[0])
    print(f'  {ii+1:2d} | {harm_mag_lms[ii]:11.6f} | {mag_dB:8.2f} | '
          f'{harm_phase_lms[ii]:12.6f} | {np.degrees(harm_phase_lms[ii]):11.2f}')
print()

# Verify Results
print('=' * 60)
print('Verification Results')
print('=' * 60)

# Check frequency detection
freq_error = abs(freq_lms - Fin/Fs)
status_freq = 'PASS' if freq_error < 1e-6 else 'FAIL'
print('Frequency Detection:')
print(f'  Expected: {Fin/Fs:.6f}')
print(f'  Detected: {freq_lms:.6f}')
print(f'  Error: {freq_error:.2e} [{status_freq}]')
print()

# Check magnitude detection
expected_mags = [A_fundamental, A_HD2, A_HD3, A_HD4]
print('Magnitude Detection:')
for ii in range(min(4, len(harm_mag_lms))):
    mag_error = abs(harm_mag_lms[ii] - expected_mags[ii])
    mag_error_pct = mag_error / expected_mags[ii] * 100
    status = 'OK' if mag_error_pct < 5 else 'FAIL'
    print(f'  H{ii+1}: Expected={expected_mags[ii]:.4f}, Detected={harm_mag_lms[ii]:.4f}, '
          f'Error={mag_error_pct:.2f}% [{status}]')
print()

# Check phase detection (LMS mode returns phases relative to fundamental)
print('Phase Detection (relative to fundamental):')
print(f'  H1 (fundamental): {harm_phase_lms[0]:.4f} rad (should be ~0 after rotation)')
if len(harm_phase_lms) >= 4:
    # The relative phases in LMS mode are: phase[i] - (i+1)*phase[1]
    print(f'  H2: Detected={harm_phase_lms[1]:.4f} rad')
    print(f'  H3: Detected={harm_phase_lms[2]:.4f} rad')
    print(f'  H4: Detected={harm_phase_lms[3]:.4f} rad')
print()

# Check noise floor is reasonable
print('Noise Floor:')
print(f'  Detected: {noise_dB_lms:.2f} dB')
status_noise = 'PASS (low noise)' if noise_dB_lms < -60 else 'WARNING (noisy)'
print(f'  Status: [{status_noise}]')
print()

print('=' * 60)
print('Testbench Complete')
print('=' * 60)
