"""
Debug jitter detection - single test case to examine intermediate values.
"""

import numpy as np

from adctoolbox.common.findBin import find_bin
from adctoolbox.common.sineFit import sine_fit

# Generate test signal with known jitter
N = 2**14
Fs = 10e9
J = find_bin(Fs, 400e6, N)
Fin = J/N * Fs

print(f"[N] = {N}")
print(f"[Fs] = {Fs/1e9:.2f} GHz")
print(f"[J] = {J}")
print(f"[Fin] = {Fin/1e9:.6f} GHz")
print()

# Test with 100fs jitter
Tj = 100e-15
A = 0.49
offset = 0.5
amp_noise = 0.00001

np.random.seed(42)  # For reproducibility

# Generate jittered signal
Ts = 1 / Fs
theta = 2 * np.pi * Fin * np.arange(N) * Ts
phase_noise_rms = 2 * np.pi * Fin * Tj
phase_jitter = np.random.randn(N) * phase_noise_rms

data = np.sin(theta + phase_jitter) * A + offset + np.random.randn(N) * amp_noise

print(f"[Set jitter] = {Tj*1e15:.2f} fs")
print(f"[Phase noise RMS (input)] = {phase_noise_rms:.6e} rad")
print()

# Manual step-by-step calculation to debug
from adctoolbox.aout.errHistSine import errHistSine

# First get the results
emean, erms, phase_code, anoi, pnoi, err, xx = errHistSine(data, bin=99, fin=J/N, disp=0)

# Now manually recalculate to see intermediate values
data_fit_temp, freq_temp, mag_temp, dc_temp, phi_temp = sine_fit(data, J/N)
asen = np.abs(np.cos(phase_code/360*2*np.pi))**2
psen = np.abs(np.sin(phase_code/360*2*np.pi))**2

valid_mask = ~np.isnan(erms)
A_mat = np.column_stack([asen[valid_mask], psen[valid_mask], np.ones(np.sum(valid_mask))])
b_vec = erms[valid_mask]**2
tmp = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]

print("=== Least Squares Fit Diagnostics ===")
print(f"[tmp[0] (amp noise squared)] = {tmp[0]:.6e}")
print(f"[tmp[1] (phase noise * mag^2)] = {tmp[1]:.6e}")
print(f"[tmp[2] (baseline)] = {tmp[2]:.6e}")
print(f"[sqrt(tmp[0])] = {np.sqrt(tmp[0]) if tmp[0] >= 0 else 'NEGATIVE!'}")
print(f"[sqrt(tmp[1])/mag] = {np.sqrt(tmp[1])/mag_temp:.6e}")
print()

# Call official errHistSine to compare
emean, erms, phase_code, anoi, pnoi, err, xx = errHistSine(data, bin=99, fin=J/N, disp=0)

# Also get sine fit params
data_fit, freq, mag, dc, phi = sine_fit(data, J/N)

print("=== Sine Fit Results ===")
print(f"[freq (normalized)] = {freq:.10f}")
print(f"[mag] = {mag:.10f}")
print(f"[dc] = {dc:.10f}")
print(f"[phi] = {phi:.10f} rad")
print(f"[A (calculated)] = {mag * np.cos(-phi):.10f}")
print(f"[B (calculated)] = {mag * np.sin(-phi):.10f}")
print()

print("=== Error Histogram Results ===")
print(f"[Number of bins] = {len(phase_code)}")
print(f"[anoi] = {anoi:.6e}")
print(f"[pnoi] = {pnoi:.6e} rad")
print(f"[anoi/mag (normalized amp noise)] = {anoi/mag:.6e}")
print()

# Calculate jitter
jitter_calculated = pnoi / (2 * np.pi * Fin)

print(f"[Calculated jitter] = {jitter_calculated*1e15:.2f} fs")
print(f"[Error] = {(jitter_calculated - Tj)/Tj*100:.1f}%")
print()

# Check some intermediate values
print("=== Diagnostics ===")
print(f"[RMS of err] = {np.std(err):.6e}")
print(f"[Mean erms] = {np.nanmean(erms):.6e}")
print(f"[Max erms] = {np.nanmax(erms):.6e}")
print(f"[Min erms] = {np.nanmin(erms):.6e}")
print(f"[Number of NaN in erms] = {np.sum(np.isnan(erms))}")
