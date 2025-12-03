"""Quantization Verification for spec_plot"""
import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import spec_plot

# Parameters
N = 2**13
J = 323
sig = 0.499 * np.sin(np.arange(N) * J * 2 * np.pi / N) + 0.5

# Single bit depth example (12 bits)
nbits = 12
sig_quantized = np.floor(sig * 2**nbits) / 2**nbits
results = spec_plot(sig_quantized, label=0, harmonic=5, is_plot=False)
enob, sndr = results[0], results[1]
print(f'[{nbits:2d} bit] ENOB = {enob:.4f}, SNDR = {sndr:.4f}')

# Sweep over bit depths from 1 to 20
bit_sweep = np.arange(1, 21)
enob_results = np.zeros(len(bit_sweep))
sndr_results = np.zeros(len(bit_sweep))

for idx, nbits in enumerate(bit_sweep):
    sig_quantized = np.floor(sig * 2**nbits) / 2**nbits

    results = spec_plot(sig_quantized, label=0, harmonic=5, is_plot=False)
    enob, sndr = results[0], results[1]

    enob_results[idx] = enob
    sndr_results[idx] = sndr

    print(f'[{nbits:2d} bit] ENoB = {enob:5.2f}, diff = {enob-nbits:9.2e}')

plt.figure(figsize=(8, 6))
plt.plot(bit_sweep, bit_sweep, 'k--', linewidth=1.5, label='Resolution')
plt.plot(bit_sweep, enob_results, 'b-o', linewidth=2, markersize=4, label='spec_plot ENOB')
plt.grid(True)
plt.xlabel('Quantization Bits', fontsize=14)
plt.ylabel('ENOB (bits)', fontsize=14)
plt.title('ENOB vs Quantization Bits', fontsize=16)
plt.legend(loc='upper left')
plt.xlim([1, max(bit_sweep)])
plt.ylim([1, max(bit_sweep)])
plt.gca().tick_params(labelsize=12)
plt.tight_layout()
# plt.show()
