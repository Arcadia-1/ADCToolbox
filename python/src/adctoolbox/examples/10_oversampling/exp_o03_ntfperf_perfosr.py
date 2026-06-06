"""Compare NTF theory with performance-vs-OSR sweep."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from adctoolbox import find_coherent_frequency, ntfperf, perfosr
from adctoolbox.siggen import ADC_Signal_Generator


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True, parents=True)

N = 2**13
Fs = 100e6
Fin, _ = find_coherent_frequency(Fs, Fs / 640, N)

gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=0.4, DC=0.0)
sig = gen.apply_noise_shaping(n_bits=10, quant_range=(-0.5, 0.5), order=2)

ntf1 = signal.TransferFunction([1, -1], [1, 0], dt=1)
ntf2 = signal.TransferFunction([1, -2, 1], [1, 0, 0], dt=1)
for osr in [8, 16, 32, 64]:
    g1 = ntfperf(ntf1, 0, 0.5 / osr)
    g2 = ntfperf(ntf2, 0, 0.5 / osr)
    print(f"OSR={osr:2d}: NTF1 improvement={g1:6.2f} dB, NTF2 improvement={g2:6.2f} dB")

fig, ax = plt.subplots(figsize=(7, 4.8))
osr_values, sndr, sfdr, enob = perfosr(sig, osr=np.array([2, 4, 8, 16, 32, 64]), disp=True, ax=ax)
for osr, s, e in zip(osr_values, sndr, enob):
    print(f"Measured sweep OSR={osr:4.0f}: SNDR={s:7.2f} dB, ENOB={e:5.2f}")

fig_path = output_dir / "exp_o03_ntfperf_perfosr.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[Save figure] -> {fig_path}")

