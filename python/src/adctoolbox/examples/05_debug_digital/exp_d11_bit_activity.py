import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, analyze_bit_activity

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 1e9
Fin, bin = find_coherent_frequency(fs=Fs, fin_target=300e6, n_fft=N)
t = np.arange(N) / Fs
A = 0.499

sine = 2 * A * np.sin(2*np.pi*Fin*t)
test_cases = [
    (sine, 'ideal', False),
    (sine + 0.01, '+1% DC Offset', False),
    (sine - 0.01, '-1% DC Offset', False),
    (sine, 'Poor contact in Bit-11', True),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))

# 12-bit SAR weights
cdac = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1]
B = len(cdac)
weight_voltage = np.array(cdac) / sum(cdac)
ideal_weights = 2.0 ** np.arange(B-1, -1, -1)

for idx, (sig, title, has_glitch) in enumerate(test_cases):
    # SAR quantization
    residue = sig.copy()
    dout = np.zeros((N, B))
    for j in range(B):
        dout[:, j] = (residue > 0).astype(int)
        delta_cdac = (2 * dout[:, j] - 1) * weight_voltage[j]
        if j < B - 1:
            residue -= delta_cdac

    if has_glitch:
        glitch_mask = np.random.rand(N) < 0.10 # 10% samples affected
        dout[glitch_mask, B-2] = 0 # Force the unfortunate bits to '0'

    bit_usage = analyze_bit_activity(dout, ax=axes[0, idx])    # Bit activity (use toolbox function)
    axes[0, idx].set_title(f'{title}\nBit Activity', fontsize=11, fontweight='bold')

    plt.sca(axes[1, idx])
    result = analyze_spectrum(dout @ ideal_weights, max_harmonic=5, osr=1, show_label=True, nf_method=0)    # Spectrum
    print(f"[{title:<24s}] [Bits = {B:2d}] [ENoB = {result['enob']:5.2f}] [Activity = {np.min(bit_usage):.1f}% - {np.max(bit_usage):.1f}%]")

plt.tight_layout()
fig_path = output_dir / f'exp_d11_bit_activity.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")
plt.close()
