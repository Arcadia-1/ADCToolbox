"""Time domain sine wave plotting"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

Fs = 1e6
Fin = 123e3
A = 0.49

# Generate signal
period = 1 / Fin
n_samples = 20
t = np.arange(n_samples) / Fs
signal = A * np.sin(2*np.pi*Fin*t)

print(f"[Sinewave] [Fs = {Fs/1e6:.1f} MHz, Fin = {Fin/1e3:.1f} kHz, A = {A}, samples = {n_samples}]")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t*1e6, signal, 'o-', markersize=4, linewidth=1.5)
ax.set_xlabel('Time (us)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title(f'Sine Wave: {n_samples} samples', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = (output_dir / 'exp_b01_plot_sine.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()