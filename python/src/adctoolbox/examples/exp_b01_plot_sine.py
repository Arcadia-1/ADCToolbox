"""Time domain sine wave plotting"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

N = 2**13
Fs = 1e6
Fin = 123e3
t = np.arange(N) / Fs
A = 0.49
DC = 0.5

signal = A * np.sin(2*np.pi*Fin*t) + DC
print(f"[Sinewave] [N={N}, Fs={Fs/1e6:.1f}MHz, Fin={Fin/1e3:.1f}kHz, A={A}, DC={DC}]")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t*1e6, signal, linewidth=0.5)
ax1.set_xlabel('Time (us)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Full Waveform')
ax1.grid(True)

period = 1 / Fin
n_periods = 3
n_zoom = int(n_periods * period * Fs)
ax2.plot(t[:n_zoom]*1e6, signal[:n_zoom], 'o-', markersize=4)
ax2.set_xlabel('Time (us)')
ax2.set_ylabel('Amplitude')
ax2.set_title('Zoomed: First 3 Periods')
ax2.grid(True)

plt.tight_layout()
fig_path = (output_dir / 'exp_b01_plot_sine.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()