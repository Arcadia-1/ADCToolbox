"""
Quick environment check: Verifies that NumPy, Matplotlib, and ADCToolbox are installed correctly.
Generates a simple spectrum plot to confirm everything works before running more complex examples.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

Fs = 1e6
Fin = 123e3
A = 0.5

# Generate signal
period = 1 / Fin
n_samples = 20
t = np.arange(n_samples) / Fs
signal = A * np.sin(2*np.pi*Fin*t)

print(f"[Analysis Parameters] Fs = {Fs/1e6:.2f} MHz, Fin = {Fin/1e6:.4f} MHz")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t*1e6, signal, 'o-', markersize=4, linewidth=1.5)
ax.set_xlabel('Time (us)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_title(f'Sine Wave: {n_samples} samples', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = (output_dir / 'exp_b01_environment_check.png').resolve()
print(f"\n[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()