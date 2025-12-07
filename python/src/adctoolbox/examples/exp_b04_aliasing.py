"""Aliasing visualization with Nyquist zones"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import alias

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

Fs = 800e6
test_fin_relative = [0.1, 0.9, 1.1, 1.9, 2.1, 2.9]
zone_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lavender', 'lightgray']

Fin_sweep = np.linspace(0, 3.0, 300)
Fin_aliased = np.array([alias(f * Fs, Fs) / Fs for f in Fin_sweep])

fig, (ax_abs, ax_norm) = plt.subplots(2, 1, figsize=(10, 8))

# Top subplot: Absolute frequency (MHz)
ax_abs.plot(Fin_sweep * Fs / 1e9, Fin_aliased * Fs / 1e6, 'b-', linewidth=2)
ax_abs.axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.6)

for i, color in enumerate(zone_colors):
    ax_abs.axvspan(i * 0.5 * Fs/1e9, (i + 1) * 0.5 * Fs/1e9, alpha=0.2, color=color)
    ax_abs.text((i + 0.5) * 0.5 * Fs/1e9, 0.92*Fs/2/1e6, f"Zone {i+1}", ha='center', fontsize=10, fontweight='bold')

for f in test_fin_relative:
    x_pos, y_pos = f * Fs / 1e9, alias(f * Fs, Fs) / 1e6
    ax_abs.plot(x_pos, y_pos, 'o', color='red', markersize=6)

    zone_idx = int(f / 0.5)
    ha = 'left' if zone_idx % 2 == 0 else 'right'
    x_offset = 0.03 if zone_idx % 2 == 0 else -0.03
    ax_abs.text(x_pos + x_offset, y_pos - 15, f'{f * Fs / 1e6:.0f}M',
                fontsize=10, ha=ha, va='top', color='red', fontweight='bold')

    if f > 0.5:  # Draw arrows for frequencies beyond 1st Nyquist zone
        ax_abs.annotate('', xy=(y_pos, y_pos), xytext=(x_pos, y_pos),
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.6))

ax_abs.set_xlabel('Absolute Input Frequency (MHz)', fontsize=11)
ax_abs.set_ylabel('Aliased Input Frequency (MHz)', fontsize=11)
ax_abs.set_xlim([0, 3.0*Fs/1e9])
ax_abs.set_ylim([0, 0.5*Fs/1e6])
xticks_MHz = np.arange(0, 3.01*Fs, Fs/2) / 1e6
ax_abs.set_xticks(xticks_MHz / 1000)
ax_abs.set_xticklabels([f'{int(x)}M' if x > 0 else '0' for x in xticks_MHz])
ax_abs.grid(True, alpha=0.3)

# Bottom subplot: Normalized frequency (relative to Fs)
ax_norm.plot(Fin_sweep, Fin_aliased, 'b-', linewidth=2)
ax_norm.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.6)

for i, color in enumerate(zone_colors):
    ax_norm.axvspan(i * 0.5, (i + 1) * 0.5, alpha=0.2, color=color)
    ax_norm.text((i + 0.5) * 0.5, 0.45, f"Zone {i+1}", ha='center', fontsize=10, fontweight='bold')

for f in test_fin_relative:  # No arrows in this subplot
    f_alias = alias(f * Fs, Fs) / Fs
    ax_norm.plot(f, f_alias, 'o', color='red', markersize=6)

    zone_idx = int(f / 0.5)
    ha = 'left' if zone_idx % 2 == 0 else 'right'
    x_offset = 0.05 if zone_idx % 2 == 0 else -0.05
    ax_norm.text(f + x_offset, f_alias - 0.03, f'{f:.1f}',
                 fontsize=10, ha=ha, va='top', color='red', fontweight='bold')

ax_norm.set_xlabel('Relative Input Frequency (Normalized to Fs)', fontsize=11)
ax_norm.set_ylabel('Aliased Input Frequency (Normalized to Fs)', fontsize=11)
ax_norm.set_xlim([0, 3.0])
ax_norm.set_ylim([0, 0.5])
ax_norm.grid(True, alpha=0.3)

fig.suptitle(f'Frequency Aliasing: Folding Diagram with 6 Nyquist Zones (Fs={Fs/1e6:.0f}MHz)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = (output_dir / 'exp_b04_aliasing.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()
