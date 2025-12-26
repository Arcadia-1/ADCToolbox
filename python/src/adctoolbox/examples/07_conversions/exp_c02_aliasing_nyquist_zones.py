"""
Demonstrates frequency aliasing across 6 Nyquist zones. Shows how different input frequencies
fold back into the baseband (0-Fs/2) due to undersampling, visualizing the "sawtooth" pattern.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import fold_frequency_to_nyquist

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

Fs = 1100e6
Fin_target = 123e6
N_ZONES = 6

# 1. Calculate the true baseband alias of the target frequency (101 MHz)
F_aliased = fold_frequency_to_nyquist(fin=Fin_target, fs=Fs)
print(f"[Aliasing] Fs = {Fs/1e6:.1f} MHz, Fin_target = {Fin_target/1e6:.1f} MHz -> F_aliased = {F_aliased/1e6:.1f} MHz")

# 2. Generate input test points that all alias to F_aliased
test_points_hz = []
for i in range(N_ZONES): 
    # K represents the Fs multiple for the start of the zone
    K = i // 2 
    
    if (i + 1) % 2 == 0:
        # Odd zones (2, 4, 6...): Mirrored folding: (K+1)*Fs - F_aliased
        Fs_multiple = (K + 1) * Fs
        F_in = Fs_multiple - F_aliased
    else:
        # Even zones (1, 3, 5...): Direct folding: K*Fs + F_aliased
        Fs_multiple = K * Fs
        F_in = Fs_multiple + F_aliased
    
    test_points_hz.append(F_in)

ratio_sweep = np.linspace(0, N_ZONES/2, 500)
freq_sweep = ratio_sweep * Fs
aliased_sweep = fold_frequency_to_nyquist(freq_sweep, Fs)

print(f"[Aliasing {len(freq_sweep)} frequencies] [Input = {freq_sweep[0]/1e6:.1f} - {freq_sweep[-1]/1e6:.1f} MHz] [Output = {aliased_sweep.min()/1e6:.2f} - {aliased_sweep.max()/1e6:.2f} MHz]\n")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(freq_sweep / 1e6, aliased_sweep / 1e6, 'b-', linewidth=2)
ax.axhline(y=F_aliased / 1e6, color='red', linestyle='--', linewidth=1, alpha=0.6)

for i in range(N_ZONES):
    color = 'lightblue' if i % 2 == 0 else 'white'
    ax.axvspan(i * 0.5 * Fs/1e6, (i + 1) * 0.5 * Fs/1e6, alpha=0.2, color=color)
    ax.text((i + 0.5) * 0.5 * Fs/1e6, 0.92*Fs/2/1e6, f"Zone {i+1}", ha='center', fontsize=10, fontweight='bold')

for f in test_points_hz:
    x_pos = f / 1e6
    y_pos = fold_frequency_to_nyquist(f, Fs) / 1e6
    
    ax.plot(x_pos, y_pos, 'o', color='red', markersize=6, zorder=10)
    zone_idx = int(f / (Fs/2))

    ha = 'left' if zone_idx % 2 == 0 else 'right'
    x_offset = Fs/1e6/50 if zone_idx % 2 == 0 else -1* Fs/1e6/50
    
    ax.text(x_pos + x_offset, y_pos*0.85, f'{f / 1e6:.0f}',
            fontsize=10, ha=ha, va='center', color='red', fontweight='bold')

ax.set_xlabel('Input Frequency (MHz)', fontsize=11)
ax.set_ylabel('Aliased Frequency (MHz)', fontsize=11)
ax.set_xlim([0, N_ZONES/2*Fs/1e6])
ax.set_ylim([0, 0.5*Fs/1e6])
xticks_MHz = np.arange(0, N_ZONES*Fs/2+0.1, Fs/2) / 1e6
ax.set_xticks(xticks_MHz)
ax.set_xticklabels([f'{int(x)}' if x > 0 else '0' for x in xticks_MHz])
ax.grid(True, alpha=0.3)

fig.suptitle(f'Frequency Aliasing within {N_ZONES} Nyquist Zones (Fs={Fs/1e6:.0f}MHz)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig_path = (output_dir / 'exp_c01_aliasing.png').resolve()
print(f"[Save fig] -> [{fig_path}]\n")
plt.savefig(fig_path, dpi=150)
plt.close()
