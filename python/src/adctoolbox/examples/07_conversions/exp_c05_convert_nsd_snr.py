"""
Converts between Noise Spectral Density (NSD) and SNR. Shows how SNR varies with bandwidth
and sampling frequency, demonstrating the relationship: SNR = Psignal - NSD - 10*log10(BW).
"""

import numpy as np
import matplotlib.pyplot as plt
from adctoolbox import snr_to_nsd, nsd_to_snr
from pathlib import Path

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# =============================================================================
# Plot SNR vs Bandwidth for Fixed NSD
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

bw_sweep = np.logspace(3, 10, 100)  # Bandwidth from 1 kHz to 10 GHz
nsd_values = [-130, -140, -150, -160]  # Add or remove values as needed

# SNR vs BW for fixed NSD using nsd_to_snr function
for nsd_val in nsd_values:
    # For each BW, set fs = 2*BW with OSR=1, so BW = fs/(2*OSR) = fs/2 = BW
    snr_sweep = [nsd_to_snr(nsd_val, fs=2*bw, osr=1) for bw in bw_sweep]
    ax.semilogx(bw_sweep / 1e6, snr_sweep, linewidth=2.5, label=f'NSD = {nsd_val} dBFS/Hz')

ax.grid(True, which='both', alpha=0.3)
ax.set_xlabel('Noise Bandwidth (MHz)', fontsize=12)
ax.set_ylabel('SNR (dB)', fontsize=12)
ax.set_title('SNR vs Noise Bandwidth for Fixed NSD\n(Psignal = 0 dBFS)', fontsize=13, fontweight='bold')
ax.legend(fontsize=13, loc='lower left')

ax.yaxis.set_major_locator(plt.MultipleLocator(10))

plt.tight_layout()
fig_path = output_dir / 'exp_c05_nsd_snr_conversions.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"[Figure saved] -> {fig_path.resolve()}")
plt.close()

# =============================================================================
# Round-trip conversion verification
# =============================================================================
print("\n[SNR -> NSD -> SNR Round-trip]")
snr_original, fs, osr = 85.3, 1e6, 128
nsd = snr_to_nsd(snr_original, fs, osr=osr)
snr_recovered = nsd_to_snr(nsd, fs, osr=osr)
print(f"  [SNR = {snr_original:.2f} dB] -> [NSD = {nsd:.2f} dBFS/Hz] -> [SNR = {snr_recovered:.2f} dB]")
