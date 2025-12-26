import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_weight_radix

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Test cases: [caps_nominal, title]
test_cases = [
    (np.array([1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1], dtype=float), 'Strict Binary Weights'),
    (np.array([1156, 642, 357, 198, 110, 61, 34, 18, 10, 5, 3, 2, 1, 1], dtype=float), 'Sub-Radix-2 Weights'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (caps_nominal, title) in enumerate(test_cases):
    n_bits = len(caps_nominal)
    weights_nominal = np.append(caps_nominal[:-1], caps_nominal[-1] * 0.5)
    nominal_resolution = np.log2(np.sum(caps_nominal) / caps_nominal[-1] * 2)

    # Plot weight scaling
    radix = analyze_weight_radix(weights_nominal, ax=axes[idx])
    axes[idx].set_title(title, fontsize=12, fontweight='bold')

    # Statistics
    radix_valid = radix[~np.isnan(radix)]
    mean_radix = np.mean(radix_valid)

    print(f"[{title:22s}] [Resolution = {nominal_resolution:5.2f} bit] [Average Radix = {mean_radix:.4f}]")

plt.tight_layout()
fig_path = output_dir / 'exp_d13_weight_scaling.png'
plt.savefig(fig_path, dpi=150)
print(f"\n[Save fig] -> [{fig_path}]")
plt.close('all')
