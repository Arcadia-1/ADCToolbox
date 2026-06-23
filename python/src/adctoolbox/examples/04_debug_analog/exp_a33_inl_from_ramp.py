"""Ramp-based INL/DNL extraction from simulated ADC output codes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import analyze_inl_from_ramp


output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


def simulate_ramp_codes(code_widths, samples_per_code=128):
    """Generate output codes from a monotonic ramp and code-width model."""
    code_widths = np.asarray(code_widths, dtype=float)
    if np.any(code_widths < 0):
        raise ValueError("code_widths must be non-negative")

    transition_edges = np.concatenate(([0.0], np.cumsum(code_widths)))
    transition_edges = transition_edges / transition_edges[-1]
    n_samples = len(code_widths) * samples_per_code
    ramp = np.linspace(0.0, np.nextafter(1.0, 0.0), n_samples)
    return np.searchsorted(transition_edges[1:-1], ramp, side="right")


num_bits = 6
n_codes = 2**num_bits

ideal_widths = np.ones(n_codes)

nonideal_widths = np.ones(n_codes)
nonideal_widths[18] = 1.50   # Wide code: DNL ~= +0.5 LSB
nonideal_widths[19] = 0.50   # Narrow code: DNL ~= -0.5 LSB
nonideal_widths[42] = 0.00   # Missing code: DNL = -1 LSB
nonideal_widths[43] = 2.00   # Neighbor absorbs the missing-code width

cases = [
    ("Ideal ramp ADC", simulate_ramp_codes(ideal_widths)),
    ("Injected DNL + missing code", simulate_ramp_codes(nonideal_widths)),
]

fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col")

print("[Ramp INL/DNL]")
for col, (title, codes) in enumerate(cases):
    result = analyze_inl_from_ramp(
        codes,
        num_bits=num_bits,
        endpoint="endpoints",
        create_plot=False,
    )

    code = result["code"]
    transition_code = result["transition_code"]
    counts = result["counts"]
    dnl = result["dnl"]
    inl = result["inl"]

    ax_counts = axes[0, col]
    ax_dnl = axes[1, col]
    ax_inl = axes[2, col]

    ax_counts.bar(code, counts, width=0.9, color="#4C78A8")
    ax_counts.axhline(result["ideal_count"], color="black", linestyle="--", linewidth=1)
    ax_counts.set_title(title, fontweight="bold")
    ax_counts.set_ylabel("Counts")

    ax_dnl.bar(code, dnl, width=0.9, color="#F58518")
    ax_dnl.axhline(0.0, color="black", linewidth=0.8)
    ax_dnl.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_dnl.axhline(-1.0, color="gray", linestyle="--", linewidth=0.8)
    ax_dnl.set_ylabel("DNL (LSB)")

    ax_inl.step(transition_code, inl, where="post", color="#54A24B")
    ax_inl.axhline(0.0, color="black", linewidth=0.8)
    ax_inl.set_xlabel("ADC transition code")
    ax_inl.set_ylabel("INL (LSB)")

    for ax in (ax_counts, ax_dnl, ax_inl):
        ax.grid(True, alpha=0.25)
        ax.set_xlim([0, n_codes - 1])

    if result["missing_codes"].size:
        for missing_code in result["missing_codes"]:
            for ax in (ax_counts, ax_dnl, ax_inl):
                ax.axvline(missing_code, color="red", linestyle=":", linewidth=1)
        ax_counts.text(
            0.02,
            0.88,
            f"missing {result['missing_codes'].tolist()}",
            transform=ax_counts.transAxes,
            color="red",
            fontweight="bold",
        )

    print(
        f"  {title}: "
        f"DNL [{result['dnl_min']:.2f}, {result['dnl_max']:.2f}] LSB, "
        f"INLpp {result['inl_pp']:.2f} LSB, "
        f"missing {result['missing_codes'].tolist()}"
    )

fig.suptitle(
    "Ramp-Based Static INL/DNL from Code Histogram\n"
    "Counts show code width; DNL is counts/mean(counts)-1; INL uses endpoint='endpoints'",
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.93])

fig_path = output_dir / "exp_a33_inl_from_ramp.png"
fig.savefig(fig_path, dpi=150)
print(f"[Save fig] -> [{fig_path}]")
plt.close(fig)
