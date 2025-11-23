"""
Overflow Check Tool for SAR ADC

Analyzes residue distribution at each bit position to detect overflow conditions.
This is useful for sub-radix-2 SAR ADC calibration and redundancy analysis.

Ported from MATLAB: overflowChk.m
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def overflow_check(
    raw_code: np.ndarray,
    weight: np.ndarray,
    ofb: int = 1,
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> dict:
    """
    Analyze residue distribution at each bit position.

    For each bit position, calculates the normalized residue (remaining bits weighted sum).
    Detects overflow conditions where residue goes outside [0, 1] range.

    Args:
        raw_code: Digital codes array, shape (N, M) where N=samples, M=bits
                  Each row is one sample, each column is one bit (MSB first)
        weight: Weight array for each bit, shape (M,)
        ofb: Overflow bit position for overflow detection (default=1, i.e., LSB)
        save_path: Path to save the figure (if None, not saved)
        show_plot: Whether to display the plot (default False)

    Returns:
        dict with keys:
            - data_decom: Decomposed/normalized residue at each bit, shape (N, M)
            - range_min: Minimum residue at each bit position
            - range_max: Maximum residue at each bit position
            - ovf_zero_pct: Percentage of samples with residue <= 0 at each bit
            - ovf_one_pct: Percentage of samples with residue >= 1 at each bit
    """
    raw_code = np.asarray(raw_code)
    weight = np.asarray(weight)

    if raw_code.ndim == 1:
        raw_code = raw_code.reshape(-1, 1)

    N, M = raw_code.shape

    if len(weight) != M:
        raise ValueError(f"Weight length ({len(weight)}) must match number of bits ({M})")

    data_decom = np.zeros((N, M))
    range_min = np.zeros(M)
    range_max = np.zeros(M)

    # Calculate normalized residue at each bit position
    for ii in range(M):
        # Weighted sum of remaining bits (from current bit to LSB)
        tmp = raw_code[:, ii:] @ weight[ii:]

        # Normalize by sum of remaining weights
        sum_weight = np.sum(weight[ii:])
        if sum_weight > 0:
            data_decom[:, ii] = tmp / sum_weight
            range_min[ii] = np.min(tmp) / sum_weight
            range_max[ii] = np.max(tmp) / sum_weight
        else:
            data_decom[:, ii] = tmp
            range_min[ii] = np.min(tmp)
            range_max[ii] = np.max(tmp)

    # Detect overflow at specified bit position
    # Note: residue = 0 or 1 is normal, only < 0 or > 1 is overflow
    ofb_idx = M - ofb  # Convert to 0-based index
    ovf_zero = data_decom[:, ofb_idx] < 0
    ovf_one = data_decom[:, ofb_idx] > 1

    # Calculate overflow percentages at each bit
    ovf_zero_pct = np.array([np.sum(data_decom[:, ii] < 0) / N * 100 for ii in range(M)])
    ovf_one_pct = np.array([np.sum(data_decom[:, ii] > 1) / N * 100 for ii in range(M)])

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Reference lines at 0 and 1
    ax.axhline(y=1, color='k', linestyle='-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)

    # Plot min/max range
    bit_positions = np.arange(1, M + 1)
    ax.plot(bit_positions, range_min, '-r', linewidth=1.5, label='Min')
    ax.plot(bit_positions, range_max, '-r', linewidth=1.5, label='Max')

    # Scatter plot for each bit position
    for ii in range(M):
        # All samples (blue)
        ax.scatter(
            np.ones(N) * (ii + 1),
            data_decom[:, ii],
            c='blue', alpha=0.01, s=10
        )

        # Overflow high samples (red, shifted left)
        n_ovf_one = np.sum(ovf_one)
        if n_ovf_one > 0:
            ax.scatter(
                np.ones(n_ovf_one) * (ii + 1) - 0.2,
                data_decom[ovf_one, ii],
                c='red', alpha=0.01, s=10
            )

        # Overflow low samples (yellow, shifted right)
        n_ovf_zero = np.sum(ovf_zero)
        if n_ovf_zero > 0:
            ax.scatter(
                np.ones(n_ovf_zero) * (ii + 1) + 0.2,
                data_decom[ovf_zero, ii],
                c='yellow', alpha=0.01, s=10
            )

        # Add percentage labels
        ax.text(ii + 1, -0.08, f'{ovf_zero_pct[ii]:.1f}%', ha='center', fontsize=8)
        ax.text(ii + 1, 1.05, f'{ovf_one_pct[ii]:.1f}%', ha='center', fontsize=8)

    ax.set_xlim(0, M + 1)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xticks(bit_positions)
    ax.set_xticklabels([str(M - i) for i in range(M)])  # MSB to LSB labels
    ax.set_xlabel('Bit')
    ax.set_ylabel('Residue Distribution')
    ax.set_title('Overflow Check - Residue Distribution per Bit')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[overflowChk] Figure saved to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        'data_decom': data_decom,
        'range_min': range_min,
        'range_max': range_max,
        'ovf_zero_pct': ovf_zero_pct,
        'ovf_one_pct': ovf_one_pct
    }


def test_overflow_check():
    """Test function with synthetic SAR ADC data."""
    import os
    import sys

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 60)
    print("Testing overflowChk.py")
    print("=" * 60)

    # Test parameters
    num_bits = 10
    num_samples = 4096

    # Generate ideal binary weights
    weights = np.array([2**(num_bits - 1 - i) for i in range(num_bits)], dtype=float)
    weights = weights / np.sum(weights)  # Normalize

    print(f"[Config] num_bits = {num_bits}, num_samples = {num_samples}")
    print(f"[Weights] = {weights}")

    # Generate sine wave input
    fs = 1e6  # Sample rate
    fin = fs / num_samples * 31  # Coherent sampling
    t = np.arange(num_samples) / fs
    v_in = 0.5 + 0.5 * np.sin(2 * np.pi * fin * t)  # [0, 1] range

    # Ideal quantization
    levels = 2 ** num_bits
    codes_decimal = np.floor(v_in * levels).astype(int)
    codes_decimal = np.clip(codes_decimal, 0, levels - 1)

    # Convert to binary (MSB first)
    raw_code = np.zeros((num_samples, num_bits), dtype=float)
    for i in range(num_bits):
        raw_code[:, i] = (codes_decimal >> (num_bits - 1 - i)) & 1

    # Run overflow check
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'output_data',
        'test_overflow_check.png'
    )

    result = overflow_check(
        raw_code,
        weights,
        ofb=1,
        save_path=output_path,
        show_plot=False
    )

    print(f"\n[Results]")
    print(f"  data_decom shape: {result['data_decom'].shape}")
    print(f"  range_min: {result['range_min']}")
    print(f"  range_max: {result['range_max']}")
    print(f"  ovf_zero_pct: {result['ovf_zero_pct']}")
    print(f"  ovf_one_pct: {result['ovf_one_pct']}")

    # For ideal ADC with full-scale sine, expect:
    # - MSB residue range should be close to [0, 1]
    # - Lower bits should have progressively narrower ranges
    # - Minimal overflow for ideal binary weights

    print(f"\n[Verification]")
    print(f"  MSB residue range: [{result['range_min'][0]:.4f}, {result['range_max'][0]:.4f}]")
    print(f"  LSB residue range: [{result['range_min'][-1]:.4f}, {result['range_max'][-1]:.4f}]")

    # Check that MSB covers near full range
    if result['range_min'][0] < 0.1 and result['range_max'][0] > 0.9:
        print("  [PASS] MSB residue covers expected range")
    else:
        print("  [WARN] MSB residue range unexpected")

    print(f"\n[Output] Figure saved to: {output_path}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    test_overflow_check()
