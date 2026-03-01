"""Unit tests for dout_dashboard with different ADC architectures.

This test file validates the generate_dout_dashboard function with digital bit data
from three different ADC architectures:
1. 2-stage pipeline ADC (gen_pipeline2s_dout.m)
2. 3-stage pipeline ADC (gen_pipeline3s_dout.m)
3. SAR ADC with different weight configurations (gen_sar_dout.m)

Common signal parameters (from common_gen_dout.m):
- N = 2^13
- Fs = 1e9
- Fin ≈ 300e6
- A = 0.499
- DC = 0.5
- Noise_rms = 50e-6
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency
from adctoolbox.toolset import generate_dout_dashboard


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


# Common signal parameters from common_gen_dout.m
N = 2**13
Fs = 1e9
Fin_target = 300e6
A = 0.499
DC = 0.5
Noise_rms = 50e-6


def bin_saturate(x, N_bits):
    """Encode bits: Prevent digital overflow/wrap-around; saturate codes exceeding the maximum.

    Matches MATLAB binSaturate() function.
    """
    x = np.floor(x).astype(int)
    x = np.clip(x, 0, 2**N_bits - 1)
    # Convert to binary bits (MSB to LSB)
    bits = (x[:, None] >> np.arange(N_bits - 1, -1, -1)) & 1
    return bits


def generate_common_signal():
    """Generate common test signal matching common_gen_dout.m."""
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)
    ideal_phase = 2 * np.pi * Fin * np.arange(N) / Fs
    signal = A * np.sin(ideal_phase) + DC
    # Note: MATLAB adds noise but we'll skip it for deterministic testing
    return signal, Fin, Fin_bin, ideal_phase


def generate_pipeline2s_bits(signal):
    """Generate 2-stage pipeline ADC bits matching gen_pipeline2s_dout.m.

    Architecture:
    - Stage 1: N1=3 bits, Gain G1=8
    - Stage 2: N2=8 bits
    - Redundancy: R1 = log2(2^N1 / G1) = log2(8/8) = 0 bits
    - Total bits: N1 + N2 = 11 bits

    MATLAB code:
        offset1 = (1 - G1 * 1 / 2^N1) / 2;
        msb = floor(sig*2^N1) / 2^N1;          % stage 1
        residue1 = sig - msb;
        lsb = floor((G1 * residue1 + offset1)*2^N2) / 2^N2;  % stage 2
        msb_bits = binSaturate(msb*2^N1, N1);
        lsb_bits = binSaturate(lsb*2^N2, N2);
        dout = [msb_bits, lsb_bits];
    """
    N1, G1, N2 = 3, 8, 8
    R1 = int(np.log2(2**N1 / G1))  # Redundancy = 0

    # Stage 1
    offset1 = (1 - G1 * 1 / 2**N1) / 2
    msb = np.floor(signal * 2**N1) / 2**N1
    residue1 = signal - msb

    # Stage 2
    lsb = np.floor((G1 * residue1 + offset1) * 2**N2) / 2**N2

    # Convert to bits
    msb_bits = bin_saturate(msb * 2**N1, N1)
    lsb_bits = bin_saturate(lsb * 2**N2, N2)

    # Concatenate bits
    dout = np.concatenate([msb_bits, lsb_bits], axis=1)

    # Compute weights (normalized)
    w1 = 2.0 ** np.arange(N1 + N2 - R1 - 1, N2 - R1 - 1, -1)
    w2 = 2.0 ** np.arange(N2 - 1, -1, -1)
    weights = np.concatenate([w1, w2])
    weights = weights / weights.sum()

    return dout, weights


def generate_pipeline3s_bits(signal):
    """Generate 3-stage pipeline ADC bits matching gen_pipeline3s_dout.m.

    Architecture:
    - Stage 1: N1=3 bits, Gain G1=4
    - Stage 2: N2=3 bits, Gain G2=4
    - Stage 3: N3=8 bits
    - Redundancy: R1 = log2(2^N1 / G1) = 1, R2 = log2(2^N2 / G2) = 1
    - Total bits: N1 + N2 + N3 = 14 bits

    MATLAB code:
        offset1 = (1 - G1 * 1 / 2^N1) / 2;
        offset2 = (1 - G2 * 1 / 2^N2) / 2;
        msb = floor(sig*2^N1) / 2^N1;          % stage 1
        residue1 = sig - msb;
        lsb = floor((G1 * residue1 + offset1)*2^N2) / 2^N2;  % stage 2
        residue2 = G1 * residue1 + offset1 - lsb;
        lsb2 = floor((G2 * residue2 + offset2)*2^N3) / 2^N3; % stage 3
    """
    N1, G1 = 3, 4
    N2, G2 = 3, 4
    N3 = 8
    R1 = int(np.log2(2**N1 / G1))  # Redundancy = 1
    R2 = int(np.log2(2**N2 / G2))  # Redundancy = 1

    # Stage 1
    offset1 = (1 - G1 * 1 / 2**N1) / 2
    msb = np.floor(signal * 2**N1) / 2**N1
    residue1 = signal - msb

    # Stage 2
    offset2 = (1 - G2 * 1 / 2**N2) / 2
    lsb = np.floor((G1 * residue1 + offset1) * 2**N2) / 2**N2
    residue2 = G1 * residue1 + offset1 - lsb

    # Stage 3
    lsb2 = np.floor((G2 * residue2 + offset2) * 2**N3) / 2**N3

    # Convert to bits
    msb_bits = bin_saturate(msb * 2**N1, N1)
    lsb_bits = bin_saturate(lsb * 2**N2, N2)
    lsb2_bits = bin_saturate(lsb2 * 2**N3, N3)

    # Concatenate bits
    dout = np.concatenate([msb_bits, lsb_bits, lsb2_bits], axis=1)

    # Compute weights (normalized)
    w1 = 2.0 ** np.arange(N1 + N2 + N3 - R2 - R1 - 1, N2 + N3 - R2 - R1 - 1, -1)
    w2 = 2.0 ** np.arange(N2 + N3 - R2 - 1, N3 - R2 - 1, -1)
    w3 = 2.0 ** np.arange(N3 - 1, -1, -1)
    weights = np.concatenate([w1, w2, w3])
    weights = weights / weights.sum()

    return dout, weights


def generate_sar_bits(signal, CDAC):
    """Generate SAR ADC bits matching gen_sar_dout.m.

    MATLAB code:
        sig = 2*A*sin(ideal_phase);  % Note: no DC offset!
        FS = 1;
        weight_voltage = CDAC / sum(CDAC) * FS;
        residue = sig;
        dout = zeros(N, B);
        for j = 1:B
            dout(:, j) = (residue > 0);
            delta_cdac = (2 * dout(:, j) - 1) * weight_voltage(j);
            if j < B
                residue = residue - delta_cdac;
            end
        end
    """
    CDAC = np.array(CDAC, dtype=float)
    B = len(CDAC)
    FS = 1.0

    weight_voltage = CDAC / CDAC.sum() * FS

    # SAR conversion process
    dout = np.zeros((len(signal), B), dtype=int)
    residue = signal.copy()

    for j in range(B):
        dout[:, j] = (residue > 0).astype(int)
        delta_cdac = (2 * dout[:, j] - 1) * weight_voltage[j]
        if j < B - 1:
            residue = residue - delta_cdac

    # Compute nominal weights (LSB dummy divided by 2)
    nominal_weight = CDAC.copy()
    nominal_weight[-1] = nominal_weight[-1] / 2  # LSB dummy
    weights = nominal_weight / nominal_weight.sum()

    return dout, weights


def test_dout_dashboard_pipeline2s():
    """Test dashboard with 2-stage pipeline ADC architecture."""
    print(f"\n[Test] 2-stage Pipeline ADC Dashboard")
    print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin≈{Fin_target/1e6:.0f} MHz, N={N}")

    # Generate signal and bits
    signal, Fin, Fin_bin, _ = generate_common_signal()
    bits, weights = generate_pipeline2s_bits(signal)

    print(f"[Pipeline2s] N1=3, G1=8, N2=8 → {bits.shape[1]} bits")
    print(f"[Pipeline2s] Weights: {weights[:5]} ... {weights[-3:]}")

    # Generate dashboard
    fig_path = output_dir / 'test_dout_dashboard_pipeline2s.png'
    fig, axes = generate_dout_dashboard(
        bits=bits,
        freq=Fin/Fs,
        weights=weights,
        output_path=fig_path
    )

    print(f"[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Verify dashboard structure
    assert fig is not None, "Figure should be created"
    assert axes is not None, "Axes should be created"
    assert len(axes) == 6, "Dashboard should have 6 subplots"


def test_dout_dashboard_pipeline3s():
    """Test dashboard with 3-stage pipeline ADC architecture."""
    print(f"\n[Test] 3-stage Pipeline ADC Dashboard")
    print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin≈{Fin_target/1e6:.0f} MHz, N={N}")

    # Generate signal and bits
    signal, Fin, Fin_bin, _ = generate_common_signal()
    bits, weights = generate_pipeline3s_bits(signal)

    print(f"[Pipeline3s] N1=3, G1=4, N2=3, G2=4, N3=8 → {bits.shape[1]} bits")
    print(f"[Pipeline3s] Weights: {weights[:5]} ... {weights[-3:]}")

    weights[2] *= 0.99
    # Generate dashboard
    fig_path = output_dir / 'test_dout_dashboard_pipeline3s.png'
    fig, axes = generate_dout_dashboard(
        bits=bits,
        freq=Fin/Fs,
        weights=weights,
        output_path=fig_path
    )

    print(f"[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Verify dashboard structure
    assert fig is not None, "Figure should be created"
    assert axes is not None, "Axes should be created"
    assert len(axes) == 6, "Dashboard should have 6 subplots"


def test_dout_dashboard_sar_binary():
    """Test dashboard with SAR ADC using binary CDAC weights."""
    print(f"\n[Test] SAR ADC Dashboard (Binary Weights)")
    print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin≈{Fin_target/1e6:.0f} MHz, N={N}")

    # Generate signal (SAR uses different signal: 2*A*sin, no DC!)
    _, Fin, Fin_bin, ideal_phase = generate_common_signal()
    signal_sar = 2 * A * np.sin(ideal_phase)  # Note: no DC offset for SAR

    CDAC = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1]
    bits, weights = generate_sar_bits(signal_sar, CDAC)

    print(f"[SAR Binary] {bits.shape[1]} bits")
    print(f"[SAR Binary] CDAC: {CDAC}")
    print(f"[SAR Binary] Weights (with LSB/2): {weights}")

    # Generate dashboard
    fig_path = output_dir / 'test_dout_dashboard_sar_binary.png'
    fig, axes = generate_dout_dashboard(
        bits=bits,
        freq=Fin/Fs,
        weights=weights,
        output_path=fig_path
    )

    print(f"[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Verify dashboard structure
    assert fig is not None, "Figure should be created"
    assert axes is not None, "Axes should be created"
    assert len(axes) == 6, "Dashboard should have 6 subplots"


def test_dout_dashboard_sar_subradix():
    """Test dashboard with SAR ADC using sub-radix CDAC weights."""
    print(f"\n[Test] SAR ADC Dashboard (Sub-radix Weights)")
    print(f"[Config] Fs={Fs/1e6:.0f} MHz, Fin≈{Fin_target/1e6:.0f} MHz, N={N}")

    # Generate signal (SAR uses different signal: 2*A*sin, no DC!)
    _, Fin, Fin_bin, ideal_phase = generate_common_signal()
    signal_sar = 2 * A * np.sin(ideal_phase)  # Note: no DC offset for SAR

    CDAC = [800, 440, 230, 122, 63, 32, 16, 8, 4, 2, 1, 1]
    bits, weights = generate_sar_bits(signal_sar, CDAC)

    print(f"[SAR Sub-radix] {bits.shape[1]} bits")
    print(f"[SAR Sub-radix] CDAC: {CDAC}")
    print(f"[SAR Sub-radix] Weights (with LSB/2): {weights}")

    # Generate dashboard
    fig_path = output_dir / 'test_dout_dashboard_sar_subradix.png'
    fig, axes = generate_dout_dashboard(
        bits=bits,
        freq=Fin/Fs,
        weights=weights,
        output_path=fig_path
    )

    print(f"[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Verify dashboard structure
    assert fig is not None, "Figure should be created"
    assert axes is not None, "Axes should be created"
    assert len(axes) == 6, "Dashboard should have 6 subplots"


if __name__ == '__main__':
    """Run dout_dashboard tests standalone"""
    print('='*80)
    print('RUNNING DOUT_DASHBOARD TESTS')
    print('='*80)

    test_dout_dashboard_pipeline2s()
    test_dout_dashboard_pipeline3s()
    test_dout_dashboard_sar_binary()
    test_dout_dashboard_sar_subradix()

    print('\n' + '='*80)
    print(f'** All dout_dashboard tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
