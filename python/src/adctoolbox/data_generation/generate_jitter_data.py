"""Generate jitter test data with sweep from 100fs to 1ns."""

import numpy as np
import os


def generate_jitter_signal(rms_jitter, seed=None, j=101, n=2**16, fs=1e9):
    """
    Generate ADC output with clock jitter.

    Args:
        rms_jitter: RMS jitter in seconds
        seed: Random seed
        j: Cycles in n samples
        n: Number of samples
        fs: Sampling frequency (Hz)

    Returns:
        Normalized output [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert time jitter to phase jitter: phase = 2*pi*Fin*t
    fin = fs * j / n
    phase_jitter_std = 2 * np.pi * fin * rms_jitter

    # Generate signal with phase jitter (correct method)
    # data = sin([0:N-1]*J*2*pi/N + randn*phase_jitter) * 0.49 + 0.5 + randn*noise
    t = np.arange(n)
    phase = t * j * 2 * np.pi / n + np.random.randn(n) * phase_jitter_std
    data = np.sin(phase) * 0.49 + 0.5 + np.random.randn(n) * 0.000001

    # Normalize to [-1, 1]
    data = (data - np.mean(data)) / ((np.max(data) - np.min(data)) / 2)

    return data


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "reference_data")
    os.makedirs(output_dir, exist_ok=True)

    # Jitter sweep: 100fs to 1ns (logspace, 20 points)
    jitter_values = np.logspace(-13, -9, 50)  # 0.1ps to 1ns

    print(f"\nGenerating {len(jitter_values)} jitter data files...")
    print(f"Range: 100 fs to 1 ns (logspace)\n")

    for i, jitter in enumerate(jitter_values):
        # Format filename
        if jitter < 1e-12:
            jitter_str = f"{jitter*1e15:.2f}fs".replace('.', 'p')
        elif jitter < 1e-9:
            jitter_str = f"{jitter*1e12:.2f}ps".replace('.', 'p')
        else:
            jitter_str = f"{jitter*1e9:.2f}ns".replace('.', 'p')

        filename = f"jitter_sweep_{i+1:02d}_{jitter_str}.csv"
        filepath = os.path.join(output_dir, filename)

        data = generate_jitter_signal(jitter, seed=2025 + i)
        np.savetxt(filepath, data.reshape(1, -1), delimiter=',', fmt='%.10f')

        print(f"[{i+1:2d}/{len(jitter_values)}] Jitter = {jitter*1e12:9.4f} ps, save into [{filename}]")

if __name__ == "__main__":
    main()
