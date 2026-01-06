"""Unit tests for INL/DNL analysis from sine sweep with different record lengths."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, analyze_spectrum, analyze_inl_from_sine


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_inl_from_sine_sweep_length():
    """Test INL/DNL analysis with different record lengths."""
    # Parameters
    n_bits = 16
    full_scale = 1.0
    fs = 800e6
    fin_target = 80e6

    # Nonidealities
    A = 0.49
    DC = 0.5
    base_noise = 50e-6
    hd2_dB, hd3_dB = -80, -66

    # Compute HD coefficients
    hd2_amp = 10**(hd2_dB/20)
    hd3_amp = 10**(hd3_dB/20)
    k2 = hd2_amp / (A / 2)
    k3 = hd3_amp / (A**2 / 4)

    N_list = [2**10, 2**14, 2**18]
    n_plots = len(N_list)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 6))

    print(f"\n[INL/DNL Sweep] [Fs = {fs/1e6:.0f} MHz, Fin = {fin_target/1e6:.0f} MHz]")
    print(f"  [HD2 = {hd2_dB} dB, HD3 = {hd3_dB} dB, Noise = {base_noise*1e6:.1f} uV]\n")

    for idx, N in enumerate(N_list):
        fin, J = find_coherent_frequency(fs, fin_target, N)
        t = np.arange(N) / fs
        sinewave = A * np.sin(2 * np.pi * fin * t)
        signal_distorted = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * base_noise

        result = analyze_spectrum(signal_distorted, fs=fs, create_plot=False)

        # Analyze INL/DNL and plot
        plt.sca(axes[idx])
        result_inl = analyze_inl_from_sine(
            signal_distorted,
            num_bits=n_bits,
            full_scale=full_scale,
            clip_percent=0.01,
            col_title=f'N = 2^{int(np.log2(N))}'
        )
        inl, dnl, code = result_inl['inl'], result_inl['dnl'], result_inl['code']

        print(f"  [N = 2^{int(np.log2(N)):2d} = {N:5d}] [ENOB = {result['enob']:5.2f}] [INL: {np.min(inl):5.2f} to {np.max(inl):5.2f}] [DNL: {np.min(dnl):5.2f} to {np.max(dnl):5.2f}] LSB")

        # Assertions
        assert 'inl' in result_inl, "Result should contain INL"
        assert 'dnl' in result_inl, "Result should contain DNL"
        assert len(inl) > 0, "INL array should not be empty"
        assert len(dnl) > 0, "DNL array should not be empty"

    fig.suptitle(f'INL/DNL Sweep: Record Length Comparison (Fs={fs/1e6:.0f} MHz, Fin={fin_target/1e6:.0f} MHz)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = output_dir / 'test_inl_from_sine_sweep_length.png'
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"\n[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


if __name__ == '__main__':
    """Run inl_from_sine_sweep_length tests standalone"""
    print('='*80)
    print('RUNNING INL_FROM_SINE_SWEEP_LENGTH TESTS')
    print('='*80)

    test_inl_from_sine_sweep_length()

    print('\n' + '='*80)
    print(f'** All inl_from_sine_sweep_length tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
