"""Unit tests for harmonic decomposition time-domain analysis with figure generation."""

import warnings

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_decomposition_polar, analyze_decomposition_time
from adctoolbox.aout import decompose_harmonic_error, decompose_harmonics


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def _clean_harmonic_signal(n=2048, freq_bin=37.25):
    t = np.arange(n)
    freq = freq_bin / n
    signal = (
        0.45 * np.sin(2 * np.pi * freq * t + 0.2)
        + 0.01 * np.sin(2 * np.pi * 2 * freq * t - 0.4)
        + 0.1
    )
    return signal, freq


def test_decompose_harmonics_basic():
    """Test harmonic decomposition for thermal noise vs nonlinearity vs glitches."""
    # Setup parameters
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.25
    DC = 0.5
    base_noise = 50e-6
    rng = np.random.default_rng(2026062215)

    sig_ac = A * np.sin(2 * np.pi * Fin * t)
    sig_ideal = sig_ac + DC
    print(f"\n[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}")

    # Case 1: Ideal ADC with Thermal Noise
    sig_noise = sig_ideal + rng.standard_normal(N) * base_noise

    # Case 2: ADC with Nonlinearity
    k2 = 0.001
    k3 = 0.005
    sig_nonlin = DC + sig_ac + k2 * sig_ac**2 + k3 * sig_ac**3 + rng.standard_normal(N) * base_noise

    # Case 3: ADC with Glitches
    glitch_prob = 0.01
    glitch_amplitude = 0.1
    glitch_mask = rng.random(N) < glitch_prob
    glitch = glitch_mask * glitch_amplitude
    sig_glitch = sig_ideal + glitch + rng.standard_normal(N) * base_noise

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Harmonic Decomposition - Thermal Noise vs Nonlinearity vs Glitches',
                 fontsize=14, fontweight='bold')

    analyze_decomposition_time(sig_noise, ax=axes[0], title='Thermal Noise Only')
    analyze_decomposition_time(sig_nonlin, ax=axes[1], title=f'Nonlinearity (k2={k2:.3f}, k3={k3:.3f})')
    analyze_decomposition_time(sig_glitch, ax=axes[2], title=f'Glitches (prob={glitch_prob*100:.2f}%, amp={glitch_amplitude:.1f})')

    plt.tight_layout()

    fig_path = output_dir / 'test_decompose_harmonics.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Save fig] -> [{fig_path.resolve()}]\n")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


@pytest.mark.parametrize("k3_value", [0.001, 0.005, 0.01])
def test_decompose_harmonics_nonlinearity_levels(k3_value):
    """Test harmonic decomposition at different nonlinearity levels."""
    N = 2**13
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.25
    DC = 0.5
    base_noise = 50e-6
    rng = np.random.default_rng(2026062216)

    sig_ac = A * np.sin(2 * np.pi * Fin * t)
    sig_nonlin = DC + sig_ac + k3_value * sig_ac**3 + rng.standard_normal(N) * base_noise

    # Just run the analysis without creating plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    analyze_decomposition_time(sig_nonlin, ax=ax, title=f'k3={k3_value:.3f}')
    plt.tight_layout()
    plt.close(fig)

    print(f"\n[k3={k3_value:.3f}] -> [PASS]")


def test_decompose_harmonic_error_uses_fixed_frequency():
    """Known frequency plus max_iterations=0 should remain fixed."""
    signal, freq = _clean_harmonic_signal()

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = decompose_harmonic_error(
            signal,
            n_harmonics=2,
            frequency=freq,
            max_iterations=0,
        )

    runtime_warnings = [w for w in records if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert result['fundamental_freq'] == pytest.approx(freq)
    assert result['residual_rms'] < 1e-12


def test_decomposition_wrappers_forward_frequency_options():
    """Time and polar wrappers should expose the core fit controls."""
    signal, freq = _clean_harmonic_signal()

    time_result = analyze_decomposition_time(
        signal,
        harmonic=2,
        frequency=freq,
        max_iterations=0,
        create_plot=False,
    )
    polar_result = analyze_decomposition_polar(
        signal,
        harmonic=2,
        frequency=freq,
        max_iterations=0,
        create_plot=False,
    )

    assert time_result['fundamental_freq'] == pytest.approx(freq)
    assert polar_result['fundamental_freq'] == pytest.approx(freq)
    assert time_result['residual_rms'] < 1e-12
    assert polar_result['residual_rms'] < 1e-12


def test_legacy_decompose_harmonics_uses_provided_frequency():
    """The legacy freq argument should drive decomposition instead of being ignored."""
    signal, freq = _clean_harmonic_signal()

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        fundamental, other, harmonic, noise = decompose_harmonics(
            signal,
            freq=freq,
            harmonic=2,
            disp=0,
        )

    runtime_warnings = [w for w in records if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert fundamental.shape == other.shape == harmonic.shape == noise.shape == signal.shape
    assert np.std(noise) < 1e-12


def test_decompose_harmonic_error_fit_options_reduce_near_dc_residual():
    """Extra fit iterations should remove near-DC false residual structure."""
    n = 8192
    freq = 0.70 / n
    t = np.arange(n)
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1

    with pytest.warns(RuntimeWarning, match="did not converge"):
        one_iter = decompose_harmonic_error(
            signal,
            n_harmonics=5,
            max_iterations=1,
        )

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        five_iter = decompose_harmonic_error(
            signal,
            n_harmonics=5,
            max_iterations=5,
        )

    runtime_warnings = [w for w in records if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert abs(one_iter['fundamental_freq'] * n - 0.70) > 0.01
    assert five_iter['fundamental_freq'] == pytest.approx(freq, abs=1e-12)
    assert five_iter['residual_rms'] < one_iter['residual_rms'] * 1e-8


def test_decompose_harmonic_error_known_frequency_handles_strong_hd2():
    """Known frequency should bypass auto-detection when HD2 is the largest tone."""
    n = 4096
    freq = 37 / n
    t = np.arange(n)
    signal = (
        0.05 * np.sin(2 * np.pi * freq * t + 0.2)
        + 0.4 * np.sin(2 * np.pi * 2 * freq * t - 0.4)
        + 0.1
    )

    with pytest.warns(RuntimeWarning, match="did not converge"):
        auto_result = decompose_harmonic_error(signal, n_harmonics=3)

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        fixed_result = decompose_harmonic_error(
            signal,
            n_harmonics=3,
            frequency=freq,
            max_iterations=0,
        )

    runtime_warnings = [w for w in records if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert auto_result['fundamental_freq'] * n == pytest.approx(2 * freq * n, abs=1e-2)
    assert fixed_result['fundamental_freq'] == pytest.approx(freq)
    assert fixed_result['magnitudes'][0] == pytest.approx(0.1)
    assert fixed_result['magnitudes'][1] == pytest.approx(0.8)
    assert fixed_result['residual_rms'] < auto_result['residual_rms'] * 1e-10


if __name__ == '__main__':
    """Run decompose_harmonics tests standalone"""
    print('='*80)
    print('RUNNING DECOMPOSE_HARMONICS TESTS')
    print('='*80)

    test_decompose_harmonics_basic()
    test_decompose_harmonics_nonlinearity_levels(0.001)
    test_decompose_harmonics_nonlinearity_levels(0.005)
    test_decompose_harmonics_nonlinearity_levels(0.01)

    print('\n' + '='*80)
    print(f'** All decompose_harmonics tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
