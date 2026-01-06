"""Unit tests for analyze_error_by_phase with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_by_phase


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_analyze_error_by_phase_pure_cases():
    """Test phase error analysis with pure noise cases (thermal, AM, PM)."""
    # Base parameters
    N = 2**16
    Fs = 800e6
    Fin = 10.1234567e6
    norm_freq = Fin / Fs
    t = np.arange(N) / Fs
    A = 0.49
    DC = 0.5
    phase_clean = 2 * np.pi * Fin * t

    print(f"\n[Config] Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, N={N}, A={A}")

    # Define pure test cases
    test_cases = [
        {'am': 0,    'pm': 0,    'thermal': 50e-6, 'label': 'Thermal Only',
         'expected': {'am': 0, 'pm': 0, 'baseline': 50}},
        {'am': 50e-6, 'pm': 0,    'thermal': 0,     'label': 'AM Only',
         'expected': {'am': 50, 'pm': 0, 'baseline': 0}},
        {'am': 0,    'pm': 50e-6, 'thermal': 0,     'label': 'PM Only',
         'expected': {'am': 0, 'pm': 50, 'baseline': 0}},
    ]

    # Generate signals
    for case in test_cases:
        am_noise = np.random.randn(N) * case['am'] if case['am'] > 0 else 0
        pm_noise = np.random.randn(N) * case['pm'] / A if case['pm'] > 0 else 0
        th_noise = np.random.randn(N) * case['thermal'] if case['thermal'] > 0 else 0
        case['signal'] = (A + am_noise) * np.sin(phase_clean + pm_noise) + DC + th_noise

    # Test both baseline modes
    for base_noise_mode in [True, False]:
        base_noise_suffix = 'with_base_noise' if base_noise_mode else 'no_base_noise'
        print(f"\n{'='*80}")
        print(f"Mode: include_base_noise={base_noise_mode}")
        print(f"{'='*80}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        mode_str = 'With Base Noise' if base_noise_mode else 'Without Base Noise'
        fig.suptitle(f'Phase Error Analysis - Pure Noise Cases ({mode_str})', fontsize=14, fontweight='bold')

        for i, case in enumerate(test_cases):
            plt.sca(axes[i])
            results = analyze_error_by_phase(case['signal'], n_bins=100,
                                            include_base_noise=base_noise_mode,
                                            title=case['label'])

            exp = case['expected']
            exp_total = np.sqrt((exp['am']**2)/2 + (exp['pm']**2)/2 + exp['baseline']**2)

            print(f"{case['label']:15s}")
            print(f"  [Expected  ] [AM={exp['am']:4.0f} uV] [PM={exp['pm']:4.0f} uV] [Base={exp['baseline']:4.0f} uV] [Total={exp_total:4.1f} uV]")
            print(f"  [Calculated] [AM={results['am_noise_rms_v']*1e6:4.1f} uV] [PM={results['pm_noise_rms_v']*1e6:4.1f} uV] [Base={results['base_noise_rms_v']*1e6:4.1f} uV] [Total={results['total_rms_v']*1e6:4.1f} uV] [R2={results['r_squared_binned']:.3f}]\n")

            # Basic assertions - should have reasonable R² and total RMS
            assert results['r_squared_binned'] > 0.5, f"R² too low for {case['label']}: {results['r_squared_binned']:.3f}"
            assert results['total_rms_v'] > 0, f"Total RMS should be positive for {case['label']}"

        plt.tight_layout()
        fig_path = output_dir / f'test_analyze_error_by_phase_pure_{base_noise_suffix}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Save fig] -> [{fig_path.resolve()}]")

        # Verify file was created
        assert fig_path.exists(), f"Figure file not created: {fig_path}"
        assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


def test_analyze_error_by_phase_mixed_cases():
    """Test phase error analysis with mixed noise cases."""
    # Base parameters
    N = 2**16
    Fs = 800e6
    Fin = 10.1234567e6
    norm_freq = Fin / Fs
    t = np.arange(N) / Fs
    A = 0.49
    DC = 0.5
    phase_clean = 2 * np.pi * Fin * t

    # Define mixed test cases
    test_cases = [
        {'am': 50e-6, 'pm': 0,    'thermal': 30e-6, 'label': 'AM + Thermal',
         'expected': {'am': 50, 'pm': 0, 'baseline': 30}},
        {'am': 0,    'pm': 50e-6, 'thermal': 30e-6, 'label': 'PM + Thermal',
         'expected': {'am': 0, 'pm': 50, 'baseline': 30}},
        {'am': 50e-6, 'pm': 50e-6, 'thermal': 30e-6, 'label': 'AM + PM + Thermal',
         'expected': {'am': 0, 'pm': 0, 'baseline': 58}},
    ]

    # Generate signals
    for case in test_cases:
        am_noise = np.random.randn(N) * case['am'] if case['am'] > 0 else 0
        pm_noise = np.random.randn(N) * case['pm'] / A if case['pm'] > 0 else 0
        th_noise = np.random.randn(N) * case['thermal'] if case['thermal'] > 0 else 0
        case['signal'] = (A + am_noise) * np.sin(phase_clean + pm_noise) + DC + th_noise

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Phase Error Analysis - Mixed Noise Cases', fontsize=14, fontweight='bold')

    for i, case in enumerate(test_cases):
        plt.sca(axes[i])
        results = analyze_error_by_phase(case['signal'], n_bins=100,
                                        include_base_noise=True,
                                        title=case['label'])

        exp = case['expected']
        exp_total = np.sqrt((exp['am']**2)/2 + (exp['pm']**2)/2 + exp['baseline']**2)

        print(f"\n{case['label']:20s}")
        print(f"  [Expected  ] [AM={exp['am']:4.0f} uV] [PM={exp['pm']:4.0f} uV] [Base={exp['baseline']:4.0f} uV] [Total={exp_total:4.1f} uV]")
        print(f"  [Calculated] [AM={results['am_noise_rms_v']*1e6:4.1f} uV] [PM={results['pm_noise_rms_v']*1e6:4.1f} uV] [Base={results['base_noise_rms_v']*1e6:4.1f} uV] [Total={results['total_rms_v']*1e6:4.1f} uV] [R2={results['r_squared_binned']:.3f}]")

        # Assertions
        assert results['r_squared_binned'] > 0.5, f"R² too low for {case['label']}: {results['r_squared_binned']:.3f}"
        assert results['total_rms_v'] > 0, f"Total RMS should be positive for {case['label']}"

    plt.tight_layout()
    fig_path = output_dir / 'test_analyze_error_by_phase_mixed.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[Save fig] -> [{fig_path.resolve()}]")

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"


@pytest.mark.parametrize("n_bins", [50, 100, 200])
def test_analyze_error_by_phase_bin_count(n_bins):
    """Test analyze_error_by_phase with different bin counts."""
    N = 2**14
    Fs = 800e6
    Fin = 10.1234567e6
    t = np.arange(N) / Fs
    A = 0.49
    DC = 0.5

    # Simple thermal noise case
    thermal_noise = 50e-6
    signal = A * np.sin(2 * np.pi * Fin * t) + DC + np.random.randn(N) * thermal_noise

    # Just run the analysis
    results = analyze_error_by_phase(signal, n_bins=n_bins, include_base_noise=True, create_plot=False)

    assert results['r_squared_binned'] > 0.5, f"R² too low with {n_bins} bins: {results['r_squared_binned']:.3f}"
    print(f"\n[Bin Count={n_bins}] R²={results['r_squared_binned']:.3f} -> [PASS]")


if __name__ == '__main__':
    """Run analyze_error_by_phase tests standalone"""
    print('='*80)
    print('RUNNING ANALYZE_ERROR_BY_PHASE TESTS')
    print('='*80)

    test_analyze_error_by_phase_pure_cases()
    test_analyze_error_by_phase_mixed_cases()
    test_analyze_error_by_phase_bin_count(50)
    test_analyze_error_by_phase_bin_count(100)
    test_analyze_error_by_phase_bin_count(200)

    print('\n' + '='*80)
    print(f'** All analyze_error_by_phase tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
