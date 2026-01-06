"""Unit tests for jitter calculation using analyze_error_by_phase with figure generation."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import analyze_error_by_phase, find_coherent_frequency, analyze_spectrum
from adctoolbox.fundamentals.metrics import calculate_jitter_limit


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


def test_jitter_recovery_at_1ghz():
    """Test jitter recovery at 100 MHz, 1 GHz, and 2 GHz without and with thermal noise."""
    # Setup
    N = 2**16
    Fs = 7e9
    A = 0.49
    DC = 0.0
    base_noise = 50e-6

    # Test 3 frequencies
    fin_targets = [100e6, 1000e6, 2000e6]  # 100 MHz, 1 GHz, 2 GHz
    fin_labels = ['100mhz', '1ghz', '2ghz']

    # Jitter sweep (logarithmic spacing) - 20 points
    jitter_levels = np.logspace(-15, -11, 20)

    # Test with and without thermal noise
    noise_levels = [0.0, base_noise]
    noise_labels = ['Without Thermal Noise', f'With {base_noise*1e6:.0f} uV Thermal Noise']

    for freq_idx, (Fin_target, freq_label) in enumerate(zip(fin_targets, fin_labels)):
        # Find coherent frequency
        Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

        # Create figure with 1 row x 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for idx, (noise_level, noise_label) in enumerate(zip(noise_levels, noise_labels)):
            # Arrays to store results
            jitter_set = []
            jitter_measured = []
            snr_measured = []

            for jitter_rms in jitter_levels:
                # Generate signal with phase jitter
                t = np.arange(N) / Fs

                # Phase jitter model
                phase_noise_rms = 2 * np.pi * Fin * jitter_rms
                phase_jitter = np.random.randn(N) * phase_noise_rms

                signal = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * noise_level

                # Measure jitter using analyze_error_by_phase
                results = analyze_error_by_phase(signal, norm_freq=Fin/Fs, n_bins=100,
                                                include_base_noise=True, create_plot=False)

                # Extract PM noise and convert to jitter
                pm_noise_rad = results['pm_noise_rms_rad']
                jitter_calc = pm_noise_rad / (2 * np.pi * Fin)

                # Calculate SNR using analyze_spectrum
                spec_results = analyze_spectrum(signal, fs=Fs, osr=1, create_plot=False)
                snr_db = spec_results['snr_dbc']

                jitter_set.append(jitter_rms)
                jitter_measured.append(jitter_calc)
                snr_measured.append(snr_db)

            # Convert to numpy arrays
            jitter_set = np.array(jitter_set)
            jitter_measured = np.array(jitter_measured)
            snr_measured = np.array(snr_measured)

            # Calculate theoretical jitter limit SNR
            snr_theoretical = calculate_jitter_limit(Fin, jitter_set)

            # Calculate metrics
            correlation = np.corrcoef(jitter_set, jitter_measured)[0, 1]
            errors_pct = np.abs(jitter_measured - jitter_set) / jitter_set * 100
            avg_error = np.mean(errors_pct)

            # Plot: Measured vs Set jitter (left axis) + SNR (right axis)
            ax1 = axes[idx]
            ax1.loglog(jitter_set*1e15, jitter_set*1e15, 'k--', linewidth=1.5, label='Set jitter')
            ax1.loglog(jitter_set*1e15, jitter_measured*1e15, 'bo', linewidth=2, markersize=8,
                       markerfacecolor='b', label='Calculated jitter')
            ax1.set_xlabel('Set Jitter (fs)', fontsize=12)
            ax1.set_ylabel('Jitter (fs)', fontsize=12, color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_ylim([jitter_set.min()*1e15*0.5, jitter_set.max()*1e15*2])
            ax1.grid(True, which='both', alpha=0.3)

            # Right axis for SNR
            ax2 = ax1.twinx()
            ax2.semilogx(jitter_set*1e15, snr_measured, 's-', color='red', linewidth=2,
                         markersize=8, label='Measured SNR')
            ax2.semilogx(jitter_set*1e15, snr_theoretical, '--', color='darkred', linewidth=2,
                         label='Theoretical SNR (Jitter Limit)')
            ax2.set_ylabel('SNR (dB)', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim([0, 120])

            # Title
            ax1.set_title(f'Fin = {Fin/1e6:.0f} MHz ({noise_label})', fontsize=11, fontweight='bold')

            # Add text annotation
            text_str = f'Corr = {correlation:.4f}\nAvg Err = {avg_error:.2f}%'
            ax1.text(0.5, 0.02, text_str, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=9, framealpha=1)

            print(f'\n[Jitter Test] Fin={Fin/1e6:.0f} MHz, {noise_label}: Corr={correlation:.4f}, Avg Err={avg_error:.2f}%')

            # Assertions
            if noise_level == 0.0:
                # Without noise: strict requirements
                assert correlation > 0.99, f"Correlation too low without noise at {Fin/1e6:.0f} MHz: {correlation:.6f} (expected > 0.99)"
                assert avg_error < 10.0, f"Average error too high at {Fin/1e6:.0f} MHz: {avg_error:.2f}% (expected < 10%)"
            else:
                # With noise: relaxed requirements
                # At lower frequencies (100 MHz), thermal noise dominates jitter, so we expect high error
                if Fin_target == 100e6:
                    assert correlation > 0.95, f"Correlation too low with noise at {Fin/1e6:.0f} MHz: {correlation:.6f} (expected > 0.95)"
                    # No avg_error check for 100 MHz - thermal noise dominates
                else:
                    assert correlation > 0.98, f"Correlation too low with noise at {Fin/1e6:.0f} MHz: {correlation:.6f} (expected > 0.98)"
                    assert avg_error < 15.0, f"Average error too high with noise at {Fin/1e6:.0f} MHz: {avg_error:.2f}% (expected < 15%)"

        # Save figure
        fig.suptitle(f'Jitter Recovery Using analyze_error_by_phase (Fs = {Fs/1e9:.0f} GHz, Fin = {Fin/1e6:.0f} MHz)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig_path = output_dir / f'test_jitter_recovery_{freq_label}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Verify file was created
        assert fig_path.exists(), f"Figure file not created: {fig_path}"
        assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

        print(f'[Figure saved] -> {fig_path}')


@pytest.mark.parametrize("fin_target,expected_correlation", [
    (100e6, 0.99),   # 100 MHz
    (1000e6, 0.99),  # 1 GHz
    (2000e6, 0.99),  # 2 GHz
])
def test_jitter_recovery_frequency_sweep(fin_target, expected_correlation):
    """Test jitter recovery at different frequencies."""
    # Setup
    N = 2**16
    Fs = 7e9
    A = 0.49
    DC = 0.0

    # Find coherent frequency
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=fin_target, n_fft=N)

    # Jitter sweep
    jitter_levels = np.logspace(-15, -11, 20)  # Fewer points for speed

    # Arrays to store results
    jitter_set = []
    jitter_measured = []

    for jitter_rms in jitter_levels:
        # Generate signal
        t = np.arange(N) / Fs
        phase_noise_rms = 2 * np.pi * Fin * jitter_rms
        phase_jitter = np.random.randn(N) * phase_noise_rms
        signal = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC

        # Measure jitter
        results = analyze_error_by_phase(signal, norm_freq=Fin/Fs, n_bins=100,
                                        include_base_noise=True, create_plot=False)
        pm_noise_rad = results['pm_noise_rms_rad']
        jitter_calc = pm_noise_rad / (2 * np.pi * Fin)

        jitter_set.append(jitter_rms)
        jitter_measured.append(jitter_calc)

    # Convert to arrays and calculate metrics
    jitter_set = np.array(jitter_set)
    jitter_measured = np.array(jitter_measured)
    correlation = np.corrcoef(jitter_set, jitter_measured)[0, 1]
    avg_error = np.mean(np.abs(jitter_measured - jitter_set) / jitter_set * 100)

    print(f'\n[Freq Sweep] Fin={Fin/1e6:.0f} MHz: Corr={correlation:.4f}, Avg Err={avg_error:.2f}%')

    # Assertion
    assert correlation > expected_correlation, \
        f"Correlation too low at {Fin/1e6:.0f} MHz: {correlation:.6f} (expected > {expected_correlation})"


if __name__ == '__main__':
    """Run jitter calculation tests standalone"""
    print('='*80)
    print('RUNNING JITTER CALCULATION TESTS')
    print('='*80)

    test_jitter_recovery_at_1ghz()
    test_jitter_recovery_frequency_sweep(100e6, 0.99)
    test_jitter_recovery_frequency_sweep(1000e6, 0.99)
    test_jitter_recovery_frequency_sweep(2000e6, 0.99)

    print('\n' + '='*80)
    print(f'** All jitter calculation tests completed! **')
    print(f'** Figures saved to: {output_dir.resolve()} **')
    print('='*80)
