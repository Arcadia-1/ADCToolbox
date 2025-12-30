"""Unit tests for window sweep with coherent and non-coherent signals.

This test sweeps through 8 window types and records SNDR and ENOB metrics
for both coherent and non-coherent signals to verify automatic side_bin selection.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox.spectrum.analyze_spectrum import analyze_spectrum
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.fundamentals.frequency import find_coherent_frequency
from adctoolbox import amplitudes_to_snr, snr_to_nsd


# Create output directory for test figures
output_dir = Path(__file__).parent / "test_output"
output_dir.mkdir(exist_ok=True)


# Window types to test (matching exp_s08_windowing_deep_dive.py)
WINDOW_TYPES = [
    'rectangular',
    'hann',
    'hamming',
    'blackman',
    'blackmanharris',
    'flattop',
    'kaiser',
    'chebwin'
]


def test_window_sweep_coherent():
    """Test all window types with COHERENT signal (automatic side_bin selection)."""
    # Signal parameters
    N_fft = 2**16
    Fs = 100e6
    A = 0.5
    noise_rms = 10e-6
    Fin_target = 10e6

    # Generate coherent frequency
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N_fft)

    # Harmonic distortion parameters
    hd2_dB = -120
    hd3_dB = -100
    hd2_amp = 10**(hd2_dB/20)
    hd3_amp = 10**(hd3_dB/20)
    k2 = hd2_amp / (A / 2)
    k3 = -hd3_amp / (A**2 / 4)

    # Generate signal with harmonics and noise
    t = np.arange(N_fft) / Fs
    sig_ideal = A * np.sin(2*np.pi*Fin*t)
    signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    print('\n' + '='*80)
    print('COHERENT SIGNAL - WINDOW SWEEP (Automatic side_bin)')
    print('='*80)
    print(f'Fs={Fs/1e6:.2f} MHz, Fin={Fin/1e6:.6f} MHz (bin {Fin_bin}), N={N_fft}, A={A:.3f} Vpeak')
    print(f'Noise RMS={noise_rms*1e6:.2f} uVrms, HD2={hd2_dB} dB, HD3={hd3_dB} dB')
    print()

    # Create single figure with 2x4 subplots
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    results = []
    for idx, win_type in enumerate(WINDOW_TYPES):
        # Use analyze_spectrum with automatic side_bin selection
        plt.sca(axes[idx])
        metrics = analyze_spectrum(signal, fs=Fs, win_type=win_type)

        # Extract metrics
        enob = metrics['enob']
        sndr_dbc = metrics['sndr_dbc']
        sfdr_dbc = metrics['sfdr_dbc']
        snr_dbc = metrics['snr_dbc']
        nsd_dbfs_hz = metrics['nsd_dbfs_hz']

        results.append({
            'window': win_type,
            'enob': enob,
            'sndr_dbc': sndr_dbc,
            'sfdr_dbc': sfdr_dbc,
            'snr_dbc': snr_dbc,
            'nsd_dbfs_hz': nsd_dbfs_hz
        })

        # Update title with window name
        axes[idx].set_title(f'{win_type.capitalize()} Window',
                            fontsize=12, fontweight='bold')

    # Set main title and save figure
    fig.suptitle(f'Coherent Signal - Window Comparison (Fin={Fin/1e6:.2f} MHz, N={N_fft})',
                 fontsize=14, fontweight='bold')
    fig_path = output_dir / 'test_window_sweep_coherent_all.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Sort by ENOB (descending) and print table
    results.sort(key=lambda x: x['enob'], reverse=True)

    print(f"{'Window':<18} {'ENOB (b)':>9} {'SNDR (dB)':>10} {'SFDR (dB)':>10} {'SNR (dB)':>9}")
    print('-'*70)
    for r in results:
        print(f"{r['window']:<18} {r['enob']:>9.2f} {r['sndr_dbc']:>10.2f} {r['sfdr_dbc']:>10.2f} "
              f"{r['snr_dbc']:>9.2f}")

    print('='*80)

    # Assertions: For coherent signals, most windows should achieve good ENOB
    # Note: Some windows may have lower ENOB due to wider main lobes
    hann_result = next(r for r in results if r['window'] == 'hann')
    hamming_result = next(r for r in results if r['window'] == 'hamming')
    assert hann_result['enob'] > 9.0, f"Hann ENOB too low: {hann_result['enob']:.2f}b (expected >9b)"
    assert hamming_result['enob'] > 9.0, f"Hamming ENOB too low: {hamming_result['enob']:.2f}b (expected >9b)"


def test_window_sweep_noncoherent():
    """Test all window types with NON-COHERENT signal (automatic side_bin selection)."""
    # Signal parameters
    N_fft = 2**13
    Fs = 100e6
    A = 0.5
    noise_rms = 10e-6
    Fin = 10e6  # Non-coherent frequency (not an integer bin)

    # Harmonic distortion parameters
    hd2_dB = -120
    hd3_dB = -100
    hd2_amp = 10**(hd2_dB/20)
    hd3_amp = 10**(hd3_dB/20)
    k2 = hd2_amp / (A / 2)
    k3 = -hd3_amp / (A**2 / 4)

    # Generate signal with harmonics and noise
    t = np.arange(N_fft) / Fs
    sig_ideal = A * np.sin(2*np.pi*Fin*t)
    signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    print('\n' + '='*80)
    print('NON-COHERENT SIGNAL - WINDOW SWEEP (Automatic side_bin)')
    print('='*80)
    print(f'Fs={Fs/1e6:.2f} MHz, Fin={Fin/1e6:.2f} MHz (non-coherent), N={N_fft}, A={A:.3f} Vpeak')
    print(f'Noise RMS={noise_rms*1e6:.2f} uVrms, HD2={hd2_dB} dB, HD3={hd3_dB} dB')
    print()

    # Create single figure with 2x4 subplots
    n_rows, n_cols = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    results = []
    for idx, win_type in enumerate(WINDOW_TYPES):
        # Use analyze_spectrum with automatic side_bin selection
        plt.sca(axes[idx])
        metrics = analyze_spectrum(
            signal,
            fs=Fs,
            win_type=win_type,
            side_bin=None,  # Automatic selection
            create_plot=True,
            show_title=False,
            show_label=True,
            plot_harmonics_up_to=3,
            ax=axes[idx]
        )

        # Extract metrics
        enob = metrics['enob']
        sndr_dbc = metrics['sndr_dbc']
        sfdr_dbc = metrics['sfdr_dbc']
        snr_dbc = metrics['snr_dbc']
        nsd_dbfs_hz = metrics['nsd_dbfs_hz']

        results.append({
            'window': win_type,
            'enob': enob,
            'sndr_dbc': sndr_dbc,
            'sfdr_dbc': sfdr_dbc,
            'snr_dbc': snr_dbc,
            'nsd_dbfs_hz': nsd_dbfs_hz
        })

        # Update title with window name
        axes[idx].set_title(f'{win_type.capitalize()} Window',
                            fontsize=12, fontweight='bold')

    # Set main title and save figure
    fig.suptitle(f'Non-coherent Signal - Window Comparison (Fin={Fin/1e6:.2f} MHz, N={N_fft})',
                 fontsize=14, fontweight='bold')
    fig_path = output_dir / 'test_window_sweep_noncoherent_all.png'
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Verify file was created
    assert fig_path.exists(), f"Figure file not created: {fig_path}"
    assert fig_path.stat().st_size > 0, f"Figure file is empty: {fig_path}"

    # Sort by ENOB (descending) and print table
    results.sort(key=lambda x: x['enob'], reverse=True)

    print(f"{'Window':<18} {'ENOB (b)':>9} {'SNDR (dB)':>10} {'SFDR (dB)':>10} {'SNR (dB)':>9}")
    print('-'*70)
    for r in results:
        print(f"{r['window']:<18} {r['enob']:>9.2f} {r['sndr_dbc']:>10.2f} {r['sfdr_dbc']:>10.2f} "
              f"{r['snr_dbc']:>9.2f}")

    print('='*80)

    # Assertions: For non-coherent signals, performance varies by window
    # Rectangular window should have poor ENOB due to spectral leakage
    rectangular_result = next(r for r in results if r['window'] == 'rectangular')
    assert rectangular_result['enob'] < 5.0, f"Rectangular ENOB should be low due to leakage: {rectangular_result['enob']:.2f}b"

    # Kaiser and Blackman-Harris should have best ENOB (>10b)
    kaiser_result = next(r for r in results if r['window'] == 'kaiser')
    blackmanharris_result = next(r for r in results if r['window'] == 'blackmanharris')
    assert kaiser_result['enob'] > 10.0, f"Kaiser ENOB too low: {kaiser_result['enob']:.2f}b (expected >10b)"
    assert blackmanharris_result['enob'] > 10.0, f"Blackman-Harris ENOB too low: {blackmanharris_result['enob']:.2f}b (expected >10b)"


@pytest.mark.parametrize("win_type", ['rectangular', 'hann', 'blackman', 'blackmanharris', 'kaiser'])
def test_noise_accuracy_with_windows(win_type):
    """Test noise floor, SNR, and NSD accuracy across window types with automatic side_bin.

    Verifies that SNR, noise floor, and NSD match theoretical values within tolerance.
    NSD should be window-independent after ENBW normalization fix.
    """
    # Signal parameters
    N_fft = 2**14  # 16384 points
    Fs = 100e6
    A = 0.5
    noise_rms = 50e-6  # 50 uVrms
    Fin_target = 10e6
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N_fft)

    # Generate signal with known noise
    t = np.arange(N_fft) / Fs
    sig_ideal = A * np.sin(2*np.pi*Fin*t)
    signal = sig_ideal + np.random.randn(N_fft) * noise_rms

    # Calculate theoretical values
    sig_pwr_theory = 0.0  # dBFS (auto-detect uses peak as reference)
    snr_theory = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
    noise_floor_theory = sig_pwr_theory - snr_theory
    nsd_theory = snr_to_nsd(snr_theory, fs=Fs, osr=1, psignal_dbfs=sig_pwr_theory)

    # Compute spectrum with automatic side_bin selection
    result = compute_spectrum(signal, fs=Fs, win_type=win_type, side_bin=None, verbose=0)

    # Extract metrics
    snr_measured = result['metrics']['snr_dbc']
    noise_floor_measured = result['metrics']['noise_floor_dbfs']
    nsd_measured = result['metrics']['nsd_dbfs_hz']

    # Calculate errors
    snr_error = abs(snr_measured - snr_theory)
    nf_error = abs(noise_floor_measured - noise_floor_theory)
    nsd_error = abs(nsd_measured - nsd_theory)

    print(f"\n[Window={win_type:15s}] A={A:.2f}, noise_rms={noise_rms*1e6:.2f}uV, N={N_fft}")
    print(f"SNR/NF/NSD: Expected=[{snr_theory:6.2f}, {noise_floor_theory:7.2f}, {nsd_theory:7.2f}], Computed=[{snr_measured:6.2f}, {noise_floor_measured:7.2f}, {nsd_measured:7.2f}], Error=[{snr_error:5.2f}, {nf_error:5.2f}, {nsd_error:5.2f}] dB")

    # Assertions: All windows should achieve accurate measurements
    assert snr_error < 1.0, f"{win_type} SNR error too large: {snr_error:.2f} dB"
    assert nf_error < 1.0, f"{win_type} Noise Floor error too large: {nf_error:.2f} dB"
    assert nsd_error < 1.0, f"{win_type} NSD error too large: {nsd_error:.2f} dB"
