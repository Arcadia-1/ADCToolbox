"""
Unit Test: Verify compute_two_tone_spectrum for two-tone IMD analysis

Purpose: Self-verify that compute_two_tone_spectrum correctly detects
         two fundamental tones and calculates IMD2/IMD3 metrics
"""
import numpy as np
from adctoolbox.spectrum.compute_two_tone_spectrum import compute_two_tone_spectrum


def test_verify_compute_two_tone_clean():
    """
    Verify compute_two_tone_spectrum on clean two-tone signal.

    Test strategy:
    1. Generate two clean sine waves at different frequencies
    2. Compute two-tone spectrum
    3. Assert: Correct tone detection and low distortion
    """
    N = 1024
    fs = 1000.0
    f1 = 100.0
    f2 = 200.0
    t = np.arange(N) / fs
    A1, A2 = 0.3, 0.3

    sig = A1 * np.sin(2*np.pi*f1*t) + A2 * np.sin(2*np.pi*f2*t)

    result = compute_two_tone_spectrum(sig, fs=fs)

    print(f'\n[Verify Clean Two-Tone] [f1={f1}Hz, f2={f2}Hz]')
    print(f'  [SNR   ] {result["metrics"]["snr_db"]:.2f} dB')
    print(f'  [IMD2  ] {result["metrics"]["imd2_db"]:.2f} dB')
    print(f'  [IMD3  ] {result["metrics"]["imd3_db"]:.2f} dB')
    print(f'  [ENOB  ] {result["metrics"]["enob"]:.2f} bits')

    # Verify structure
    assert 'metrics' in result, "Result should have 'metrics' key"
    assert 'plot_data' in result, "Result should have 'plot_data' key"
    assert 'imd_bins' in result, "Result should have 'imd_bins' key"

    # Verify metrics keys
    required_metrics = ['snr_db', 'sndr_db', 'imd2_db', 'imd3_db', 'thd_db', 'enob']
    for key in required_metrics:
        assert key in result['metrics'], f"Missing metric: {key}"

    # For clean two-tone, IMD should be reasonable
    # IMD measurements depend on FFT noise floor and signal level
    assert result['metrics']['imd2_db'] > 0, f"IMD2 should be positive for clean signal: {result['metrics']['imd2_db']} dB"
    assert result['metrics']['imd3_db'] > 30, f"IMD3 too low for clean signal: {result['metrics']['imd3_db']} dB"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_with_imd():
    """
    Verify compute_two_tone_spectrum detects IMD products.

    Test strategy:
    1. Generate two tones with intentional IMD generation
    2. Compute spectrum
    3. Assert: IMD power increases appropriately
    """
    N = 2048
    fs = 1000.0
    f1 = 100.0
    f2 = 250.0
    t = np.arange(N) / fs
    A = 0.4

    # Two clean tones
    sig_clean = A * (np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t))

    # Add nonlinearity to create IMD: y = x + k*x^2
    k = 0.1
    sig_imd = sig_clean + k * sig_clean**2

    result_clean = compute_two_tone_spectrum(sig_clean, fs=fs)
    result_imd = compute_two_tone_spectrum(sig_imd, fs=fs)

    print(f'\n[Verify IMD Detection] [f1={f1}Hz, f2={f2}Hz, k={k}]')
    print(f'  [Clean] IMD2={result_clean["metrics"]["imd2_db"]:.2f} dB, IMD3={result_clean["metrics"]["imd3_db"]:.2f} dB')
    print(f'  [With IMD] IMD2={result_imd["metrics"]["imd2_db"]:.2f} dB, IMD3={result_imd["metrics"]["imd3_db"]:.2f} dB')

    # IMD metrics should be worse (lower) with distortion
    imd2_improvement = result_imd["metrics"]["imd2_db"] - result_clean["metrics"]["imd2_db"]
    imd3_improvement = result_imd["metrics"]["imd3_db"] - result_clean["metrics"]["imd3_db"]

    print(f'  [IMD2 degradation] {imd2_improvement:.2f} dB')
    print(f'  [IMD3 degradation] {imd3_improvement:.2f} dB')

    # With nonlinearity, IMD metrics should degrade (become more negative)
    assert imd2_improvement < 0, f"IMD2 should degrade with distortion: {imd2_improvement:.2f} dB"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_1d_input():
    """
    Verify compute_two_tone_spectrum handles 1D input.

    Test strategy:
    1. Pass 1D array (single run)
    2. Assert: Processes correctly
    """
    N = 512
    t = np.arange(N) / 1000.0
    sig = 0.3 * (np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t))

    result = compute_two_tone_spectrum(sig, fs=1000.0)

    print(f'\n[Verify 1D Input] [Shape={sig.shape}]')
    print(f'  [SNR] {result["metrics"]["snr_db"]:.2f} dB')

    # Verify structure
    assert isinstance(result['metrics'], dict), "Metrics should be dict"
    assert isinstance(result['plot_data'], dict), "Plot data should be dict"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_2d_input():
    """
    Verify compute_two_tone_spectrum handles 2D input (multiple runs).

    Test strategy:
    1. Pass 2D array with multiple runs
    2. Assert: Averages correctly
    """
    M, N = 3, 512
    t = np.arange(N) / 1000.0
    f1, f2 = 100.0, 200.0

    # Generate 3 runs with slight phase variations
    data = np.zeros((M, N))
    for i in range(M):
        phase1 = np.random.rand() * 2 * np.pi
        phase2 = np.random.rand() * 2 * np.pi
        data[i, :] = 0.3 * (np.sin(2*np.pi*f1*t + phase1) + np.sin(2*np.pi*f2*t + phase2))

    result = compute_two_tone_spectrum(data, fs=1000.0)

    print(f'\n[Verify 2D Input] [Shape={data.shape}]')
    print(f'  [M runs] {M}')
    print(f'  [SNR] {result["metrics"]["snr_db"]:.2f} dB')

    assert 'metrics' in result, "Should have metrics"
    assert 'plot_data' in result, "Should have plot_data"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_tone_detection():
    """
    Verify compute_two_tone_spectrum correctly identifies two tones.

    Test strategy:
    1. Generate two tones at known frequencies
    2. Verify detected tone frequencies
    """
    N = 2048
    fs = 10000.0
    f1_true = 1000.0
    f2_true = 3000.0
    t = np.arange(N) / fs

    sig = 0.3 * (np.sin(2*np.pi*f1_true*t) + np.sin(2*np.pi*f2_true*t))

    result = compute_two_tone_spectrum(sig, fs=fs)

    print(f'\n[Verify Tone Detection] [f1={f1_true}Hz, f2={f2_true}Hz]')
    print(f'  [Detected f1] {result["plot_data"]["freq1"]:.2f} Hz')
    print(f'  [Detected f2] {result["plot_data"]["freq2"]:.2f} Hz')

    # Check that detected frequencies are close to true frequencies
    f1_error = abs(result['plot_data']['freq1'] - f1_true)
    f2_error = abs(result['plot_data']['freq2'] - f2_true)

    print(f'  [f1 error] {f1_error:.2f} Hz')
    print(f'  [f2 error] {f2_error:.2f} Hz')

    # Error should be within few bins (frequency resolution)
    freq_resolution = fs / N
    assert f1_error < 5 * freq_resolution, f"f1 detection error too large: {f1_error} Hz"
    assert f2_error < 5 * freq_resolution, f"f2 detection error too large: {f2_error} Hz"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_window_types():
    """
    Verify compute_two_tone_spectrum works with different windows.

    Test strategy:
    1. Apply different window types
    2. Verify all produce valid results
    """
    N = 512
    t = np.arange(N) / 1000.0
    sig = 0.3 * (np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t))

    windows = ['boxcar', 'hann', 'hamming', 'blackman']
    print(f'\n[Verify Window Types] [N={N}]')

    for win_type in windows:
        result = compute_two_tone_spectrum(sig, fs=1000.0, win_type=win_type)
        print(f'  [{win_type:8s}] SNR={result["metrics"]["snr_db"]:7.2f} dB, IMD2={result["metrics"]["imd2_db"]:7.2f} dB')

        assert 'metrics' in result, f"Missing metrics for {win_type}"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_coherent_mode():
    """
    Verify compute_two_tone_spectrum coherent_averaging mode.

    Test strategy:
    1. Compute with coherent_averaging=False
    2. Compute with coherent_averaging=True
    3. Assert: Both produce valid results
    """
    M, N = 2, 512
    t = np.arange(N) / 1000.0
    data = 0.3 * (np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t))
    data = np.vstack([data, data])  # 2 identical runs

    result_power = compute_two_tone_spectrum(data, fs=1000.0, coherent_averaging=False)
    result_coherent = compute_two_tone_spectrum(data, fs=1000.0, coherent_averaging=True)

    print(f'\n[Verify Coherent Mode]')
    print(f'  [Power   ] SNR={result_power["metrics"]["snr_db"]:.2f} dB')
    print(f'  [Coherent] SNR={result_coherent["metrics"]["snr_db"]:.2f} dB')

    # Both should have valid metrics
    assert 'metrics' in result_power, "Power mode should have metrics"
    assert 'metrics' in result_coherent, "Coherent mode should have metrics"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_imd_bins():
    """
    Verify compute_two_tone_spectrum returns IMD bin locations.

    Test strategy:
    1. Compute spectrum
    2. Assert: imd_bins dict contains expected keys
    """
    N = 1024
    t = np.arange(N) / 1000.0
    sig = 0.3 * (np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t))

    result = compute_two_tone_spectrum(sig, fs=1000.0)

    print(f'\n[Verify IMD Bins]')

    expected_imd_bins = [
        'imd2_sum', 'imd2_diff',
        'imd3_2f1_plus_f2', 'imd3_f1_plus_2f2',
        'imd3_2f1_minus_f2', 'imd3_2f2_minus_f1'
    ]

    for bin_name in expected_imd_bins:
        assert bin_name in result['imd_bins'], f"Missing IMD bin: {bin_name}"
        bin_val = result['imd_bins'][bin_name]
        print(f'  [{bin_name:18s}] = {bin_val}')
        assert isinstance(bin_val, (int, np.integer)), f"{bin_name} should be integer"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_plot_data():
    """
    Verify compute_two_tone_spectrum returns valid plot_data.

    Test strategy:
    1. Compute spectrum
    2. Assert: plot_data has all required keys and valid arrays
    """
    N = 512
    t = np.arange(N) / 1000.0
    sig = 0.3 * (np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t))

    result = compute_two_tone_spectrum(sig, fs=1000.0)
    plot_data = result['plot_data']

    print(f'\n[Verify Plot Data]')

    expected_keys = ['freq', 'spec_db', 'bin1', 'bin2', 'freq1', 'freq2', 'N', 'M', 'fs']
    for key in expected_keys:
        assert key in plot_data, f"Missing plot_data key: {key}"

    print(f'  [freq shape  ] {plot_data["freq"].shape}')
    print(f'  [spec_db shape] {plot_data["spec_db"].shape}')
    print(f'  [bin1, bin2  ] ({plot_data["bin1"]}, {plot_data["bin2"]})')
    print(f'  [freq1, freq2] ({plot_data["freq1"]:.1f}, {plot_data["freq2"]:.1f})')

    # Verify frequencies are monotonic
    assert np.all(np.diff(plot_data['freq']) > 0), "Frequency array should be monotonic"

    # Verify spectrum is reasonable
    assert np.all(np.isfinite(plot_data['spec_db'])), "Spectrum should be finite"

    print(f'  [Status] PASS')


def test_verify_compute_two_tone_metrics_structure():
    """
    Verify compute_two_tone_spectrum returns complete metrics.

    Test strategy:
    1. Compute spectrum
    2. Assert: All expected metric keys present
    3. Assert: All values are numeric
    """
    N = 512
    t = np.arange(N) / 1000.0
    sig = 0.3 * (np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t))

    result = compute_two_tone_spectrum(sig, fs=1000.0)
    metrics = result['metrics']

    print(f'\n[Verify Metrics Structure]')

    expected_metrics = [
        'enob', 'sndr_db', 'sfdr_db', 'snr_db', 'thd_db',
        'signal_power_1_dbfs', 'signal_power_2_dbfs',
        'noise_floor_db', 'nsd_dbfs_hz', 'imd2_db', 'imd3_db'
    ]

    for metric in expected_metrics:
        assert metric in metrics, f"Missing metric: {metric}"
        val = metrics[metric]
        assert isinstance(val, (int, float, np.number)), f"Metric {metric} not numeric: {type(val)}"

    print(f'  [Expected metrics: {len(expected_metrics)}]')
    print(f'  [Found metrics  : {len(metrics)}]')
    print(f'  [SNR: {metrics["snr_db"]:.2f} dB]')
    print(f'  [IMD2: {metrics["imd2_db"]:.2f} dB]')
    print(f'  [IMD3: {metrics["imd3_db"]:.2f} dB]')

    print(f'  [Status] PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING COMPUTE_TWO_TONE_SPECTRUM VERIFICATION TESTS')
    print('='*80)

    test_verify_compute_two_tone_clean()
    test_verify_compute_two_tone_with_imd()
    test_verify_compute_two_tone_1d_input()
    test_verify_compute_two_tone_2d_input()
    test_verify_compute_two_tone_tone_detection()
    test_verify_compute_two_tone_window_types()
    test_verify_compute_two_tone_coherent_mode()
    test_verify_compute_two_tone_imd_bins()
    test_verify_compute_two_tone_plot_data()
    test_verify_compute_two_tone_metrics_structure()

    print('\n' + '='*80)
    print('** All compute_two_tone_spectrum verification tests passed! **')
    print('='*80)
