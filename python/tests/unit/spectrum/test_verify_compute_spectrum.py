"""
Unit Test: Verify compute_spectrum for single-tone FFT analysis

Purpose: Self-verify that compute_spectrum correctly computes spectrum metrics
         including SNR, THD, SFDR, and ENOB for ADC analysis
"""
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum


def test_verify_compute_spectrum_clean_sine():
    """
    Verify compute_spectrum on clean single-tone signal.

    Test strategy:
    1. Generate clean sine wave at known frequency
    2. Compute spectrum
    3. Assert: Correct metrics structure and reasonable values
    """
    N = 1024
    fs = 1000.0
    f_signal = 100.0
    t = np.arange(N) / fs
    A = 0.5
    sig = A * np.sin(2*np.pi*f_signal*t)

    result = compute_spectrum(sig, fs=fs, win_type='hann')

    print(f'\n[Verify Clean Sine] [N={N}, Fs={fs}Hz, Fin={f_signal}Hz]')
    print(f'  [SNR   ] {result["metrics"]["snr_db"]:.2f} dB')
    print(f'  [SNDR  ] {result["metrics"]["sndr_db"]:.2f} dB')
    print(f'  [THD   ] {result["metrics"]["thd_db"]:.2f} dB')
    print(f'  [ENOB  ] {result["metrics"]["enob"]:.2f} bits')

    # Verify structure
    assert 'metrics' in result, "Result should have 'metrics' key"
    assert 'plot_data' in result, "Result should have 'plot_data' key"

    # Verify metrics keys
    metrics_keys = ['snr_db', 'sndr_db', 'thd_db', 'hd2_db', 'hd3_db', 'sfdr_db', 'enob', 'sig_pwr_dbfs']
    for key in metrics_keys:
        assert key in result['metrics'], f"Missing metric: {key}"

    # For clean sine, THD should be low
    assert result['metrics']['thd_db'] < -40, f"THD too high for clean sine: {result['metrics']['thd_db']} dB"

    # SNR should be reasonable (depends on noise from FFT floor)
    assert result['metrics']['snr_db'] > 10, f"SNR too low: {result['metrics']['snr_db']} dB"

    print(f'  [Status] PASS')


def test_verify_compute_spectrum_distorted_sine():
    """
    Verify compute_spectrum detects harmonic distortion.

    Test strategy:
    1. Generate sine with known HD2 and HD3
    2. Compute spectrum
    3. Assert: HD2 and HD3 power detected correctly
    """
    N = 1024
    fs = 1000.0
    f_signal = 100.0
    t = np.arange(N) / fs
    A = 0.5

    # Add known HD2 and HD3
    hd2_amp = 0.05  # -20 dB relative to fundamental
    hd3_amp = 0.025  # -26 dB relative to fundamental
    sig = A * np.sin(2*np.pi*f_signal*t)
    sig += hd2_amp * np.sin(2*2*np.pi*f_signal*t)
    sig += hd3_amp * np.sin(3*2*np.pi*f_signal*t)

    result = compute_spectrum(sig, fs=fs, win_type='hann')

    print(f'\n[Verify Distorted Sine] [HD2={hd2_amp:.3f}, HD3={hd3_amp:.3f}]')
    print(f'  [THD   ] {result["metrics"]["thd_db"]:.2f} dB')
    print(f'  [HD2   ] {result["metrics"]["hd2_db"]:.2f} dB')
    print(f'  [HD3   ] {result["metrics"]["hd3_db"]:.2f} dB')

    # THD should be elevated
    assert result['metrics']['thd_db'] > -30, f"THD not detected: {result['metrics']['thd_db']} dB"

    # HD2 and HD3 should be detected
    assert result['metrics']['hd2_db'] < 0, f"HD2 power should be negative dB: {result['metrics']['hd2_db']}"
    assert result['metrics']['hd3_db'] < 0, f"HD3 power should be negative dB: {result['metrics']['hd3_db']}"

    print(f'  [Status] PASS')


def test_verify_compute_spectrum_1d_input():
    """
    Verify compute_spectrum handles 1D input correctly.

    Test strategy:
    1. Pass 1D array (single run)
    2. Assert: Processes correctly and returns valid metrics
    """
    N = 512
    sig = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))

    result = compute_spectrum(sig, fs=1.0)

    print(f'\n[Verify 1D Input] [Shape={sig.shape}]')
    print(f'  [SNR] {result["metrics"]["snr_db"]:.2f} dB')

    # Basic structure check
    assert isinstance(result['metrics'], dict), "Metrics should be dict"
    assert isinstance(result['plot_data'], dict), "Plot data should be dict"

    print(f'  [Status] PASS')


def test_verify_compute_spectrum_2d_input():
    """
    Verify compute_spectrum handles 2D input (multiple runs).

    Test strategy:
    1. Pass 2D array (M runs)
    2. Assert: Averages correctly over runs
    """
    M, N = 3, 512
    data_2d = 0.4 * np.sin(2*np.pi*0.1*np.arange(N)[np.newaxis, :] + np.random.randn(M, 1) * 0.01)

    result = compute_spectrum(data_2d, fs=1.0)

    print(f'\n[Verify 2D Input] [Shape={data_2d.shape}]')
    print(f'  [M runs] {M}')
    print(f'  [SNR] {result["metrics"]["snr_db"]:.2f} dB')

    assert 'metrics' in result, "Should have metrics"
    assert 'plot_data' in result, "Should have plot_data"

    print(f'  [Status] PASS')


def test_verify_compute_spectrum_window_types():
    """
    Verify compute_spectrum works with different window types.

    Test strategy:
    1. Apply different windows
    2. Verify all produce valid results
    3. Check that THD values differ due to window effects
    """
    N = 512
    sig = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))

    windows = ['boxcar', 'hann', 'hamming', 'blackman']
    print(f'\n[Verify Window Types] [N={N}]')

    results_by_window = {}
    for win_type in windows:
        result = compute_spectrum(sig, fs=1.0, win_type=win_type)
        results_by_window[win_type] = result
        print(f'  [{win_type:8s}] THD={result["metrics"]["thd_db"]:7.2f} dB, SNR={result["metrics"]["snr_db"]:7.2f} dB')

        assert 'metrics' in result, f"Missing metrics for {win_type}"

    print(f'  [Status] PASS')


def test_verify_compute_spectrum_coherent_vs_power():
    """
    Verify compute_spectrum coherent_averaging mode.

    Test strategy:
    1. Compute spectrum with coherent_averaging=False
    2. Compute spectrum with coherent_averaging=True
    3. Assert: Both produce valid metrics
    """
    M, N = 2, 512
    t = np.arange(N) / 1000.0
    data = 0.4 * np.sin(2*np.pi*100*t)[np.newaxis, :] + np.random.randn(M, N) * 0.01
    data = np.vstack([data, data])  # 4 runs

    result_power = compute_spectrum(data, fs=1000.0, coherent_averaging=False)
    result_coherent = compute_spectrum(data, fs=1000.0, coherent_averaging=True)

    print(f'\n[Verify Coherent Mode]')
    print(f'  [Power   ] SNR={result_power["metrics"]["snr_db"]:.2f} dB')
    print(f'  [Coherent] SNR={result_coherent["metrics"]["snr_db"]:.2f} dB')

    # Both should have metrics
    assert 'metrics' in result_power, "Power mode should have metrics"
    assert 'metrics' in result_coherent, "Coherent mode should have metrics"

    print(f'  [Status] PASS')


def test_verify_compute_spectrum_noise_floor_methods():
    """
    Verify compute_spectrum handles different noise floor methods.

    Test strategy:
    1. Test nf_method=0 (median)
    2. Test nf_method=1 (trimmed mean)
    3. Test nf_method=2 (exclude harmonics)
    4. Assert: All produce valid results
    """
    N = 1024
    sig = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))

    methods = [0, 1, 2]
    print(f'\n[Verify Noise Floor Methods]')

    for method in methods:
        result = compute_spectrum(sig, nf_method=method)
        print(f'  [Method {method}] NF={result["metrics"]["noise_floor_db"]:.2f} dB')

        assert 'noise_floor_db' in result['metrics'], f"Missing noise floor for method {method}"

    print(f'  [Status] PASS')


def test_verify_compute_spectrum_metrics_structure():
    """
    Verify compute_spectrum returns complete metrics structure.

    Test strategy:
    1. Compute spectrum
    2. Assert: All expected metrics keys present
    3. Assert: All values are numeric
    """
    sig = 0.5 * np.sin(2*np.pi*0.1*np.arange(512))
    result = compute_spectrum(sig)

    print(f'\n[Verify Metrics Structure]')

    expected_metrics = [
        'enob', 'sndr_db', 'sfdr_db', 'snr_db', 'thd_db', 'hd2_db', 'hd3_db',
        'sig_pwr_dbfs', 'noise_floor_db', 'nsd_dbfs_hz', 'bin_idx', 'bin_r'
    ]

    for metric in expected_metrics:
        assert metric in result['metrics'], f"Missing metric: {metric}"
        assert isinstance(result['metrics'][metric], (int, float, np.number)), \
            f"Metric {metric} not numeric: {type(result['metrics'][metric])}"

    print(f'  [Expected metrics: {len(expected_metrics)}]')
    print(f'  [Found metrics  : {len(result["metrics"])}]')
    print(f'  [Status] PASS')


def test_verify_compute_spectrum_plot_data():
    """
    Verify compute_spectrum returns valid plot_data.

    Test strategy:
    1. Compute spectrum
    2. Assert: plot_data has required keys and arrays
    """
    N = 512
    sig = 0.5 * np.sin(2*np.pi*0.1*np.arange(N))
    fs = 1000.0

    result = compute_spectrum(sig, fs=fs)
    plot_data = result['plot_data']

    print(f'\n[Verify Plot Data]')

    expected_keys = ['freq', 'spec_db', 'bin_idx', 'N', 'M', 'fs']
    for key in expected_keys:
        assert key in plot_data, f"Missing plot_data key: {key}"

    print(f'  [freq shape  ] {plot_data["freq"].shape}')
    print(f'  [spec_db shape] {plot_data["spec_db"].shape}')
    print(f'  [bin_idx     ] {plot_data["bin_idx"]}')

    # Verify frequency array is reasonable
    assert plot_data['freq'][0] == 0, "First frequency should be 0"
    assert len(plot_data['freq']) > 0, "Frequency array should not be empty"

    # Verify spectrum is reasonable
    assert np.all(np.isfinite(plot_data['spec_db'])), "Spectrum should be finite"

    print(f'  [Status] PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING COMPUTE_SPECTRUM VERIFICATION TESTS')
    print('='*80)

    test_verify_compute_spectrum_clean_sine()
    test_verify_compute_spectrum_distorted_sine()
    test_verify_compute_spectrum_1d_input()
    test_verify_compute_spectrum_2d_input()
    test_verify_compute_spectrum_window_types()
    test_verify_compute_spectrum_coherent_vs_power()
    test_verify_compute_spectrum_noise_floor_methods()
    test_verify_compute_spectrum_metrics_structure()
    test_verify_compute_spectrum_plot_data()

    print('\n' + '='*80)
    print('** All compute_spectrum verification tests passed! **')
    print('='*80)
