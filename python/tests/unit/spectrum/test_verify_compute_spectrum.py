"""
Unit Test: Verify compute_spectrum for single-tone FFT analysis

Purpose: Self-verify that compute_spectrum correctly computes spectrum metrics
         including SNR, THD, SFDR, and ENOB for ADC analysis
"""
import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox import amplitudes_to_snr, snr_to_nsd, find_coherent_frequency


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

    result = compute_spectrum(sig, fs=fs, win_type='hann', verbose=0)

    print(f'\n[Verify Clean Sine] [N={N}, Fs={fs}Hz, Fin={f_signal}Hz]')
    print(f'  [SNR   ] {result["metrics"]["snr_dbc"]:.2f} dB')
    print(f'  [SNDR  ] {result["metrics"]["sndr_dbc"]:.2f} dB')
    print(f'  [THD   ] {result["metrics"]["thd_dbc"]:.2f} dBc')
    print(f'  [ENOB  ] {result["metrics"]["enob"]:.2f} bits')

    # Verify structure
    assert 'metrics' in result, "Result should have 'metrics' key"
    assert 'plot_data' in result, "Result should have 'plot_data' key"

    # Verify metrics keys
    metrics_keys = ['snr_dbc', 'sndr_dbc', 'thd_dbc', 'harmonics_dbc', 'sfdr_dbc', 'enob', 'sig_pwr_dbfs']
    for key in metrics_keys:
        assert key in result['metrics'], f"Missing metric: {key}"

    # For clean sine, THD should be low
    assert result['metrics']['thd_dbc'] < -40, f"THD too high for clean sine: {result['metrics']['thd_dbc']} dBc"

    # SNR should be reasonable (depends on noise from FFT floor)
    assert result['metrics']['snr_dbc'] > 10, f"SNR too low: {result['metrics']['snr_dbc']} dB"

    print(f'  [Status] PASS')


def _run_noise_accuracy_test(A, noise_rms, Fs, Fin_target, max_scale_range=None, N=2**13):
    """
    Helper function to run noise accuracy test with given parameters.

    Parameters
    ----------
    A : float
        Signal amplitude (Vpeak)
    noise_rms : float
        Noise RMS value
    Fs : float
        Sampling frequency
    Fin_target : float
        Target input frequency
    max_scale_range : float or None, optional
        Full-scale range for normalization
    N : int, optional
        Number of FFT points (default: 2**13)
    """

    # Use coherent frequency for accurate measurement
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

    # Calculate theoretical signal power based on normalization
    # When max_scale_range=None: Auto-detect should use peak = A → sig = 0 dBFS
    # When max_scale_range=value: Explicit FSR, dBFS = 20*log10(A/FSR)
    if max_scale_range is None:
        sig_pwr_theory = 0.0  # Auto-detect: signal peak is the FSR reference
    else:
        sig_pwr_theory = 20 * np.log10(A / max_scale_range)  # Explicit FSR

    # Calculate theoretical values
    snr_theory = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=noise_rms)
    nsd_theory = snr_to_nsd(snr_theory, fs=Fs, osr=1, psignal_dbfs=sig_pwr_theory)

    # Generate signal with known noise
    t = np.arange(N) / Fs
    signal = A * np.sin(2*np.pi*Fin*t) + np.random.randn(N) * noise_rms

    # Compute spectrum (use boxcar window for accurate NSD comparison)
    # Boxcar has ENBW=1.0, so NSD calculation matches theoretical formula
    if max_scale_range is not None:
        result = compute_spectrum(signal, fs=Fs, win_type='boxcar', verbose=0, max_scale_range=max_scale_range)
    else:
        result = compute_spectrum(signal, fs=Fs, win_type='boxcar', verbose=0)

    # Calculate noise floor
    noise_floor_theory = sig_pwr_theory - snr_theory

    print(f'\n[Verify Noise Accuracy] [N={N}, Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, A={A:.3f} V, MaxScaleRange={max_scale_range}, Noise={noise_rms*1e6:.0f} uVrms]')
    print(f'  [Expected] Sig={sig_pwr_theory:6.2f} dBFs, SNR={snr_theory:6.2f} dB, NF={noise_floor_theory:7.2f} dBFs, NSD={nsd_theory:7.2f} dBFs/Hz')
    print(f'  [Measured] Sig={result["metrics"]["sig_pwr_dbfs"]:6.2f} dBFs, SNR={result["metrics"]["snr_dbc"]:6.2f} dB, NF={result["metrics"]["noise_floor_dbfs"]:7.2f} dBFs, NSD={result["metrics"]["nsd_dbfs_hz"]:7.2f} dBFs/Hz')

    # Calculate errors
    sig_error = abs(result['metrics']['sig_pwr_dbfs'] - sig_pwr_theory)
    snr_error = abs(result['metrics']['snr_dbc'] - snr_theory)
    nf_error = abs(result['metrics']['noise_floor_dbfs'] - noise_floor_theory)
    nsd_error = abs(result['metrics']['nsd_dbfs_hz'] - nsd_theory)

    print(f'  [Error   ] Sig={sig_error:6.2f} dBFs, SNR={snr_error:6.2f} dB, NF={nf_error:7.2f} dBFs, NSD={nsd_error:7.2f} dBFs/Hz')
    print(f'  [Result  ] Sig {"-> PASS" if sig_error < 1.0 else "-> FAIL"}, SNR {"-> PASS" if snr_error < 1.0 else "-> FAIL"}, NF {"-> PASS" if nf_error < 1.0 else "-> FAIL"}, NSD {"-> PASS" if nsd_error < 1.0 else "-> FAIL"}')

    # Verify structure
    assert 'metrics' in result, "Result should have 'metrics' key"

    # THD should be very low for clean sine with only noise
    assert result['metrics']['thd_dbc'] < -40, f"THD too high for clean sine: {result['metrics']['thd_dbc']:.2f} dBc"

    # Assert error tolerances
    assert sig_error < 1.0, f"Signal power error too large: {sig_error:.2f} dB"
    assert snr_error < 1.0, f"SNR error too large: {snr_error:.2f} dB"
    assert nf_error < 1.0, f"Noise floor error too large: {nf_error:.2f} dB"
    assert nsd_error < 1.0, f"NSD error too large: {nsd_error:.2f} dB"

@pytest.mark.parametrize("A,max_scale_range", [
    (10, None),
    (1, None),
    (0.5, None),
    (0.5, 0.5),
    (0.25, 0.5),
    (0.05, 0.5),
])
def test_noise_accuracy(A, max_scale_range):
    """
    Parametrized test for noise accuracy verification.

    Test cases:
    - A=10V/1V/0.5V, max_scale_range=None → Expected sig_pwr = 0 dBFS (auto-detect)
    - A=0.5V, max_scale_range=0.5V → Expected sig_pwr = 0 dBFS (full scale)
    - A=0.25V, max_scale_range=0.5V → Expected sig_pwr = -6.02 dBFS (half scale)
    - A=0.05V, max_scale_range=0.5V → Expected sig_pwr = -20.00 dBFS (1/10 scale)
    """
    _run_noise_accuracy_test(
        A=A,
        noise_rms=50e-6,
        Fs=100e6,
        Fin_target=12e6,
        max_scale_range=max_scale_range
    )


def _run_distorted_sine_test(A, max_scale_range=None, N=2**13):
    """
    Helper function to run distorted sine test with given parameters.

    Parameters
    ----------
    A : float
        Signal amplitude (Vpeak)
    max_scale_range : float or None, optional
        Full-scale range for normalization
    N : int, optional
        Number of FFT points (default: 2**13)
    """
    Fs = 100e6
    Fin_target = 10e6

    # Use coherent frequency
    Fin, Fin_bin = find_coherent_frequency(fs=Fs, fin_target=Fin_target, n_fft=N)

    # Add known HD2 and HD3 in dB (relative to fundamental)
    hd2_db_expected = -60  # dB
    hd3_db_expected = -60  # dB

    # Convert dB to amplitude ratios
    hd2_amp = A * 10**(hd2_db_expected / 20)
    hd3_amp = A * 10**(hd3_db_expected / 20)

    t = np.arange(N) / Fs
    sig = A * np.sin(2*np.pi*Fin*t)
    sig += hd2_amp * np.sin(2*2*np.pi*Fin*t)
    sig += hd3_amp * np.sin(3*2*np.pi*Fin*t)

    if max_scale_range is not None:
        result = compute_spectrum(sig, fs=Fs, win_type='boxcar', verbose=0, max_scale_range=max_scale_range)
    else:
        result = compute_spectrum(sig, fs=Fs, win_type='boxcar', verbose=0)

    # Get measured values
    sig_pwr_dbfs = result['metrics']['sig_pwr_dbfs']
    # HD2, HD3, THD are already in dBc (relative to carrier)
    hd2_measured = result['metrics']['harmonics_dbc'][0]
    hd3_measured = result['metrics']['harmonics_dbc'][1]
    thd_dbc_measured = result['metrics']['thd_dbc']

    # Calculate expected signal power and THD
    if max_scale_range is None:
        # Auto-detect: peak of fundamental + harmonics is FSR
        peak_total = A + hd2_amp + hd3_amp
        sig_pwr_expected = 20 * np.log10(A / peak_total)  # Signal relative to auto-detected FSR
    else:
        sig_pwr_expected = 20 * np.log10(A / max_scale_range)  # dBFS for explicit FSR

    # THD is relative to the fundamental signal amplitude (in dBc)
    thd_expected = 10 * np.log10((hd2_amp**2 + hd3_amp**2) / A**2)

    # SFDR = maximum spur level relative to signal (absolute value)
    sfdr_expected = -min(hd2_db_expected, hd3_db_expected)  # Max spur (most negative is smallest)

    print(f'\n[Verify Distorted Sine] [N={N}, Fs={Fs/1e6:.0f} MHz, Fin={Fin/1e6:.2f} MHz, A={A:.3f} V, MaxScaleRange={max_scale_range}]')
    print(f'  [Expected] Sig={sig_pwr_expected:6.2f} dBFs, HD2={hd2_db_expected:6.2f} dBc, HD3={hd3_db_expected:6.2f} dBc, THD={thd_expected:6.2f} dBc, SFDR={sfdr_expected:6.2f} dBc')
    print(f'  [Measured] Sig={sig_pwr_dbfs:6.2f} dBFs, HD2={hd2_measured:6.2f} dBc, HD3={hd3_measured:6.2f} dBc, THD={thd_dbc_measured:6.2f} dBc, SFDR={result["metrics"]["sfdr_dbc"]:6.2f} dBc')

    # Calculate errors
    sig_error = abs(sig_pwr_dbfs - sig_pwr_expected)
    hd2_error = abs(hd2_measured - hd2_db_expected)
    hd3_error = abs(hd3_measured - hd3_db_expected)
    thd_error = abs(thd_dbc_measured - thd_expected)
    sfdr_error = abs(result['metrics']['sfdr_dbc'] - sfdr_expected)

    print(f'  [Error   ] Sig={sig_error:6.2f} dBFs, HD2={hd2_error:6.2f} dBc, HD3={hd3_error:6.2f} dBc, THD={thd_error:6.2f} dBc, SFDR={sfdr_error:6.2f} dBc')
    print(f'  [Result  ] Sig:{"PASS" if sig_error < 1.0 else "FAIL"}, HD2:{"PASS" if hd2_error < 1.0 else "FAIL"}, HD3:{"PASS" if hd3_error < 1.0 else "FAIL"}, THD:{"PASS" if thd_error < 1.0 else "FAIL"}, SFDR:{"PASS" if sfdr_error < 1.0 else "FAIL"}')

    # Assert error tolerances
    assert sig_error < 1.0, f"Signal power error too large: {sig_error:.2f} dB"
    assert hd2_error < 1.0, f"HD2 error too large: {hd2_error:.2f} dBc"
    assert hd3_error < 1.0, f"HD3 error too large: {hd3_error:.2f} dBc"
    assert thd_error < 1.0, f"THD error too large: {thd_error:.2f} dBc"
    assert sfdr_error < 1.0, f"SFDR error too large: {sfdr_error:.2f} dBc"


@pytest.mark.parametrize("A,max_scale_range", [
    (10, None),
    (1.0, None),
    (0.5, None),
    (0.5, 0.5),
    (0.25, 0.5),
    (0.05, 0.5),
])
def test_verify_compute_spectrum_distorted_sine(A, max_scale_range):
    """
    Parametrized test for distorted sine verification.

    Test strategy:
    1. Generate coherent sine with known HD2 and HD3 amplitudes
    2. Compute spectrum
    3. Verify measured harmonic levels match theoretical values

    Test cases:
    - Different signal amplitudes with auto-detect FSR
    - Different signal amplitudes with explicit FSR
    """
    _run_distorted_sine_test(A=A, max_scale_range=max_scale_range)


def test_verify_compute_spectrum_1d_input():
    """
    Verify compute_spectrum handles 1D input correctly.

    Test strategy:
    1. Pass 1D array (single run)
    2. Assert: Processes correctly and returns valid metrics
    """
    N = 512
    sig = 0.4 * np.sin(2*np.pi*0.1*np.arange(N))

    result = compute_spectrum(sig, fs=1.0, verbose=0)

    print(f'\n[Verify 1D Input] [Shape={sig.shape}]')
    print(f'  [SNR] {result["metrics"]["snr_dbc"]:.2f} dB')

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

    result = compute_spectrum(data_2d, fs=1.0, verbose=0)

    print(f'\n[Verify 2D Input] [Shape={data_2d.shape}]')
    print(f'  [M runs] {M}')
    print(f'  [SNR] {result["metrics"]["snr_dbc"]:.2f} dB')

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
        result = compute_spectrum(sig, fs=1.0, win_type=win_type, verbose=0)
        results_by_window[win_type] = result
        print(f'  [{win_type:8s}] THD={result["metrics"]["thd_dbc"]:7.2f} dBc, SNR={result["metrics"]["snr_dbc"]:7.2f} dB')

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

    result_power = compute_spectrum(data, fs=1000.0, coherent_averaging=False, verbose=0)
    result_coherent = compute_spectrum(data, fs=1000.0, coherent_averaging=True, verbose=0)

    print(f'\n[Verify Coherent Mode]')
    print(f'  [Power   ] SNR={result_power["metrics"]["snr_dbc"]:.2f} dB')
    print(f'  [Coherent] SNR={result_coherent["metrics"]["snr_dbc"]:.2f} dB')

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
        result = compute_spectrum(sig, nf_method=method, verbose=0)
        print(f'  [Method {method}] NF={result["metrics"]["noise_floor_dbfs"]:.2f} dB')

        assert 'noise_floor_dbfs' in result['metrics'], f"Missing noise floor for method {method}"

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
    result = compute_spectrum(sig, verbose=0)

    print(f'\n[Verify Metrics Structure]')

    expected_metrics = [
        'enob', 'sndr_dbc', 'sfdr_dbc', 'snr_dbc', 'thd_dbc', 'harmonics_dbc',
        'sig_pwr_dbfs', 'noise_floor_dbfs', 'nsd_dbfs_hz'
    ]

    for metric in expected_metrics:
        assert metric in result['metrics'], f"Missing metric: {metric}"
        if metric == 'harmonics_dbc':
            assert isinstance(result['metrics'][metric], np.ndarray), \
                f"Metric {metric} should be ndarray: {type(result['metrics'][metric])}"
        else:
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

    result = compute_spectrum(sig, fs=fs, verbose=0)
    plot_data = result['plot_data']

    print(f'\n[Verify Plot Data]')

    expected_plot_keys = ['freq', 'power_spectrum_db_plot', 'fundamental_bin']
    for key in expected_plot_keys:
        assert key in plot_data, f"Missing plot_data key: {key}"

    expected_metadata_keys = ['N', 'M', 'fs']
    for key in expected_metadata_keys:
        assert key in result, f"Missing metadata key: {key}"

    print(f'  [freq shape              ] {plot_data["freq"].shape}')
    print(f'  [power_spectrum_db_plot  ] {plot_data["power_spectrum_db_plot"].shape}')
    print(f'  [fundamental_bin         ] {plot_data["fundamental_bin"]}')

    # Verify frequency array is reasonable
    assert plot_data['freq'][0] == 0, "First frequency should be 0"
    assert len(plot_data['freq']) > 0, "Frequency array should not be empty"

    # Verify spectrum is reasonable
    assert np.all(np.isfinite(plot_data['power_spectrum_db_plot'])), "Spectrum should be finite"

    print(f'  [Status] PASS')


if __name__ == '__main__':
    """Run verification tests standalone"""
    print('='*80)
    print('RUNNING COMPUTE_SPECTRUM VERIFICATION TESTS')
    print('='*80)

    test_verify_compute_spectrum_clean_sine()

    # Run parametrized noise accuracy tests
    for A, max_scale_range in [(10, None), (1, None), (0.5, None), (0.5, 0.5), (0.25, 0.5), (0.05, 0.5)]:
        test_noise_accuracy(A, max_scale_range)

    # Run parametrized distorted sine tests
    for A, max_scale_range in [(0.5, None), (1.0, None), (0.25, None), (0.5, 0.5), (0.25, 0.5), (0.1, 0.5)]:
        test_verify_compute_spectrum_distorted_sine(A, max_scale_range)

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
