"""Test noise floor calculation methods (MATLAB plotspec NFMethod numbering)."""

import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum


def test_nf_methods_comparison():
    """Test that all noise floor methods produce reasonable results."""

    # Generate test signal
    N_fft = 8192
    Fs = 100e6
    Fin = 12e6
    A = 0.5
    noise_rms = 200e-6

    # Create coherent signal
    bin_target = int(N_fft * Fin / Fs)
    Fin_coherent = bin_target * Fs / N_fft

    t = np.arange(N_fft) / Fs
    signal = A * np.sin(2*np.pi*Fin_coherent*t) + np.random.randn(N_fft) * noise_rms

    # MATLAB numbering: 0=auto, 1=median, 2=trimmed, 3=exclude
    results_auto = compute_spectrum(signal, fs=Fs, nf_method=0)
    results_median = compute_spectrum(signal, fs=Fs, nf_method=1)
    results_trimmed = compute_spectrum(signal, fs=Fs, nf_method=2)
    results_exclude = compute_spectrum(signal, fs=Fs, nf_method=3)

    metrics_auto = results_auto['metrics']
    metrics_median = results_median['metrics']
    metrics_trimmed = results_trimmed['metrics']
    metrics_exclude = results_exclude['metrics']

    print("\nNoise Floor Method Comparison:")
    print("="*70)
    print(f"Method 0 (Auto):          ENOB={metrics_auto['enob']:.2f} b, SNDR={metrics_auto['sndr_dbc']:.2f} dB, SNR={metrics_auto['snr_dbc']:.2f} dB")
    print(f"Method 1 (Median):        ENOB={metrics_median['enob']:.2f} b, SNDR={metrics_median['sndr_dbc']:.2f} dB, SNR={metrics_median['snr_dbc']:.2f} dB")
    print(f"Method 2 (Trimmed Mean):    ENOB={metrics_trimmed['enob']:.2f} b, SNDR={metrics_trimmed['sndr_dbc']:.2f} dB, SNR={metrics_trimmed['snr_dbc']:.2f} dB")
    print(f"Method 3 (Exclude Harm):  ENOB={metrics_exclude['enob']:.2f} b, SNDR={metrics_exclude['sndr_dbc']:.2f} dB, SNR={metrics_exclude['snr_dbc']:.2f} dB")
    print("="*70)

    for label, metrics in [
        ("auto", metrics_auto),
        ("median", metrics_median),
        ("trimmed", metrics_trimmed),
        ("exclude", metrics_exclude),
    ]:
        assert 9 < metrics['enob'] < 12, f"Method {label} ENOB {metrics['enob']} out of range"

    sig_rms = A / np.sqrt(2)
    theoretical_snr = 20 * np.log10(sig_rms / noise_rms)

    print(f"\nTheoretical SNR: {theoretical_snr:.2f} dB")
    print(f"Method 0 (auto) SNR error: {abs(metrics_auto['snr_dbc'] - theoretical_snr):.2f} dB")

    assert abs(metrics_auto['snr_dbc'] - theoretical_snr) < 2.0, "Auto SNR error too large"

    print("\n[PASS] All noise floor methods working correctly!")


if __name__ == "__main__":
    test_nf_methods_comparison()
