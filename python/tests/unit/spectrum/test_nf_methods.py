"""Test noise floor calculation methods."""

import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum


def test_nf_methods_comparison():
    """Test that all three noise floor methods produce reasonable results."""

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

    # Test all three methods
    results_method0 = compute_spectrum(signal, fs=Fs, nf_method=0)
    results_method1 = compute_spectrum(signal, fs=Fs, nf_method=1)
    results_method2 = compute_spectrum(signal, fs=Fs, nf_method=2)

    # Extract metrics
    metrics0 = results_method0['metrics']
    metrics1 = results_method1['metrics']
    metrics2 = results_method2['metrics']

    print("\nNoise Floor Method Comparison:")
    print("="*70)
    print(f"Method 0 (Median):        ENOB={metrics0['enob']:.2f} b, SNDR={metrics0['sndr_dbc']:.2f} dB, SNR={metrics0['snr_dbc']:.2f} dB")
    print(f"Method 1 (Trimmed Mean):  ENOB={metrics1['enob']:.2f} b, SNDR={metrics1['sndr_dbc']:.2f} dB, SNR={metrics1['snr_dbc']:.2f} dB")
    print(f"Method 2 (Exclude Harm):  ENOB={metrics2['enob']:.2f} b, SNDR={metrics2['sndr_dbc']:.2f} dB, SNR={metrics2['snr_dbc']:.2f} dB")
    print("="*70)

    # All methods should produce reasonable ENOB values (within reasonable range)
    assert 9 < metrics0['enob'] < 12, f"Method 0 ENOB {metrics0['enob']} out of range"
    assert 9 < metrics1['enob'] < 12, f"Method 1 ENOB {metrics1['enob']} out of range"
    assert 9 < metrics2['enob'] < 12, f"Method 2 ENOB {metrics2['enob']} out of range"

    # Method 2 should be most accurate (closest to theoretical SNR)
    # Theoretical SNR for 200uV noise and 0.5V amplitude
    sig_rms = A / np.sqrt(2)
    theoretical_snr = 20 * np.log10(sig_rms / noise_rms)

    print(f"\nTheoretical SNR: {theoretical_snr:.2f} dB")
    print(f"Method 2 SNR error: {abs(metrics2['snr_dbc'] - theoretical_snr):.2f} dB")

    # SNR should be within 2 dB of theoretical
    assert abs(metrics2['snr_dbc'] - theoretical_snr) < 2.0, "SNR error too large"

    print("\n[PASS] All noise floor methods working correctly!")


if __name__ == "__main__":
    test_nf_methods_comparison()
