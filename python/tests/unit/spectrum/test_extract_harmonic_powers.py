"""Unit tests for harmonic detection in compute_spectrum using distorted sinewaves."""

import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.fundamentals.frequency import fold_bin_to_nyquist


@pytest.mark.parametrize("hd2_target", [-80, -90])
@pytest.mark.parametrize("hd3_target", [-80, -90])
def test_harmonic_detection(hd2_target, hd3_target):
    """Test compute_spectrum's ability to detect HD2 and HD3 harmonics in distorted sinewave."""
    # Signal parameters
    N_fft = 2**13
    Fs = 100e6
    A = 0.5
    noise_rms = 10e-6
    Fin = 123 / N_fft * Fs  # Coherent frequency

    # Compute nonlinearity coefficients for HD2 and HD3
    # HD2: k2 * A^2 / 2 = hd2_amp * A  →  k2 = hd2_amp / (A/2)
    # HD3: k3 * A^3 / 4 = hd3_amp * A  →  k3 = hd3_amp / (A^2/4)
    hd2_amp = 10**(hd2_target/20)
    hd3_amp = 10**(hd3_target/20)
    k2 = hd2_amp / (A / 2)
    k3 = hd3_amp / (A**2 / 4)

    # Generate signal with both HD2 and HD3 distortion
    t = np.arange(N_fft) / Fs
    sig_ideal = A * np.sin(2 * np.pi * Fin * t)
    signal = sig_ideal + k2 * sig_ideal**2 + k3 * sig_ideal**3 + np.random.randn(N_fft) * noise_rms

    # Run spectrum analysis
    result = compute_spectrum(signal, fs=Fs, max_harmonic=6, side_bin=1)

    # Extract detected harmonic levels
    harmonics_dbc = result['metrics']['harmonics_dbc']
    hd2_computed = harmonics_dbc[0]  # HD2 at index 0
    hd3_computed = harmonics_dbc[1]  # HD3 at index 1

    # Calculate errors
    hd2_error = abs(hd2_computed - hd2_target)
    hd3_error = abs(hd3_computed - hd3_target)

    # Get harmonic bins for printing
    harmonic_bins_temp = result['plot_data']['harmonic_bins']
    harmonic_bins_str = '[' + ', '.join(f'{h:3d}' for h in harmonic_bins_temp) + ']'

    print(f"\n[Harmonic Detection] hd2_target=[{hd2_target}], hd3_target=[{hd3_target}], Harmonic_bins={harmonic_bins_str}")
    print(f"HD2=[Computed={hd2_computed:6.2f}, Error={hd2_error:5.2f} dB], HD3=[Computed={hd3_computed:6.2f}, Error={hd3_error:5.2f} dB]")

    # Verify harmonic detection accuracy (within 1 dB)
    assert hd2_computed == pytest.approx(hd2_target, abs=0.5)
    assert hd3_computed == pytest.approx(hd3_target, abs=0.5)
