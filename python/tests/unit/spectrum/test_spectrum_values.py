"""Unit tests for power_spectrum_db_plot values at key bin locations."""

import pytest
import numpy as np
from adctoolbox.spectrum.compute_spectrum import compute_spectrum
from adctoolbox.fundamentals.frequency import fold_bin_to_nyquist


@pytest.mark.parametrize("hd2_target", [-74, -82, -90])
@pytest.mark.parametrize("hd3_target", [-67, -80, -93])
def test_spectrum_values_at_bins(hd2_target, hd3_target):
    """Test power_spectrum_db_plot values at fundamental, spur, and harmonic bins."""
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

    # Extract data
    power_spectrum_db_plot = result['plot_data']['power_spectrum_db_plot']
    fundamental_bin = result['plot_data']['fundamental_bin']
    spur_bin_idx = result['plot_data']['spur_bin_idx']
    harmonic_bins = result['plot_data']['harmonic_bins']

    # Get spectrum values at key locations
    fundamental_value = power_spectrum_db_plot[fundamental_bin]
    spur_value = power_spectrum_db_plot[spur_bin_idx]

    harmonic_values = np.zeros(len(harmonic_bins))
    for i, h_bin in enumerate(harmonic_bins):
        harmonic_values[i] = power_spectrum_db_plot[h_bin]

    # Format harmonic bins and values for printing
    harmonic_bins_str = '[' + ', '.join(f'{h:3d}' for h in harmonic_bins) + ']'
    harmonic_values_str = '[' + ', '.join(f'{v:7.2f}' for v in harmonic_values) + ']'

    print(f"\n[Spectrum Values] hd2_target=[{hd2_target}], hd3_target=[{hd3_target}]")
    print(f"Fundamental: bin=[{fundamental_bin}], value=[{fundamental_value:7.2f} dB] | Spur: bin=[{spur_bin_idx}], value=[{spur_value:7.2f} dB]")
    print(f"Harmonics:   bins={harmonic_bins_str}, values={harmonic_values_str}")

    # Verify fundamental is the highest peak
    assert fundamental_value > spur_value, "Fundamental should be higher than spur"

    # Verify harmonic values are reasonable (should be close to target levels relative to fundamental)
    # HD2 should be at fundamental_value + hd2_target (in dBc)
    # HD3 should be at fundamental_value + hd3_target (in dBc)
    hd2_expected = fundamental_value + hd2_target
    hd3_expected = fundamental_value + hd3_target

    assert harmonic_values[0] == pytest.approx(hd2_expected, abs=1.0), f"HD2 value mismatch"
    assert harmonic_values[1] == pytest.approx(hd3_expected, abs=1.0), f"HD3 value mismatch"

    # Verify all harmonic values are below fundamental
    for i, h_val in enumerate(harmonic_values):
        assert h_val < fundamental_value, f"Harmonic H{i+2} should be below fundamental"

@pytest.mark.parametrize("hd2_target", [-74, -82, -90])
@pytest.mark.parametrize("hd3_target", [-67, -80, -93])
def test_spectrum_values_at_bins_max_scale_range(hd2_target, hd3_target):
    """Test power_spectrum_db_plot values at fundamental, spur, and harmonic bins."""
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
    result = compute_spectrum(signal, fs=Fs, max_harmonic=6, side_bin=1, max_scale_range=[-1, 1])

    # Extract data
    power_spectrum_db_plot = result['plot_data']['power_spectrum_db_plot']
    fundamental_bin = result['plot_data']['fundamental_bin']
    spur_bin_idx = result['plot_data']['spur_bin_idx']
    harmonic_bins = result['plot_data']['harmonic_bins']

    # Get spectrum values at key locations
    fundamental_value = power_spectrum_db_plot[fundamental_bin]
    spur_value = power_spectrum_db_plot[spur_bin_idx]

    harmonic_values = np.zeros(len(harmonic_bins))
    for i, h_bin in enumerate(harmonic_bins):
        harmonic_values[i] = power_spectrum_db_plot[h_bin]

    # Format harmonic bins and values for printing
    harmonic_bins_str = '[' + ', '.join(f'{h:3d}' for h in harmonic_bins) + ']'
    harmonic_values_str = '[' + ', '.join(f'{v:7.2f}' for v in harmonic_values) + ']'

    print(f"\n[Spectrum Values] hd2_target=[{hd2_target}], hd3_target=[{hd3_target}]")
    print(f"Fundamental: bin=[{fundamental_bin}], value=[{fundamental_value:7.2f} dB] | Spur: bin=[{spur_bin_idx}], value=[{spur_value:7.2f} dB]")
    print(f"Harmonics:   bins={harmonic_bins_str}, values={harmonic_values_str}")

    # Verify fundamental is the highest peak
    assert fundamental_value > spur_value, "Fundamental should be higher than spur"

    # Verify harmonic values are reasonable (should be close to target levels relative to fundamental)
    # HD2 should be at fundamental_value + hd2_target (in dBc)
    # HD3 should be at fundamental_value + hd3_target (in dBc)
    hd2_expected = fundamental_value + hd2_target
    hd3_expected = fundamental_value + hd3_target

    assert harmonic_values[0] == pytest.approx(hd2_expected, abs=1.0), f"HD2 value mismatch"
    assert harmonic_values[1] == pytest.approx(hd3_expected, abs=1.0), f"HD3 value mismatch"

    # Verify all harmonic values are below fundamental
    for i, h_val in enumerate(harmonic_values):
        assert h_val < fundamental_value, f"Harmonic H{i+2} should be below fundamental"
