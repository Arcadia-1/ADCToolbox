"""apply_jitter(input_signal=...) must add the jitter of the generator's tone.

The chained path interpolated the already sampled signal with a cubic spline.
Near Nyquist the spline has the wrong slope and above Nyquist it follows the
alias, so 1 ps of jitter produced an SNR 10 dB too high at fin/fs = 0.45 and
19 dB too high at fin/fs = 0.9.
"""
import numpy as np
import pytest

from adctoolbox import find_coherent_frequency
from adctoolbox.siggen import ADC_Signal_Generator

N, FS, JITTER_RMS = 2**15, 1e9, 1e-12


def _residual_snr_db(signal, norm_freq):
    t = np.arange(len(signal))
    design = np.column_stack(
        (np.ones(len(signal)), np.cos(2 * np.pi * norm_freq * t), np.sin(2 * np.pi * norm_freq * t))
    )
    coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
    residual = signal - design @ coefficients
    amplitude = np.hypot(coefficients[1], coefficients[2])
    return 10 * np.log10(amplitude**2 / 2 / np.mean(residual**2))


@pytest.mark.parametrize("fin_target", [10e6, 250e6, 450e6, 490e6, 900e6])
def test_chained_jitter_matches_the_jitter_limited_snr(fin_target):
    np.random.seed(11)
    fin, _ = find_coherent_frequency(FS, fin_target, N)
    generator = ADC_Signal_Generator(N, FS, fin, 0.49, 0.5)

    chained = generator.apply_jitter(input_signal=generator.get_clean_signal(), jitter_rms=JITTER_RMS)

    expected = -20 * np.log10(2 * np.pi * fin * JITTER_RMS)
    assert _residual_snr_db(chained, fin / FS) == pytest.approx(expected, abs=0.3)


def test_chained_and_source_jitter_agree_for_the_same_jitter_sequence():
    fin, _ = find_coherent_frequency(FS, 450e6, N)
    generator = ADC_Signal_Generator(N, FS, fin, 0.49, 0.5)

    np.random.seed(5)
    from_source = generator.apply_jitter(jitter_rms=JITTER_RMS)
    np.random.seed(5)
    chained = generator.apply_jitter(input_signal=generator.get_clean_signal(), jitter_rms=JITTER_RMS)

    np.testing.assert_allclose(chained, from_source, rtol=0, atol=1e-12)


def test_chained_jitter_keeps_the_errors_already_in_the_input():
    fin, _ = find_coherent_frequency(FS, 100e6, N)
    generator = ADC_Signal_Generator(N, FS, fin, 0.49, 0.5)
    np.random.seed(2)
    noisy = generator.apply_thermal_noise(noise_rms=1e-3)
    earlier_error = noisy - generator.get_clean_signal()

    np.random.seed(3)
    chained = generator.apply_jitter(input_signal=noisy, jitter_rms=JITTER_RMS)
    np.random.seed(3)
    jitter_only = generator.apply_jitter(jitter_rms=JITTER_RMS)

    np.testing.assert_allclose(chained - jitter_only, earlier_error, rtol=0, atol=1e-12)


def test_zero_jitter_returns_the_input_unchanged():
    generator = ADC_Signal_Generator(1024, FS, 123e6, 0.49, 0.5)
    signal = generator.get_clean_signal() + 0.01

    np.testing.assert_array_equal(generator.apply_jitter(input_signal=signal, jitter_rms=0.0), signal)
