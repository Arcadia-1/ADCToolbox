"""Coherent FFT checks of requested output harmonic levels."""

import numpy as np
import pytest

from adctoolbox.siggen import ADC_Signal_Generator


@pytest.mark.parametrize('orders', [(2,), (3,), (4,), (5,), (2, 3, 4, 5)])
@pytest.mark.parametrize('target', [-20, -40, -60])
def test_harmonic_targets_are_independent(orders, target):
    size, tone_bin, amplitude = 8192, 137, .4
    generator = ADC_Signal_Generator(size, float(size), float(tone_bin), amplitude, .5)
    levels = {f'hd{order}_dB': target for order in orders}
    output = generator.apply_static_nonlinearity_hd(**levels)
    spectrum = np.fft.rfft(output-.5)
    fundamental = abs(spectrum[tone_bin])
    assert 2*fundamental/size == pytest.approx(amplitude, abs=1e-12)
    for order in range(2, 6):
        ratio = abs(spectrum[order*tone_bin])/fundamental
        if order in orders:
            assert 20*np.log10(ratio) == pytest.approx(target, abs=1e-9)
        else:
            assert ratio < 1e-12
