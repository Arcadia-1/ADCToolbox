"""Regression tests for ranking SFDR spurs by integrated lobe power.

With a Hann window, a half-bin spur splits its power between two bins. Its
tallest bin can therefore be lower than that of a weaker coherent spur, even
though its integrated lobe, which defines SFDR, is larger.
"""

import numpy as np
import pytest

from adctoolbox.spectrum import compute_spectrum


N_SAMPLES = 8192
SAMPLE_INDEX = np.arange(N_SAMPLES)
FUNDAMENTAL_BIN = 123
FUNDAMENTAL_AMPLITUDE = 0.8


def _tone(bin_position: float, amplitude: float, phase: float = 0.0) -> np.ndarray:
    return amplitude * np.sin(
        2 * np.pi * bin_position * SAMPLE_INDEX / N_SAMPLES + phase
    )


@pytest.mark.parametrize(
    ("coherent_bin", "half_bin"),
    [
        (1000, 1700.5),  # The configuration from upstream issue 110.
        (1700, 1000.5),  # Same spurs with their frequency order swapped.
    ],
)
@pytest.mark.parametrize(
    ("coherent_amplitude", "half_bin_amplitude"),
    [
        (8e-4, 9e-4),  # Half-bin spur is stronger but has the lower peak bin.
        (9e-4, 8e-4),  # Coherent spur is stronger.
    ],
)
def test_sfdr_selects_spur_with_largest_integrated_power(
    coherent_bin, half_bin, coherent_amplitude, half_bin_amplitude
):
    signal = (
        _tone(FUNDAMENTAL_BIN, FUNDAMENTAL_AMPLITUDE)
        + _tone(coherent_bin, coherent_amplitude, phase=0.1)
        + _tone(half_bin, half_bin_amplitude, phase=0.7)
    )

    result = compute_spectrum(
        signal,
        max_scale_range=[-1, 1],
        win_type="hann",
        side_bin=3,
        max_harmonic=1,
        nf_method=3,
    )

    spur_bin = result["plot_data"]["spur_bin_idx"]
    if half_bin_amplitude > coherent_amplitude:
        assert abs(spur_bin - half_bin) == 0.5
    else:
        assert spur_bin == coherent_bin
    strongest_amplitude = max(coherent_amplitude, half_bin_amplitude)
    expected_sfdr = 20 * np.log10(FUNDAMENTAL_AMPLITUDE / strongest_amplitude)
    assert result["metrics"]["sfdr_dbc"] == pytest.approx(expected_sfdr, abs=0.01)


def test_spur_ranking_includes_masked_lobe_compensation():
    # The spur at bin 507 loses one Hann side bin to the fundamental mask.
    # Its visible lobe is smaller than the full lobe of the weaker spur at
    # bin 305; ranking must use the compensated lobe, as SFDR does.
    n = np.arange(1024)
    signal = (
        np.sin(2 * np.pi * 509 * n / 1024)
        + 1e-4 * np.sin(2 * np.pi * 507 * n / 1024)
        + 0.95e-4 * np.sin(2 * np.pi * 305 * n / 1024)
    )

    result = compute_spectrum(
        signal,
        fs=100e6,
        max_scale_range=2,
        win_type="hann",
        side_bin=1,
    )

    assert result["plot_data"]["spur_bin_idx"] == 507
    assert result["metrics"]["sfdr_dbc"] == pytest.approx(80.0, abs=0.01)
