"""Regression tests for SFDR lobes clipped by analysis masks."""

import numpy as np
import pytest

from adctoolbox.spectrum import compute_spectrum


N_SAMPLES = 1024
SAMPLE_INDEX = np.arange(N_SAMPLES)


def _tone(bin_index: int, amplitude: float) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * bin_index * SAMPLE_INDEX / N_SAMPLES)


@pytest.mark.parametrize(
    ("fundamental_bin", "spur_bin"),
    [
        (509, 507),  # Spur lobe touches the lower fundamental-mask edge.
        (301, 303),  # Spur lobe touches the upper fundamental-mask edge.
        (301, 1),  # Spur lobe touches the DC mask.
        (301, 511),  # Spur lobe touches the Nyquist boundary.
        (509, 511),  # Both spur side bins are unavailable.
        (301, 305),  # Control: the full spur lobe is visible.
    ],
)
def test_sfdr_recovers_masked_hann_spur_lobe(fundamental_bin, spur_bin):
    signal = _tone(fundamental_bin, 1.0) + _tone(spur_bin, 1e-4)

    metrics = compute_spectrum(
        signal,
        fs=100e6,
        max_scale_range=2,
        win_type="hann",
        side_bin=1,
    )["metrics"]

    assert metrics["sfdr_dbc"] == pytest.approx(80.0, abs=0.01)


@pytest.mark.parametrize("spur_bin", [303, 1, 511])
def test_assumed_signal_power_does_not_make_sfdr_location_dependent(spur_bin):
    fundamental_amplitude = 0.5
    spur_amplitude = fundamental_amplitude * 1e-4

    def sfdr_for(candidate_spur_bin):
        signal = _tone(301, fundamental_amplitude) + _tone(
            candidate_spur_bin, spur_amplitude
        )
        return compute_spectrum(
            signal,
            fs=100e6,
            max_scale_range=2,
            win_type="hann",
            side_bin=1,
            assumed_sig_pwr_dbfs=0.0,
        )["metrics"]["sfdr_dbc"]

    full_lobe_sfdr = sfdr_for(305)
    assert sfdr_for(spur_bin) == pytest.approx(full_lobe_sfdr, abs=0.01)

