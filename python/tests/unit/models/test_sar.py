"""Unit tests for ``adctoolbox.models.sar`` SAR forward model.

The headline sanity check: at 0 dBFS coherent sine with no comparator
noise and no cap mismatch, measured ENoB on the reconstruction should
match the architectural N to within finite-FFT noise (±0.05 b for
N >= 8, looser for smaller N where the FFT noise floor approaches the
Hann window sidelobe level).
"""
import numpy as np
import pytest

from adctoolbox import (
    analyze_spectrum,
    sar_convert,
    sar_reconstruct,
    sar_ideal_weights,
    sar_apply_mismatch,
)


# ──────────────────────────── basic structure ────────────────────────────

def test_ideal_weights_use_adc_lsb_convention():
    for n in [4, 8, 12, 16, 20]:
        w = sar_ideal_weights(n)
        assert w.shape == (n,)
        assert np.isclose(w[-1], 1.0 / 2**n)
        assert np.isclose(w.sum(), 1.0 - 1.0 / 2**n)
        # MSB at index 0
        assert w[0] > w[-1]


def test_ideal_weights_binary_progression():
    w = sar_ideal_weights(4)
    expected = np.array([8, 4, 2, 1]) / 16.0
    assert np.allclose(w, expected)


def test_ideal_weights_with_redundancy():
    w = sar_ideal_weights(4, redundant_bit=1)
    # Redundancy is normalized by sum(bit weights) + one LSB: 8+4+4+2+1+1.
    expected = np.array([8, 4, 4, 2, 1]) / 20.0
    assert w.shape == (5,)
    assert np.allclose(w, expected)
    assert np.isclose(w[1], w[2])
    assert w.sum() > (1.0 - 1.0 / 16.0)
    assert np.isclose(w.sum(), 19.0 / 20.0)


def test_apply_mismatch_preserves_shape_and_changes_values():
    w = sar_ideal_weights(10)
    rng = np.random.default_rng(42)
    w_mis = sar_apply_mismatch(w, sigma=0.05, rng=rng)
    assert w_mis.shape == w.shape
    assert not np.allclose(w, w_mis)


# ────────────────────────── encode + reconstruct ─────────────────────────

def test_encode_dc_extremes():
    """vin = 0 → all zeros; vin = 1 → all ones."""
    w = sar_ideal_weights(8)
    rng = np.random.default_rng(0)
    codes = sar_convert(np.array([0.0, 1.0]), w, rng=rng)
    assert codes[0].sum() == 0          # vin=0 → no bits set
    assert codes[1].sum() == 8          # vin=1 → all bits set


def test_ideal_transfer_thresholds_and_full_scale_code():
    """A 4-bit ideal ADC has 1/16 LSBs and max reconstruction 15/16."""
    w = sar_ideal_weights(4)
    vin = np.array([
        0.0,
        np.nextafter(1 / 16, 0.0),
        1 / 16,
        np.nextafter(15 / 16, 0.0),
        15 / 16,
        1.0,
    ])
    codes = sar_convert(vin, w, rng=np.random.default_rng(0))
    aout = sar_reconstruct(codes, w)

    assert np.array_equal(
        codes,
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
    )
    assert np.allclose(aout, [0, 0, 1/16, 14/16, 15/16, 15/16])
    assert np.isclose(aout[-1], 15 / 16)


def test_quant_range_scales_input_thresholds_and_output():
    """The user-facing range is a two-endpoint voltage range."""
    w = sar_ideal_weights(4)
    eps = 1e-12
    vin = np.array([
        -0.5,
        -0.5 + 1 / 16 - eps,
        -0.5 + 1 / 16,
        -0.5 + 15 / 16 - eps,
        -0.5 + 15 / 16,
        0.5,
    ])
    codes = sar_convert(vin, w, quant_range=(-0.5, 0.5), rng=np.random.default_rng(0))
    aout = sar_reconstruct(codes, w, quant_range=(-0.5, 0.5))

    assert np.array_equal(
        codes,
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
    )
    assert np.allclose(aout, [-0.5, -0.5, -7/16, 6/16, 7/16, 7/16])


def test_reconstruct_is_codes_at_weights():
    w = sar_ideal_weights(6)
    codes = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], dtype=np.int8)
    aout = sar_reconstruct(codes, w)
    assert np.allclose(aout, codes @ w)


# ──────────────────── headline: ENoB ≈ N at 0 dBFS ───────────────────────

@pytest.mark.parametrize(
    "num_bits, tolerance",
    [
        (4, 0.20),   # low N: FFT noise floor approaches Hann sidelobe
        (8, 0.10),
        (12, 0.05),
        (16, 0.05),
        (20, 0.05),
    ],
)
def test_ideal_sar_enob_matches_n_at_0dBFS(num_bits, tolerance):
    fs = 1.0e9
    n_samples = 16384
    cycles = 131  # coprime to 2^14 -> coherent FFT
    w = sar_ideal_weights(num_bits)
    # 0 dBFS sine in normalized [0, 1]: amp=0.5, offset=0.5
    n = np.arange(n_samples)
    vin = 0.5 + 0.5 * np.sin(2 * np.pi * cycles * n / n_samples)
    codes = sar_convert(vin, w, rng=np.random.default_rng(42))
    assert codes.shape == (n_samples, num_bits)
    aout = sar_reconstruct(codes, w) - 0.5
    metrics = analyze_spectrum(aout, fs=fs, create_plot=False)
    enob = metrics["enob"]
    assert abs(enob - num_bits) < tolerance, (
        f"N={num_bits}: ENoB={enob:.3f}, expected ~{num_bits}, "
        f"|Δ|={abs(enob-num_bits):.3f} > {tolerance}"
    )


# ─────────────────────── noise sensitivity ───────────────────────────────

def test_higher_comparator_noise_degrades_enob():
    fs = 1.0e9
    n_samples = 16384
    cycles = 131
    num_bits = 12
    w = sar_ideal_weights(num_bits)
    n = np.arange(n_samples)
    vin = 0.5 + 0.5 * np.sin(2 * np.pi * cycles * n / n_samples)

    enobs = []
    for noise_rms in [0.0, 1e-3, 1e-2]:
        codes = sar_convert(
            vin,
            w,
            comparator_noise_rms=noise_rms,
            rng=np.random.default_rng(7),
        )
        aout = sar_reconstruct(codes, w) - 0.5
        enobs.append(analyze_spectrum(aout, fs=fs, create_plot=False)["enob"])

    # Strictly monotonic: more noise → lower ENoB
    assert enobs[0] > enobs[1] > enobs[2]


def test_higher_sampling_noise_degrades_enob():
    fs = 1.0e9
    n_samples = 16384
    cycles = 131
    num_bits = 12
    w = sar_ideal_weights(num_bits)
    n = np.arange(n_samples)
    vin = 0.5 + 0.5 * np.sin(2 * np.pi * cycles * n / n_samples)

    enobs = []
    for noise_rms in [0.0, 1e-3, 1e-2]:
        codes = sar_convert(
            vin,
            w,
            sampling_noise_rms=noise_rms,
            rng=np.random.default_rng(7),
        )
        aout = sar_reconstruct(codes, w) - 0.5
        enobs.append(analyze_spectrum(aout, fs=fs, create_plot=False)["enob"])

    assert enobs[0] > enobs[1] > enobs[2]


# ─────────────────────── cap mismatch sensitivity ────────────────────────

def test_cap_mismatch_degrades_enob_without_cal():
    """5% σ_cap should drop ENoB substantially below architectural N."""
    fs = 1.0e9
    n_samples = 16384
    cycles = 131
    num_bits = 12
    w_nom = sar_ideal_weights(num_bits)
    w_actual = sar_apply_mismatch(w_nom, sigma=0.05,
                                  rng=np.random.default_rng(20260514))

    n = np.arange(n_samples)
    vin = 0.5 + 0.5 * np.sin(2 * np.pi * cycles * n / n_samples)
    codes = sar_convert(vin, w_actual, rng=np.random.default_rng(0))
    # Naive reconstruction with NOMINAL weights (no cal), matching sar_convert.
    aout = sar_reconstruct(codes, w_nom) - 0.5
    enob = analyze_spectrum(aout, fs=fs, create_plot=False)["enob"]
    # 5% mismatch typically drops ENoB by 4-6 b — exact value depends on
    # mismatch realization, but it must be well below the architectural N
    assert enob < num_bits - 3, (
        f"5%% mismatch should drop ENoB substantially; got {enob:.2f} for N={num_bits}"
    )
