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
    sar_encode,
    sar_reconstruct,
    sar_ideal_weights,
    sar_apply_mismatch,
)


# ──────────────────────────── basic structure ────────────────────────────

def test_ideal_weights_sum_to_one():
    for n in [4, 8, 12, 16, 20]:
        w = sar_ideal_weights(n)
        assert w.shape == (n,)
        assert np.isclose(w.sum(), 1.0)
        # MSB at index 0
        assert w[0] > w[-1]


def test_ideal_weights_binary_progression():
    w = sar_ideal_weights(4)
    expected = np.array([8, 4, 2, 1]) / 15.0
    assert np.allclose(w, expected)


def test_ideal_weights_with_redundancy():
    w = sar_ideal_weights(4, redundant_bit=1)
    # Should be [8, 4, 4, 2, 1] / 19
    assert w.shape == (5,)
    assert np.isclose(w[1], w[2])


def test_apply_mismatch_preserves_shape_and_changes_values():
    w = sar_ideal_weights(10)
    rng = np.random.default_rng(42)
    w_mis = sar_apply_mismatch(w, sigma=0.05, rng=rng)
    assert w_mis.shape == w.shape
    assert not np.allclose(w, w_mis)


# ────────────────────────── encode + reconstruct ─────────────────────────

def test_encode_dc_extremes():
    """vin = 0 → all zeros; vin = 1 → all ones (modulo last-cap tie)."""
    w = sar_ideal_weights(8)
    rng = np.random.default_rng(0)
    codes = sar_encode(np.array([0.0, 1.0]), w, noise_rms=0.0, rng=rng)
    assert codes[0].sum() == 0          # vin=0 → no bits set
    assert codes[1].sum() == 8          # vin=1 → all bits set


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
    codes = sar_encode(vin, w, noise_rms=0.0, rng=np.random.default_rng(42))
    aout = sar_reconstruct(codes, w) - 0.5
    metrics = analyze_spectrum(aout, fs=fs, create_plot=False)
    enob = metrics["enob"]
    assert abs(enob - num_bits) < tolerance, (
        f"N={num_bits}: ENoB={enob:.3f}, expected ~{num_bits}, "
        f"|Δ|={abs(enob-num_bits):.3f} > {tolerance}"
    )


# ───────────────────── comparator noise sensitivity ──────────────────────

def test_higher_noise_degrades_enob():
    fs = 1.0e9
    n_samples = 16384
    cycles = 131
    num_bits = 12
    w = sar_ideal_weights(num_bits)
    n = np.arange(n_samples)
    vin = 0.5 + 0.5 * np.sin(2 * np.pi * cycles * n / n_samples)

    enobs = []
    for noise_rms in [0.0, 1e-3, 1e-2]:
        codes = sar_encode(vin, w, noise_rms=noise_rms,
                           rng=np.random.default_rng(7))
        aout = sar_reconstruct(codes, w) - 0.5
        enobs.append(analyze_spectrum(aout, fs=fs, create_plot=False)["enob"])

    # Strictly monotonic: more noise → lower ENoB
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
    codes = sar_encode(vin, w_actual, noise_rms=0.0,
                       rng=np.random.default_rng(0))
    # Naive reconstruction with NOMINAL weights (no cal)
    aout = sar_reconstruct(codes, w_nom) - 0.5
    enob = analyze_spectrum(aout, fs=fs, create_plot=False)["enob"]
    # 5% mismatch typically drops ENoB by 4-6 b — exact value depends on
    # mismatch realization, but it must be well below the architectural N
    assert enob < num_bits - 3, (
        f"5%% mismatch should drop ENoB substantially; got {enob:.2f} for N={num_bits}"
    )
