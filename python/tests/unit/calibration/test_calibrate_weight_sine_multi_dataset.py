"""Regression tests for multi-dataset sine weight calibration.

Datasets share one weight vector but keep their own DC term, fundamental
amplitude/phase and fitted harmonics. The solver fixes only the first
dataset's fundamental to one, so the returned references, offsets and the
unit-sine normalization must all follow that global coefficient layout.
"""

import numpy as np
import pytest

import adctoolbox.calibration._lstsq_solver as lstsq_solver
from adctoolbox.calibration import calibrate_weight_sine, scale_calibration_output


N_BITS = 10
# (samples, cycles, amplitude, phase, dc); amplitude and dc are fractions of
# full scale. Lengths, frequencies, amplitudes, phases and DCs all differ.
CAPTURES = [
    (4096, 173, 0.45, 0.23, 0.50),
    (3000, 228.65, 0.20, 1.90, 0.43),
    (2048, 150.5, 0.35, -2.50, 0.56),
]


@pytest.fixture
def chosen_bases(monkeypatch):
    """Record which dual-basis assumption each least-squares solve selects.

    The last entry belongs to the solve whose coefficients are returned.
    """
    seen = []
    solve = lstsq_solver._dual_basis_lstsq

    def spy(*args, **kwargs):
        coeffs, basis_choice, err = solve(*args, **kwargs)
        seen.append(basis_choice)
        return coeffs, basis_choice, err

    monkeypatch.setattr(lstsq_solver, "_dual_basis_lstsq", spy)
    return seen


def _ideal_adc_captures(captures, n_bits=N_BITS):
    """Quantize each sine with an ideal floor ADC; bits are MSB first."""
    bits, analog, codes, freqs = [], [], [], []
    for n_samples, cycles, amplitude, phase, dc in captures:
        n = np.arange(n_samples)
        freq = cycles / n_samples
        x = 2**n_bits * (dc + amplitude * np.sin(2 * np.pi * freq * n + phase))
        code = np.floor(x).astype(int)
        bits.append((code[:, None] >> np.arange(n_bits - 1, -1, -1)) & 1)
        analog.append(x)
        codes.append(code)
        freqs.append(freq)
    return bits, analog, codes, np.array(freqs)


def _fit_tone(y, freq):
    """Return the least-squares amplitude and sine phase of one tone in y."""
    n = np.arange(len(y))
    basis = np.column_stack([
        np.ones(len(y)),
        np.sin(2 * np.pi * freq * n),
        np.cos(2 * np.pi * freq * n),
    ])
    _, sin_part, cos_part = np.linalg.lstsq(basis, y, rcond=None)[0]
    return np.hypot(sin_part, cos_part), np.arctan2(cos_part, sin_part)


@pytest.mark.parametrize(
    ("harmonic_order", "frequency_mode"),
    [(1, "known"), (3, "known"), (1, "refined"), (3, "refined"), (1, "auto")],
)
@pytest.mark.parametrize(
    ("first_phase", "basis"),
    [
        (0.23, 1),  # First fundamental is mostly sine: sine-unity basis.
        (1.35, 0),  # First fundamental is mostly cosine: cosine-unity basis.
    ],
)
def test_multi_dataset_calibration_matches_ideal_adc(
    harmonic_order, frequency_mode, first_phase, basis, chosen_bases
):
    captures = [CAPTURES[0][:3] + (first_phase, CAPTURES[0][4]), *CAPTURES[1:]]
    bits, analog, codes, freqs = _ideal_adc_captures(captures)
    lengths = np.array([capture[0] for capture in captures])
    amplitudes = np.array([capture[2] for capture in captures])
    phases = np.array([capture[3] for capture in captures])
    dcs = np.array([capture[4] for capture in captures])
    frequency_args = {
        "known": dict(freq=freqs, force_search=False),
        "refined": dict(freq=freqs + 0.3 / lengths, force_search=True),
        "auto": dict(freq=None),
    }[frequency_mode]

    result = calibrate_weight_sine(
        bits, harmonic_order=harmonic_order, **frequency_args
    )

    assert chosen_bases[-1] == basis

    # The first capture's fundamental sets the unit-sine scale, so one ADC
    # code is 1 / (2**N_BITS * A_0) for every capture. Quantization moves the
    # least-squares weights by a few percent of a code at most.
    code_unit = 1.0 / (2**N_BITS * amplitudes[0])
    np.testing.assert_allclose(
        result["weight"], 2.0 ** np.arange(N_BITS - 1, -1, -1) * code_unit,
        rtol=0, atol=0.05 * code_unit,
    )
    # A floor quantizer outputs x - 1/2 on average.
    np.testing.assert_allclose(
        result["offset"], (2**N_BITS * dcs - 0.5) * code_unit, atol=1e-4
    )
    frequency_error_bins = (result["refined_frequency"] - freqs) * lengths
    np.testing.assert_allclose(frequency_error_bins, 0, atol=1e-3)

    for k in range(len(captures)):
        amplitude, phase = _fit_tone(result["ideal"][k], freqs[k])
        assert amplitude == pytest.approx(amplitudes[k] / amplitudes[0], rel=2e-4)
        assert np.angle(np.exp(1j * (phase - phases[k]))) == pytest.approx(0, abs=1e-3)

        np.testing.assert_allclose(
            result["error"][k],
            result["calibrated_signal"][k] - result["offset"][k] - result["ideal"][k],
            rtol=0, atol=1e-12,
        )

        # Independent oracle: the ideal quantizer's own error against the
        # known analog sine.
        quantization_error = codes[k] - analog[k]
        signal_power = np.mean((analog[k] - 2**N_BITS * dcs[k]) ** 2)
        oracle_snr = 10 * np.log10(signal_power / np.var(quantization_error))
        assert result["snr_db"][k] == pytest.approx(oracle_snr, abs=0.1)


@pytest.mark.parametrize("harmonic_order", [1, 3])
@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize(
    ("first_cos", "first_sin", "basis"),
    [(0.9, 0.3, 0), (0.3, 0.9, 1)],
)
def test_multi_dataset_layout_on_exact_linear_model(
    harmonic_order, refine, first_cos, first_sin, basis, chosen_bases
):
    """Columns built so known weights reproduce every dataset's waveform."""
    rng = np.random.default_rng(20261105)
    weights = np.array([8.0, 4.1, 1.9, 1.03])
    lengths = np.array([600, 900, 1100])
    freqs = np.array([37, 101.3, 151]) / lengths
    dcs = np.array([0.4, -0.7, 1.3])
    cos_true = rng.uniform(-0.3, 0.3, (3, harmonic_order))
    sin_true = rng.uniform(-0.3, 0.3, (3, harmonic_order))
    cos_true[:, 0] = [first_cos, 0.5, -0.2]
    sin_true[:, 0] = [first_sin, -0.1, 0.7]

    columns, references = [], []
    harmonics = np.arange(1, harmonic_order + 1)
    for k, n_samples in enumerate(lengths):
        phase = 2 * np.pi * freqs[k] * np.outer(np.arange(n_samples), harmonics)
        reference = np.cos(phase) @ cos_true[k] + np.sin(phase) @ sin_true[k]
        # Tiny noise makes the dual-basis selection deterministic.
        target = dcs[k] + reference + 1e-7 * rng.standard_normal(n_samples)
        others = rng.uniform(0.0, 1.0, (n_samples, weights.size - 1))
        first = (target - others @ weights[1:]) / weights[0]
        columns.append(np.column_stack([first, others]))
        references.append(reference)

    result = calibrate_weight_sine(
        columns,
        freq=freqs + (0.25 / lengths if refine else 0.0),
        force_search=refine,
        harmonic_order=harmonic_order,
    )

    assert chosen_bases[-1] == basis
    unit = np.hypot(first_cos, first_sin)
    np.testing.assert_allclose(result["refined_frequency"], freqs, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result["weight"], weights / unit, rtol=1e-6)
    np.testing.assert_allclose(result["offset"], dcs / unit, rtol=0, atol=1e-6)
    for k in range(lengths.size):
        np.testing.assert_allclose(
            result["ideal"][k], references[k] / unit, rtol=0, atol=1e-6
        )


def test_joint_residual_snr_matches_separate_calibration():
    n_samples = 8192
    n = np.arange(n_samples)
    bits, freqs, amplitudes = [], [173 / n_samples, 607 / n_samples], [0.42, 0.17]
    for freq, amplitude, phase in zip(freqs, amplitudes, [0.23, 1.07]):
        code = np.floor(256 * (0.5 + amplitude * np.sin(2 * np.pi * freq * n + phase)))
        bits.append((code.astype(int)[:, None] >> np.arange(7, -1, -1)) & 1)

    joint = calibrate_weight_sine(bits, freq=freqs, force_search=False)
    separate = [
        calibrate_weight_sine(b, freq=f, force_search=False)["snr_db"]
        for b, f in zip(bits, freqs)
    ]

    np.testing.assert_allclose(joint["snr_db"], separate, rtol=0, atol=0.05)
    ideal_ratio = (
        _fit_tone(joint["ideal"][1], freqs[1])[0]
        / _fit_tone(joint["ideal"][0], freqs[0])[0]
    )
    assert ideal_ratio == pytest.approx(amplitudes[1] / amplitudes[0], rel=1e-3)


def test_offset_is_scalar_for_one_dataset_and_per_dataset_otherwise():
    bits, _, _, freqs = _ideal_adc_captures(CAPTURES)

    single = calibrate_weight_sine(bits[0], freq=freqs[0])
    single_list = calibrate_weight_sine([bits[0]], freq=freqs[0])
    multi = calibrate_weight_sine(bits, freq=freqs)

    assert np.ndim(single["offset"]) == 0
    assert single_list["offset"] == single["offset"]
    assert isinstance(multi["offset"], np.ndarray)
    assert multi["offset"].shape == (len(CAPTURES),)
    assert len(set(multi["offset"])) == len(CAPTURES)

    scaled = scale_calibration_output(multi, scale=2.0)
    np.testing.assert_allclose(scaled["offset"], 2.0 * multi["offset"])
