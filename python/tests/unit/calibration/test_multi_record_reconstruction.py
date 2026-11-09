"""Joint calibration must retain each capture's fitted amplitude and offset."""

import numpy as np
import pytest

from adctoolbox.calibration import calibrate_weight_sine, scale_calibration_output


@pytest.mark.parametrize('harmonics', [1, 3])
@pytest.mark.parametrize('search', [False, True])
@pytest.mark.parametrize('first_phase', [.23, 1.3])
def test_joint_reference_matches_independent_fit(harmonics, search, first_phase):
    records, frequencies = [], []
    for size, k, amplitude, dc, phase in [
        (4096, 173, .42, .5, first_phase),
        (4096, 607, .17, .4, 1.07),
        (3072, 311, .28, .6, -.8),
    ]:
        angle = 2*np.pi*k*np.arange(size)/size + phase
        x = dc + amplitude*np.sin(angle)
        if harmonics > 1:
            x += .005*amplitude*np.cos(2*angle)
        codes = np.floor(1024*x).astype(int)
        records.append((codes[:, None] >> np.arange(9, -1, -1)) & 1)
        frequencies.append(k/size)
    guesses = np.array(frequencies) + (.03/4096 if search else 0)
    result = calibrate_weight_sine(records, freq=guesses, force_search=search,
                                   harmonic_order=harmonics)
    assert min(result['snr_db']) > 50
    amplitudes = []
    for i, (signal, ref, offset, residual, frequency) in enumerate(zip(
        result['calibrated_signal'], result['ideal'], result['offset'],
        result['error'], result['refined_frequency'],
    )):
        angle = 2*np.pi*frequency*np.arange(len(signal))[:, None]*np.arange(1, harmonics+1)
        basis = np.column_stack([np.ones(len(signal)), np.cos(angle), np.sin(angle)])
        coefficients = np.linalg.lstsq(basis, signal, rcond=None)[0]
        projected_reference = basis[:, 1:]@coefficients[1:]
        if i == 0:
            # This record fixes the solver's scale; its constrained reference
            # need not equal a second, unconstrained least-squares projection.
            assert np.linalg.norm(ref-projected_reference) <= np.linalg.norm(residual)
        else:
            np.testing.assert_allclose(ref, projected_reference, atol=1e-8)
        assert offset == pytest.approx(coefficients[0], abs=1e-8)
        np.testing.assert_allclose(residual, signal-offset-ref, atol=1e-14)
        amplitudes.append(np.hypot(coefficients[1], coefficients[1+harmonics]))
    np.testing.assert_allclose(amplitudes, [1, .17/.42, .28/.42], atol=3e-4)
    scaled = scale_calibration_output(result, scale=2)
    np.testing.assert_allclose(scaled['offset'], np.array(result['offset'])*2)
