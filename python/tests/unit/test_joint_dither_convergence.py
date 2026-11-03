"""Identify bias independently and preserve the baseline and causal guards."""
import importlib

import numpy as np
import pytest

jm = importlib.import_module('adctoolbox.examples.11_calibration.joint_dither_model')
jc = importlib.import_module('adctoolbox.examples.11_calibration.joint_dither_convergence')
ex = importlib.import_module('adctoolbox.examples.11_calibration.exp_cal07_joint_convergence')


def test_pooled_moments_reproduce_original_estimator():
    original, pooled = jm.JointTracker(.2), jc.ConvergenceTracker()
    plant = jm.JointPlant(64, .2)
    for block in range(24):
        x = .8*np.sin(.861*np.arange(block*4096, (block+1)*4096))
        c, _ = plant.convert(x, jm.FINAL)
        assert original.update(c) == pooled.update(c)
        np.testing.assert_allclose(pooled.coefficients, original.coefficients, atol=3e-12)


def test_balanced_joint_identification_and_independent_noise_bias():
    rows = ex.balanced_controls()
    for row in rows:
        if row['bits'] == 'unquantized':
            assert row['c1_error'] == pytest.approx(row['c1_noise_prediction'], abs=2e-12)
        if row['bits'] == 'unquantized' and row['noise_sigma'] == 0:
            for k in ('gain', 'beta', 'history'):
                assert abs(row[k+'_error']) < 2e-11
        if row['bits'] == 13:
            assert abs(row['gain_error']) < .001
            assert abs(row['beta_error']) < .001


@pytest.mark.parametrize('kwargs,peak,expected', [
    ({'binary': True}, .8, 'held_rank'),
    ({'amplitude': .4}, .8, 'held_rail'),
    ({}, .1, 'held_excitation'),
])
def test_grouped_guards(kwargs, peak, expected):
    plant = jm.JointPlant(13, **kwargs)
    tracker = jc.ConvergenceTracker(plant.amplitude, grouped=True)
    for _ in range(3):
        # Constant inner-code input isolates missing capacitor excitation from
        # the separate finite-record electronics bounds guard.
        x = np.full(4096, peak) if expected == 'held_excitation' else peak*np.sin(np.arange(4096)*.861)
        c, _ = plant.convert(x, jm.FINAL)
        status = tracker.update(c)
    assert expected in status.values()


def test_grouped_stream_is_causal_without_truth_initialization():
    rows = ex.run_case(64, 'sine', 'step', blocks=16, size=2048)
    assert max(r['causality_gap'] for r in rows) == 0
    first = [r for r in rows if r['block'] == 0]
    nominal = next(r for r in first if r['method'] == 'nominal')
    for row in first:
        if row['method'] in ex.METHODS:
            assert row['spectral_ratio_db'] == nominal['spectral_ratio_db']
    assert np.isfinite([r['spectral_ratio_db'] for r in rows]).all()


def test_convergence_cli_artifacts_and_no_overwrite(tmp_path):
    args = ['--output', str(tmp_path), '--blocks', '8', '--long-blocks', '8',
            '--block-size', '1024', '--seeds', '64']
    ex.main(args)
    for filename in ('configuration.json', 'summary.csv', 'long_summary.csv',
                     'balanced_controls.csv', 'audit.json', 'fixed_probe_convergence.png'):
        assert (tmp_path/filename).stat().st_size > 100
    with pytest.raises(SystemExit):
        ex.main(args)
