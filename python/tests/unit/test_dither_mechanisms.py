"""Identify physical parameters, reject insufficient excitation, preserve causality."""
import importlib

import numpy as np
import pytest

dm = importlib.import_module('adctoolbox.examples.11_calibration.dither_mechanisms')
ex = importlib.import_module('adctoolbox.examples.11_calibration.exp_cal02_dither_coverage')


def test_lfsr_period_balance_autocorrelation_and_streaming():
    generator = dm.PRBS(13, bits=7)
    whole = generator.draw(254)
    np.testing.assert_array_equal(whole[:127], whole[127:])
    assert abs(whole[:127].sum()) == 1
    for lag in (1, 2, 31, 63):
        assert whole[:127]@np.roll(whole[:127], lag) == -1
    generator.reset()
    np.testing.assert_array_equal(np.r_[generator.draw(23), generator.draw(231)], whole)


@pytest.mark.parametrize('cls,params', [(dm.MemoryPlant, [3.88, .12]),
                                      (dm.CapacitorPlant, [.006]),
                                      (dm.NonlinearPlant, [1/3.88, .16/3.88])])
def test_plants_preserve_sequences_noise_and_memory_across_chunks(cls, params):
    x = .8*np.sin(np.arange(10003)*.113)
    whole = cls(42).convert(x, params)
    plant = cls(42)
    pieces = [plant.convert(x[a:b], params) for a, b in ((0, 139), (139, 8017), (8017, len(x)))]
    for key in ('p', 'q', 'r'):
        np.testing.assert_array_equal(whole[key], np.concatenate([piece[key] for piece in pieces]))
    assert whole['overload'] == sum(piece['overload'] for piece in pieces)


def test_memory_inverse_is_exact_and_streaming():
    x = .8*np.sin(np.arange(17001)*.313)
    plant = dm.MemoryPlant(1, bits=24, noise=0.)
    c = plant.convert(x, [3.88, .18])
    corrected = dm.MemoryCorrector().apply(c, [3.88, .18])
    assert max(abs(corrected-x)) < 2e-8
    apply = dm.MemoryCorrector()
    parts = []
    for start, end in ((0, 15), (15, 8017), (8017, len(x))):
        capture = {key: c[key][start:end] for key in ('q', 'r', 'p')}
        parts.append(apply.apply(capture, [3.88, .18]))
    np.testing.assert_array_equal(np.concatenate(parts), corrected)


def test_memory_covariance_handles_correlated_probe_columns():
    rng = np.random.default_rng(7)
    p = rng.choice([-1., 1.], 8192)
    pp = p.copy()
    pp[::3] *= -1
    r = .0625*(3.88*p+.18*pp)+.12
    tracker = dm.MemoryTracker(rate=1.)
    assert tracker.update(dict(p=p, pp=pp, r=r, rail=False)) == 'updated'
    np.testing.assert_allclose(tracker.parameters, [3.88, .18], atol=1e-12)


def test_exact_capacitor_ratio_correction_and_instrument():
    x = np.tile(np.linspace(-.8, .8, 10001), 2)
    p = np.repeat([-1., 1.], len(x)//2)
    q = np.where(x > .25, 1., np.where(x < -.25, -1., 0.))
    delta = .012
    cf, cs = 1-delta*p, 1+delta*p
    r = (cf+cs)/cf*x-cs/cf*q
    y = (q+r)/2
    c = dict(y=y, q=q, p=p, r=r, rail=False)
    np.testing.assert_allclose(dm.capacitor_correct(c, [delta]), x, atol=3e-16)
    tracker = dm.CapacitorTracker(rate=1.)
    assert tracker.update(c) == 'updated'
    assert tracker.parameters[0] == pytest.approx(delta, abs=1e-14)
    # Symmetric input makes unweighted PN correlation unable to identify delta.
    naive = dm.CapacitorTracker(code_weighted=False)
    assert naive.update(c) == 'held_rank'


def test_capacitor_identification_requires_visited_outer_codes():
    c = dm.CapacitorPlant(3).convert(.1*np.sin(np.arange(4096)), [.01])
    tracker = dm.CapacitorTracker()
    assert tracker.update(c) == 'held_excitation'
    assert tracker.parameters[0] == 0.


def test_nonlinear_moments_recover_inverse_without_input_labels():
    # Cartesian product provides exact independence for this algebraic test.
    residual = np.tile(np.linspace(-.12, .12, 10001), 4)
    p = np.repeat([-1., -1/3, 1/3, 1.], 10001)
    r = dm.compressed_amplifier(residual+.1*p, 3.88, .16)
    tracker = dm.NonlinearTracker()
    assert tracker.update(dict(p=p, r=r, rail=False)) == 'updated'
    np.testing.assert_allclose(tracker.parameters, [1/3.88, .16/3.88], atol=1e-12)
    scalar = dm.NonlinearTracker(scalar=True)
    scalar.update(dict(p=p, r=r, rail=False))
    assert abs(scalar.parameters[0]-1/3.88) > .005


def test_binary_probe_cannot_support_two_odd_instruments():
    p = np.tile([-1., 1.], 2048)
    tracker = dm.NonlinearTracker()
    status = tracker.update(dict(p=p, r=.4*p, rail=False))
    assert status == 'held_rank'
    np.testing.assert_array_equal(tracker.parameters, [.25, 0.])


@pytest.mark.parametrize('tracker', [dm.MemoryTracker, dm.CapacitorTracker, dm.NonlinearTracker])
def test_observable_rail_guard_does_not_use_overload_truth(tracker):
    instance = tracker()
    initial = instance.parameters.copy()
    assert instance.update({'rail': True}) == 'held_rail'
    np.testing.assert_array_equal(instance.parameters, initial)


@pytest.mark.parametrize('mechanism', ['memory', 'capacitor', 'nonlinear'])
def test_dedicated_foreground_sine_recovers_coefficients(mechanism):
    parameters = np.array(ex.INITIAL[mechanism])
    cls = {'memory': dm.MemoryPlant, 'capacitor': dm.CapacitorPlant, 'nonlinear': dm.NonlinearPlant}[mechanism]
    actual = dm.foreground_fit(cls(33, bits=24, noise=0.), parameters, mechanism, size=8192, blocks=2)
    np.testing.assert_allclose(actual, parameters, atol=2e-7)


def test_end_to_end_memory_tracking_is_causal_and_beats_scalar():
    rows, result = ex.run_case('memory', 'sine', 'step', 11, 64, 4096, foreground_blocks=2)
    assert result['max_causality_gap'] == 0
    assert result['overloads'] == 0
    assert rows[0]['matched_background_spectral_ratio_db'] == rows[0]['nominal_spectral_ratio_db']
    final = result['windows']['final_quarter']
    assert final['matched_background']['spectral_ratio_db'] > final['scalar_correlation']['spectral_ratio_db']+12


def test_subtractive_dither_improves_spurs_but_not_upstream_distortion():
    rows = ex.linearization_sweep([11])
    pick = lambda method, amplitude, cubic: next(r for r in rows if (r['method'], r['amplitude'], r['front_cubic']) == (method, amplitude, cubic))
    assert pick('subtractive', .01, 0.)['spur_ratio_db'] > pick('off', .01, 0.)['spur_ratio_db']+10
    assert abs(pick('subtractive', .8, .03)['spur_ratio_db']-pick('off', .8, .03)['spur_ratio_db']) < 1
    assert all(r['overloads'] == 0 for r in rows)


def test_failure_diagnostics_expose_bias_and_rank_loss():
    result = ex.diagnostics()['failure_controls']
    assert abs(result['correct_alignment']-3.88) < .02
    assert abs(result['wrong_alignment_one_sample']) < .05
    assert result['probe_amplitude_actual_0.070_assumed_0.0625'] > 4.3
    assert result['input_correlated_with_probe_0.02'] > 5.
    assert result['binary_cubic_update_status'] == 'held_rank'
    assert result['overload_update_status'] == 'held_rail'


def test_cli_writes_reviewable_artifacts_and_refuses_overwrite(tmp_path):
    argv = ['--output', str(tmp_path), '--blocks', '16', '--block-size', '1024',
            '--foreground-blocks', '1', '--seeds', '11', '--mechanisms', 'gain',
            '--waveforms', 'sine', '--scenarios', 'constant']
    ex.main(argv)
    for name in ('configuration.json', 'summary.json', 'aggregate.csv', 'linearization.csv', 'diagnostics.json', 'linearization.png'):
        assert (tmp_path/name).stat().st_size > 100
    with pytest.raises(SystemExit):
        ex.main(argv)
