"""Public foreground correction and packaged experiment workflow tests."""
import importlib
import json

import numpy as np
import pytest

from adctoolbox import (apply_calibration, build_calibration_features,
                       fit_calibration_polynomial, fit_calibration_mlp)


def test_history_streaming_matches_whole_record_and_does_not_invent_startup():
    x = np.arange(46.).reshape(23, 2)
    whole = build_calibration_features(x, history=3)
    state, parts, indices, start = None, [], [], 0
    for size in [1, 1, 5, 3, 13]:
        result = build_calibration_features(x[start:start+size], history=3, state=state)
        state = result['state']
        parts.append(result['features'])
        indices.extend(result['sample_indices']+start)
        start += size
    np.testing.assert_array_equal(np.vstack(parts), whole['features'])
    np.testing.assert_array_equal(indices, whole['sample_indices'])
    np.testing.assert_array_equal(whole['features'][0], x[[3, 2, 1, 0]].ravel())
    restarted = build_calibration_features(x[:2], history=3)
    assert restarted['features'].shape == (0, 8)


def test_polynomial_recovers_cross_terms_and_json_round_trip():
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, (500, 2))
    target = .2+1.3*x[:, 0]-.1*x[:, 1]**2+.07*x[:, 0]*x[:, 1]**2
    fitted = fit_calibration_polynomial(x, target, degree=3, ridge=0)
    model = json.loads(json.dumps(fitted['model'], allow_nan=False))
    xt = rng.uniform(-.9, .9, (100, 2))
    truth = .2+1.3*xt[:, 0]-.1*xt[:, 1]**2+.07*xt[:, 0]*xt[:, 1]**2
    np.testing.assert_allclose(apply_calibration(xt, model)['calibrated_signal'], truth, atol=2e-14)
    assert fitted['diagnostics']['design_rank'] == 10
    assert not fitted['diagnostics']['rank_deficient']


def test_rank_and_extrapolation_are_reported_without_hidden_clipping():
    x = np.linspace(-1, 1, 100)
    result = fit_calibration_polynomial(np.c_[x, x], x, degree=2)
    assert result['diagnostics']['rank_deficient']
    assert result['diagnostics']['design_condition_number'] is None
    json.dumps(result, allow_nan=False)
    applied = apply_calibration([[2, 2]], result['model'])
    assert applied['outside_training_range'][0]
    assert applied['calibrated_signal'][0] > 1.9


def test_polynomial_joint_excitation_and_near_collinearity_diagnostics():
    rng = np.random.default_rng(91)
    phase = rng.uniform(0, 2*np.pi, (1000, 1))
    frequency = rng.uniform(.02, .45, (1000, 1))
    x = rng.uniform(.05, .9, (1000, 1))*np.sin(phase-2*np.pi*frequency*np.arange(4))
    # All zero-DC single tones share a quadratic relation across four samples.
    single = fit_calibration_polynomial(x, x[:, 0], degree=2)['diagnostics']
    assert single['design_rank'] == 14 and single['parameters'] == 15
    assert single['design_condition_number'] is None
    two = x+.2*np.sin(rng.uniform(0, 2*np.pi, (1000, 1))
                      -2*np.pi*rng.uniform(.02, .45, (1000, 1))*np.arange(4))
    dual = fit_calibration_polynomial(two, two[:, 0], degree=2)['diagnostics']
    assert dual['design_rank'] == 15 and dual['design_condition_number'] < 100
    # Full numerical rank need not mean a well-conditioned design.
    near = np.c_[x[:, 0], x[:, 0]+1e-6*rng.normal(size=len(x))]
    diagnostics = fit_calibration_polynomial(near, x[:, 0], degree=1)['diagnostics']
    assert not diagnostics['rank_deficient']
    assert diagnostics['design_condition_number'] > 1e5
    under = fit_calibration_polynomial([[0, 0], [1, 1]], [0, 1], degree=2)
    assert under['diagnostics']['design_condition_number'] is None


def test_mlp_generalizes_and_selects_validation_best_with_frozen_normalization():
    x = np.linspace(-.9, .9, 320)
    val = np.linspace(-.88, .88, 113)
    raw = x+.08*x**3
    args = dict(validation_features=val+.08*val**3, validation_reference=val,
                hidden_units=8, max_iterations=120, seed=71)
    result = fit_calibration_mlp(raw, x, **args)
    np.testing.assert_allclose(result['model']['input_mean'], [raw.mean()], atol=1e-16)
    chosen = result['history'][result['diagnostics']['selected_iteration']]
    assert chosen['validation_mse'] == min(row['validation_mse'] for row in result['history'])
    test = np.linspace(-.86, .86, 127)
    model = json.loads(json.dumps(result['model'], allow_nan=False))
    pred = apply_calibration(test+.08*test**3, model)['calibrated_signal']
    assert np.sqrt(np.mean((pred-test)**2)) < 3e-4
    original = json.dumps(model, sort_keys=True)
    apply_calibration([5.], model)
    assert json.dumps(model, sort_keys=True) == original


@pytest.mark.parametrize('unobserved_final', [False, True])
def test_mlp_history_records_final_parameters_once_when_best_is_earlier(monkeypatch, unobserved_final):
    from types import SimpleNamespace
    from adctoolbox.calibration import foreground

    def optimizer(fun, initial, *, callback, **kwargs):
        # Deliberately move far from the initial model so the validation best
        # precedes the last iterate; the final point may or may not be reported.
        final = initial.copy()
        final[-1] += 1e6
        callback(final)
        if unobserved_final:
            final = final.copy()
            final[-1] += 1e6
        return SimpleNamespace(x=final, nit=1, success=False, message='controlled endpoint')

    monkeypatch.setattr(foreground, 'minimize', optimizer)
    x = np.linspace(-1, 1, 64)
    fitted = fit_calibration_mlp(x, x+.2*x**3, validation_features=x,
                                validation_reference=x+.2*x**3, hidden_units=3)
    assert fitted['diagnostics']['selected_iteration'] == 0
    assert len(fitted['history']) == (3 if unobserved_final else 2)


def test_mlp_objective_gradient_against_finite_differences(monkeypatch):
    from types import SimpleNamespace
    from adctoolbox.calibration import foreground
    rng = np.random.default_rng(778)

    def optimizer(fun, initial, **kwargs):
        theta = initial+rng.normal(0, .1, len(initial))
        _, analytic = fun(theta)
        numerical = []
        for index in range(len(theta)):
            step = np.zeros_like(theta)
            step[index] = 1e-6
            numerical.append((fun(theta+step)[0]-fun(theta-step)[0])/2e-6)
        np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-5)
        return SimpleNamespace(x=initial, nit=0, success=False, message='gradient checked')

    monkeypatch.setattr(foreground, 'minimize', optimizer)
    x = rng.normal(size=(37, 3))
    y = .7*x[:, 0]+.3*x[:, 1]**2-.04*x[:, 2]**3
    fit_calibration_mlp(x, y, validation_features=x+.01, validation_reference=y,
                        hidden_units=5, l2=.003)


@pytest.mark.parametrize('x,y', [([1., np.nan], [1., 2.]), ([1., 2.], [1.]), ([], [])])
def test_bad_training_data_rejected(x, y):
    with pytest.raises(ValueError):
        fit_calibration_polynomial(x, y)


def test_model_and_history_contracts():
    with pytest.raises(ValueError):
        build_calibration_features([1, 2], history=-1)
    with pytest.raises(ValueError):
        build_calibration_features([1, 2], history=1, state=[[1, 2]])
    with pytest.raises(ValueError):
        fit_calibration_polynomial(np.ones((5, 20)), np.ones(5), degree=8)
    model = fit_calibration_polynomial([1, 2, 3], [2, 4, 6], degree=1)['model']
    with pytest.raises(ValueError):
        apply_calibration([[1, 2]], model)
    model['input_scale'] = [0]
    with pytest.raises(ValueError):
        apply_calibration([1], model)


@pytest.mark.parametrize('memory', [0., .04])
def test_frontend_uses_repository_models_and_discards_memory_wrap(monkeypatch, memory):
    fb = importlib.import_module('adctoolbox.examples.12_foreground.foreground_benchmark')
    generator_type = fb.ADC_Signal_Generator
    called = []
    for name in ('apply_static_nonlinearity', 'apply_memory_effect', 'apply_quantization_noise'):
        original = getattr(generator_type, name)
        def spy(self, *args, _name=name, _original=original, **kwargs):
            called.append(_name)
            result = _original(self, *args, **kwargs)
            if _name == 'apply_memory_effect':
                result[0] = 123.  # The np.roll wrap must not reach any evaluated sample.
            return result
        monkeypatch.setattr(generator_type, name, spy)
    config = fb.FrontendConfig(memory=memory, noise_rms=0.)
    kwargs = dict(size=1024, seed=11, frequencies=[137/1024], amplitudes=[.8], phases=[.2])
    capture = fb.capture(config, **kwargs)
    assert called == ['apply_static_nonlinearity']+(['apply_memory_effect'] if memory else [])+['apply_quantization_noise']*2
    n = np.arange(-67, 1024)
    clean = .8*np.sin(2*np.pi*137/1024*n+.2)
    generator = generator_type(len(n), 1., 0., 1., 0.)
    analog = generator.apply_static_nonlinearity(clean, k2=.015, k3=.04)
    if memory:
        analog = generator.apply_memory_effect(analog, memory_strength=memory)
    expected = generator.apply_quantization_noise(analog, n_bits=14, quant_range=(-1, 1))+2**-14
    np.testing.assert_array_equal(capture['raw'], expected[64:])
    assert capture['overloads'] == 0 and len(capture['raw']) == 1027
    with pytest.raises(ValueError, match='Legacy/custom'):
        fb.FrontendConfig.from_manifest({'memory': memory})


def test_reference_channel_is_explicit():
    fb = importlib.import_module('adctoolbox.examples.12_foreground.foreground_benchmark')
    biased = fb.capture(fb.FrontendConfig(monitor_gain=.9), size=2048, seed=11,
                        frequencies=[137/2048], amplitudes=[.8], phases=[.2])
    # The fitting label comes from the configured monitor, not silently from truth.
    assert np.std(biased['reference'])/np.std(biased['truth']) == pytest.approx(.9, abs=1e-4)


def test_foreground_cli_exports_models_independent_splits_and_refuses_overwrite(tmp_path):
    ex = importlib.import_module('adctoolbox.examples.12_foreground.exp_cal03_foreground_nonlinear')
    argv = ['--output', str(tmp_path), '--size', '1024', '--records', '4',
            '--validation-records', '2', '--iterations', '3', '--stride', '8',
            '--seeds', '11', '--plants', 'static']
    ex.main(argv)
    manifest = json.loads((tmp_path/'static_seed11/data_manifest.json').read_text())
    train = {r['tone_bin'] for r in manifest['training']}
    val = {r['tone_bin'] for r in manifest['validation']}
    tests = {k for c in manifest['tests'] for k in c['bins']}
    assert not train & val and not train & tests and not val & tests
    models = json.loads((tmp_path/'static_seed11/models.json').read_text())
    assert set(models) >= {'poly5_history', 'mlp_history', 'poly5_output_fit'}
    assert (tmp_path/'foreground_comparison.png').stat().st_size > 1000
    with pytest.raises(SystemExit):
        ex.main(argv)




@pytest.mark.parametrize('retrain_mlp', [False, True])
def test_acceptance_audit_replays_foreground_artifacts(tmp_path, retrain_mlp):
    fg = importlib.import_module('adctoolbox.examples.12_foreground.exp_cal03_foreground_nonlinear')
    audit = importlib.import_module('adctoolbox.examples.12_foreground.audit_foreground_acceptance')
    excitation = importlib.import_module('adctoolbox.examples.12_foreground.exp_cal05_foreground_excitation')
    foreground = tmp_path/'fg'
    fg.main(['--output', str(foreground), '--size', '1024', '--records', '4',
             '--validation-records', '2', '--iterations', '3', '--stride', '8',
             '--seeds', '11', '--plants', 'static'])
    path = tmp_path/'audit.json'
    control = tmp_path/'excitation'
    excitation.main(['--baseline', str(foreground), '--output', str(control)]
                    + (['--retrain-mlp'] if retrain_mlp else []))
    audit.main(['--foreground', str(foreground), '--excitation', str(control), '--output', str(path)])
    result = json.loads(path.read_text())
    assert result['status'] == 'passed'
    assert result['foreground']['frozen_model_metric_rows_replayed'] == 180
    assert result['excitation']['mixed_model_metric_rows_replayed'] == (100 if retrain_mlp else 80)
    assert (foreground/'source_snapshot/foreground.py').is_file()
    if retrain_mlp:
        transfer = importlib.import_module('adctoolbox.examples.12_foreground.audit_foreground_transfer')
        transfer_output = tmp_path/'transfer'
        transfer.main(['--baseline', str(foreground), '--mixed', str(control),
                       '--output', str(transfer_output), '--records-per-group', '1'])
        captures = json.loads((transfer_output/'test_manifest.json').read_text())
        assert len(captures) == 3
        for capture in captures:
            assert not set(capture['tone_bins']) & set(capture['excluded_bins'])
            assert capture['overloads'] == 0
        import csv
        with (transfer_output/'test_metrics.csv').open(newline='') as stream:
            assert len(list(csv.DictReader(stream))) == 21
        independent = importlib.import_module('adctoolbox.examples.12_foreground.audit_repository_foreground')
        verified = independent.audit(foreground, control, transfer_output)
        assert verified['rows_replayed'] == 421
        assert verified['max_independent_fft_difference_db'] < 1e-7
        assert verified['frozen_models_unchanged']
        saved_models = json.loads((control/'static_seed11/models.json').read_text())
        saved_models['poly5_static']['coefficients'][0] += .01
        (control/'static_seed11/models.json').write_text(json.dumps(saved_models))
        with pytest.raises(AssertionError):
            independent.audit(foreground, control, transfer_output)
        with pytest.raises(SystemExit):
            transfer.main(['--baseline', str(foreground), '--mixed', str(control),
                           '--output', str(transfer_output)])
    with pytest.raises(SystemExit):
        audit.main(['--foreground', str(foreground), '--output', str(path)])
