"""Combined error injection, isolated identification and frozen-cascade tests."""
import importlib
import json

import numpy as np
import pytest

ex = importlib.import_module('adctoolbox.examples.12_foreground.exp_cal11_combined_foreground')


def test_original_nonlinearity_is_used_and_bypass_preserves_mismatched_converter(monkeypatch):
    device = ex.make_device('redundant')
    record = dict(tone_bins=[101], amplitudes=[.85], phases=[.2], capture_seed=123)
    original = ex.ADC_Signal_Generator.apply_static_nonlinearity
    calls = []

    def apply(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ex.ADC_Signal_Generator, 'apply_static_nonlinearity', apply)
    combined = ex.capture(device, record, 2048)
    bypass = ex.capture(device, record, 2048, bypass=True)
    assert len(calls) == 1
    assert calls[0]['k2'] == .015 and calls[0]['k3'] == .04
    assert not np.array_equal(combined['bits'], bypass['bits'])
    nominal_device = {**device, 'actual_weights': device['nominal_weights']}
    no_mismatch = ex.capture(nominal_device, record, 2048, bypass=True)
    assert not np.array_equal(no_mismatch['bits'], bypass['bits'])
    np.testing.assert_array_equal(combined['truth'], bypass['truth'])
    np.testing.assert_array_equal(combined['reference'], bypass['reference'])
    assert not combined['bypass'] and bypass['bypass']


def test_bit_fit_has_no_truth_access_and_dedicated_calibration_reconstructs_sine():
    device = ex.make_device('redundant')
    record = dict(tone_bins=[509], amplitudes=[.9], phases=[.3], capture_seed=71)
    observed = ex.capture(device, record, 8192, bypass=True)
    observable = {key: observed[key] for key in ('bits', 'reference')}
    model, diagnostics = ex.fit_bit_stage(observable, 509/8192, device['nominal_weights'])
    corrected = ex.apply_bits(observed['bits'], model)
    metrics = ex.prior.spectral_metrics(corrected, observed['truth'], [509])
    assert metrics['sndr_db'] > 84
    assert np.sqrt(np.mean((corrected-observed['truth'])**2)) < 5e-5
    assert diagnostics['rank_patch_applied'] is False
    data = ex.dataset(device, [record], 8192, 8, model)
    assert set(data) == {'raw', 'bit_corrected', 'bits', 'reference'}
    assert len(data['raw']) == 1024


def test_known_nonlinearity_inverse_control():
    device = ex.make_device('binary')
    x = np.linspace(-.95, .95, 1000)
    gen = ex.ADC_Signal_Generator(len(x), 1, .1, 1, 0)
    y = gen.apply_static_nonlinearity(x, k2=device['k2'], k3=device['k3'])
    np.testing.assert_allclose(ex.inverse_control(y, device), x, atol=1e-14)


def test_combined_cli_cascade_audit_and_model_tamper(tmp_path):
    directory = tmp_path/'combined'
    ex.main(['--output', str(directory), '--records', '8', '--validation-records', '2',
             '--size', '2048', '--iterations', '3', '--seeds', '11', '--architectures', 'redundant'])
    audit = importlib.import_module('adctoolbox.examples.12_foreground.audit_combined_foreground')
    result = audit.audit(directory)
    assert result['captures'] == 18 and result['frozen_rows_replayed'] == 162
    summary = ex.read_json(directory/'summary.json')
    score = {r['method']: r['sndr_db_mean'] for r in summary if r['group'] == 'sine'}
    assert score['bits_poly'] > score['bits_only']+20
    assert score['bits_poly'] > score['poly_only']+20
    models_path = directory/'redundant_seed11/models.json'
    models = ex.read_json(models_path)
    models['bit_stage']['offset'] += .01
    models_path.write_text(json.dumps(models))
    with pytest.raises(AssertionError):
        audit.audit(directory)
    with pytest.raises(SystemExit):
        ex.main(['--output', str(directory)])
