"""Physical observation and spectral contracts for the frozen comparison."""
import csv
import importlib
import json

import numpy as np
import pytest

ex = importlib.import_module('adctoolbox.examples.12_foreground.exp_cal10_foreground_bitweights')


def test_midrise_bit_decode_exact_and_invalid_range():
    codes = np.arange(2**14)
    raw = (codes+.5)/8192-1
    bits = ex.decode_midrise(raw)
    np.testing.assert_array_equal(bits@2**np.arange(13, -1, -1), codes)
    np.testing.assert_array_equal(2*(bits@ex.sar_ideal_weights(14))-1+2**-14, raw)
    with pytest.raises(ValueError):
        ex.decode_midrise([2.])


def test_spectral_ratios_include_harmonics_and_separate_multitone():
    n = np.arange(4096)
    fundamental = .7*np.sin(2*np.pi*103*n/len(n))
    distortion = .007*np.cos(2*np.pi*206*n/len(n))
    result = ex.spectral_metrics(fundamental+distortion, fundamental, [103])
    assert result['sndr_db'] == pytest.approx(40., abs=1e-8)
    assert result['sfdr_db'] == pytest.approx(40., abs=1e-8)
    assert result['public_fft_max_difference_db'] < 1e-8
    multi = ex.spectral_metrics(fundamental+distortion, fundamental+distortion, [103, 206])
    assert multi['sndr_db'] is None and multi['sfdr_db'] is None
    assert multi['spectral_ratio_db'] > 200


def test_sar_redundancy_and_internal_bits_have_physical_meaning():
    nominal = ex.sar_ideal_weights(14, redundant_bit=2)
    actual = nominal.copy()
    actual[0] *= 1.01
    device = dict(kind='sar', nominal_weights=nominal.tolist(), actual_weights=actual.tolist())
    record = dict(tone_bins=[509], amplitudes=[.85], phases=[.3], capture_seed=101)
    observed = ex.capture(device, record, 8192)
    outputs, truth = ex.predictions(observed, {}, device)
    raw = ex.spectral_metrics(outputs['raw'], truth, [509])
    known = ex.spectral_metrics(outputs['known_weights_control'], truth, [509])
    assert known['sndr_db'] > raw['sndr_db']+25
    assert observed['overloads'] == 0
    # Redundant internal decision patterns may collapse to the same final code.
    raw_integer = observed['bits']@(nominal/nominal[-1]).round().astype(int)
    grouped = {}
    for code, pattern in zip(raw_integer, observed['bits']):
        grouped.setdefault(int(code), set()).add(tuple(pattern))
    assert any(len(patterns) > 1 for patterns in grouped.values())


def test_example_frozen_cli_splits_and_artifacts(tmp_path):
    directory = tmp_path/'comparison'
    ex.main(['--output', str(directory), '--sar-records', '6', '--sar-validation-records', '2',
             '--sar-size', '1024', '--sar-iterations', '3', '--sar-plants', 'redundant', '--seeds', '11'])
    case = directory/'sar_redundant_seed11'
    manifest = ex.read_json(case/'manifest.json')
    train = {k for r in manifest['training'] for k in r['tone_bins']}
    valid = {k for r in manifest['validation'] for k in r['tone_bins']}
    test = {k for r in manifest['tests'] for k in r['bins']}
    assert not train & valid and not train & test and not valid & test
    models = ex.read_json(case/'models.json')
    before = json.dumps(models, sort_keys=True)
    condition = manifest['tests'][0]
    observed = ex.capture(manifest['device'], dict(tone_bins=condition['bins'],
        amplitudes=condition['amplitudes'], phases=condition['phases'], capture_seed=30011), 1024)
    outputs, _ = ex.predictions(observed, models, manifest['device'])
    assert len(outputs) == 7 and len(outputs['raw']) == 1024
    assert json.dumps(models, sort_keys=True) == before
    with (directory/'test_metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 119
    assert all(r['sndr_db'] and r['sfdr_db'] for r in rows if r['group'] == 'sine')
    assert all(not r['sndr_db'] and not r['sfdr_db'] for r in rows if r['group'] == 'multitone')
    assert (directory/'spectral_comparison.png').is_file()
    assert (directory/'source_snapshot/models/sar.py').is_file()
    audit = importlib.import_module('adctoolbox.examples.12_foreground.audit_foreground_bitweights')
    checked = audit.audit(directory)
    assert checked['frozen_rows_replayed'] == 119 and checked['status'] == 'passed'
    changed = dict(models)
    changed['bits_sine'] = {**models['bits_sine'], 'offset': models['bits_sine']['offset']+.01}
    (case/'models.json').write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        audit.audit(directory)
    with pytest.raises(SystemExit):
        ex.main(['--output', str(directory)])
