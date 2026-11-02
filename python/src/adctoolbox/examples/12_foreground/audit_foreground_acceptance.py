"""Audit saved foreground runs and replay frozen models without retraining."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import inspect
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy
from adctoolbox import apply_calibration, build_calibration_features
try:
    from . import exp_cal03_foreground_nonlinear as fg
    from . import exp_cal05_foreground_excitation as excitation
except ImportError:
    import exp_cal03_foreground_nonlinear as fg
    import exp_cal05_foreground_excitation as excitation


def read_json(path):
    def invalid(value):
        raise ValueError(f'nonfinite JSON constant {value} in {path}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def check_sources(directory, config):
    paths = [Path(fg.__file__), Path(excitation.__file__), Path(fg.fb.__file__),
             Path(inspect.getfile(fg.metrics)), Path(inspect.getfile(fg.fit_calibration_mlp)),
             Path(inspect.getfile(fg.fb.ADC_Signal_Generator))]
    lookup = {p.name: p for p in paths}
    for name, expected in config['source_sha256'].items():
        if name not in lookup:
            raise ValueError(f'Historical source {name} is not in the current example; '
                             'use the matching revision/source snapshot before replay')
        actual = hashlib.sha256(lookup[name].read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Source changed since run: {name}; use matching source before replay')
    # Preserve exact executed source alongside data for portable review.
    snapshots = directory/'source_snapshot'
    snapshots.mkdir(exist_ok=True)
    for name in config['source_sha256']:
        target = snapshots/name
        if target.exists() and target.read_bytes() != lookup[name].read_bytes():
            raise ValueError(f'Conflicting source snapshot: {target}')
        target.write_bytes(lookup[name].read_bytes())


def audit_foreground(directory):
    config = read_json(directory/'configuration.json')
    check_sources(directory, config)
    cases = predictions = 0
    max_error = 0.
    optimizer_budget_cases = []
    for plant in config['plants']:
        for seed in config['seeds']:
            case = directory/f'{plant}_seed{seed}'
            manifest = read_json(case/'data_manifest.json')
            assert len(manifest['training']) == config['records']
            assert len(manifest['validation']) == config['validation_records']
            train = {r['tone_bin'] for r in manifest['training']}
            valid = {r['tone_bin'] for r in manifest['validation']}
            tests = {k for r in manifest['tests'] for k in r['bins']}
            assert not train & valid and not train & tests and not valid & tests
            assert sum(r['overloads'] for part in ('training', 'validation') for r in manifest[part]) == 0
            models = read_json(case/'models.json')
            diagnostic = read_json(case/'training_diagnostics.json')['mlp_history']
            losses = read_csv(case/'mlp_history.csv')
            chosen = losses[diagnostic['selected_iteration']]
            assert float(chosen['validation_mse']) == min(float(r['validation_mse']) for r in losses)
            if not diagnostic['optimizer_converged']:
                optimizer_budget_cases.append({'case': case.name, 'message': diagnostic['optimizer_message']})
            expected = {(r['condition'], r['method']): r for r in read_csv(case/'test_metrics.csv')}
            assert len(expected) == len(manifest['tests'])*(len(models)+1)
            device = fg.fb.FrontendConfig.from_manifest(manifest['plant'])
            for ci, condition in enumerate(manifest['tests']):
                target_device = replace(device, gain=device.gain*1.02, cubic=device.cubic*1.5) if condition.get('drift') else device
                capture = fg.fb.capture(target_device, size=config['size'], seed=seed+30000+ci,
                    frequencies=[k/config['size'] for k in condition['bins']],
                    amplitudes=condition['amplitudes'], phases=condition['phases'], chirp=condition.get('chirp', False))
                built = build_calibration_features(capture['raw'], history=3)
                truth = capture['truth'][built['sample_indices']]
                for name in ['raw', *models]:
                    if name == 'raw':
                        y = built['features'][:, 0]
                    else:
                        model = models[name]
                        y = apply_calibration(built['features'][:, :model['feature_count']], model)['calibrated_signal']
                    measured = fg.metrics(y, truth, condition['bins']) if condition['bins'] else {
                        'error_snr_db': float(10*np.log10(np.var(truth)/np.mean((y-truth)**2)))}
                    row = expected[condition['name'], name]
                    assert int(row['overloads']) == capture['overloads']
                    for key, value in measured.items():
                        error = abs(value-float(row[key]))
                        max_error = max(max_error, error)
                        assert error < 1e-7, (case.name, condition['name'], name, key, error)
                    predictions += 1
            cases += 1
    # Recompute all aggregate cells from the exported per-test table.
    rows = read_csv(directory/'test_metrics.csv')
    for row in rows:
        row['seed'] = int(row['seed'])
        for field in ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db', 'sndr_db', 'sfdr_db'):
            row[field] = float(row[field]) if row[field] else None
    rebuilt = fg.summarize(rows)
    saved = read_json(directory/'summary.json')
    assert len(rebuilt) == len(saved)
    for actual, expected in zip(rebuilt, saved):
        for key, value in actual.items():
            if isinstance(value, float):
                assert abs(value-expected[key]) < 1e-10
            else:
                assert value == expected[key]
    return {'cases': cases, 'frozen_model_metric_rows_replayed': predictions,
            'maximum_replay_error_db': max_error, 'aggregate_recomputed': True,
            'optimizer_not_converged': optimizer_budget_cases}




def audit_excitation(directory, baseline):
    config = read_json(directory/'configuration.json')
    check_sources(directory, config)
    assert hashlib.sha256((baseline/'configuration.json').read_bytes()).hexdigest() == config['baseline_configuration_sha256']
    rows = read_csv(directory/'test_metrics.csv')
    for row in rows:
        row['seed'] = int(row['seed'])
        for metric in ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db', 'sndr_db', 'sfdr_db'):
            row[metric] = float(row[metric]) if row[metric] else None
    predictions = cases = 0
    for plant in config['plants']:
        for seed in config['seeds']:
            case = directory/f'{plant}_seed{seed}'
            manifest = read_json(baseline/f'{plant}_seed{seed}/data_manifest.json')
            training = read_json(case/'training_manifest.json')
            assert len(training) == config['records'] and not sum(r['overloads'] for r in training)
            assert sum(len(r['tone_bins']) == 2 for r in training) == len(training)//2
            train_bins = {k for r in training for k in r['tone_bins']}
            assert not train_bins & {k for r in manifest['tests'] for k in r['bins']}
            assert not train_bins & {r['tone_bin'] for r in manifest['validation']}
            models = read_json(case/'models.json')
            if config.get('retrain_mlp', False):
                validation = read_json(case/'validation_manifest.json')
                assert len(validation) == config['validation_records']
                assert not sum(r['overloads'] for r in validation)
                valid_bins = {k for r in validation for k in r['tone_bins']}
                test_bins = {k for r in manifest['tests'] for k in r['bins']}
                assert not valid_bins & (train_bins | test_bins)
                assert not {r['capture_seed'] for r in training} & {r['capture_seed'] for r in validation}
                diagnostics = read_json(case/'training_diagnostics.json')['mlp_history']
                history = read_csv(case/'mlp_history.csv')
                assert float(history[diagnostics['selected_iteration']]['validation_mse']) == min(
                    float(r['validation_mse']) for r in history)
                assert diagnostics['iterations'] <= config['iterations']
                assert len(models['mlp_history']['hidden_bias']) == config['hidden_units']
            device = fg.fb.FrontendConfig.from_manifest(manifest['plant'])
            for ci, condition in enumerate(manifest['tests']):
                test_device = replace(device, gain=device.gain*1.02, cubic=device.cubic*1.5) if condition.get('drift') else device
                capture = fg.fb.capture(test_device, size=config['size'], seed=seed+30000+ci,
                    frequencies=[k/config['size'] for k in condition['bins']], amplitudes=condition['amplitudes'],
                    phases=condition['phases'], chirp=condition.get('chirp', False))
                built = build_calibration_features(capture['raw'], history=3)
                truth = capture['truth'][built['sample_indices']]
                for name, model in models.items():
                    y = apply_calibration(built['features'][:, :model['feature_count']], model)['calibrated_signal']
                    measured = fg.metrics(y, truth, condition['bins']) if condition['bins'] else {
                        'error_snr_db': float(10*np.log10(np.var(truth)/np.mean((y-truth)**2)))}
                    row = next(r for r in rows if (r['plant'], r['seed'], r['condition'], r['method'], r['training_excitation']) ==
                               (plant, seed, condition['name'], name, 'mixed'))
                    for metric, value in measured.items():
                        assert abs(value-row[metric]) < 1e-7
                    predictions += 1
            cases += 1
    rebuilt = excitation.aggregate(rows)
    saved = read_json(directory/'summary.json')
    assert len(rebuilt) == len(saved)
    for actual, expected in zip(rebuilt, saved):
        for key, value in actual.items():
            if isinstance(value, float):
                assert abs(value-expected[key]) < 1e-10
            else:
                assert value == expected[key]
    return {'cases': cases, 'mixed_model_metric_rows_replayed': predictions, 'aggregate_recomputed': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foreground', type=Path, required=True)
    parser.add_argument('--excitation', type=Path)
    parser.add_argument('--output', type=Path, default=Path('calibration_audit.json'))
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a new audit output file')
    result = {'foreground': audit_foreground(args.foreground),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'status': 'passed'}
    if args.excitation:
        result['excitation'] = audit_excitation(args.excitation, args.foreground)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fg.write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
