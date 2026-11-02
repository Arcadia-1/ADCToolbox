"""Check extracted code against existing full-run frozen outputs, without refitting.

Requires the five original output directories supplied with --source-output.
Records current code hashes separately from historical experiment provenance.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import csv
import hashlib
import importlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from adctoolbox import apply_calibration, build_calibration_features

prefix = 'adctoolbox.examples.12_foreground.'
ex = importlib.import_module(prefix+'exp_cal03_foreground_nonlinear')
bit_audit = importlib.import_module(prefix+'audit_foreground_bitweights')
combined_audit = importlib.import_module(prefix+'audit_combined_foreground')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay(root):
    maximum, count = 0., 0
    input_hashes = {}

    def record(path):
        input_hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)

    def check(row, signal, truth, bins):
        nonlocal maximum, count
        measured = bit_audit.independent_metrics(signal, truth, bins) if bins else {
            'error_snr_db': float(10*np.log10(np.var(truth)/np.mean((signal-truth)**2)))}
        for key, value in measured.items():
            error = abs(value-float(row[key]))
            assert error < 1e-7, (row['condition'], row['method'], key, error)
            maximum = max(maximum, error)
        count += 1

    baseline = root/'foreground_repository_20261102'
    mixed = root/'excitation_repository_20261102'
    fresh = root/'transfer_repository_20261102'
    cfg = record(baseline/'configuration.json')
    models, manifests = {}, {}
    for plant in cfg['plants']:
        for seed in cfg['seeds']:
            case = f'{plant}_seed{seed}'
            manifests[case] = record(baseline/case/'data_manifest.json')
            models[case] = {label: record(folder/case/'models.json')
                            for label, folder in [('single', baseline), ('mixed', mixed)]}
    before = json.dumps(models, sort_keys=True)
    for folder in (baseline, mixed, fresh):
        path = folder/'test_metrics.csv'
        input_hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        captures = {}
        fresh_records = record(fresh/'test_manifest.json') if folder == fresh else []
        for row in rows:
            plant, seed = row['plant'], int(row['seed'])
            case = f'{plant}_seed{seed}'
            manifest = manifests[case]
            key = (case, row['condition'])
            if key not in captures:
                device = ex.fb.FrontendConfig.from_manifest(manifest['plant'])
                if folder == fresh:
                    condition = next(r for r in fresh_records if (r['plant'], r['seed'], r['condition']) == (plant, seed, row['condition']))
                    bins = condition['tone_bins']
                    capture_seed = condition['capture_seed']
                else:
                    ci, condition = next((i, c) for i, c in enumerate(manifest['tests']) if c['name'] == row['condition'])
                    bins, capture_seed = condition['bins'], seed+30000+ci
                    if condition.get('drift'):
                        device = replace(device, gain=device.gain*1.02, cubic=device.cubic*1.5)
                capture = ex.fb.capture(device, size=cfg['size'], seed=capture_seed,
                    frequencies=np.array(bins)/cfg['size'], amplitudes=condition['amplitudes'],
                    phases=condition['phases'], chirp=condition.get('chirp', False))
                features = build_calibration_features(capture['raw'], history=3)
                captures[key] = (features['features'], capture['truth'][features['sample_indices']], bins)
            features, truth, bins = captures[key]
            if row['method'] == 'raw':
                signal = features[:, 0]
            else:
                model = models[case][row.get('training_excitation', 'single')][row['method']]
                signal = apply_calibration(features[:, :model['feature_count']], model)['calibrated_signal']
            check(row, signal, truth, bins)
    assert json.dumps(models, sort_keys=True) == before
    bits = bit_audit.audit(root/'bitweight_repository_20261102')
    combined = combined_audit.audit(root/'combined_foreground_acceptance_20261102')
    package = Path(ex.__file__).parents[2]
    sources = list(Path(ex.__file__).parent.glob('*.py'))+[package/'calibration/foreground.py',
        Path(inspect.getfile(ex.fb.ADC_Signal_Generator)), package/'models/sar.py']
    sources += list((package/'spectrum').glob('*.py'))
    return dict(status='passed', foreground_rows=count, bitweight_replay=bits,
        combined_replay=combined, total_rows=count+bits['frozen_rows_replayed']+combined['frozen_rows_replayed'],
        max_independent_difference_db=max(maximum, bits['max_independent_fft_difference_db'], combined['max_independent_fft_difference_db']),
        frozen_models_unchanged=True, fitting_performed=False,
        current_source_sha256={p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        historical_input_sha256=input_hashes,
        policy='Historical hashes preserved; this is a separate replay with the extracted code, not a new training run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('choose a new output file')
    result = replay(args.source_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f"Passed: {result['total_rows']} rows; maximum difference {result['max_independent_difference_db']:.3g} dB")


if __name__ == '__main__':
    main()
