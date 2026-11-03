"""Replay combined calibration; independently check FFT ratios and data contracts."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from . import exp_cal11_combined_foreground as ex
    from .audit_foreground_bitweights import independent_metrics
except ImportError:
    import exp_cal11_combined_foreground as ex
    from audit_foreground_bitweights import independent_metrics


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def audit(directory):
    config = ex.read_json(directory/'configuration.json')
    assert config['status'] == 'complete'
    for name, digest in config['source_sha256'].items():
        assert hashlib.sha256((directory/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    saved = read_csv(directory/'test_metrics.csv')
    lookup = {(r['plant'], int(r['seed']), r['condition'], r['method']): r for r in saved}
    assert len(lookup) == len(saved) == config['metric_rows']
    recomputed, maximum, single_checks, captures = [], 0., 0, 0
    hashes, optimizer = {}, {}
    for architecture in config['architectures']:
        for seed in config['seeds']:
            case = directory/f'{architecture}_seed{seed}'
            manifest = ex.read_json(case/'manifest.json')
            models = ex.read_json(case/'models.json')
            diagnostics = ex.read_json(case/'diagnostics.json')
            sets = [{k for r in manifest[part] for k in r['tone_bins']} for part in ('training', 'validation')]
            sets += [{k for r in manifest['tests'] for k in r['bins']}, set(manifest['bypass_calibration']['tone_bins'])]
            for i, one in enumerate(sets):
                assert all(not one & other for other in sets[i+1:])
            train_seeds = {r['capture_seed'] for r in manifest['training']}
            valid_seeds = {r['capture_seed'] for r in manifest['validation']}
            assert not train_seeds & valid_seeds
            assert manifest['bypass_calibration']['capture_seed'] not in train_seeds | valid_seeds
            nominal = np.array(manifest['device']['nominal_weights'])
            actual = np.array(manifest['device']['actual_weights'])
            assert actual[0] == nominal[0]*1.01 and np.array_equal(actual[1:], nominal[1:])
            assert manifest['device']['k2'] != 0 and manifest['device']['k3'] != 0
            for name in ('direct_poly', 'direct_mlp', 'cascade_poly', 'cascade_mlp'):
                assert models[name]['feature_count'] == 1
                assert diagnostics[name]['samples'] == config['training_rows_per_case']
            for name in ('direct_mlp', 'cascade_mlp'):
                history = read_csv(case/(name+'_history.csv'))
                diag = diagnostics[name]
                assert float(history[diag['selected_iteration']]['validation_mse']) == min(float(r['validation_mse']) for r in history)
                assert diag['iterations'] <= config['iterations']
                optimizer[f'{case.name}/{name}'] = diag
            before = json.dumps(models, sort_keys=True)
            for i, condition in enumerate(manifest['tests']):
                observed = ex.capture(manifest['device'], ex.test_record(condition, seed, i), manifest['size'])
                assert observed['bypass'] is False and observed['overloads'] == 0
                outputs = ex.predictions(observed, models, manifest['device'])
                assert set(ex.PRIMARY).issubset(outputs)
                for method, signal in outputs.items():
                    row = lookup[(architecture, seed, condition['name'], method)]
                    measured = independent_metrics(signal, observed['truth'], condition['bins'])
                    for key, value in measured.items():
                        difference = abs(value-float(row[key]))
                        maximum = max(maximum, difference)
                        assert difference < 1e-7, (case.name, method, key, difference)
                    if len(condition['bins']) == 1:
                        assert float(row['sndr_db']) == float(row['spectral_ratio_db'])
                        assert float(row['sfdr_db']) == float(row['spur_ratio_db'])
                        if row['public_fft_matching_fundamental'] == 'True':
                            assert float(row['public_fft_max_difference_db']) < 1e-6
                            single_checks += 1
                    else:
                        assert row['sndr_db'] == row['sfdr_db'] == ''
                    recomputed.append({**row, 'seed': seed, **measured,
                        'sndr_db': float(row['sndr_db']) if row['sndr_db'] else None,
                        'sfdr_db': float(row['sfdr_db']) if row['sfdr_db'] else None})
                captures += 1
            assert json.dumps(models, sort_keys=True) == before
            for path in case.iterdir():
                if path.is_file():
                    hashes[f'{case.name}/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(recomputed) == len(saved)
    summary = ex.prior.summarize(recomputed)
    published = ex.read_json(directory/'summary.json')
    assert len(summary) == len(published)
    for a, b in zip(summary, published):
        for key in a:
            assert abs(a[key]-b[key]) < 1e-7 if isinstance(a[key], float) else a[key] == b[key]
    return dict(status='passed', cases=len(config['architectures'])*len(config['seeds']),
        captures=captures, frozen_rows_replayed=len(recomputed), public_single_tone_checks=single_checks,
        max_independent_fft_difference_db=maximum, both_errors_in_all_tests=True,
        overloads=0, frozen_models_unchanged=True, aggregate_recomputed=True,
        optimizer_diagnostics=optimizer, case_sha256=hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('choose a new output file')
    result = audit(args.directory)
    ex.write_json(args.output, result)
    print(f"Passed: {result['cases']} cases, {result['frozen_rows_replayed']} rows; "
          f"FFT difference {result['max_independent_fft_difference_db']:.3g} dB")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
