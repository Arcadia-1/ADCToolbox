"""Replay current repository-model foreground runs with an independent full FFT.

Checks single-tone, mixed-training and fresh multi-tone runs together. Does not
fit, choose models, rewrite results, or support legacy custom-plant artifacts.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from adctoolbox import apply_calibration, build_calibration_features
try:
    from . import audit_foreground_acceptance as acceptance
    from . import audit_foreground_transfer as transfer
    from .audit_foreground_bitweights import independent_metrics
except ImportError:
    import audit_foreground_acceptance as acceptance
    import audit_foreground_transfer as transfer
    from audit_foreground_bitweights import independent_metrics

ex = transfer.ex
read_json, read_csv = acceptance.read_json, acceptance.read_csv


def audit(baseline, mixed, fresh):
    configs = [read_json(root/'configuration.json') for root in (baseline, mixed, fresh)]
    assert all(c.get('plant_model') == ex.fb.MODEL_ID for c in configs)
    checked = {'baseline': acceptance.audit_foreground(baseline),
               'mixed': acceptance.audit_excitation(mixed, baseline)}
    cfg = configs[0]
    assert configs[1]['retrain_mlp']
    assert Path(configs[2]['baseline']).resolve() == baseline.resolve()
    assert Path(configs[2]['mixed']).resolve() == mixed.resolve()
    source_paths = [Path(transfer.__file__), Path(ex.__file__), Path(ex.fb.__file__),
                    Path(inspect.getfile(apply_calibration)), Path(inspect.getfile(transfer.aggregate)),
                    Path(inspect.getfile(ex.metrics)), Path(inspect.getfile(ex.fb.ADC_Signal_Generator))]
    source_lookup = {p.name: p for p in source_paths}
    for name, digest in configs[2]['source_sha256'].items():
        assert hashlib.sha256((fresh/'source_snapshot'/name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256(source_lookup[name].read_bytes()).hexdigest() == digest
    for name, digest in configs[2]['input_sha256'].items():
        label, relative = name.split('/', 1)
        assert hashlib.sha256(((baseline if label == 'single' else mixed)/relative).read_bytes()).hexdigest() == digest

    rows = [read_csv(root/'test_metrics.csv') for root in (baseline, mixed, fresh)]
    keys = ('plant', 'seed', 'condition', 'method', 'training_excitation')
    lookups = []
    for table in rows:
        for row in table:
            row.setdefault('training_excitation', 'none' if row['method'] == 'raw' else 'single')
        lookup = {tuple(row[k] for k in keys): row for row in table}
        assert len(lookup) == len(table)
        lookups.append(lookup)
    consumed = [set(), set(), set()]
    max_difference, public_checks, mismatched_peaks = 0., 0, 0
    fresh_manifests = read_json(fresh/'test_manifest.json')
    assert len(fresh_manifests) == len(cfg['plants'])*len(cfg['seeds'])*3*configs[2]['records_per_group']
    file_hashes, optimizer = {}, []

    def verify(table_index, plant, seed, condition, method, excitation, signal, truth, bins):
        nonlocal max_difference, public_checks, mismatched_peaks
        key = (plant, str(seed), condition, method, excitation)
        row = lookups[table_index][key]
        measured = independent_metrics(signal, truth, bins) if bins else {
            'error_snr_db': float(10*np.log10(np.var(truth)/max(np.mean((signal-truth)**2), 1e-300)))}
        for metric, value in measured.items():
            difference = abs(value-float(row[metric]))
            assert difference < 1e-7, (key, metric, difference)
            max_difference = max(max_difference, difference)
        if len(bins) == 1:
            assert float(row['sndr_db']) == float(row['spectral_ratio_db'])
            assert float(row['sfdr_db']) == float(row['spur_ratio_db'])
            public = ex.measure(signal, truth, bins)
            assert str(public['public_fft_matching_fundamental']) == row['public_fft_matching_fundamental']
            if public['public_fft_matching_fundamental']:
                assert abs(public['public_fft_max_difference_db']-float(row['public_fft_max_difference_db'])) < 1e-7
                public_checks += 1
            else:
                mismatched_peaks += 1
        else:
            assert row['sndr_db'] == row['sfdr_db'] == ''
        assert key not in consumed[table_index]
        consumed[table_index].add(key)

    for plant in cfg['plants']:
        for seed in cfg['seeds']:
            case = f'{plant}_seed{seed}'
            manifest = read_json(baseline/case/'data_manifest.json')
            device = ex.fb.FrontendConfig.from_manifest(manifest['plant'])
            model_sets = {label: read_json(root/case/'models.json') for label, root in [('single', baseline), ('mixed', mixed)]}
            before = json.dumps(model_sets, sort_keys=True)
            excluded = {r['tone_bin'] for part in ('training', 'validation') for r in manifest[part]}
            excluded |= {k for r in manifest['tests'] for k in r['bins']}
            for filename in ('training_manifest.json', 'validation_manifest.json'):
                excluded |= {k for r in read_json(mixed/case/filename) for k in r['tone_bins']}
            for label, root in [('single', baseline), ('mixed', mixed)]:
                diag = read_json(root/case/'training_diagnostics.json')['mlp_history']
                optimizer.append(dict(case=case, excitation=label, **diag))
                for filename in ('models.json', 'training_diagnostics.json', 'mlp_history.csv'):
                    file_hashes[f'{label}/{case}/{filename}'] = hashlib.sha256((root/case/filename).read_bytes()).hexdigest()
            for ci, condition in enumerate(manifest['tests']):
                target = replace(device, gain=device.gain*1.02, cubic=device.cubic*1.5) if condition.get('drift') else device
                observed = ex.fb.capture(target, size=cfg['size'], seed=seed+30000+ci,
                    frequencies=[k/cfg['size'] for k in condition['bins']], amplitudes=condition['amplitudes'],
                    phases=condition['phases'], chirp=condition.get('chirp', False))
                if condition['group'] in ('sine', 'multitone', 'chirp'):
                    assert observed['overloads'] == 0
                built = build_calibration_features(observed['raw'], history=3)
                truth = observed['truth'][built['sample_indices']]
                signals = [('none', 'raw', built['features'][:, 0])]
                for label, models in model_sets.items():
                    for method, model in models.items():
                        signal = apply_calibration(built['features'][:, :model['feature_count']], model)['calibrated_signal']
                        signals.append((label, method, signal))
                for label, method, signal in signals:
                    args = (plant, seed, condition['name'], method, label, signal, truth, condition['bins'])
                    if label in ('none', 'single'):
                        verify(0, *args)
                    if label in ('none', 'mixed') or method in ('affine_history', 'poly5_static', 'poly3_history', 'poly5_history', 'mlp_history'):
                        verify(1, *args)
            for record in [r for r in fresh_manifests if (r['plant'], r['seed']) == (plant, seed)]:
                assert set(record['excluded_bins']) == excluded
                assert not set(record['tone_bins']) & excluded
                observed = ex.fb.capture(device, size=cfg['size'], seed=record['capture_seed'],
                    frequencies=np.array(record['tone_bins'])/cfg['size'],
                    amplitudes=record['amplitudes'], phases=record['phases'])
                assert observed['overloads'] == record['overloads'] == 0
                built = build_calibration_features(observed['raw'], history=3)
                truth = observed['truth'][built['sample_indices']]
                verify(2, plant, seed, record['condition'], 'raw', 'none', built['features'][:, 0], truth, record['tone_bins'])
                for label, models in model_sets.items():
                    for method in ('poly5_static', 'poly5_history', 'mlp_history'):
                        model = models[method]
                        signal = apply_calibration(built['features'][:, :model['feature_count']], model)['calibrated_signal']
                        verify(2, plant, seed, record['condition'], method, label, signal, truth, record['tone_bins'])
            assert json.dumps(model_sets, sort_keys=True) == before
    assert [len(c) for c in consumed] == [len(t) for t in rows]
    fresh_rows = rows[2]
    for row in fresh_rows:
        row['seed'] = int(row['seed'])
        for key in ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db', 'sndr_db', 'sfdr_db'):
            row[key] = float(row[key]) if row[key] else None
    rebuilt = transfer.aggregate(fresh_rows)
    saved = read_json(fresh/'summary.json')
    assert len(rebuilt) == len(saved)
    for actual, expected in zip(rebuilt, saved):
        for key, value in actual.items():
            assert abs(value-expected[key]) < 1e-7 if isinstance(value, float) else value == expected[key]
    return dict(status='passed', plant_model=ex.fb.MODEL_ID, inherited_checks=checked,
                rows_by_run=[len(t) for t in rows], rows_replayed=sum(len(t) for t in rows),
                fresh_captures=len(fresh_manifests), max_independent_fft_difference_db=max_difference,
                public_single_tone_checks=public_checks, auto_peak_mismatches=mismatched_peaks,
                frozen_models_unchanged=True, aggregate_recomputed=True, normal_capture_overloads=0,
                optimizer_diagnostics=optimizer, model_artifact_sha256=file_hashes,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'mixed', 'fresh', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('choose a new output file')
    result = audit(args.baseline, args.mixed, args.fresh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ex.write_json(args.output, result)
    print(f"Passed: {result['rows_replayed']} rows; independent FFT max difference {result['max_independent_fft_difference_db']:.3g} dB")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
