"""Fixed-model control for single-tone versus mixed-excitation training.

Consumes a completed exp_cal03 run. Refit the SAME polynomial structures using
the same capture/row budget, with half single-tone and half two-tone captures.
By default the saved single-tone-trained MLP is only a reference. --retrain-mlp
also trains it on the mixed captures with separate mixed validation captures,
retaining the baseline network size, optimizer budget and validation row budget.
Test conditions remain identical and do not select degrees or regularization.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import (build_calibration_features, fit_calibration_polynomial,
                       fit_calibration_mlp, apply_calibration)
try:
    from . import exp_cal03_foreground_nonlinear as ex
except ImportError:
    import exp_cal03_foreground_nonlinear as ex


POLYNOMIALS = [('affine_history', 1, 4), ('poly5_static', 5, 1),
               ('poly3_history', 3, 4), ('poly5_history', 5, 4)]


def mixed_dataset(config, *, records, size, stride, seed, exclude_bins):
    rng = np.random.default_rng(seed)
    candidates = np.setdiff1d(np.arange(max(3, int(.02*size)) | 1, int(.45*size), 2), exclude_bins)
    count = records+records//2
    if count > len(candidates):
        raise ValueError('Not enough distinct tone bins for mixed-excitation records')
    bins = rng.choice(candidates, count, replace=False)
    features, references, manifest, cursor = [], [], [], 0
    for record in range(records):
        tones = 1+record % 2
        record_bins = bins[cursor:cursor+tones]
        cursor += tones
        peak_budget = rng.uniform(.05, .90)
        split = rng.uniform(.3, .7) if tones == 2 else 1.
        amplitudes = [peak_budget*split, peak_budget*(1-split)] if tones == 2 else [peak_budget]
        phases = rng.uniform(0, 2*np.pi, tones)
        capture_seed = int(rng.integers(0, 2**31-1))
        capture = ex.fb.capture(config, size=size, seed=capture_seed, frequencies=record_bins/size,
                                amplitudes=amplitudes, phases=phases)
        built = build_calibration_features(capture['raw'], history=3)
        features.append(built['features'][::stride])
        references.append(capture['reference'][built['sample_indices'][::stride]])
        manifest.append({'record': record, 'tone_bins': record_bins.tolist(), 'amplitudes': amplitudes,
                         'phases': phases.tolist(), 'capture_seed': capture_seed, 'overloads': capture['overloads']})
    return np.vstack(features), np.concatenate(references), manifest


def aggregate(rows):
    summaries = []
    keys = sorted({(r['plant'], r['group'], r['method'], r['training_excitation']) for r in rows})
    for key in keys:
        selected = [r for r in rows if tuple(r[k] for k in ('plant', 'group', 'method', 'training_excitation')) == key]
        record = dict(zip(('plant', 'group', 'method', 'training_excitation'), key))
        for metric in ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db', 'sndr_db', 'sfdr_db'):
            values = [np.mean([r[metric] for r in selected if r['seed'] == seed])
                      for seed in sorted({r['seed'] for r in selected})] if selected[0][metric] is not None else []
            record[metric+'_mean'] = float(np.mean(values)) if values else None
            record[metric+'_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else (0. if values else None)
        summaries.append(record)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--retrain-mlp', action='store_true',
                        help='Train a mixed-excitation MLP with independent mixed validation records')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'output'/'foreground_excitation')
    args = parser.parse_args(argv)
    cfg = json.loads((args.baseline/'configuration.json').read_text())
    if cfg.get('plant_model') != ex.fb.MODEL_ID:
        parser.error('Legacy/custom baseline: rerun exp_cal03 with the repository model')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(ex.__file__), Path(ex.fb.__file__),
               Path(inspect.getfile(fit_calibration_polynomial)), Path(inspect.getfile(ex.metrics)),
               Path(inspect.getfile(ex.fb.ADC_Signal_Generator))]
    snapshots = args.output/'source_snapshot'
    snapshots.mkdir()
    for source in sources:
        (snapshots/source.name).write_bytes(source.read_bytes())
    ex.write_json(args.output/'configuration.json', {
        'baseline': str(args.baseline.resolve()), 'records': cfg['records'], 'size': cfg['size'], 'stride': cfg['stride'],
        'plants': cfg['plants'], 'seeds': cfg['seeds'], 'ridge': 1e-10, 'plant_model': ex.fb.MODEL_ID,
        'retrain_mlp': args.retrain_mlp, 'validation_records': cfg['validation_records'],
        'hidden_units': cfg['hidden_units'], 'iterations': cfg['iterations'], 'mlp_l2': 1e-6,
        'baseline_configuration_sha256': hashlib.sha256((args.baseline/'configuration.json').read_bytes()).hexdigest(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'control': 'same model sizes and capture/row budgets; new half single/half dual tone records; '
                   + ('MLP retrained with independent mixed validation; same optimizer budget'
                      if args.retrain_mlp else 'MLP baseline is NOT retrained')})
    rows = []
    for plant in cfg['plants']:
        for seed in cfg['seeds']:
            baseline = args.baseline/f'{plant}_seed{seed}'
            manifest = json.loads((baseline/'data_manifest.json').read_text())
            device = ex.fb.FrontendConfig.from_manifest(manifest['plant'])
            excluded = sorted({k for c in manifest['tests'] for k in c['bins']} |
                              {r['tone_bin'] for r in manifest['validation']})
            x, target, training_manifest = mixed_dataset(device, records=cfg['records'], size=cfg['size'],
                stride=cfg['stride'], seed=seed+40000, exclude_bins=excluded)
            assert not sum(r['overloads'] for r in training_manifest)
            directory = args.output/f'{plant}_seed{seed}'
            directory.mkdir()
            ex.write_json(directory/'training_manifest.json', training_manifest)
            models, diagnostics = {}, {}
            for name, degree, columns in POLYNOMIALS:
                fitted = fit_calibration_polynomial(x[:, :columns], target, degree=degree, ridge=1e-10)
                models[name], diagnostics[name] = fitted['model'], fitted['diagnostics']
            if args.retrain_mlp:
                train_bins = {k for r in training_manifest for k in r['tone_bins']}
                xv, yv, validation_manifest = mixed_dataset(device, records=cfg['validation_records'],
                    size=cfg['size'], stride=cfg['stride'], seed=seed+50000,
                    exclude_bins=sorted(set(excluded) | train_bins))
                assert not sum(r['overloads'] for r in validation_manifest)
                ex.write_json(directory/'validation_manifest.json', validation_manifest)
                fitted = fit_calibration_mlp(x, target, validation_features=xv, validation_reference=yv,
                    hidden_units=cfg['hidden_units'], max_iterations=cfg['iterations'], seed=seed)
                models['mlp_history'], diagnostics['mlp_history'] = fitted['model'], fitted['diagnostics']
                ex.write_csv(directory/'mlp_history.csv', fitted['history'])
            ex.write_json(directory/'models.json', models)
            ex.write_json(directory/'training_diagnostics.json', diagnostics)
            # Predictions use JSON-loaded, frozen models; reference labels are not available here.
            models = json.loads((directory/'models.json').read_text())
            for ci, condition in enumerate(manifest['tests']):
                test_device = replace(device, gain=device.gain*1.02, cubic=device.cubic*1.5) if condition.get('drift') else device
                capture = ex.fb.capture(test_device, size=cfg['size'], seed=seed+30000+ci,
                    frequencies=[k/cfg['size'] for k in condition['bins']], amplitudes=condition['amplitudes'],
                    phases=condition['phases'], chirp=condition.get('chirp', False))
                built = build_calibration_features(capture['raw'], history=3)
                truth = capture['truth'][built['sample_indices']]
                for name, model in models.items():
                    applied = apply_calibration(built['features'][:, :model['feature_count']], model)
                    prediction = applied['calibrated_signal']
                    measured = ex.measure(prediction, truth, condition['bins'])
                    rows.append({'plant': plant, 'seed': seed, 'condition': condition['name'], 'group': condition['group'],
                                 'method': name, 'training_excitation': 'mixed', **measured})
            print(f'{plant}/seed{seed}: mixed-reference fits complete ({len(x)} rows, MLP retrained={args.retrain_mlp})', flush=True)
    import csv
    with (args.baseline/'test_metrics.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            if r['method'] in [m[0] for m in POLYNOMIALS]+['mlp_history', 'raw']:
                rows.append({'plant': r['plant'], 'seed': int(r['seed']), 'condition': r['condition'], 'group': r['group'],
                             'method': r['method'], 'training_excitation': 'none' if r['method'] == 'raw' else 'single',
                             **{k: float(r[k]) if r[k] else None for k in ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db', 'sndr_db', 'sfdr_db')},
                             'public_fft_matching_fundamental': {'True': True, 'False': False, '': None}[r['public_fft_matching_fundamental']],
                             'public_fft_max_difference_db': float(r['public_fft_max_difference_db']) if r['public_fft_max_difference_db'] else None})
    ex.write_csv(args.output/'test_metrics.csv', rows)
    summaries = aggregate(rows)
    ex.write_csv(args.output/'summary.csv', summaries)
    ex.write_json(args.output/'summary.json', summaries)
    fig, axes = plt.subplots(1, len(cfg['plants']), figsize=(7*len(cfg['plants']), 5), squeeze=False, layout='constrained')
    for ax, plant in zip(axes[0], cfg['plants']):
        keys = [('poly5_history', 'single'), ('poly5_history', 'mixed'), ('mlp_history', 'single')]
        labels = ['Poly5: single-tone train', 'Poly5: mixed-tone train', 'MLP: single-tone train']
        if args.retrain_mlp:
            keys.append(('mlp_history', 'mixed'))
            labels.append('MLP: mixed-tone train')
        for offset, group in [(-.2, 'sine'), (.2, 'multitone')]:
            selected = [next(r for r in summaries if (r['plant'], r['group'], r['method'], r['training_excitation']) == (plant, group, m, excitation)) for m, excitation in keys]
            ax.bar(np.arange(len(keys))+offset, [r['error_snr_db_mean'] for r in selected], width=.4,
                   yerr=[r['error_snr_db_sd'] for r in selected], capsize=3, label=group+' test')
        ax.set_xticks(np.arange(len(keys)), labels, rotation=20, ha='right')
        ax.set_ylabel('Absolute error SNR (dB); seed mean +/- SD')
        ax.set_title(plant+' plant: fixed model, changed excitation')
        ax.grid(axis='y', alpha=.25)
        ax.legend()
    fig.savefig(args.output/'excitation_comparison.png', dpi=160)
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
