"""Frozen polynomial / MLP calibration with independent reference and test data.

Default: two plants x three seeds; 32 training and 8 validation records, each
8192 points. A fixed stride of 16 gives 16384 training / 4096 validation rows.
Validation selects MLP iterates only. All test records are held out in full.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
import inspect
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import (analyze_spectrum, apply_calibration, build_calibration_features,
                       fit_calibration_polynomial, fit_calibration_mlp)
try:
    from . import foreground_benchmark as fb
    from .foreground_metrics import metrics
except ImportError:
    import foreground_benchmark as fb
    from foreground_metrics import metrics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measure(signal, truth, bins):
    """Explicit single-tone SNDR/SFDR; multi-tone and chirp stay distinct."""
    measured = metrics(signal, truth, bins) if len(bins) else {
        'spectral_ratio_db': None, 'spur_ratio_db': None,
        'error_snr_db': float(10*np.log10(np.var(truth)/max(np.mean((signal-truth)**2), 1e-300)))}
    single = len(bins) == 1
    difference, matching = None, None
    if single:
        matching = int(np.argmax(abs(np.fft.rfft(signal-signal.mean()))[1:]))+1 == bins[0]
        if matching:
            public = analyze_spectrum(signal, win_type='boxcar', side_bin=0,
                                      max_scale_range=None, create_plot=False, show_label=False)
            difference = max(abs(public['sndr_dbc']-measured['spectral_ratio_db']),
                             abs(public['sfdr_dbc']-measured['spur_ratio_db']))
            if difference > 1e-6:
                raise AssertionError(f'FFT metric disagreement: {difference}')
    return {**measured, 'sndr_db': measured['spectral_ratio_db'] if single else None,
            'sfdr_db': measured['spur_ratio_db'] if single else None,
            'public_fft_matching_fundamental': matching, 'public_fft_max_difference_db': difference}


def test_conditions(size):
    for amplitude in (.1, .5, .85):
        for f in (.031, .113, .239, .371, .467):
            yield {'name': f'sine_a{amplitude}_f{f}', 'group': 'sine',
                   'bins': [int(round(f*size)) | 1], 'amplitudes': [amplitude], 'phases': [.29]}
    yield {'name': 'two_tone', 'group': 'multitone', 'bins': [int(.083*size) | 1, int(.217*size) | 1],
           'amplitudes': [.45, .35], 'phases': [.3, 1.1]}
    yield {'name': 'three_tone', 'group': 'multitone', 'bins': [int(f*size) | 1 for f in (.057, .183, .327)],
           'amplitudes': [.3, .25, .2], 'phases': [.2, 1.3, 2.1]}
    yield {'name': 'chirp', 'group': 'chirp', 'bins': [], 'amplitudes': [], 'phases': [], 'chirp': True}
    yield {'name': 'outside_amplitude', 'group': 'overload_stress', 'bins': [int(.113*size) | 1],
           'amplitudes': [.98], 'phases': [.29]}
    yield {'name': 'changed_device', 'group': 'drift_stress', 'bins': [int(.239*size) | 1],
           'amplitudes': [.85], 'phases': [.29], 'drift': True}


def run_case(args, plant_kind, seed):
    config = fb.FrontendConfig(memory=0. if plant_kind == 'static' else .04)
    conditions = list(test_conditions(args.size))
    excluded = sorted({k for condition in conditions for k in condition['bins']})
    training = fb.dataset(config, records=args.records, size=args.size, seed=seed+10000,
                          stride=args.stride, exclude_bins=excluded)
    validation = fb.dataset(config, records=args.validation_records, size=args.size, seed=seed+20000,
                            stride=args.stride, exclude_bins=excluded+[m['tone_bin'] for m in training['manifest']])
    directory = args.output/f'{plant_kind}_seed{seed}'
    directory.mkdir()
    write_json(directory/'data_manifest.json', {'training': training['manifest'], 'validation': validation['manifest'],
               'tests': conditions, 'plant': training['plant'], 'reference': 'independent 16-bit source monitor, known-frequency sine projection; amplitude/phase/DC fitted'})
    x, y = training['features'], training['reference']
    models, diagnostics = {}, {}
    for name, degree, columns in [('affine_static', 1, 1), ('affine_history', 1, 4),
                                  ('poly3_static', 3, 1), ('poly5_static', 5, 1),
                                  ('poly3_history', 3, 4), ('poly5_history', 5, 4),
                                  ('poly5_output_fit', 5, 4)]:
        target = training['output_fit_reference'] if name == 'poly5_output_fit' else y
        fitted = fit_calibration_polynomial(x[:, :columns], target, degree=degree, ridge=1e-10)
        models[name] = fitted['model']
        diagnostics[name] = fitted['diagnostics']
    start = time.perf_counter()
    fitted = fit_calibration_mlp(x, y, validation_features=validation['features'],
                                validation_reference=validation['reference'], seed=seed,
                                hidden_units=args.hidden_units, max_iterations=args.iterations)
    models['mlp_history'] = fitted['model']
    diagnostics['mlp_history'] = {**fitted['diagnostics'], 'wall_seconds': time.perf_counter()-start}
    write_json(directory/'models.json', models)
    write_json(directory/'training_diagnostics.json', diagnostics)
    write_csv(directory/'mlp_history.csv', fitted['history'])
    models = json.loads((directory/'models.json').read_text())
    rows = []
    for ci, condition in enumerate(conditions):
        test_config = replace(config, gain=config.gain*1.02, cubic=config.cubic*1.5) if condition.get('drift') else config
        c = fb.capture(test_config, size=args.size, seed=seed+30000+ci,
                       frequencies=[k/args.size for k in condition['bins']], amplitudes=condition['amplitudes'],
                       phases=condition['phases'], chirp=condition.get('chirp', False))
        built = build_calibration_features(c['raw'], history=3)
        xt, indices = built['features'], built['sample_indices']
        truth = c['truth'][indices]  # Evaluator only; never passed to fitting.
        predictions = {'raw': {'calibrated_signal': xt[:, 0], 'outside_training_range': np.zeros(len(xt), bool)}}
        for name, model in models.items():
            predictions[name] = apply_calibration(xt[:, :model['feature_count']], model)
        for name, applied in predictions.items():
            prediction = applied['calibrated_signal']
            measured = measure(prediction, truth, condition['bins'])
            rows.append({'plant': plant_kind, 'seed': seed, 'condition': condition['name'], 'group': condition['group'],
                         'method': name, **measured, 'rmse': float(np.sqrt(np.mean((prediction-truth)**2))),
                         'outside_range_fraction': float(np.mean(applied['outside_training_range'])),
                         'overloads': c['overloads'], 'rails': c['rails']})
    write_csv(directory/'test_metrics.csv', rows)
    print(f'{plant_kind}/seed{seed}: MLP {diagnostics["mlp_history"]["wall_seconds"]:.1f}s, '
          f'validation RMSE={diagnostics["mlp_history"]["validation_rmse"]:.3g}', flush=True)
    return rows


def summarize(rows):
    result = []
    for plant in sorted({r['plant'] for r in rows}):
        for group in sorted({r['group'] for r in rows}):
            for method in sorted({r['method'] for r in rows}):
                selected = [r for r in rows if (r['plant'], r['group'], r['method']) == (plant, group, method)]
                row = {'plant': plant, 'group': group, 'method': method}
                for metric in ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db', 'sndr_db', 'sfdr_db'):
                    # First average conditions within seed, then summarize independent seeds.
                    values = [np.mean([r[metric] for r in selected if r['seed'] == seed and r[metric] is not None])
                              for seed in sorted({r['seed'] for r in selected})] if selected[0][metric] is not None else []
                    row[metric+'_mean'] = float(np.mean(values)) if values else None
                    row[metric+'_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else (0. if values else None)
                result.append(row)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'output'/'foreground_nonlinear')
    parser.add_argument('--size', type=int, default=8192)
    parser.add_argument('--records', type=int, default=32)
    parser.add_argument('--validation-records', type=int, default=8)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--hidden-units', type=int, default=16)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--plants', nargs='+', choices=['static', 'memory'], default=['static', 'memory'])
    args = parser.parse_args(argv)
    if args.size < 1024 or args.size % 2 or min(args.records, args.validation_records, args.stride, args.iterations, args.hidden_units) < 1 or args.stride > args.size:
        parser.error('Use even size >=1024, positive counts, and stride <= size')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.plants)) != len(args.plants):
        parser.error('Seeds and plants must not repeat')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    source = [Path(__file__), Path(fb.__file__), Path(inspect.getfile(fb.ADC_Signal_Generator)),
              Path(inspect.getfile(fit_calibration_mlp)), Path(inspect.getfile(metrics))]
    snapshots = args.output/'source_snapshot'
    snapshots.mkdir()
    for path in source:
        (snapshots/path.name).write_bytes(path.read_bytes())
    write_json(args.output/'configuration.json', {**vars(args), 'output': str(args.output.resolve()),
               'history': 3, 'plant_model': fb.MODEL_ID,
               'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
               'selection': 'MLP iterate selected by independent monitor-reference validation MSE; test data unused',
               'training_rows_per_case': args.records*len(range(0, args.size, args.stride))})
    rows = []
    for plant in args.plants:
        for seed in args.seeds:
            rows += run_case(args, plant, seed)
    write_csv(args.output/'test_metrics.csv', rows)
    summary = summarize(rows)
    write_csv(args.output/'summary.csv', summary)
    write_json(args.output/'summary.json', summary)
    methods = ['raw', 'affine_history', 'poly5_static', 'poly5_history', 'mlp_history', 'poly5_output_fit']
    fig, axes = plt.subplots(1, len(args.plants), figsize=(7*len(args.plants), 5), squeeze=False, layout='constrained')
    for ax, plant in zip(axes[0], args.plants):
        for offset, metric, label in [(-.2, 'sndr_db', 'SNDR'), (.2, 'sfdr_db', 'SFDR')]:
            data = [next(r for r in summary if (r['plant'], r['group'], r['method']) == (plant, 'sine', m)) for m in methods]
            ax.bar(np.arange(len(methods))+offset, [r[metric+'_mean'] for r in data], width=.4,
                   yerr=[r[metric+'_sd'] for r in data], label=label, capsize=3)
        ax.set_xticks(np.arange(len(methods)), methods, rotation=30, ha='right')
        ax.set_ylabel('dB; condition mean, then seed mean +/- sample SD')
        ax.set_title(plant+' plant | held-out sine tests')
        ax.grid(axis='y', alpha=.25)
        ax.legend()
    fig.savefig(args.output/'foreground_comparison.png', dpi=160)
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
