"""Evaluate frozen foreground models on fresh random two/three/five-tone records.

No fitting or model selection occurs here. Frequencies exclude all saved training,
validation and original test bins for each case. Compare identical captures across
the saved single-tone and mixed-excitation models, including the static baseline.
This is a behavioral transfer check, not a measured-ADC or arbitrary-waveform claim.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from adctoolbox import apply_calibration, build_calibration_features
try:
    from . import exp_cal03_foreground_nonlinear as ex
    from .exp_cal05_foreground_excitation import aggregate
except ImportError:
    import exp_cal03_foreground_nonlinear as ex
    from exp_cal05_foreground_excitation import aggregate


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--mixed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records-per-group', type=int, default=4)
    args = parser.parse_args(argv)
    cfg = read_json(args.baseline/'configuration.json')
    mix_cfg = read_json(args.mixed/'configuration.json')
    if cfg.get('plant_model') != ex.fb.MODEL_ID or mix_cfg.get('plant_model') != ex.fb.MODEL_ID:
        parser.error('Legacy/custom models: rerun exp_cal03 and exp_cal05 with the repository model')
    if args.records_per_group < 1:
        parser.error('records-per-group must be positive')
    if not mix_cfg.get('retrain_mlp'):
        parser.error('mixed run must include --retrain-mlp')
    if hashlib.sha256((args.baseline/'configuration.json').read_bytes()).hexdigest() != mix_cfg['baseline_configuration_sha256']:
        parser.error('mixed run does not refer to this baseline configuration')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(ex.__file__), Path(ex.fb.__file__),
               Path(inspect.getfile(apply_calibration)), Path(inspect.getfile(aggregate)),
               Path(inspect.getfile(ex.metrics)), Path(inspect.getfile(ex.fb.ADC_Signal_Generator))]
    snapshots = args.output/'source_snapshot'
    snapshots.mkdir()
    for source in sources:
        (snapshots/source.name).write_bytes(source.read_bytes())
    config = {'baseline': str(args.baseline.resolve()), 'mixed': str(args.mixed.resolve()),
              'records_per_group': args.records_per_group, 'size': cfg['size'],
              'plants': cfg['plants'], 'seeds': cfg['seeds'], 'plant_model': ex.fb.MODEL_ID,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'input_sha256': {}, 'policy': 'frozen models; new random 2/3/5 tones; no tuning or selection'}
    rows, manifests = [], []
    for plant in cfg['plants']:
        for seed in cfg['seeds']:
            case = f'{plant}_seed{seed}'
            original = read_json(args.baseline/case/'data_manifest.json')
            training = read_json(args.mixed/case/'training_manifest.json')
            validation = read_json(args.mixed/case/'validation_manifest.json')
            excluded = {r['tone_bin'] for part in ('training', 'validation') for r in original[part]}
            excluded |= {k for r in original['tests'] for k in r['bins']}
            excluded |= {k for r in training+validation for k in r['tone_bins']}
            candidates = np.setdiff1d(np.arange(max(3, int(.02*cfg['size'])) | 1, int(.45*cfg['size']), 2), sorted(excluded))
            if len(candidates) < 5:
                raise ValueError('Not enough unused tone bins for the transfer check')
            model_sets = {}
            for label, root in (('single', args.baseline), ('mixed', args.mixed)):
                path = root/case/'models.json'
                config['input_sha256'][f'{label}/{case}/models.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
                models = read_json(path)
                model_sets[label] = {name: models[name] for name in ('poly5_static', 'poly5_history', 'mlp_history')}
            rng = np.random.default_rng(500000+seed)
            device = ex.fb.FrontendConfig.from_manifest(original['plant'])
            for tones in (2, 3, 5):
                for record in range(args.records_per_group):
                    bins = rng.choice(candidates, tones, replace=False)
                    budget = float(rng.uniform(.2, .85))
                    amplitudes = budget*rng.dirichlet(np.ones(tones))
                    phases = rng.uniform(0, 2*np.pi, tones)
                    capture_seed = int(rng.integers(0, 2**31-1))
                    capture = ex.fb.capture(device, size=cfg['size'], seed=capture_seed,
                        frequencies=bins/cfg['size'], amplitudes=amplitudes, phases=phases)
                    built = build_calibration_features(capture['raw'], history=3)
                    truth = capture['truth'][built['sample_indices']]  # Evaluator only.
                    name = f'random_{tones}_tone_{record}'
                    manifests.append({'plant': plant, 'seed': seed, 'condition': name,
                        'tone_bins': bins.tolist(), 'amplitudes': amplitudes.tolist(),
                        'phases': phases.tolist(), 'capture_seed': capture_seed,
                        'overloads': capture['overloads'], 'excluded_bins': sorted(excluded)})
                    predictions = [('none', 'raw', {'calibrated_signal': built['features'][:, 0],
                                                    'outside_training_range': np.zeros(len(truth), bool)})]
                    for excitation, models in model_sets.items():
                        for method, model in models.items():
                            predictions.append((excitation, method, apply_calibration(
                                built['features'][:, :model['feature_count']], model)))
                    for excitation, method, applied in predictions:
                        rows.append({'plant': plant, 'seed': seed, 'group': f'{tones}_tone',
                            'condition': name, 'method': method, 'training_excitation': excitation,
                            **ex.measure(applied['calibrated_signal'], truth, bins.tolist()),
                            'outside_range_fraction': float(applied['outside_training_range'].mean()),
                            'overloads': capture['overloads']})
    ex.write_json(args.output/'configuration.json', config)
    ex.write_json(args.output/'test_manifest.json', manifests)
    ex.write_csv(args.output/'test_metrics.csv', rows)
    summary = aggregate(rows)
    ex.write_csv(args.output/'summary.csv', summary)
    ex.write_json(args.output/'summary.json', summary)
    print(f'Fresh frozen transfer check: {len(manifests)} captures, {len(rows)} metric rows', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
