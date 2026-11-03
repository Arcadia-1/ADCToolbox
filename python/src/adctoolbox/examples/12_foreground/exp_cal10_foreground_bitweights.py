"""Frozen spectral comparison: bit weights, polynomials and MLP.

Optional saved smooth-front-end experiments are replayed without retraining their
polynomial/MLP models. New binary/redundant SAR experiments isolate MSB mismatch.
Supervised bit weights share reference rows with polynomial/MLP. The existing
sine-weight solver is a separate one-full-record calibration protocol.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from adctoolbox import (analyze_spectrum, apply_calibration, build_calibration_features,
                       calibrate_weight_sine, scale_calibration_output,
                       fit_calibration_polynomial, fit_calibration_mlp)
from adctoolbox.models import sar_convert, sar_ideal_weights, sar_reconstruct
try:
    from . import exp_cal03_foreground_nonlinear as ex
except ImportError:
    import exp_cal03_foreground_nonlinear as ex


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def decode_midrise(raw, bits=14):
    """Observable final-code binary digits, not extra analog information."""
    codes = np.rint((np.asarray(raw)+1)*2**(bits-1)-.5).astype(np.int64)
    if np.any((codes < 0) | (codes >= 2**bits)):
        raise ValueError('not an in-range midrise output')
    return ((codes[:, None] >> np.arange(bits-1, -1, -1)) & 1).astype(np.int8)


def sar_capture(device, *, size, seed, frequencies, amplitudes, phases):
    n = np.arange(-67, size)
    clean = sum(a*np.sin(2*np.pi*f*n+p) for f, a, p in zip(frequencies, amplitudes, phases))
    adc_seed, monitor_seed = np.random.SeedSequence(seed).spawn(2)
    sampled = clean+np.random.default_rng(adc_seed).normal(0, 1e-5, len(clean))
    decisions = sar_convert(sampled, np.array(device['actual_weights']), quant_range=(-1, 1))
    nominal = np.array(device['nominal_weights'])
    raw = sar_reconstruct(decisions, nominal, quant_range=(-1, 1))+nominal[-1]
    monitor = clean+np.random.default_rng(monitor_seed).normal(0, 2e-4, len(clean))
    monitor = (np.clip(np.floor((monitor+1)*32768), 0, 65535)+.5)/32768-1
    return {'raw': raw[64:], 'bits': decisions[64:], 'truth': clean[64:],
            'reference': ex.fb.tone_projection(monitor[64:], frequencies),
            'overloads': int((abs(sampled[67:]) >= 1).sum())}


def capture(device, record, size):
    kw = dict(size=size, seed=record['capture_seed'],
              frequencies=np.array(record['tone_bins'])/size,
              amplitudes=record['amplitudes'], phases=record['phases'])
    if device['kind'] == 'smooth':
        result = ex.fb.capture(ex.fb.FrontendConfig.from_manifest(device['config']), **kw)
        result['bits'] = decode_midrise(result['raw'], device['config']['bits'])
        return result
    return sar_capture(device, **kw)


def new_records(size, count, seed, excluded):
    rng = np.random.default_rng(seed)
    candidates = np.setdiff1d(np.arange(max(3, int(.02*size)) | 1, int(.45*size), 2), list(excluded))
    if count+count//2 > len(candidates):
        raise ValueError('too few unused tone bins')
    chosen = rng.choice(candidates, count+count//2, replace=False)
    records, cursor = [], 0
    for i in range(count):
        tones = 1+i % 2
        bins = chosen[cursor:cursor+tones]
        cursor += tones
        budget = rng.uniform(.05, .90)
        split = rng.uniform(.3, .7) if tones == 2 else 1.
        amps = [budget*split, budget*(1-split)] if tones == 2 else [budget]
        records.append(dict(tone_bins=bins.tolist(), amplitudes=amps,
                            phases=rng.uniform(0, 2*np.pi, tones).tolist(),
                            capture_seed=int(rng.integers(0, 2**31-1))))
    return records


def dataset(device, records, size, stride):
    features, bits, labels = [], [], []
    for record in records:
        observed = capture(device, record, size)
        if observed['overloads']:
            raise ValueError('training/validation overload')
        built = build_calibration_features(observed['raw'], history=3)
        index = built['sample_indices'][::stride]
        features.append(built['features'][::stride])
        bits.append(observed['bits'][index])
        labels.append(observed['reference'][index])
    return np.vstack(features), np.vstack(bits), np.concatenate(labels)


def fit_bit_models(device, records, size, bit_rows, reference):
    fitted = fit_calibration_polynomial(bit_rows, reference, degree=1, ridge=1e-10)
    # Fixed input-amplitude criterion, independent of calibration or test scores.
    chosen = max((r for r in records if len(r['tone_bins']) == 1), key=lambda r: r['amplitudes'][0])
    observed = capture(device, chosen, size)
    nominal = np.array(device['nominal_weights'])*2
    sine = calibrate_weight_sine(observed['bits'][3:], freq=chosen['tone_bins'][0]/size,
                                force_search=False, nominal_weights=nominal, harmonic_order=1)
    sine = scale_calibration_output(sine, target_weights=nominal)
    models = {'bits_reference': fitted['model'],
              'bits_sine': {'kind': 'bit_weights', 'weight': sine['weight'].tolist(),
                            'offset': -float(sine['offset'])}}
    model = fitted['model']
    weights = np.array(model['coefficients'][1:])/model['input_scale']
    diagnostics = {'bits_reference': {**fitted['diagnostics'], 'weight': weights.tolist(),
                    'offset': float(model['coefficients'][0]-weights@model['input_mean'])},
                   'bits_sine': {'samples': size, 'record': chosen,
                    'rank_patch_applied': bool(sine['rank_patch']['applied']),
                    'protocol': 'one largest-amplitude training sine, full record; nominal-sum scale; no external labels'}}
    return models, diagnostics


def predictions(observed, models, device):
    built = build_calibration_features(observed['raw'], history=3)
    x, index = built['features'], built['sample_indices']
    bits = observed['bits'][index]
    result = {'raw': x[:, 0]}
    for name, model in models.items():
        if name == 'bits_sine':
            result[name] = bits@model['weight']+model['offset']
        else:
            inputs = bits if name == 'bits_reference' else x[:, :model['feature_count']]
            result[name] = apply_calibration(inputs, model)['calibrated_signal']
    if device['kind'] == 'sar':
        actual = np.array(device['actual_weights'])
        result['known_weights_control'] = sar_reconstruct(bits, actual, quant_range=(-1, 1))+actual[-1]
    return result, observed['truth'][index]


def spectral_metrics(signal, truth, bins):
    result = ex.metrics(signal, truth, bins)
    single = len(bins) == 1
    # The public analyzer locates the largest line. A failed correction may have
    # a spur larger than the intended fundamental; retain that failure instead
    # of silently calling the spur the desired signal or aborting the experiment.
    strongest_bin = int(np.argmax(abs(np.fft.rfft(signal-signal.mean()))[1:]))+1
    matching_fundamental = single and strongest_bin == bins[0]
    difference = None
    if matching_fundamental:
        public = analyze_spectrum(signal, win_type='boxcar', side_bin=0,
                                  max_scale_range=None, create_plot=False, show_label=False)
        difference = max(abs(public['sndr_dbc']-result['spectral_ratio_db']),
                         abs(public['sfdr_dbc']-result['spur_ratio_db']))
        if difference > 1e-6:
            raise AssertionError(f'FFT metric disagreement: {difference}')
    return {**result, 'sndr_db': result['spectral_ratio_db'] if single else None,
            'sfdr_db': result['spur_ratio_db'] if single else None,
            'public_fft_matching_fundamental': matching_fundamental if single else None,
            'public_fft_max_difference_db': difference}


def evaluate(device, models, conditions, size, seed, plant, directory):
    rows = []
    for i, condition in enumerate(conditions):
        record = dict(tone_bins=condition['bins'], amplitudes=condition['amplitudes'],
                      phases=condition['phases'], capture_seed=seed+30000+i)
        observed = capture(device, record, size)
        outputs, truth = predictions(observed, models, device)
        for method, signal in outputs.items():
            rows.append(dict(plant=plant, seed=seed, condition=condition['name'],
                group=condition['group'], amplitude_sum=sum(condition['amplitudes']),
                frequency_fs=condition['bins'][0]/size if len(condition['bins']) == 1 else None,
                method=method, **spectral_metrics(signal, truth, condition['bins']),
                overloads=observed['overloads']))
        if condition['name'] == 'sine_a0.85_f0.239':
            spectrum = {'frequency_fs': np.fft.rfftfreq(size)}
            for method, signal in outputs.items():
                power = abs(np.fft.rfft(signal-signal.mean()))**2
                power[1:-1] *= 2
                spectrum[method] = 10*np.log10(np.maximum(power/power[condition['bins'][0]], 1e-30))
            ex.write_csv(directory/'spectrum.csv', [dict(zip(spectrum, values)) for values in zip(*spectrum.values())])
    ex.write_csv(directory/'test_metrics.csv', rows)
    return rows


def summarize(rows):
    result = []
    for key in sorted({(r['plant'], r['group'], r['method']) for r in rows}):
        selected = [r for r in rows if (r['plant'], r['group'], r['method']) == key]
        row = dict(zip(('plant', 'group', 'method'), key))
        for metric in ('sndr_db', 'sfdr_db', 'spectral_ratio_db', 'spur_ratio_db', 'error_snr_db'):
            values = [np.mean([r[metric] for r in selected if r['seed'] == seed])
                      for seed in sorted({r['seed'] for r in selected})] if selected[0][metric] is not None else []
            row[metric+'_mean'] = float(np.mean(values)) if values else None
            row[metric+'_sd'] = float(np.std(values, ddof=1)) if len(values) > 1 else (0. if values else None)
        result.append(row)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smooth-baseline', type=Path)
    parser.add_argument('--smooth-mixed', type=Path)
    parser.add_argument('--sar-records', type=int, default=64)
    parser.add_argument('--sar-validation-records', type=int, default=8)
    parser.add_argument('--sar-size', type=int, default=8192)
    parser.add_argument('--sar-stride', type=int, default=8)
    parser.add_argument('--sar-iterations', type=int, default=1000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--sar-plants', nargs='*', choices=['binary', 'redundant'], default=['binary', 'redundant'])
    args = parser.parse_args(argv)
    if bool(args.smooth_baseline) != bool(args.smooth_mixed):
        parser.error('supply both smooth baseline and mixed paths')
    if args.sar_size < 1024 or args.sar_size % 2 or min(args.sar_records, args.sar_validation_records, args.sar_stride, args.sar_iterations) < 1 or args.sar_stride > args.sar_size:
        parser.error('invalid SAR data or optimizer budget')
    if not args.sar_plants and not args.smooth_baseline:
        parser.error('select at least one experiment')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.sar_plants)) != len(args.sar_plants):
        parser.error('seeds and plants must not repeat')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('choose a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    package = Path(inspect.getfile(fit_calibration_mlp)).parent.parent
    sources = [Path(__file__), Path(ex.__file__), Path(ex.fb.__file__), Path(inspect.getfile(ex.metrics))]
    sources += list((package/'calibration').glob('*.py'))+list((package/'spectrum').glob('*.py'))+[Path(inspect.getfile(sar_convert)), Path(inspect.getfile(ex.fb.ADC_Signal_Generator))]
    hashes = {}
    for source in sources:
        relative = source.relative_to(package)
        target = args.output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(source_sha256=hashes, input_sha256={}, history=3, hidden_units=16,
                  sar_msb_relative_error=.01, sar_bits=14,
                  metric_policy='coherent boxcar, DC excluded, full Nyquist band, all harmonics included in SNDR; single-tone cross-check public analyzer',
                  selection='validation reference MSE only; no test selection')
    rows = []
    if args.smooth_baseline:
        original_cfg = read_json(args.smooth_baseline/'configuration.json')
        mixed_cfg = read_json(args.smooth_mixed/'configuration.json')
        if original_cfg.get('plant_model') != ex.fb.MODEL_ID or mixed_cfg.get('plant_model') != ex.fb.MODEL_ID:
            raise ValueError('Legacy/custom smooth models: rerun exp_cal03 and exp_cal05')
        if not mixed_cfg.get('retrain_mlp') or hashlib.sha256((args.smooth_baseline/'configuration.json').read_bytes()).hexdigest() != mixed_cfg['baseline_configuration_sha256']:
            raise ValueError('mismatched baseline/mixed inputs')
        for plant in original_cfg['plants']:
            for seed in original_cfg['seeds']:
                case = f'{plant}_seed{seed}'
                directory = args.output/f'smooth_{case}'
                directory.mkdir()
                for source in [args.smooth_baseline/'configuration.json', args.smooth_baseline/case/'data_manifest.json',
                               args.smooth_mixed/'configuration.json', args.smooth_mixed/case/'models.json',
                               args.smooth_mixed/case/'training_manifest.json']:
                    config['input_sha256'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
                old = read_json(args.smooth_baseline/case/'data_manifest.json')
                records = read_json(args.smooth_mixed/case/'training_manifest.json')
                device = dict(kind='smooth', config=old['plant'], nominal_weights=sar_ideal_weights(old['plant']['bits']).tolist())
                size, stride = original_cfg['size'], original_cfg['stride']
                _, bit_rows, reference = dataset(device, records, size, stride)
                models, diagnostics = fit_bit_models(device, records, size, bit_rows, reference)
                saved = read_json(args.smooth_mixed/case/'models.json')
                models.update({name: saved[name] for name in ('poly5_static', 'poly5_history', 'mlp_history')})
                conditions = [r for r in old['tests'] if r['group'] in ('sine', 'multitone')]
                ex.write_json(directory/'manifest.json', dict(device=device, training=records, tests=conditions, size=size, stride=stride))
                ex.write_json(directory/'models.json', models)
                ex.write_json(directory/'diagnostics.json', diagnostics)
                rows += evaluate(device, read_json(directory/'models.json'), conditions, size, seed, 'smooth_'+plant, directory)
                print(f'smooth/{case}: saved polynomial/MLP replayed; bit calibrations complete', flush=True)
    for architecture in args.sar_plants:
        nominal = sar_ideal_weights(14, redundant_bit=2 if architecture == 'redundant' else None)
        actual = nominal.copy()
        actual[0] *= 1.01
        device = dict(kind='sar', architecture=architecture, nominal_weights=nominal.tolist(), actual_weights=actual.tolist())
        for seed in args.seeds:
            start = time.perf_counter()
            plant = 'sar_'+architecture
            directory = args.output/f'{plant}_seed{seed}'
            directory.mkdir()
            size = args.sar_size
            conditions = [r for r in ex.test_conditions(size) if r['group'] in ('sine', 'multitone')]
            excluded = {k for r in conditions for k in r['bins']}
            records = new_records(size, args.sar_records, seed+610000, excluded)
            valid = new_records(size, args.sar_validation_records, seed+620000,
                                excluded | {k for r in records for k in r['tone_bins']})
            x, bits, reference = dataset(device, records, size, args.sar_stride)
            xv, _, rv = dataset(device, valid, size, args.sar_stride)
            models, diagnostics = fit_bit_models(device, records, size, bits, reference)
            for name, columns in [('poly5_static', 1), ('poly5_history', 4)]:
                fitted = fit_calibration_polynomial(x[:, :columns], reference, degree=5, ridge=1e-10)
                models[name], diagnostics[name] = fitted['model'], fitted['diagnostics']
            fitted = fit_calibration_mlp(x, reference, validation_features=xv, validation_reference=rv,
                                         hidden_units=16, max_iterations=args.sar_iterations, seed=seed)
            models['mlp_history'], diagnostics['mlp_history'] = fitted['model'], fitted['diagnostics']
            ex.write_csv(directory/'mlp_history.csv', fitted['history'])
            ex.write_json(directory/'manifest.json', dict(device=device, training=records, validation=valid,
                          tests=conditions, size=size, stride=args.sar_stride))
            ex.write_json(directory/'models.json', models)
            ex.write_json(directory/'diagnostics.json', diagnostics)
            rows += evaluate(device, read_json(directory/'models.json'), conditions, size, seed, plant, directory)
            print(f'{plant}/seed{seed}: {time.perf_counter()-start:.1f}s; MLP {diagnostics["mlp_history"]["optimizer_message"]}', flush=True)
    ex.write_json(args.output/'configuration.json', config)
    ex.write_csv(args.output/'test_metrics.csv', rows)
    summary = summarize(rows)
    ex.write_csv(args.output/'summary.csv', summary)
    ex.write_json(args.output/'summary.json', summary)
    methods = ['raw', 'bits_sine', 'bits_reference', 'poly5_static', 'poly5_history', 'mlp_history']
    plants = sorted({r['plant'] for r in rows})
    fig, axes = plt.subplots(2, len(plants), figsize=(5*len(plants), 8), squeeze=False, layout='constrained')
    for col, plant in enumerate(plants):
        for row, metric in enumerate(('sndr_db', 'sfdr_db')):
            selected = [next(r for r in summary if (r['plant'], r['group'], r['method']) == (plant, 'sine', method)) for method in methods]
            ax = axes[row, col]
            ax.bar(methods, [r[metric+'_mean'] for r in selected],
                   yerr=[r[metric+'_sd'] for r in selected], capsize=3)
            ax.set_title(plant+' / '+metric.upper())
            ax.tick_params(axis='x', rotation=55)
            ax.set_ylabel('dB; condition mean, then seed mean +/- SD')
            ax.grid(axis='y', alpha=.3)
    fig.savefig(args.output/'spectral_comparison.png', dpi=150)
    plt.close(fig)
    print(f'Finished: {len(rows)} frozen test metric rows', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
