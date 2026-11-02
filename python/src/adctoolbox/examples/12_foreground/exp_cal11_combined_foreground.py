"""Combined foreground correction: smooth front-end distortion + SAR mismatch.

Reuse ADC_Signal_Generator.apply_static_nonlinearity and public SAR conversion.
A separate sine injected at the SAR input identifies bit weights; frozen weights
precede a learned scalar polynomial/MLP inverse. Test captures always contain
BOTH errors. No fitting routine receives clean input or actual analog weights.
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

from adctoolbox import (apply_calibration, calibrate_weight_sine,
                       scale_calibration_output, fit_calibration_polynomial, fit_calibration_mlp)
from adctoolbox.models import sar_convert, sar_ideal_weights, sar_reconstruct
from adctoolbox.siggen import ADC_Signal_Generator
try:
    from . import exp_cal10_foreground_bitweights as prior
except ImportError:
    import exp_cal10_foreground_bitweights as prior

read_json, write_json, write_csv = prior.read_json, prior.ex.write_json, prior.ex.write_csv
PRIMARY = ['raw', 'bits_only', 'poly_only', 'mlp_only', 'bits_poly', 'bits_mlp']


def make_device(architecture):
    nominal = sar_ideal_weights(14, redundant_bit=2 if architecture == 'redundant' else None)
    actual = nominal.copy()
    actual[0] *= 1.01
    return dict(architecture=architecture, nominal_weights=nominal.tolist(),
                actual_weights=actual.tolist(), k2=.015, k3=.04,
                sampling_noise_rms=1e-5, monitor_noise_rms=2e-4)


def capture(device, record, size, *, bypass=False):
    n = np.arange(size)
    frequencies = np.array(record['tone_bins'])/size
    clean = sum(a*np.sin(2*np.pi*f*n+p) for f, a, p in
                zip(frequencies, record['amplitudes'], record['phases']))
    generator = ADC_Signal_Generator(size, 1., frequencies[0], 1., 0.)
    analog = clean if bypass else generator.apply_static_nonlinearity(
        input_signal=clean, k2=device['k2'], k3=device['k3'])
    adc_seed, monitor_seed = np.random.SeedSequence(record['capture_seed']).spawn(2)
    sampled = analog+np.random.default_rng(adc_seed).normal(0, device['sampling_noise_rms'], size)
    bits = sar_convert(sampled, np.array(device['actual_weights']), quant_range=(-1, 1))
    nominal = np.array(device['nominal_weights'])
    raw = sar_reconstruct(bits, nominal, quant_range=(-1, 1))+nominal[-1]
    monitor = clean+np.random.default_rng(monitor_seed).normal(0, device['monitor_noise_rms'], size)
    # Existing quantizer returns lower-bin values; add half an LSB for midpoint.
    monitor = generator.apply_quantization_noise(monitor, n_bits=16, quant_range=(-1, 1))+2**-16
    return dict(raw=raw, bits=bits, reference=prior.ex.fb.tone_projection(monitor, frequencies),
                truth=clean, overloads=int((abs(sampled) >= 1).sum()),
                bypass=bypass)


def apply_bits(bits, model):
    return bits@np.array(model['weight'])+model['offset']


def fit_bit_stage(observed, frequency, nominal):
    """Only observable binary columns, projected monitor and nominal weights."""
    result = calibrate_weight_sine(observed['bits'], freq=frequency,
        force_search=False, nominal_weights=2*np.array(nominal), harmonic_order=1)
    reference = observed['reference']
    peak = float(np.sqrt(2*np.var(reference)))
    scaled = scale_calibration_output(result, target_sine_peak=peak)
    return (dict(weight=scaled['weight'].tolist(), offset=float(reference.mean()-scaled['offset'])),
            dict(samples=len(reference), monitor_peak=peak,
                 rank_patch_applied=bool(result['rank_patch']['applied']),
                 protocol='dedicated bypass sine; calibrated monitor peak and DC; no actual weights'))


def dataset(device, records, size, stride, bit_model):
    raw, corrected, bits, reference = [], [], [], []
    for record in records:
        observed = capture(device, record, size)
        if observed['overloads']:
            raise ValueError('overloaded fitting capture')
        raw.append(observed['raw'][::stride])
        corrected.append(apply_bits(observed['bits'], bit_model)[::stride])
        bits.append(observed['bits'][::stride])
        reference.append(observed['reference'][::stride])
    return dict(raw=np.concatenate(raw), bit_corrected=np.concatenate(corrected),
                bits=np.vstack(bits), reference=np.concatenate(reference))


def fit_corrections(training, validation, *, iterations, seed):
    """Both paths see identical end-to-end labels, rows and model budgets."""
    models, diagnostics, histories = {}, {}, {}
    for prefix, key in [('direct', 'raw'), ('cascade', 'bit_corrected')]:
        fitted = fit_calibration_polynomial(training[key], training['reference'], degree=5, ridge=1e-10)
        models[prefix+'_poly'], diagnostics[prefix+'_poly'] = fitted['model'], fitted['diagnostics']
        fitted = fit_calibration_mlp(training[key], training['reference'],
            validation_features=validation[key], validation_reference=validation['reference'],
            hidden_units=16, max_iterations=iterations, seed=seed)
        models[prefix+'_mlp'], diagnostics[prefix+'_mlp'] = fitted['model'], fitted['diagnostics']
        histories[prefix+'_mlp'] = fitted['history']
    # No-bypass identification control. These weights can absorb both errors.
    fitted = fit_calibration_polynomial(training['bits'], training['reference'], degree=1, ridge=1e-10)
    models['end_to_end_bits'], diagnostics['end_to_end_bits'] = fitted['model'], fitted['diagnostics']
    z = apply_calibration(training['bits'], fitted['model'])['calibrated_signal']
    fitted = fit_calibration_polynomial(z, training['reference'], degree=5, ridge=1e-10)
    models['end_to_end_poly'], diagnostics['end_to_end_poly'] = fitted['model'], fitted['diagnostics']
    return models, diagnostics, histories


def inverse_control(voltage, device):
    """Evaluator-only inverse using known front-end coefficients; not a fitter."""
    x = np.asarray(voltage).copy()
    for _ in range(10):
        value = x+device['k2']*x*x+device['k3']*x**3
        derivative = 1+2*device['k2']*x+3*device['k3']*x*x
        x -= (value-voltage)/derivative
    return x


def predictions(observed, models, device):
    raw, bits = observed['raw'], observed['bits']
    z = apply_bits(bits, models['bit_stage'])
    result = dict(raw=raw, bits_only=z)
    for method, key, inputs in [('poly_only', 'direct_poly', raw), ('mlp_only', 'direct_mlp', raw),
                                ('bits_poly', 'cascade_poly', z), ('bits_mlp', 'cascade_mlp', z)]:
        result[method] = apply_calibration(inputs, models[key])['calibrated_signal']
    end_to_end = apply_calibration(bits, models['end_to_end_bits'])['calibrated_signal']
    result['end_to_end_bits'] = end_to_end
    result['end_to_end_bits_poly'] = apply_calibration(end_to_end, models['end_to_end_poly'])['calibrated_signal']
    actual = np.array(device['actual_weights'])
    voltage = sar_reconstruct(bits, actual, quant_range=(-1, 1))+actual[-1]
    result['known_weights_inverse'] = inverse_control(voltage, device)
    return result


def conditions(size):
    records = [r for r in prior.ex.test_conditions(size) if r['group'] in ('sine', 'multitone')]
    records.append(dict(name='five_tone', group='multitone',
        bins=[int(f*size) | 1 for f in (.037, .103, .173, .283, .397)],
        amplitudes=[.18, .16, .14, .12, .1], phases=[.1, .7, 1.4, 2.2, 2.8]))
    return records


def test_record(condition, seed, index):
    return dict(tone_bins=condition['bins'], amplitudes=condition['amplitudes'],
                phases=condition['phases'], capture_seed=seed+730000+index)


def evaluate(directory, manifest, models, seed):
    rows = []
    device, size = manifest['device'], manifest['size']
    for i, condition in enumerate(manifest['tests']):
        observed = capture(device, test_record(condition, seed, i), size)
        outputs = predictions(observed, models, device)
        for method, signal in outputs.items():
            rows.append(dict(plant=device['architecture'], seed=seed, method=method,
                condition=condition['name'], group=condition['group'],
                amplitude_sum=sum(condition['amplitudes']),
                frequency_fs=condition['bins'][0]/size if len(condition['bins']) == 1 else None,
                **prior.spectral_metrics(signal, observed['truth'], condition['bins']),
                overloads=observed['overloads']))
        if condition['name'] == 'sine_a0.85_f0.239':
            spectrum = dict(frequency_fs=np.fft.rfftfreq(size))
            for method, signal in outputs.items():
                power = abs(np.fft.rfft(signal-signal.mean()))**2
                power[1:-1] *= 2
                spectrum[method] = 10*np.log10(np.maximum(power/power[condition['bins'][0]], 1e-30))
            write_csv(directory/'spectrum.csv', [dict(zip(spectrum, values)) for values in zip(*spectrum.values())])
    write_csv(directory/'test_metrics.csv', rows)
    return rows


def run_case(args, architecture, seed):
    start = time.perf_counter()
    directory = args.output/f'{architecture}_seed{seed}'
    directory.mkdir()
    device, tests = make_device(architecture), conditions(args.size)
    excluded = {k for r in tests for k in r['bins']}
    train = prior.new_records(args.size, args.records, seed+710000, excluded)
    excluded |= {k for r in train for k in r['tone_bins']}
    validation = prior.new_records(args.size, args.validation_records, seed+720000, excluded)
    excluded |= {k for r in validation for k in r['tone_bins']}
    candidates = np.setdiff1d(np.arange(int(.02*args.size) | 1, int(.45*args.size), 2), list(excluded))
    if not len(candidates):
        raise ValueError('no unused bypass-calibration frequency')
    chosen = int(candidates[np.argmin(abs(candidates/args.size-.071))])
    bypass = dict(tone_bins=[chosen], amplitudes=[.9], phases=[.41], capture_seed=seed+700000)
    observed = capture(device, bypass, args.size, bypass=True)
    if observed['overloads']:
        raise ValueError('overloaded bypass calibration')
    bit_model, bit_diagnostic = fit_bit_stage(observed, chosen/args.size, device['nominal_weights'])
    manifest = dict(device=device, size=args.size, stride=args.stride, training=train,
        validation=validation, bypass_calibration=bypass, tests=tests,
        reference='independent synchronized 16-bit monitor; known-tone projection; no true input labels')
    write_json(directory/'manifest.json', manifest)
    training = dataset(device, train, args.size, args.stride, bit_model)
    valid = dataset(device, validation, args.size, args.stride, bit_model)
    fitted, diagnostics, histories = fit_corrections(training, valid, iterations=args.iterations, seed=seed)
    models = dict(bit_stage=bit_model, **fitted)
    diagnostics['bit_stage'] = bit_diagnostic
    for name, history in histories.items():
        write_csv(directory/(name+'_history.csv'), history)
    write_json(directory/'models.json', models)
    write_json(directory/'diagnostics.json', diagnostics)
    rows = evaluate(directory, manifest, read_json(directory/'models.json'), seed)
    print(f'{architecture}/seed{seed}: {time.perf_counter()-start:.1f}s; '+
          '; '.join(f'{name}: iter={diagnostics[name]["iterations"]}, converged={diagnostics[name]["optimizer_converged"]}'
                    for name in ('direct_mlp', 'cascade_mlp')), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=int, default=64)
    parser.add_argument('--validation-records', type=int, default=8)
    parser.add_argument('--size', type=int, default=8192)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--architectures', nargs='+', choices=['binary', 'redundant'], default=['binary', 'redundant'])
    args = parser.parse_args(argv)
    if args.size < 1024 or args.size % 2 or min(args.records, args.validation_records, args.stride, args.iterations) < 1 or args.stride > args.size:
        parser.error('invalid size, count or optimizer budget')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.architectures)) != len(args.architectures):
        parser.error('seeds and architectures must not repeat')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('choose a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    package = Path(inspect.getfile(fit_calibration_mlp)).parent.parent
    sources = list((package/'calibration').glob('*.py'))+list((package/'spectrum').glob('*.py'))
    sources += [Path(__file__), Path(prior.__file__), Path(prior.ex.__file__), Path(prior.ex.fb.__file__),
                Path(inspect.getfile(prior.ex.metrics)), Path(inspect.getfile(sar_convert)),
                Path(inspect.getfile(ADC_Signal_Generator))]
    hashes = {}
    for source in sources:
        relative = source.relative_to(package)
        target = args.output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {**vars(args), 'output': str(args.output.resolve()), 'status': 'running',
        'source_sha256': hashes, 'history': 0, 'hidden_units': 16, 'degree': 5,
        'ridge': 1e-10, 'mlp_l2': 1e-6, 'bit_stage_extra_samples': args.size,
        'training_rows_per_case': args.records*len(range(0, args.size, args.stride)),
        'policy': 'both errors in every test; dedicated SAR-input bypass calibration; freeze bit weights before fitting downstream correction; test unused for selection'}
    write_json(args.output/'configuration.json', config)
    rows = []
    for architecture in args.architectures:
        for seed in args.seeds:
            rows += run_case(args, architecture, seed)
    summary = prior.summarize(rows)
    write_csv(args.output/'test_metrics.csv', rows)
    write_csv(args.output/'summary.csv', summary)
    write_json(args.output/'summary.json', summary)
    fig, axes = plt.subplots(2, len(args.architectures), figsize=(7*len(args.architectures), 8), squeeze=False, layout='constrained')
    for col, architecture in enumerate(args.architectures):
        for row, metric in enumerate(('sndr_db', 'sfdr_db')):
            data = [next(r for r in summary if (r['plant'], r['group'], r['method']) == (architecture, 'sine', method)) for method in PRIMARY]
            ax = axes[row, col]
            ax.bar(PRIMARY, [r[metric+'_mean'] for r in data], yerr=[r[metric+'_sd'] for r in data], capsize=3)
            ax.set_title(architecture+' SAR + smooth nonlinearity / '+metric.upper())
            ax.set_ylabel('dB; condition mean, then seed mean +/- SD')
            ax.tick_params(axis='x', rotation=35)
            ax.grid(axis='y', alpha=.3)
    fig.savefig(args.output/'combined_comparison.png', dpi=150)
    plt.close(fig)
    config.update(status='complete', metric_rows=len(rows))
    write_json(args.output/'configuration.json', config)
    print(f'Finished: {len(rows)} frozen metric rows', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
