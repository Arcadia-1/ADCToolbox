"""Replay saved bit-weight comparisons and independently check full-FFT ratios."""
import os
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
try:
    from . import exp_cal10_foreground_bitweights as ex
except ImportError:
    import exp_cal10_foreground_bitweights as ex


def independent_metrics(signal, truth, bins):
    # Full complex FFT, explicitly remove positive/negative desired lines.
    power = abs(np.fft.fft(signal-signal.mean()))**2
    n = len(signal)
    desired = [int(k) for k in bins]+[n-int(k) for k in bins]
    desired_power = float(power[desired].sum())
    strongest = max(float(power[k]+power[n-k]) for k in bins)
    residual = power.copy()
    residual[[0]+desired] = 0
    unwanted = residual[1:n//2]+residual[-1:n//2:-1]
    spur = max(float(unwanted.max()), float(residual[n//2]))
    return dict(spectral_ratio_db=float(10*np.log10(desired_power/max(residual.sum(), 1e-300))),
                spur_ratio_db=float(10*np.log10(strongest/max(spur, 1e-300))),
                error_snr_db=float(10*np.log10(np.var(truth)/max(np.mean((signal-truth)**2), 1e-300))))


def audit(directory):
    cfg = ex.read_json(directory/'configuration.json')
    for name, digest in cfg['source_sha256'].items():
        assert hashlib.sha256((directory/'source_snapshot'/name).read_bytes()).hexdigest() == digest
    for name, digest in cfg['input_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    with (directory/'test_metrics.csv').open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    lookup = {(r['plant'], int(r['seed']), r['condition'], r['method']): r for r in saved}
    assert len(lookup) == len(saved)
    recomputed, max_difference, cases, single_checked, auto_peak_mismatches = [], 0., 0, 0, 0
    input_hashes = {}
    for case in sorted(directory.glob('*_seed*')):
        if not case.is_dir():
            continue
        plant, seed_text = case.name.rsplit('_seed', 1)
        seed = int(seed_text)
        manifest, models = ex.read_json(case/'manifest.json'), ex.read_json(case/'models.json')
        train_bins = {k for r in manifest['training'] for k in r['tone_bins']}
        test_bins = {k for r in manifest['tests'] for k in r['bins']}
        assert not train_bins & test_bins
        diagnostics = ex.read_json(case/'diagnostics.json')
        assert diagnostics['bits_reference']['samples'] == len(manifest['training'])*len(range(0, manifest['size'], manifest['stride']))
        if 'validation' in manifest:
            validation_bins = {k for r in manifest['validation'] for k in r['tone_bins']}
            assert not validation_bins & (train_bins | test_bins)
            train_seeds = {r['capture_seed'] for r in manifest['training']}
            assert not train_seeds & {r['capture_seed'] for r in manifest['validation']}
            with (case/'mlp_history.csv').open(newline='') as stream:
                history = list(csv.DictReader(stream))
            diag = diagnostics['mlp_history']
            assert float(history[diag['selected_iteration']]['validation_mse']) == min(float(r['validation_mse']) for r in history)
            assert diag['iterations'] <= cfg['sar_iterations']
        for i, condition in enumerate(manifest['tests']):
            record = dict(tone_bins=condition['bins'], amplitudes=condition['amplitudes'],
                          phases=condition['phases'], capture_seed=seed+30000+i)
            observed = ex.capture(manifest['device'], record, manifest['size'])
            assert observed['overloads'] == 0
            outputs, truth = ex.predictions(observed, models, manifest['device'])
            for method, signal in outputs.items():
                row = lookup[(plant, seed, condition['name'], method)]
                measured = independent_metrics(signal, truth, condition['bins'])
                for key, value in measured.items():
                    difference = abs(value-float(row[key]))
                    max_difference = max(max_difference, difference)
                    assert difference < 1e-7, (plant, seed, method, key, difference)
                if len(condition['bins']) == 1:
                    assert float(row['sndr_db']) == float(row['spectral_ratio_db'])
                    assert float(row['sfdr_db']) == float(row['spur_ratio_db'])
                    if row['public_fft_matching_fundamental'] == 'True':
                        assert float(row['public_fft_max_difference_db']) < 1e-6
                        single_checked += 1
                    else:
                        auto_peak_mismatches += 1
                else:
                    assert row['sndr_db'] == row['sfdr_db'] == ''
                recomputed.append({**row, 'seed': seed, **measured,
                                   'sndr_db': float(row['sndr_db']) if row['sndr_db'] else None,
                                   'sfdr_db': float(row['sfdr_db']) if row['sfdr_db'] else None})
        for name in ('manifest.json', 'models.json', 'diagnostics.json', 'test_metrics.csv'):
            input_hashes[f'{case.name}/{name}'] = hashlib.sha256((case/name).read_bytes()).hexdigest()
        cases += 1
    assert len(recomputed) == len(saved)
    summary = ex.summarize(recomputed)
    published = ex.read_json(directory/'summary.json')
    for a, b in zip(summary, published):
        for key in a:
            if isinstance(a[key], float):
                assert abs(a[key]-b[key]) < 1e-7
            else:
                assert a[key] == b[key]
    assert len(summary) == len(published)
    return dict(status='passed', cases=cases, frozen_rows_replayed=len(recomputed),
                max_independent_fft_difference_db=max_difference,
                public_single_tone_checks=single_checked, auto_peak_mismatches=auto_peak_mismatches,
                overloads=0, aggregate_recomputed=True, case_sha256=input_hashes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('choose a new output file')
    result = audit(args.directory)
    ex.ex.write_json(args.output, result)
    print(f"Passed: {result['cases']} cases, {result['frozen_rows_replayed']} rows; "
          f"full-FFT maximum difference {result['max_independent_fft_difference_db']:.3g} dB")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
