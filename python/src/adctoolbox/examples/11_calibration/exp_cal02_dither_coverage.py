"""Dither mechanisms: randomization, gain, memory, capacitor and nonlinearity.

Run this file directly; outputs are written to a new directory. Defaults run
48 continuous experiments (4 mechanisms x 2 waveforms x 2 scenarios x 3 seeds),
18 subtractive-dither sweeps and identifiability diagnostics. No ideal input
labels enter background identification. See DITHER_COVERAGE.md.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import dither_mechanisms as dm
    from .exp_cal01_dither_background import metrics, write_json
except ImportError:
    import dither_mechanisms as dm
    from exp_cal01_dither_background import metrics, write_json


FS = 100e6
METHODS = ('nominal', 'scalar_correlation', 'foreground_frozen', 'matched_background',
           'oracle', 'nominal_probe_off', 'foreground_probe_off', 'oracle_probe_off')
INITIAL = {'gain': [3.88, 0.], 'memory': [3.88, .12], 'capacitor': [.006],
           'nonlinear': [1/3.88, .16/3.88]}
FINAL = {'gain': [4.08, 0.], 'memory': [4.08, .24], 'capacitor': [.012],
         'nonlinear': [1/4.08, .24/4.08]}
NOMINAL = {'gain': [4., 0.], 'memory': [4., 0.], 'capacitor': [0.], 'nonlinear': [.25, 0.]}


def waveform(n, size, kind):
    if kind == 'sine':
        bins, amplitudes = [int(round(.137*size)) | 1], [.8]
    else:
        bins, amplitudes = [int(round(.083*size)) | 1, int(round(.217*size)) | 1], [.46, .34]
    return sum(a*np.sin(2*np.pi*k*n/size+.37+.81*j)
               for j, (a, k) in enumerate(zip(amplitudes, bins))), bins


def make_plant(mechanism, seed):
    cls = (dm.MemoryPlant if mechanism in ('gain', 'memory')
           else dm.CapacitorPlant if mechanism == 'capacitor' else dm.NonlinearPlant)
    return cls(seed)


def make_tracker(mechanism, scalar=False):
    if mechanism in ('gain', 'memory'):
        return dm.MemoryTracker(scalar=scalar or mechanism == 'gain')
    if mechanism == 'capacitor':
        return dm.CapacitorTracker(code_weighted=not scalar)
    return dm.NonlinearTracker(scalar=scalar)


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(mechanism, kind, scenario, seed, blocks, size, output=None, foreground_blocks=16):
    initial = np.array(INITIAL[mechanism])
    fg = dm.foreground_fit(make_plant(mechanism, seed+100000), initial, mechanism,
                           size=size, blocks=foreground_blocks)
    plant, off_plant = make_plant(mechanism, seed), make_plant(mechanism, seed)
    matched, scalar = make_tracker(mechanism), make_tracker(mechanism, True)
    correctors = {name: dm.MemoryCorrector() for name in METHODS}
    rows = []
    for block in range(blocks):
        n = np.arange(block*size, (block+1)*size)
        clean, bins = waveform(n, size, kind)
        truth = np.array(FINAL[mechanism] if scenario == 'step' and block >= blocks//2
                         else INITIAL[mechanism])
        c, off = plant.convert(clean, truth), off_plant.convert(clean, truth, probe_on=False)
        before, scalar_before = matched.parameters.copy(), scalar.parameters.copy()
        params = dict(nominal=NOMINAL[mechanism], scalar_correlation=scalar_before,
                      foreground_frozen=fg, matched_background=before, oracle=truth,
                      nominal_probe_off=NOMINAL[mechanism], foreground_probe_off=fg,
                      oracle_probe_off=truth)
        row = dict(block=block, time_ms=(block+1)*size/FS*1e3, overloads=c['overload'],
                   off_overloads=off['overload'])
        for name, parameter in params.items():
            capture = off if name.endswith('probe_off') else c
            if mechanism in ('gain', 'memory'):
                result = correctors[name].apply(capture, parameter)
            elif mechanism == 'capacitor':
                result = dm.capacitor_correct(capture, parameter)
            else:
                result = dm.nonlinear_correct(capture, parameter)
            row.update({name+'_'+key: value for key, value in metrics(result, clean, bins).items()})
        # Estimates from this block are applied only to the following block.
        row['status'] = matched.update(c)
        row['scalar_status'] = scalar.update(c)
        for index in range(len(truth)):
            row[f'true_{index}'] = float(truth[index])
            row[f'used_{index}'] = float(before[index])
            row[f'next_{index}'] = float(matched.parameters[index])
            row[f'foreground_{index}'] = float(fg[index])
        rows.append(row)
    windows = {'initial_quarter': rows[:blocks//4], 'final_quarter': rows[3*blocks//4:]}
    if scenario == 'step':
        windows['step_transient'] = rows[blocks//2:3*blocks//4]
    summary = dict(mechanism=mechanism, waveform=kind, scenario=scenario, seed=seed,
                   foreground=fg.tolist(), final_parameters=matched.parameters.tolist(),
                   overloads=sum(r['overloads'] for r in rows),
                   off_overloads=sum(r['off_overloads'] for r in rows),
                   held_updates=sum(r['status'] != 'updated' for r in rows),
                   scalar_held_updates=sum(r['scalar_status'] != 'updated' for r in rows),
                   max_causality_gap=max(abs(rows[j][f'used_{i}']-rows[j-1][f'next_{i}'])
                                         for j in range(1, blocks) for i in range(len(truth))),
                   windows={window: {method: {key: float(np.mean([r[method+'_'+key]
                                                                  for r in records]))
                              for key in ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db')}
                              for method in METHODS} for window, records in windows.items()})
    if output is not None:
        output.mkdir()
        save_csv(output/'trajectory.csv', rows)
        write_json(output/'summary.json', summary)
    return rows, summary


def linearization_sweep(seeds, size=131072):
    """Monotone periodic threshold INL, ideal subtractive DAC, optional upstream HD3."""
    bits, period, inl_lsb = 14, 64, 2.
    lsb = 2/2**bits
    j = np.arange(1, 2**bits)
    thresholds = -1+j*lsb+inl_lsb*lsb*np.sin(2*np.pi*j/period)
    assert np.all(np.diff(thresholds) > 0)
    n, tone = np.arange(size), int(round(.137*size)) | 1
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        d = rng.uniform(-period*lsb/2, period*lsb/2, size)
        for amplitude in (.01, .1, .8):
            clean = amplitude*np.sin(2*np.pi*tone*n/size+.31)
            for front_cubic in (0., .03):
                # Dither is added AFTER this upstream nonlinearity.
                front = clean+front_cubic*clean**3
                for method, injected, removed in (('off', 0., 0.), ('unsubtracted', d, 0.),
                                                  ('subtractive', d, d),
                                                  ('dac_scale_error_1pct', 1.01*d, d)):
                    analog = front+injected
                    result = -1+(np.searchsorted(thresholds, analog)+.5)*lsb-removed
                    row = dict(seed=seed, amplitude=amplitude, front_cubic=front_cubic,
                               method=method, overloads=int(np.sum(abs(analog) >= 1)))
                    row.update(metrics(result, clean, [tone]))
                    rows.append(row)
    return rows


def diagnostics(size=524288):
    """Explicit failure controls, separate from the normal background runs."""
    p = dm.PRBS(17).draw(size)
    nuisance = .1*np.sin(2*np.pi*.137*np.arange(size)+.31)
    response = 3.88*(nuisance+.0625*p)
    estimate = lambda probe, observed: float(dm.covariance(probe, observed)[0, 0]
                                             /dm.covariance(probe, probe)[0, 0]/.0625)
    sequence = []
    for count in (1024, 8192, 65536, size):
        sequence.append(dict(samples=count, estimate=estimate(p[:count], response[:count])))
    failures = {
        'correct_alignment': estimate(p, response),
        'wrong_alignment_one_sample': estimate(np.roll(p, 1), response),
        'probe_amplitude_actual_0.070_assumed_0.0625': estimate(p, 3.88*(nuisance+.070*p)),
        'input_correlated_with_probe_0.02': estimate(p, 3.88*(nuisance+(.0625+.02)*p)),
    }
    short = dm.PRBS(17, bits=7).draw(size)
    # Deliberately adversarial periodic input repeats the short PRBS; this
    # demonstrates a broken exogeneity assumption, not typical user traffic.
    failures['short_period_input_lock'] = estimate(short, 3.88*(.02*short+.0625*short))
    binary = dm.NonlinearPlant(17, binary=True)
    x = .8*np.sin(2*np.pi*.137*np.arange(32768))
    c = binary.convert(x, INITIAL['nonlinear'])
    tracker = dm.NonlinearTracker()
    failures['binary_cubic_update_status'] = tracker.update(c)
    failures['binary_cubic_parameters_unchanged'] = tracker.parameters.tolist()
    # Test absence of outer coarse codes: selected instrument cannot identify.
    cap = dm.CapacitorPlant(17)
    small = cap.convert(.1*np.sin(np.arange(32768)*.73), INITIAL['capacitor'])
    failures['capacitor_missing_outer_codes'] = dm.CapacitorTracker().update(small)
    over = dm.MemoryPlant(17, amplitude=.25).convert(x, INITIAL['memory'])
    failures['overload_count'] = over['overload']
    failures['overload_update_status'] = dm.MemoryTracker(amplitude=.25).update(over)
    return dict(true_gain=3.88, sequence_convergence=sequence, failure_controls=failures,
                long_prbs_period=2**31-1, short_prbs_period=127)


def aggregate(summaries):
    rows = []
    for mechanism in sorted({s['mechanism'] for s in summaries}):
        for kind in sorted({s['waveform'] for s in summaries}):
            for scenario in sorted({s['scenario'] for s in summaries}):
                selected = [s for s in summaries
                            if (s['mechanism'], s['waveform'], s['scenario'])
                            == (mechanism, kind, scenario)]
                for window in selected[0]['windows']:
                    for method in METHODS:
                        row = dict(mechanism=mechanism, waveform=kind, scenario=scenario,
                                   window=window, method=method, seeds=len(selected))
                        for key in ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db'):
                            values = [s['windows'][window][method][key] for s in selected]
                            row[key+'_mean'] = float(np.mean(values))
                            row[key+'_sd'] = (float(np.std(values, ddof=1)) if len(values) > 1
                                              else 0.)
                        rows.append(row)
    return rows


def plot_results(traces, linearization, output):
    selected = [(key, rows) for key, rows in traces.items() if key[1:] == ('sine', 'step')]
    if selected:
        fig, axes = plt.subplots(2, len(selected), figsize=(4.5*len(selected), 7.5), squeeze=False,
                                 layout='constrained')
        for col, (key, rows) in enumerate(selected):
            t = [r['time_ms'] for r in rows]
            index = 1 if key[0] in ('memory', 'nonlinear') else 0
            for prefix, label, color in [('true', 'Truth (evaluation)', 'black'),
                                         ('foreground', 'Foreground frozen', '#bb761a'),
                                         ('used', 'Background, causal', '#147caa')]:
                axes[0, col].plot(t, [r[f'{prefix}_{index}'] for r in rows], label=label,
                                  color=color)
            axes[0, col].set_title(key[0].capitalize())
            axes[0, col].set_ylabel(('G', 'H')[index] if key[0] in ('gain', 'memory')
                                    else 'Cap mismatch delta' if key[0] == 'capacitor'
                                    else 'Inverse cubic coefficient')
            for name, color in [('nominal', '#999999'), ('scalar_correlation', '#9467bd'),
                                ('foreground_frozen', '#bb761a'), ('matched_background', '#147caa'),
                                ('oracle', '#31955b')]:
                axes[1, col].plot(t, [r[name+'_spectral_ratio_db'] for r in rows], color=color,
                                  label=name.replace('_', ' '))
            axes[1, col].set_xlabel('Normal-operation time (ms)')
            axes[1, col].set_ylabel('SNDR (dB)')
            for ax in axes[:, col]:
                ax.grid(alpha=.25)
        axes[0, 0].legend(fontsize=7)
        axes[1, 0].legend(fontsize=7)
        fig.suptitle('Mechanism-specific dither calibration | first seed, same probe-on captures\n'
                     'Different plant models: columns are not an algorithm ranking')
        fig.savefig(output/'mechanisms.png', dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    amplitudes = sorted({r['amplitude'] for r in linearization})
    for method in ('off', 'unsubtracted', 'subtractive', 'dac_scale_error_1pct'):
        for ax, metric in zip(axes, ('spectral_ratio_db', 'spur_ratio_db')):
            means = [np.mean([r[metric] for r in linearization if r['method'] == method
                              and r['amplitude'] == a and r['front_cubic'] == 0])
                     for a in amplitudes]
            ax.plot(20*np.log10(amplitudes), means, 'o-', label=method.replace('_', ' '))
            ax.set_xlabel('Input amplitude (dBFS)')
            ax.grid(alpha=.25)
    axes[0].set_ylabel('SNDR (dB)')
    axes[1].set_ylabel('SFDR (dB)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Dither randomization without parameter learning | periodic threshold INL')
    fig.savefig(output/'linearization.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent/'output'/'dither_coverage')
    parser.add_argument('--blocks', type=int, default=256)
    parser.add_argument('--block-size', type=int, default=32768)
    parser.add_argument('--foreground-blocks', type=int, default=16)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--mechanisms', nargs='+', choices=list(INITIAL), default=list(INITIAL))
    parser.add_argument('--waveforms', nargs='+', choices=['sine', 'multitone'],
                        default=['sine', 'multitone'])
    parser.add_argument('--scenarios', nargs='+', choices=['constant', 'step'],
                        default=['constant', 'step'])
    args = parser.parse_args(argv)
    if (args.blocks < 16 or args.blocks % 4 or args.block_size < 1024 or args.block_size % 2
            or args.foreground_blocks < 1):
        parser.error(
            'Require blocks >=16 divisible by 4, even block-size >=1024, foreground-blocks >=1')
    for values in (args.seeds, args.mechanisms, args.waveforms, args.scenarios):
        if len(values) != len(set(values)):
            parser.error('Duplicate case selections are not allowed')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new or empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    source = [Path(__file__), Path(dm.__file__),
              Path(__file__).with_name('exp_cal01_dither_background.py'),
              Path(__file__).with_name('pipeline_dither_model.py')]
    config = {**vars(args), 'output': str(args.output.resolve()), 'fs_hz': FS,
              'initial_parameters': INITIAL, 'final_parameters': FINAL,
              'normal_samples_per_case': args.blocks*args.block_size,
              'foreground_samples_per_case': args.foreground_blocks*args.block_size,
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source}}
    write_json(args.output/'configuration.json', config)
    summaries, traces = [], {}
    for mechanism in args.mechanisms:
        for kind in args.waveforms:
            for scenario in args.scenarios:
                for seed in args.seeds:
                    name = f'{mechanism}_{kind}_{scenario}_seed{seed}'
                    rows, summary = run_case(mechanism, kind, scenario, seed, args.blocks,
                                             args.block_size, args.output/name,
                                             foreground_blocks=args.foreground_blocks)
                    summaries.append(summary)
                    if seed == args.seeds[0]:
                        traces[(mechanism, kind, scenario)] = rows
                    final = summary['windows']['final_quarter']
                    print(name+': '+', '.join(f'{m}={final[m]["spectral_ratio_db"]:.2f}'
                                              for m in METHODS[:5])
                          +f'; held={summary["held_updates"]}; overloads={summary["overloads"]}',
                          flush=True)
    table = aggregate(summaries)
    linearization = linearization_sweep(args.seeds)
    save_csv(args.output/'aggregate.csv', table)
    save_csv(args.output/'linearization.csv', linearization)
    write_json(args.output/'summary.json', dict(runs=summaries, aggregate=table))
    write_json(args.output/'diagnostics.json', diagnostics())
    plot_results(traces, linearization, args.output)
    print(f'Saved {len(summaries)} causal runs and {len(linearization)} randomization '
          f'measurements to {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
