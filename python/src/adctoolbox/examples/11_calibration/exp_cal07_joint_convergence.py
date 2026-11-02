"""Separate finite averaging, weak identification and bias at fixed dither.

Run directly from a copied examples tree. See JOINT_CONVERGENCE.md.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

try:
    from . import joint_dither_model as jm
    from . import joint_dither_convergence as jc
    from . import exp_cal06_joint_dither as previous
except ImportError:
    import joint_dither_model as jm
    import joint_dither_convergence as jc
    import exp_cal06_joint_dither as previous

METHODS = ('pooled_04', 'pooled_01', 'cumulative', 'grouped_04', 'grouped_01')
BANDS = previous.RELATIVE_BANDS


def make_trackers():
    return dict(pooled_04=jc.ConvergenceTracker(rate=.04),
                pooled_01=jc.ConvergenceTracker(rate=.01),
                cumulative=jc.ConvergenceTracker(rate=None),
                grouped_04=jc.ConvergenceTracker(rate=.04, grouped=True),
                grouped_01=jc.ConvergenceTracker(rate=.01, grouped=True))


def run_case(seed, kind, scenario, blocks=256, size=32768):
    trackers = make_trackers()
    correctors = {name: jm.JointCorrector(.2) for name in (*METHODS, 'nominal', 'oracle')}
    plant = jm.JointPlant(seed, .2)
    rows = []
    last_next = {}
    for block in range(blocks):
        x, bins = previous.waveform(np.arange(block*size, (block+1)*size), size, kind)
        truth = jm.INITIAL if scenario == 'step' and block < blocks//2 else jm.FINAL
        c, diag = plant.convert(x, truth)
        for method, corrector in correctors.items():
            coeff = (trackers[method].coefficients.copy() if method in trackers else
                     truth.inverse() if method == 'oracle' else jm.NOMINAL.inverse())
            estimate = jm.physical(coeff)
            row = dict(seed=seed, waveform=kind, scenario=scenario, block=block,
                       method=method, **diag, **{'used_'+k: v for k, v in estimate.items()},
                       **{'true_'+k: v for k, v in asdict(truth).items()})
            row.update(previous.metrics(corrector.apply(c, coeff)['signal'], x, bins))
            row['in_band'] = all(abs(estimate[k]-getattr(truth, k)) <= BANDS[k]*abs(getattr(truth, k)) for k in BANDS)
            row['causality_gap'] = float(np.max(abs(coeff-last_next[method]))) if method in last_next else 0.
            if method in trackers:
                row.update(trackers[method].update(c))
                row['condition'] = trackers[method].condition
                last_next[method] = trackers[method].coefficients.copy()
            else:
                row.update(electronics='reference', capacitor='reference', condition=0.)
            rows.append(row)
    return rows


def summarize(rows, blocks):
    result = []
    for seed, kind, scenario, method in sorted({(r['seed'], r['waveform'], r['scenario'], r['method']) for r in rows}):
        selected = [r for r in rows if (r['seed'], r['waveform'], r['scenario'], r['method']) == (seed, kind, scenario, method)]
        for window, lo, hi in [('pre_midpoint', blocks//4, blocks//2),
                               ('post_step_transient', blocks//2, 3*blocks//4),
                               ('final_quarter', 3*blocks//4, blocks)]:
            records = selected[lo:hi]
            entry = dict(seed=seed, waveform=kind, scenario=scenario, method=method, window=window,
                         in_band_fraction=float(np.mean([r['in_band'] for r in records])),
                         sndr=float(np.mean([r['spectral_ratio_db'] for r in records])),
                         sfdr=float(np.mean([r['spur_ratio_db'] for r in records])))
            for k in BANDS:
                errors = np.array([r['used_'+k]-r['true_'+k] for r in records])
                entry[k+'_bias'] = float(errors.mean())
                entry[k+'_rmse'] = float(np.sqrt(np.mean(errors**2)))
            gain_error = np.array([r['used_gain']-r['true_gain'] for r in records])
            beta_error = np.array([r['used_beta']-r['true_beta'] for r in records])
            entry['gain_beta_error_correlation'] = (float(np.corrcoef(gain_error, beta_error)[0, 1])
                if np.std(gain_error) > 1e-15 and np.std(beta_error) > 1e-15 else 0.)
            result.append(entry)
    return result


def long_record(seed, blocks=1024, size=32768):
    """Same input/probe streams: noisy joint, clean joint, clean H=delta=0.

    Only inverse moment statistics are retained; fixed-c3 is explicitly an
    oracle diagnostic and is never used for corrected-output results.
    """
    plants = dict(noisy_joint=jm.JointPlant(seed, .2),
                  clean_joint=jm.JointPlant(seed, .2, bits=24, noise=0),
                  clean_no_memory_cap=jm.JointPlant(seed, .2, bits=24, noise=0))
    arrays = {key: [] for key in plants}
    for block in range(blocks):
        x, _ = previous.waveform(np.arange(block*size, (block+1)*size), size, 'sine')
        for label, plant in plants.items():
            truth = jm.Parameters(jm.FINAL.gain, 0., 0., jm.FINAL.beta) if label.endswith('no_memory_cap') else jm.FINAL
            c, d = plant.convert(x, truth)
            if d['overloads']:
                raise RuntimeError('Long-record diagnostic clipped')
            a, b, _, _ = jc.grouped_moments(c, .2, jm.NOMINAL.inverse())
            arrays[label].append(np.r_[a.sum(axis=0).ravel(), b.sum(axis=0)])
    return {k: np.array(v) for k, v in arrays.items()}


def summarize_long(arrays, seed, size):
    rows = []
    for label, stats in arrays.items():
        for method, rate in [('ema04', .04), ('ema01', .01), ('cumulative', None), ('fixed_c3_oracle', .04)]:
            a, b = np.zeros((3, 3)), np.zeros(3)
            estimates = []
            cosines, conditions = [], []
            algebra_gap = 0.
            for i, stat in enumerate(stats):
                w = 1/(i+1) if rate is None or i == 0 else rate
                a += w*(stat[:9].reshape(3, 3)-a)
                b += w*(stat[9:]-b)
                # Orthogonalizing/rescaling an exactly identified system must
                # change numerical conditioning, not its statistical answer.
                q, triangular = np.linalg.qr(a)
                theta = np.linalg.solve(a, b)
                reparameterized = np.linalg.solve(triangular, q.T@b)
                algebra_gap = max(algebra_gap, float(np.max(abs(theta-reparameterized))))
                if method == 'fixed_c3_oracle':
                    c3 = jm.FINAL.inverse()[1]
                    c1, h = np.linalg.lstsq(a[:, [0, 2]], b-a[:, 1]*c3, rcond=None)[0]
                    theta = np.array([c1, c3, h])
                estimates.append(jm.physical(np.r_[theta, 0.]))
                scaled = a/np.linalg.norm(a, axis=0)
                cosines.append(float(scaled[:, 0] @ scaled[:, 1]))
                conditions.append(float(np.linalg.cond(scaled)))
                count = i+1
                if count not in (64, 128, 256, 512, 1024, len(stats)):
                    continue
                tail = estimates[3*count//4:]
                row = dict(seed=seed, plant=label, method=method, blocks=count, samples=count*size,
                           column_cosine=float(np.mean(cosines[3*count//4:])),
                           scaled_condition=float(np.mean(conditions[3*count//4:])),
                           max_basis_change_gap=algebra_gap)
                for k in ('gain', 'beta', 'history'):
                    truth = 0. if k == 'history' and label.endswith('no_memory_cap') else getattr(jm.FINAL, k)
                    error = np.array([t[k]-truth for t in tail])
                    row[k+'_rmse'] = float(np.sqrt(np.mean(error**2)))
                    row[k+'_bias'] = float(error.mean())
                    row[k+'_endpoint_error'] = estimates[-1][k]-truth
                rows.append(row)
    return rows


def balanced_controls():
    """Exact Cartesian excitation removes finite PN/input correlations.

    Symmetric two-point backend noise integrates the cubic noise moments
    exactly. It is a bias control, not a substitute for the Gaussian runs.
    """
    mesh = np.meshgrid(np.linspace(-.8, .8, 101), [-1., -1/3, 1/3, 1.],
                       [-1., -1/3, 1/3, 1.], [-1., 1.], [-1., 1.], [-1., 1.], indexing='ij')
    x, p, pp, s, sp, noise_sign = [v.ravel() for v in mesh]
    xp = .6-.2*x
    q = np.where(x > .25, 1., np.where(x < -.25, -1., 0.))
    qp = np.where(xp > .25, 1., np.where(xp < -.25, -1., 0.))
    truth = jm.FINAL
    v = (2*x-(1+truth.delta*s)*q)/(2*(1-truth.delta*s))+.2*p
    vp = (2*xp-(1+truth.delta*sp)*qp)/(2*(1-truth.delta*sp))+.2*pp
    r = jm.compressed_amplifier(truth.gain*v+truth.history*vp, 1., truth.beta)
    rows = []
    for sigma, bits in [(0., None), (0., 13), (.0002, 13), (.02, None)]:
        measured = r+sigma*noise_sign
        if bits is not None:
            measured, rail, _ = jm.quantize(measured, bits)
            if rail:
                raise RuntimeError('Balanced control clipped')
        c = dict(r=measured, p=p, pp=pp, q=q, qp=qp, s=s, rail=False)
        for grouped in (False, True):
            tracker = jc.ConvergenceTracker(grouped=grouped)
            tracker.update(c)
            estimate = jm.physical(tracker.coefficients)
            rows.append(dict(noise_sigma=sigma, bits=bits or 'unquantized', grouped=grouped,
                             **{k+'_error': estimate[k]-getattr(truth, k) for k in ('gain', 'beta', 'history')},
                             c1_error=float(tracker.coefficients[0]-truth.inverse()[0]),
                             c1_noise_prediction=float(-3*sigma**2*truth.inverse()[1])))
    return rows


def plot(rows, long_rows, output, seed, size):
    selected = [r for r in rows if r['seed'] == seed and r['waveform'] == 'sine' and r['scenario'] == 'step']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    colors = dict(pooled_04='#d47720', pooled_01='#94999e', grouped_04='#147caa', grouped_01='#148a58')
    for method, color in colors.items():
        record = [r for r in selected if r['method'] == method]
        t = [(r['block']+1)*size/previous.FS*1000 for r in record]
        for ax, key in zip(axes.flat, ('used_gain', 'used_beta', 'spectral_ratio_db', 'spur_ratio_db')):
            ax.plot(t, [r[key] for r in record], label=method, color=color, lw=1)
    for ax, k in zip(axes[0], ('gain', 'beta')):
        record = [r for r in selected if r['method'] == 'oracle']
        truth = np.array([r['true_'+k] for r in record])
        ax.plot(t, truth, color='black', lw=1)
        ax.fill_between(t, truth*(1-BANDS[k]), truth*(1+BANDS[k]), color='black', alpha=.07)
        ax.set_ylabel(k)
    axes[1, 0].set_ylabel('SNDR (dB)'); axes[1, 1].set_ylabel('SFDR (dB)')
    for ax in axes.flat:
        ax.set_xlabel('Time (ms)'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Fixed probe peak 0.2; held-out seed {seed}')
    fig.savefig(output/'fixed_probe_convergence.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for method in ('ema04', 'ema01', 'cumulative', 'fixed_c3_oracle'):
        entries = [r for r in long_rows if r['plant'] == 'noisy_joint' and r['method'] == method]
        counts = sorted({r['blocks'] for r in entries})
        for ax, k in zip(axes, ('gain', 'beta')):
            values = [np.sqrt(np.mean([r[k+'_rmse']**2 for r in entries if r['blocks'] == n])) for n in counts]
            ax.loglog(np.array(counts)*size, values, 'o-', label=method)
            ax.set_xlabel('Acquired samples'); ax.set_ylabel(k+' RMSE, last quarter'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output/'sample_budget.png', dpi=160); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=256)
    parser.add_argument('--long-blocks', type=int, default=1024)
    parser.add_argument('--block-size', type=int, default=32768)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101, 202, 303])
    args = parser.parse_args(argv)
    if min(args.blocks, args.long_blocks) < 8 or args.block_size < 256:
        parser.error('Require at least 8 blocks and block-size >= 256')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(jc.__file__), Path(jm.__file__), Path(previous.__file__),
               Path(jm.__file__).with_name('dither_mechanisms.py')]
    config = dict(blocks=args.blocks, long_blocks=args.long_blocks, block_size=args.block_size,
                  seeds=args.seeds, amplitude=.2, peak=.8, sample_rate=previous.FS,
                  methods=list(METHODS), bands=BANDS, pilot_seed=44,
                  parameter_policy='constant FINAL; step INITIAL to FINAL at midpoint; no estimator reset',
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    previous.write_json(args.output/'configuration.json', config)
    rows, long_rows = [], []
    for seed in args.seeds:
        for kind in ('sine', 'multitone'):
            for scenario in ('constant', 'step'):
                records = run_case(seed, kind, scenario, args.blocks, args.block_size)
                previous.save_csv(args.output/f'{seed}_{kind}_{scenario}.csv', records)
                rows.extend(records)
                print(f'completed {seed} {kind} {scenario}', flush=True)
        arrays = long_record(seed, args.long_blocks, args.block_size)
        np.savez_compressed(args.output/f'{seed}_long_moments.npz', **arrays)
        long_rows.extend(summarize_long(arrays, seed, args.block_size))
        print(f'completed {seed} long-record diagnostics', flush=True)
    summary = summarize(rows, args.blocks)
    previous.save_csv(args.output/'summary.csv', summary)
    previous.save_csv(args.output/'long_summary.csv', long_rows)
    previous.save_csv(args.output/'balanced_controls.csv', balanced_controls())
    plot(rows, long_rows, args.output, args.seeds[0], args.block_size)
    previous.write_json(args.output/'audit.json', dict(
        main_runs=4*len(args.seeds), main_samples=4*len(args.seeds)*args.blocks*args.block_size,
        diagnostic_samples=3*len(args.seeds)*args.long_blocks*args.block_size,
        minimum_headroom=min(r['headroom'] for r in rows),
        overloads=sum(r['overloads'] for r in rows if r['method'] == 'nominal'),
        max_causality_gap=max(r['causality_gap'] for r in rows),
        finite_numeric_results=all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float))),
        all_outputs_use_start_of_block_coefficients=True))
    print(f'finished {args.output}', flush=True)


if __name__ == '__main__':
    main()
