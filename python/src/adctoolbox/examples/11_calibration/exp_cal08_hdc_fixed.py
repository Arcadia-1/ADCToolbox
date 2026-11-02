"""Three-PN HDC hardware arithmetic experiment; see HDC_FIXED.md.

Run with optional numba for full-length experiments. Output directories are
immutable. Oracle paths are evaluation-only and never feed any estimator.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from dataclasses import asdict
import hashlib
from adctoolbox.calibration import _dither_hdc
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from .hdc_fixed import Plant, Tracker, INITIAL, FINAL, resource_budget
    from .exp_cal06_joint_dither import waveform, metrics, save_csv, write_json, FS
except ImportError:
    from hdc_fixed import Plant, Tracker, INITIAL, FINAL, resource_budget
    from exp_cal06_joint_dither import waveform, metrics, save_csv, write_json, FS

PARAMETERS = ('gain', 'cubic', 'history', 'delta')
BANDS = dict(gain=.01, cubic=.10, history=.10, delta=.10)


def oracle(c, truth, exact=False, dither_peak=.2):
    """Evaluation only, 64-sample warm-up before the measured FFT segment."""
    g, k, h = truth.gain/2, truth.cubic, truth.history/truth.gain
    linear = c['r']/g
    if exact:
        w = linear.copy()
        for _ in range(8):
            w -= (g*w+k*w**3-c['r'])/(g+3*k*w*w)
    else:
        w = linear-(k/g)*linear**3
    v = lfilter([1], [1, h], w)
    return c['q']/2+v-dither_peak/3*c['p'].sum(axis=1)+truth.delta/2*c['s']*c['q']


def run_case(output, seed, kind, total_log2, window_log2, formats=False,
             scenario='step', device=None, peak=.8, dither_peak=.2,
             backend_bits=13, noise=.0002):
    size, fft_size = 65536, 32768
    blocks = 2**total_log2//size
    plant = Plant(seed, peak=dither_peak, bits=backend_bits, noise=noise)
    short_log2 = min(20, window_log2)
    trackers = dict(float=Tracker(False, window_log2=window_log2, dither_peak=dither_peak),
                    fixed24=Tracker(window_log2=window_log2, dither_peak=dither_peak),
                    short24=Tracker(window_log2=short_log2, dither_peak=dither_peak))
    if formats:
        trackers.update({f'fixed{f}':Tracker(fraction=f, bits=f+4, window_log2=window_log2, dither_peak=dither_peak)
                         for f in (16, 20)})
    clean, bins = waveform(np.arange(size), fft_size, kind, peak)
    rows = []
    diag = dict(minimum_headroom=1., overloads=0, maximum_probe_peak=0.,
                maximum_internal_peak=0., minimum_slope=1.)
    t0 = time.monotonic()
    for block in range(blocks):
        truth = device if device is not None else (FINAL if scenario == 'constant' or block >= blocks//2 else INITIAL)
        c, d = plant.convert(clean, truth)
        diag['minimum_headroom'] = min(diag['minimum_headroom'], d['headroom'])
        diag['overloads'] += d['overloads']
        diag['maximum_probe_peak'] = max(diag['maximum_probe_peak'], d['probe_peak'])
        diag['maximum_internal_peak'] = max(diag['maximum_internal_peak'], d['internal_peak'])
        diag['minimum_slope'] = min(diag['minimum_slope'], d['minimum_slope'])
        evaluate = block % 16 == 15 or block == blocks-1
        row = dict(block=block, sample=(block+1)*size, time_ms=(block+1)*size/FS*1000,
                   **{'true_'+k:v for k,v in asdict(truth).items()})
        for name, tracker in trackers.items():
            y = tracker.process(c)
            if evaluate:
                row.update({name+'_'+k:v for k,v in metrics(y[-fft_size:], clean[-fft_size:], bins).items()})
                row.update({name+'_'+k:v for k,v in tracker.physical().items()})
        if evaluate:
            nominal = c['q']/2+c['r']-dither_peak/3*c['p'].sum(axis=1)
            tail = {k:(v[-fft_size-64:] if isinstance(v, np.ndarray) else v) for k,v in c.items()}
            for name, y in [('nominal',nominal), ('oracle_hdc',oracle(tail, truth, dither_peak=dither_peak)),
                            ('oracle_exact',oracle(tail, truth, True, dither_peak=dither_peak))]:
                row.update({name+'_'+k:v for k,v in metrics(y[-fft_size:],clean[-fft_size:],bins).items()})
            rows.append(row)
    final = [r for r in rows if r['sample'] > 3*2**total_log2//4]
    summaries = {}
    for name in list(trackers)+['nominal','oracle_hdc','oracle_exact']:
        item = {m:float(np.mean([r[name+'_'+m] for r in final]))
                for m in ('spectral_ratio_db','spur_ratio_db','error_snr_db')}
        if name in trackers:
            tracker = trackers[name]
            item.update(parameter_mean={k:float(np.mean([r[name+'_'+k] for r in final])) for k in PARAMETERS},
                        parameter_rmse={k:float(np.sqrt(np.mean([(r[name+'_'+k]-r['true_'+k])**2 for r in final]))) for k in PARAMETERS},
                        parameter_relative_rmse={k:float(np.sqrt(np.mean([((r[name+'_'+k]-r['true_'+k])/r['true_'+k])**2 for r in final])))
                            for k in PARAMETERS if final[0]['true_'+k] != 0},
                        in_band_fraction=float(np.mean([all(abs(r[name+'_'+k]-r['true_'+k]) <= BANDS[k]*abs(r['true_'+k])
                            for k in PARAMETERS) for r in final])),
                        accepted=tracker.updates, held=tracker.held,
                        saturations=tracker.saturations, accumulator_saturations=tracker.accumulator_saturations,
                        controller_saturations=tracker.controller_saturations)
        summaries[name] = item
    result = dict(seed=seed, kind=kind, scenario=scenario, total_log2=total_log2,
                  window_log2=window_log2, peak=peak, dither_peak=dither_peak,
                  backend_bits=backend_bits, noise=noise, diagnostics=diag, methods=summaries,
                  elapsed_seconds=time.monotonic()-t0)
    output.mkdir()
    save_csv(output/'trajectory.csv', rows)
    write_json(output/'summary.json', result)
    write_json(output/'updates.json', {k:v.events for k,v in trackers.items()})
    return rows, result


def plot(rows, output):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    t = [r['time_ms'] for r in rows]
    for ax, key in zip(axes.flat[:4], PARAMETERS):
        ax.plot(t, [r['true_'+key] for r in rows], 'k--', label='Truth (evaluation only)')
        for name in ('short24','float','fixed24'):
            ax.plot(t, [r[name+'_'+key] for r in rows], label=name, alpha=.8)
        ax.set(title=key, xlabel='Time at 100 MS/s (ms)'); ax.grid(alpha=.2)
    for ax, key, title in zip(axes.flat[4:], ('spectral_ratio_db','spur_ratio_db'), ('SNDR (dB)','SFDR (dB)')):
        for name in ('nominal','short24','fixed24','oracle_hdc','oracle_exact'):
            ax.plot(t, [r[name+'_'+key] for r in rows], label=name)
        ax.set(title=title, xlabel='Time at 100 MS/s (ms)'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); axes[2,0].legend(fontsize=8)
    fig.savefig(output, dpi=150); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--total-log2', type=int, default=27)
    parser.add_argument('--window-log2', type=int, default=24)
    parser.add_argument('--seeds', type=int, nargs='+', default=[101,202,303])
    parser.add_argument('--kinds', nargs='+', choices=['sine','multitone'], default=['sine','multitone'])
    parser.add_argument('--scenario', choices=['step','constant'], default='step')
    parser.add_argument('--dither-peak', type=float, default=.2)
    parser.add_argument('--input-peak', type=float, default=.8)
    parser.add_argument('--backend-bits', type=int, default=13)
    parser.add_argument('--noise-rms', type=float, default=.0002)
    args = parser.parse_args(argv)
    if not 20 <= args.total_log2 <= 30 or not 16 <= args.window_log2 <= args.total_log2-2:
        parser.error('Require 20 <= total-log2 <= 30 and 16 <= window-log2 <= total-log2-2')
    if not np.isfinite(args.input_peak) or not 0 < args.input_peak < 1:
        parser.error('input-peak must be finite and in (0,1)')
    if not 2 <= args.backend_bits <= 24 or not np.isfinite(args.noise_rms) or args.noise_rms < 0:
        parser.error('Require backend-bits in [2,24] and finite nonnegative noise-rms')
    try:
        # Every run includes the first-case format controls, including F=16.
        Tracker(fraction=16, bits=20, dither_peak=args.dither_peak)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    sources = ['hdc_fixed.py','exp_cal08_hdc_fixed.py','exp_cal06_joint_dither.py','dither_mechanisms.py']
    source_paths = {name: here/name for name in sources}
    source_paths['_dither_hdc.py'] = Path(_dither_hdc.__file__)
    write_json(args.output/'config.json', dict(arguments={**vars(args),'output':str(args.output)},
        sources={name:hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()},
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        initial=asdict(INITIAL), final=asdict(FINAL), physical_dither_peak=args.dither_peak, pn_lengths=[31,29,23],
        cap_switch_length=27, fraction=24, word_bits=28, fft_size=32768,
        noise_rms_backend=args.noise_rms, backend_bits=args.backend_bits, controller_latency_samples=512,
        bands=BANDS, final_evaluation='last quarter, sample active coefficients; arithmetic mean of per-FFT dB metrics',
        multitone_metrics='desired-tone power / other power; strongest desired tone / largest unwanted bin, not single-tone SNDR/SFDR'))
    write_json(args.output/'resources.json', resource_budget(window_log2=args.window_log2))
    all_summaries, table = [], []
    for kind in args.kinds:
        for seed in args.seeds:
            rows, result = run_case(args.output/f'{kind}_{seed}', seed, kind, args.total_log2,
                                   args.window_log2, formats=(kind == args.kinds[0] and seed == args.seeds[0]), scenario=args.scenario,
                                   peak=args.input_peak, dither_peak=args.dither_peak,
                                   backend_bits=args.backend_bits, noise=args.noise_rms)
            all_summaries.append(result)
            for method, item in result['methods'].items():
                table.append(dict(kind=kind, seed=seed, method=method,
                    **{k:item[k] for k in ('spectral_ratio_db','spur_ratio_db','error_snr_db')},
                    **{k:result['diagnostics'][k] for k in ('minimum_headroom','overloads')}))
            print(kind, seed, 'fixed24', result['methods']['fixed24'], flush=True)
            if kind == args.kinds[0] and seed == args.seeds[0]:
                plot(rows, args.output/'convergence.png')
    write_json(args.output/'summary.json', all_summaries)
    save_csv(args.output/'performance.csv', table)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
