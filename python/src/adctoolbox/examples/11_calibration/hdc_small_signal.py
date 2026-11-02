"""DC/small-signal HDC acceptance controls; this does not implement 2013 RD.

Balanced enumeration isolates quantization-correlated bias. Streaming tests
use observable-only Trackers, independent seeds and identical captures for
continued adaptation, freezing, shorter windows and nominal startup.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
from itertools import product
from pathlib import Path
import time
import numpy as np
try:
    from .hdc_fixed import Plant, Tracker, FINAL
    from .exp_cal08_hdc_fixed import waveform, metrics, save_csv, write_json
except ImportError:
    from hdc_fixed import Plant, Tracker, FINAL
    from exp_cal08_hdc_fixed import waveform, metrics, save_csv, write_json


CASES = [('dc_zero', 0., 0.), ('tiny_inner', .137, 1e-5),
         ('small_inner', .137, .001), ('tiny_outer', .3, 1e-5)]


def balanced_scan(offsets=None, amplitudes=None, phase_count=256, dither_peak=.2):
    """Exact PN expectation, uniform phase quadrature, no memory/DAC error.

    No random-noise assumption is made for the quantizer. Repeating this
    Cartesian table changes neither the means nor the coefficient bias.
    """
    if offsets is None:
        offsets = np.unique(np.r_[np.linspace(-.24, .24, 513), -.6, -.3, -.251, -.249,
                                 .137, .249, .251, .3, .6])
    if amplitudes is None: amplitudes = [0., 1e-6, 1e-5, 1e-4, .001, .01, .05]
    p = np.array(list(product((-1, 1), repeat=3)))
    phase = np.arange(phase_count)*2*np.pi/phase_count
    a = dither_peak/3
    g, k = FINAL.gain/2, FINAL.cubic
    rows = []
    for offset, amplitude in product(offsets, amplitudes):
        x = offset+amplitude*np.sin(phase)
        q = np.where(x > .25, 1, np.where(x < -.25, -1, 0))
        z = (x-q/2)[:, None]
        v = z+a*p.sum(axis=1)
        exact = g*v+k*v**3
        for bits in (0, 13, 24):
            r = exact if bits == 0 else (np.clip(np.floor((2*exact+1)*2**(bits-1)),
                 0, 2**bits-1)+.5)/2**bits-.5
            e, ss = r-exact, r-a*p.sum(axis=1)
            gamma3 = float(np.mean(ss*np.prod(p, axis=1))/(6*a**3))
            gamma1 = float(np.mean(ss*p[:, 0])/a)
            eta = float(np.mean(ss*ss))
            gain = 1+gamma1-(7*a*a+3*eta)*gamma3
            rows.append(dict(offset=float(offset), amplitude=float(amplitude), backend_bits=bits,
                estimated_gain=2*gain, estimated_cubic=gamma3,
                gain_error=2*gain-FINAL.gain, cubic_error=gamma3-k,
                quantization_cubic_bias=float(np.mean(e*np.prod(p, axis=1))/(6*a**3)),
                eta_gain_bias=6*k*float(np.mean(z*z)-eta),
                headroom=float(1-np.max(abs(2*exact))),
                cap_occupancy=float(np.mean(q*q))))
    return rows


def branch_tracker(source, window_log2=None):
    """Snapshot ACTIVE coefficients and filter state at the phase boundary.

    Discard uncommitted estimates and old-input moments. This makes freezing
    immediate and prevents a mixed large/small-input acquisition window.
    """
    target = copy.deepcopy(source)
    if window_log2 is not None:
        target.window_log2, target.window = window_log2, 1 << window_log2
    target.pending = target.commit_at = None
    target.sums[:] = 0
    target.count = target.samples = target.updates = target.held = 0
    target.saturations = target.accumulator_saturations = target.controller_saturations = 0
    target.window_bad = False
    target.events = []
    return target


def score(y, clean, bins):
    row = dict(mean_error=float(np.mean(y-clean)), error_rms=float(np.sqrt(np.mean((y-clean)**2))),
               ac_error_rms=float(np.std(y-clean)), sndr_db=None, sfdr_db=None)
    if bins:
        measured = metrics(y, clean, bins)
        row.update(sndr_db=measured['spectral_ratio_db'], sfdr_db=measured['spur_ratio_db'])
    return row


def train(seed, noise, train_log2, window_log2, dither_peak):
    plant = Plant(seed, noise=noise, peak=dither_peak)
    trackers = {name:Tracker(fixed=fixed, window_log2=window_log2, dither_peak=dither_peak)
                for name, fixed in [('float', False), ('fixed', True)]}
    size = 65536
    x, _ = waveform(np.arange(size), 32768, 'sine', .78)
    headroom, overloads = 1., 0
    for _ in range((1 << train_log2)//size):
        capture, diag = plant.convert(x, FINAL)
        headroom, overloads = min(headroom, diag['headroom']), overloads+diag['overloads']
        for tracker in trackers.values(): tracker.process(capture)
    # Finish the controller delay on the same large signal, then snapshot.
    capture, diag = plant.convert(x[:1024], FINAL)
    for tracker in trackers.values(): tracker.process(capture, learn=False)
    headroom, overloads = min(headroom, diag['headroom']), overloads+diag['overloads']
    result = dict(seed=seed, noise=noise, samples=(1 << train_log2)+1024,
        headroom=headroom, overloads=overloads,
        methods={name:dict(parameters=t.physical(), accepted=t.updates, held=t.held,
            active=t.values().tolist(), saturations=t.saturations,
            accumulator_saturations=t.accumulator_saturations,
            controller_saturations=t.controller_saturations) for name,t in trackers.items()})
    return plant, trackers, result


def stream_case(plant_source, trained, name, offset, amplitude, test_log2, window_log2,
                short_log2, seed, noise, dither_peak):
    plant = copy.deepcopy(plant_source)
    trackers = dict(continue_fixed=branch_tracker(trained['fixed']),
                    continue_float=branch_tracker(trained['float']),
                    short_fixed=branch_tracker(trained['fixed'], short_log2),
                    frozen_fixed=branch_tracker(trained['fixed']),
                    cold_fixed=Tracker(window_log2=window_log2, dither_peak=dither_peak))
    initial_frozen = trackers['frozen_fixed'].values().copy()
    size, fft_size = 65536, 32768
    wave, bins = waveform(np.arange(size), fft_size, 'sine', amplitude)
    clean = offset+wave
    if amplitude == 0: bins = []
    rows, headroom, overloads = [], 1., 0
    # Include active-coefficient trajectories at shorter-window cadence.
    cadence = 1 << max(0, short_log2-16)
    for block in range((1 << test_log2)//size):
        capture, diag = plant.convert(clean, FINAL)
        headroom, overloads = min(headroom, diag['headroom']), overloads+diag['overloads']
        for method, tracker in trackers.items():
            y = tracker.process(capture, learn=method != 'frozen_fixed')
            if (block+1) % cadence == 0:
                rows.append(dict(case=name, seed=seed, noise=noise, method=method,
                    sample=(block+1)*size, offset=offset, amplitude=amplitude,
                    **tracker.physical(), **score(y[-fft_size:], clean[-fft_size:], bins),
                    accepted=tracker.updates, held=tracker.held,
                    saturations=tracker.saturations, accumulator_saturations=tracker.accumulator_saturations,
                    controller_saturations=tracker.controller_saturations))
    np.testing.assert_array_equal(trackers['frozen_fixed'].values(), initial_frozen)
    # Independent high-amplitude/frequency test reveals bad small-input learning;
    # it is evaluation-only and never feeds coefficients back into the stream.
    hold_x, hold_bins = waveform(np.arange(size), fft_size, 'sine', .65)
    hold_x = .65*np.sin(2*np.pi*5669*np.arange(size)/fft_size+.91)
    hold_bins = [5669]
    hold_capture, hold_diag = Plant(seed+5000, noise=noise, peak=dither_peak).convert(hold_x, FINAL)
    methods = {}
    for method, tracker in trackers.items():
        evaluator = branch_tracker(tracker)
        evaluator.previous = 0 if evaluator.fixed else 0.
        y = evaluator.process(hold_capture, learn=False)
        last = next(r for r in reversed(rows) if r['method'] == method)
        methods[method] = dict(parameters=tracker.physical(), small_signal=last,
            heldout_large_signal=score(y[-fft_size:], hold_x[-fft_size:], hold_bins),
            final_shadow=tracker.events[-1] if tracker.events else None)
    return rows, dict(case=name, seed=seed, noise=noise, offset=offset, amplitude=amplitude,
        test_samples=1 << test_log2, minimum_headroom=headroom, overloads=overloads,
        heldout_headroom=hold_diag['headroom'], heldout_overloads=hold_diag['overloads'],
        probe_peak=dither_peak, methods=methods)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[111, 222, 333])
    parser.add_argument('--noise-rms', nargs='+', type=float, default=[0., .0002])
    parser.add_argument('--dither-peak', type=float, default=.2)
    parser.add_argument('--train-log2', type=int, default=27)
    parser.add_argument('--test-log2', type=int, default=26)
    parser.add_argument('--window-log2', type=int, default=24)
    parser.add_argument('--short-log2', type=int, default=22)
    args = parser.parse_args(argv)
    if not 16 <= args.short_log2 < args.window_log2 <= min(args.train_log2, args.test_log2)-2 <= 28:
        parser.error('Require 16 <= short < window <= min(train,test)-2 <= 28')
    if any(not np.isfinite(n) or n < 0 for n in args.noise_rms): parser.error('noise-rms must be finite and nonnegative')
    try: Tracker(dither_peak=args.dither_peak)
    except ValueError as error: parser.error(str(error))
    if args.output.exists() and any(args.output.iterdir()): parser.error('Output directory is not empty')
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['hdc_small_signal.py','hdc_fixed.py','exp_cal08_hdc_fixed.py',
             'exp_cal06_joint_dither.py','dither_mechanisms.py']
    source = {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in names}
    write_json(args.output/'configuration.json', dict(arguments={**vars(args),'output':str(args.output)},
        sources=source, truth=asdict(FINAL), backend_bits=13, fraction=24, bits=28,
        cases=CASES, controller_latency=512, phase_count=256,
        balanced_scope='No memory/DAC/noise; phase quadrature and exact PN expectation',
        streaming_scope='Joint gain/cubic/memory/DAC, three PN peak unchanged; active final estimates, not best-window selection'))
    start = time.monotonic()
    save_csv(args.output/'balanced_scan.csv', balanced_scan(dither_peak=args.dither_peak))
    all_rows, results, training = [], [], []
    for seed, noise in product(args.seeds, args.noise_rms):
        plant, trained, summary = train(seed, noise, args.train_log2, args.window_log2, args.dither_peak)
        training.append(summary)
        print('trained', seed, noise, summary['methods']['fixed']['parameters'], flush=True)
        for name, offset, amplitude in CASES:
            rows, summary = stream_case(plant, trained, name, offset, amplitude,
                args.test_log2, args.window_log2, args.short_log2, seed, noise, args.dither_peak)
            all_rows.extend(rows); results.append(summary)
            print('completed', seed, noise, name, flush=True)
    save_csv(args.output/'trajectories.csv', all_rows)
    write_json(args.output/'training.json', training)
    write_json(args.output/'summary.json', results)
    write_json(args.output/'completion.json', dict(elapsed_seconds=time.monotonic()-start,
        cases=len(results), source_hashes_match=all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in source.items())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
