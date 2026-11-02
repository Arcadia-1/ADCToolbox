"""Joint gain/memory/capacitor/nonlinearity background calibration experiment.

Run directly from a copied examples tree. Uses only NumPy/SciPy/Matplotlib.
Never overwrites a nonempty output directory. See JOINT_DITHER.md.
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

try:
    from . import joint_dither_model as jm
    from . import dither_mechanisms as dm
except ImportError:
    import joint_dither_model as jm
    import dither_mechanisms as dm

FS = 100e6
METRICS = ('spectral_ratio_db', 'spur_ratio_db', 'error_snr_db')
METHODS = ('nominal', 'joint', 'unconditioned', 'gain_only', 'without_memory', 'without_capacitor',
           'without_nonlinearity', 'frozen', 'oracle', 'nominal_probe_off', 'oracle_probe_off')
# Evaluation only; fixed before the acceptance run. No truth enters training.
RELATIVE_BANDS = dict(gain=.01, history=.10, delta=.10, beta=.10)


def metrics(signal, clean, bins):
    """Coherent real FFT: SNDR/SFDR for sine; desired-tone ratios for multitone."""
    power = abs(np.fft.rfft(signal-signal.mean()))**2
    power[1:-1] *= 2
    desired, strongest = float(power[bins].sum()), float(power[bins].max())
    power[bins], power[0] = 0., 0.
    return dict(spectral_ratio_db=float(10*np.log10(desired/max(power.sum(), 1e-300))),
                spur_ratio_db=float(10*np.log10(strongest/max(power.max(), 1e-300))),
                error_snr_db=float(10*np.log10(np.var(clean)/max(np.mean((signal-clean)**2), 1e-300))))


def waveform(n, size, kind, peak=.8):
    if kind == 'sine':
        bins, amplitudes = [int(round(.137*size)) | 1], [peak]
    else:
        bins = [int(round(.083*size)) | 1, int(round(.217*size)) | 1]
        amplitudes = [peak*.575, peak*.425]
    signal = sum(a*np.sin(2*np.pi*b*n/size+.37+.81*i)
                 for i, (a, b) in enumerate(zip(amplitudes, bins)))
    return signal, bins


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ablations(coefficients):
    """Correction-only ablations using identical estimates; not separate training."""
    result = {name: coefficients.copy() for name in
              ('gain_only', 'without_memory', 'without_capacitor', 'without_nonlinearity')}
    result['gain_only'][1:] = 0.
    result['without_memory'][2] = 0.
    result['without_capacitor'][3] = 0.
    result['without_nonlinearity'][1] = 0.
    return result


def first_streak(rows, start, stop, length=8):
    streak = 0
    for row in rows[start:stop]:
        streak = streak+1 if row['all_parameters_in_band'] else 0
        if streak == length:
            # End of first qualifying streak, measured from segment start.
            return row['block']-start+1
    return None


def run_case(seed=11, kind='sine', scenario='step', blocks=256, size=32768,
             amplitude=.2, peak=.8, rate=.04, output=None):
    plant = jm.JointPlant(seed, amplitude)
    off_plant = jm.JointPlant(seed, amplitude)
    tracker = jm.JointTracker(amplitude, rate, rate)
    unconditioned = jm.JointTracker(amplitude, rate, rate, condition_codes=False)
    correctors = {name: jm.JointCorrector(amplitude) for name in METHODS}
    frozen = jm.NOMINAL.inverse()
    rows = []
    for block in range(blocks):
        clean, bins = waveform(np.arange(block*size, (block+1)*size), size, kind, peak)
        truth = jm.FINAL if scenario == 'step' and block >= blocks//2 else jm.INITIAL
        capture, diag = plant.convert(clean, truth)
        off, off_diag = off_plant.convert(clean, truth, probe_on=False)
        used = tracker.coefficients.copy()
        # Freeze at the midpoint for both scenarios; its filter keeps running.
        if block <= blocks//2:
            frozen = used.copy()
        parameters = dict(nominal=jm.NOMINAL.inverse(), joint=used, unconditioned=unconditioned.coefficients.copy(),
                          **ablations(used), frozen=frozen, oracle=truth.inverse(),
                          nominal_probe_off=jm.NOMINAL.inverse(), oracle_probe_off=truth.inverse())
        row = dict(block=block, time_ms=(block+1)*size/FS*1000,
                   **diag, off_backend_peak=off_diag['backend_peak'],
                   off_headroom=off_diag['headroom'], off_overloads=off_diag['overloads'])
        for method, coefficients in parameters.items():
            c = off if method.endswith('probe_off') else capture
            corrected = correctors[method].apply(c, coefficients)['signal']
            row.update({method+'_'+k: v for k, v in metrics(corrected, clean, bins).items()})
        status = tracker.update(capture)  # Current estimates only affect the NEXT block.
        raw_status = unconditioned.update(capture)
        row.update(electronics_status=status['electronics'], capacitor_status=status['capacitor'],
                   unconditioned_electronics_status=raw_status['electronics'],
                   unconditioned_capacitor_status=raw_status['capacitor'],
                   condition=tracker.condition, cap_denominator=tracker.denominator)
        used_physical, next_physical = jm.physical(used), jm.physical(tracker.coefficients)
        in_band = []
        for key, true_value in asdict(truth).items():
            row['true_'+key], row['used_'+key], row['next_'+key] = true_value, used_physical[key], next_physical[key]
            in_band.append(abs(used_physical[key]-true_value) <= RELATIVE_BANDS[key]*abs(true_value))
        row['all_parameters_in_band'] = bool(all(in_band))
        rows.append(row)
    final = rows[3*blocks//4:]
    summary = dict(seed=seed, waveform=kind, scenario=scenario, blocks=blocks, block_size=size,
        amplitude=amplitude, peak=peak, rate=rate,
        overloads=sum(r['overloads'] for r in rows), off_overloads=sum(r['off_overloads'] for r in rows),
        minimum_headroom=min(r['headroom'] for r in rows), minimum_off_headroom=min(r['off_headroom'] for r in rows),
        electronics_statuses=dict(Counter(r['electronics_status'] for r in rows)),
        capacitor_statuses=dict(Counter(r['capacitor_status'] for r in rows)),
        maximum_condition=max((r['condition'] for r in rows if r['condition'] is not None), default=None),
        final_parameter_means={k:float(np.mean([r['used_'+k] for r in final])) for k in RELATIVE_BANDS},
        final_parameter_rmse={k:float(np.sqrt(np.mean([(r['used_'+k]-r['true_'+k])**2 for r in final]))) for k in RELATIVE_BANDS},
        final_in_band_fraction=float(np.mean([r['all_parameters_in_band'] for r in final])),
        first_initial_eight_block_streak=first_streak(rows, 0, blocks//2),
        first_post_midpoint_eight_block_streak=first_streak(rows, blocks//2, blocks),
        max_causality_gap=max(abs(rows[i]['used_'+k]-rows[i-1]['next_'+k])
                              for i in range(1, blocks) for k in RELATIVE_BANDS),
        windows={window:{method:{metric:float(np.mean([r[method+'_'+metric] for r in records]))
                                 for metric in METRICS} for method in METHODS}
                 for window,records in [('pre_midpoint',rows[blocks//4:blocks//2]),
                                        ('post_midpoint_transient',rows[blocks//2:3*blocks//4]),
                                        ('final_quarter',final)]})
    if output is not None:
        output.mkdir()
        save_csv(output/'trajectory.csv', rows)
        write_json(output/'summary.json', summary)
    return rows, summary


def aggregate(summaries):
    table=[]
    for kind,scenario in sorted({(s['waveform'],s['scenario']) for s in summaries}):
        cases=[s for s in summaries if (s['waveform'],s['scenario']) == (kind,scenario)]
        for window in cases[0]['windows']:
            for method in METHODS:
                row=dict(waveform=kind,scenario=scenario,window=window,method=method,seeds=len(cases))
                for metric in METRICS:
                    values=[s['windows'][window][method][metric] for s in cases]
                    row[metric+'_mean']=float(np.mean(values))
                    row[metric+'_sd']=float(np.std(values,ddof=1)) if len(cases)>1 else 0.
                table.append(row)
    return table


def diagnostics(size=32768):
    """Independent headroom, missing excitation, delay and amplitude controls."""
    x,_=waveform(np.arange(size),size,'sine')
    rows=[]
    for peak in (.1,.8,.95):
        for amplitude in (0.,.02,.05,.1,.2,.4):
            for phase,parameters in [('initial',jm.INITIAL),('final',jm.FINAL)]:
                c,d=jm.JointPlant(17,amplitude).convert(x*peak/.8,parameters)
                tr=jm.JointTracker(max(amplitude,.001))
                status=tr.update(c)
                rows.append(dict(peak=peak,amplitude=amplitude,phase=phase,**d,
                                 electronics=status['electronics'],capacitor=status['capacitor']))
    failure={}
    for label,plant,signal in [
            ('binary_probe',jm.JointPlant(17,binary=True),x),
            ('missing_outer_codes',jm.JointPlant(17),.1*x/.8),
            ('overload',jm.JointPlant(17,amplitude=.4),x)]:
        tr=jm.JointTracker(plant.amplitude)
        statuses=[]
        for _ in range(4):
            c,d=plant.convert(signal,jm.FINAL)
            statuses.append(tr.update(c))
        failure[label]=dict(statuses=statuses,coefficients=tr.coefficients.tolist(),last_diagnostics=d)
    # Several blocks expose bias under invalid injection-scale/alignment assumptions.
    for label,actual,delay in [('valid',.1,False),('scale_error_10pct',.11,False),('wrong_delay',.1,True)]:
        tr=jm.JointTracker(.1)
        plant=jm.JointPlant(17,amplitude=actual)
        for _ in range(32):
            c,_=plant.convert(x,jm.FINAL)
            if delay:
                c={**c,'p':np.roll(c['p'],1),'pp':np.roll(c['pp'],1)}
            status=tr.update(c)
        failure[label]=dict(status=status,estimate=jm.physical(tr.coefficients),truth=asdict(jm.FINAL))
    return rows,failure


def plot_results(traces, headroom_rows, output):
    for (kind,scenario),rows in traces.items():
        if (kind,scenario) != ('sine','step'):
            continue
        fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
        t=[r['time_ms'] for r in rows]
        for ax,key in zip(axes.flat,RELATIVE_BANDS):
            ax.plot(t,[r['true_'+key] for r in rows],color='black',label='Device truth (evaluation)')
            ax.plot(t,[r['used_'+key] for r in rows],color='#147caa',label='Causal background estimate')
            ax.set_ylabel(key); ax.grid(alpha=.25)
        for method in ('nominal','joint','frozen','oracle'):
            axes[2,0].plot(t,[r[method+'_spectral_ratio_db'] for r in rows],label=method)
        axes[2,0].set_ylabel('SNDR (dB)'); axes[2,0].legend(fontsize=8)
        for method in ('joint','oracle'):
            axes[2,1].plot(t,[r[method+'_spur_ratio_db'] for r in rows],label=method)
        axes[2,1].set_ylabel('SFDR (dB)'); axes[2,1].legend(fontsize=8)
        for ax in axes[2]: ax.set_xlabel('Normal-operation time (ms)'); ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Joint capacitor / gain / memory / nonlinear calibration | first seed\nFour-level injection + independent capacitor switching; no foreground initialization')
        fig.savefig(output/'joint_convergence.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for peak in sorted({r['peak'] for r in headroom_rows}):
        selected=[r for r in headroom_rows if r['peak']==peak and r['phase']=='final']
        ax.plot([r['amplitude'] for r in selected],[r['headroom'] for r in selected],'o-',label=f'Input peak {peak}')
    ax.axhline(0,color='black',linestyle='--')
    ax.set(xlabel='Injected probe peak a (w/2 units)',ylabel='1 - peak |backend analog input|',
           title='Joint model headroom | negative values mean clipping')
    ax.grid(alpha=.25); ax.legend(); fig.savefig(output/'joint_headroom.png',dpi=160); plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output'/'joint_dither')
    parser.add_argument('--blocks',type=int,default=256)
    parser.add_argument('--block-size',type=int,default=32768)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    parser.add_argument('--waveforms',nargs='+',choices=['sine','multitone'],default=['sine','multitone'])
    parser.add_argument('--scenarios',nargs='+',choices=['constant','step'],default=['constant','step'])
    parser.add_argument('--amplitude',type=float,default=.2)
    parser.add_argument('--peak',type=float,default=.8)
    parser.add_argument('--rate',type=float,default=.04)
    args=parser.parse_args(argv)
    if args.blocks<16 or args.blocks%4 or args.block_size<1024 or args.block_size%2:
        parser.error('Require blocks >=16 divisible by 4, even block-size >=1024')
    if not (0 < args.amplitude <= 1 and 0 < args.peak <= 1 and 0 < args.rate <= 1):
        parser.error('amplitude, peak, rate must lie in (0,1]')
    for values in (args.seeds,args.waveforms,args.scenarios):
        if len(values)!=len(set(values)): parser.error('Duplicate selections are not allowed')
    if args.output.exists() and any(args.output.iterdir()): parser.error('Choose a new or empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):
        revision=None
    sources=[Path(__file__),Path(jm.__file__),Path(dm.__file__)]
    config=dict(**{**vars(args),'output':str(args.output.resolve())},fs_hz=FS,
                initial_parameters=asdict(jm.INITIAL),final_parameters=asdict(jm.FINAL),
                backend_bits=13,backend_noise_rms=.0002,relative_parameter_bands=RELATIVE_BANDS,
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                matplotlib=matplotlib.__version__,git_head=revision,
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                note='Source hashes identify actual files, including uncommitted new example code; git_head alone is insufficient.')
    write_json(args.output/'configuration.json',config)
    summaries,traces=[],{}
    for kind in args.waveforms:
        for scenario in args.scenarios:
            for seed in args.seeds:
                rows,summary=run_case(seed,kind,scenario,args.blocks,args.block_size,args.amplitude,args.peak,args.rate,
                                      args.output/f'{kind}_{scenario}_seed{seed}')
                summaries.append(summary)
                if seed==args.seeds[0]: traces[(kind,scenario)]=rows
                final=summary['windows']['final_quarter']['joint']
                print(f'{kind}/{scenario}/seed{seed}: ratio={final["spectral_ratio_db"]:.2f} dB, '
                      f'spur={final["spur_ratio_db"]:.2f} dB, headroom={summary["minimum_headroom"]:.4f}, '
                      f'in-band={summary["final_in_band_fraction"]:.3f}',flush=True)
    table=aggregate(summaries)
    headroom,failures=diagnostics()
    save_csv(args.output/'aggregate.csv',table)
    save_csv(args.output/'headroom.csv',headroom)
    write_json(args.output/'summary.json',dict(runs=summaries,aggregate=table))
    write_json(args.output/'diagnostics.json',failures)
    plot_results(traces,headroom,args.output)
    print(f'Saved {len(summaries)} joint runs to {args.output.resolve()}',flush=True)


if __name__=='__main__':
    main()
