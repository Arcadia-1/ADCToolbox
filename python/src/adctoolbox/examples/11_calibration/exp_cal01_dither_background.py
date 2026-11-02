"""PN background gain calibration: convergence, drift and frozen foreground.

Run from a copied examples tree or directly in the repository:
    python 11_calibration/exp_cal01_dither_background.py
    python 11_calibration/exp_cal01_dither_background.py --blocks 128 --block-size 8192 --seeds 11 --waveforms sine

Outputs JSON/CSV and noninteractive figures to output/dither_background by
default. The default experiment is 3 scenarios x 2 waveforms x 3 seeds.
No external ML package or measured data is required. See the adjacent README.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .pipeline_dither_model import PipelineConfig, DitherPipeline, DitherGainTracker, reconstruct, fit_foreground_gain
except ImportError:
    from pipeline_dither_model import PipelineConfig, DitherPipeline, DitherGainTracker, reconstruct, fit_foreground_gain


METHODS = ("uncalibrated", "foreground_frozen", "background", "oracle",
           "uncalibrated_probe_off", "foreground_frozen_probe_off", "oracle_probe_off")
INITIAL_GAIN = 3.88
FS = 100e6


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def waveform(indices, block_size, kind):
    bins = [int(round(.137*block_size)) | 1] if kind == "sine" else [int(round(.083*block_size)) | 1, int(round(.217*block_size)) | 1]
    amplitudes = [.85] if kind == "sine" else [.5, .35]
    signal = sum(a*np.sin(2*np.pi*k*indices/block_size+.37+.81*i) for i,(a,k) in enumerate(zip(amplitudes,bins)))
    return signal, bins


def gain_trajectory(indices, total_samples, scenario):
    if scenario == "constant":
        return np.full(len(indices), INITIAL_GAIN)
    if scenario == "ramp":
        # Continuous per-sample drift, rather than resetting a static gain at
        # every observation window: -3% -> +3% over the middle half of the run.
        return INITIAL_GAIN + .24*np.clip((indices-total_samples/4)/(total_samples/2),0,1)
    if scenario == "step":
        return np.where(indices < total_samples/2, INITIAL_GAIN, 4.08)
    raise ValueError(scenario)


def metrics(prediction, clean, tone_bins):
    # Known bins are used ONLY by the evaluator. For one tone this is SNDR /
    # SFDR; for multiple tones use total desired-tone power / noise+distortion
    # and strongest desired tone / largest unwanted bin, respectively.
    power = abs(np.fft.rfft(prediction-prediction.mean()))**2
    # One-sided Parseval weights; DC is excluded and Nyquist is not doubled.
    power[1:-1] *= 2
    desired = float(power[tone_bins].sum())
    strongest = float(power[tone_bins].max())
    power[tone_bins] = 0.
    power[0] = 0.
    return {"spectral_ratio_db": float(10*np.log10(desired/max(power.sum(),1e-300))),
            "spur_ratio_db": float(10*np.log10(strongest/max(power.max(),1e-300))),
            "error_snr_db": float(10*np.log10(np.var(clean)/max(np.mean((prediction-clean)**2),1e-300)))}


def foreground_fit(config, seed, block_size, blocks):
    adc = DitherPipeline(config, seed=seed+100000)
    def captures():
        for block in range(blocks):
            n = np.arange(block*block_size,(block+1)*block_size)
            signal,_ = waveform(n,block_size,"sine")
            yield adc.convert(signal, INITIAL_GAIN, probe_enabled=False)
    return fit_foreground_gain(captures(), (int(round(.137*block_size)) | 1)/block_size)


def run_case(config, *, seed, scenario, kind, blocks, block_size, learning_rate,
             foreground_blocks=32, output=None):
    fg = foreground_fit(config,seed,block_size,foreground_blocks)
    tracker = DitherGainTracker(config.probe_amplitude, learning_rate=learning_rate, backend_bits=config.backend_bits)
    plant = DitherPipeline(config,seed)
    probe_off_plant = DitherPipeline(config,seed)
    rows=[]
    for block in range(blocks):
        indices=np.arange(block*block_size,(block+1)*block_size)
        clean,bins=waveform(indices,block_size,kind)
        true_gain=gain_trajectory(indices,blocks*block_size,scenario)
        capture=plant.convert(clean,true_gain)
        off=probe_off_plant.convert(clean,true_gain,probe_enabled=False)
        q,r,pn=capture['coarse'],capture['backend'],capture['pn']
        # Causal: the present block uses the estimate available at its start.
        gain_used=tracker.gain_estimate
        predicted={name:reconstruct(q,r,pn,g,config.probe_amplitude) for name,g in
                   (("uncalibrated",4.),("foreground_frozen",fg['gain_estimate']),("background",gain_used))}
        # Evaluator-only oracle. It knows instantaneous gain even during ramp.
        predicted['oracle']=q+r/true_gain-config.probe_amplitude*pn
        for name,g in (("uncalibrated_probe_off",4.),("foreground_frozen_probe_off",fg['gain_estimate'])):
            predicted[name]=reconstruct(off['coarse'],off['backend'],off['pn'],g,config.probe_amplitude)
        predicted['oracle_probe_off']=off['coarse']+off['backend']/true_gain
        update=tracker.update(r,pn,capture['backend_code'])
        row={"block":block,"sample_end":int(indices[-1]+1),"time_ms":float((indices[-1]+1)/FS*1e3),
             "true_gain_mean":float(true_gain.mean()),"background_gain_used":gain_used,
             "background_gain_next":update['gain_next'],"observed_gain":update['observed_gain'],
             "update_status":update['status'],"foreground_gain":fg['gain_estimate'],
             "relative_gain_error_rms":float(np.sqrt(np.mean((gain_used/true_gain-1)**2))),
             "input_overloads":capture['input_overloads'],"backend_overloads":capture['backend_overloads'],
             "probe_off_backend_overloads":off['backend_overloads']}
        for name,prediction in predicted.items():
            row.update({name+'_'+key:value for key,value in metrics(prediction,clean,bins).items()})
        rows.append(row)
    summary=summarize_case(rows,scenario,block_size)
    result={"scenario":scenario,"waveform":kind,"seed":seed,"foreground":fg,"summary":summary}
    if output:
        output.mkdir(parents=True,exist_ok=True)
        with (output/'trajectory.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        write_json(output/'summary.json',result)
    return rows,result


def summarize_case(rows,scenario,block_size):
    n=len(rows)
    windows={"final_quarter":rows[3*n//4:],"initial_quarter":rows[:n//4]}
    if scenario=='ramp':windows['active_ramp']=rows[n//4:3*n//4]
    if scenario=='step':windows['first_quarter_after_step']=rows[n//2:3*n//4]
    result={"windows":{}}
    for window,records in windows.items():
        result['windows'][window]={name:{key:float(np.mean([r[name+'_'+key] for r in records]))
                                       for key in ('spectral_ratio_db','spur_ratio_db','error_snr_db')} for name in METHODS}
        result['windows'][window]['background_gain_relative_rms']=float(np.sqrt(np.mean([r['relative_gain_error_rms']**2 for r in records])))
    # A completed 16-block RMS window below 0.2%, not the first lucky crossing.
    # This diagnostic uses truth; it is not a stopping criterion for learning.
    def settling(start):
        for end in range(start+16,n+1):
            error=np.sqrt(np.mean([r['relative_gain_error_rms']**2 for r in rows[end-16:end]]))
            if error<.002:
                return {"blocks":end-start,"samples":(end-start)*block_size,"milliseconds":(end-start)*block_size/FS*1e3}
        return None
    result.update(startup_settling=settling(0),step_recovery=settling(n//2) if scenario=='step' else None,
                  settling_rule='first completed 16-block RMS relative gain-error window below 0.2%; evaluator only',
                  backend_overloads=sum(r['backend_overloads'] for r in rows),
                  input_overloads=sum(r['input_overloads'] for r in rows),
                  held_updates=sum(r['update_status']!='updated' for r in rows),
                  updates=len(rows),
                  max_causality_gap=max(abs(rows[i]['background_gain_used']-rows[i-1]['background_gain_next']) for i in range(1,n)))
    return result


def plot_cases(trajectories,output,block_size):
    colors={'uncalibrated':'#9b9b9b','foreground_frozen':'#d47b12','background':'#1678b4','oracle':'#299247'}
    for kind in sorted(set(k[0] for k in trajectories)):
        scenarios=[name for name in ('constant','ramp','step') if (kind,name) in trajectories]
        fig,axes=plt.subplots(2,len(scenarios),figsize=(5*len(scenarios),8),squeeze=False,layout='constrained')
        for col,scenario in enumerate(scenarios):
            rows=trajectories[(kind,scenario)]
            t=np.array([r['time_ms'] for r in rows])
            axes[0,col].plot(t,[r['true_gain_mean'] for r in rows],color='black',label='True gain (evaluation only)')
            axes[0,col].plot(t,[r['foreground_gain'] for r in rows],color=colors['foreground_frozen'],label='Foreground frozen')
            axes[0,col].plot(t,[r['background_gain_used'] for r in rows],color=colors['background'],label='Background, causal estimate')
            axes[0,col].set_title(scenario.capitalize());axes[0,col].set_ylabel('Interstage gain (V/V)')
            for name,color in colors.items():
                axes[1,col].plot(t,[r[name+'_spectral_ratio_db'] for r in rows],color=color,label=name.replace('_',' '),alpha=.9)
            axes[1,col].set_xlabel('Normal-operation time (ms)')
            axes[1,col].set_ylabel('SNDR (dB)' if kind=='sine' else 'Total-tone / noise+distortion (dB)')
            for ax in axes[:,col]:ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
        fig.suptitle(f'{kind}: same probe-on capture for all four methods | first seed\n{block_size} samples/update; foreground training time excluded; oracle is not an estimator input')
        fig.savefig(output/f'tracking_{kind}.png',dpi=160);plt.close(fig)


def aggregate(results):
    table=[]
    for kind in sorted(set(r['waveform'] for r in results)):
        for scenario in sorted(set(r['scenario'] for r in results)):
            selected=[r for r in results if r['waveform']==kind and r['scenario']==scenario]
            for window in selected[0]['summary']['windows']:
                row={'waveform':kind,'scenario':scenario,'window':window,'seeds':len(selected)}
                for method in METHODS:
                    values=[r['summary']['windows'][window][method]['spectral_ratio_db'] for r in selected]
                    row[method+'_mean']=float(np.mean(values));row[method+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
                table.append(row)
    return table


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'output'/'dither_background')
    parser.add_argument('--blocks',type=int,default=512)
    parser.add_argument('--block-size',type=int,default=32768)
    parser.add_argument('--foreground-blocks',type=int,default=32)
    parser.add_argument('--learning-rate',type=float,default=.08)
    parser.add_argument('--probe-amplitude',type=float,default=.0625)
    parser.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    parser.add_argument('--scenarios',nargs='+',choices=['constant','ramp','step'],default=['constant','ramp','step'])
    parser.add_argument('--waveforms',nargs='+',choices=['sine','multitone'],default=['sine','multitone'])
    args=parser.parse_args(argv)
    if args.blocks<64 or args.blocks%4 or args.block_size<1024 or args.block_size%2 or args.foreground_blocks<1:
        parser.error('Use blocks >=64 divisible by 4, even block-size >=1024, and foreground-blocks >=1')
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.scenarios))!=len(args.scenarios) or len(set(args.waveforms))!=len(args.waveforms):
        parser.error('Do not repeat seeds, scenarios or waveforms')
    config=PipelineConfig(probe_amplitude=args.probe_amplitude)
    # Validate estimator options before writing any output.
    DitherGainTracker(config.probe_amplitude,learning_rate=args.learning_rate)
    args.output.mkdir(parents=True,exist_ok=True)
    configuration={**vars(args),'output':str(args.output.resolve()),'plant':asdict(config),'fs_hz':FS,
                   'initial_true_gain':INITIAL_GAIN,'ramp_final_gain':4.12,'step_final_gain':4.08,
                   'foreground_training_samples':args.foreground_blocks*args.block_size,
                   'normal_samples_per_case':args.blocks*args.block_size,
                   'primary_comparison':'same probe-on captures, identical noise and codes',
                   'probe_off_controls':'counterfactual plants with identical input and analog-noise draws; probe off',
                   'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       (Path(__file__), Path(__file__).with_name('pipeline_dither_model.py'),
                        Path(inspect.getfile(DitherGainTracker)))}}
    config_path=args.output/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:
        parser.error('Output has another configuration/source version; choose a fresh --output directory')
    write_json(config_path,configuration)
    results=[];trajectories={}
    for kind in args.waveforms:
        for scenario in args.scenarios:
            for seed in args.seeds:
                directory=args.output/f'{kind}_{scenario}_seed{seed}'
                rows,result=run_case(config,seed=seed,scenario=scenario,kind=kind,blocks=args.blocks,
                    block_size=args.block_size,learning_rate=args.learning_rate,foreground_blocks=args.foreground_blocks,output=directory)
                results.append(result)
                if seed==args.seeds[0]:trajectories[(kind,scenario)]=rows
                tail=result['summary']['windows']['final_quarter']
                print(f"{kind}/{scenario}/{seed}: final nominal={tail['uncalibrated']['spectral_ratio_db']:.2f}, frozen={tail['foreground_frozen']['spectral_ratio_db']:.2f}, background={tail['background']['spectral_ratio_db']:.2f}, oracle={tail['oracle']['spectral_ratio_db']:.2f} dB; held={result['summary']['held_updates']}",flush=True)
    table=aggregate(results)
    write_json(args.output/'summary.json',{'runs':results,'aggregate':table})
    with (args.output/'aggregate.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    plot_cases(trajectories,args.output,args.block_size)
    print(f'Saved {len(results)} runs to {args.output.resolve()}',flush=True)


if __name__=='__main__':
    main()
