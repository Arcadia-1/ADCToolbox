"""Shared three-PN architecture experiments; see DITHER_GUIDE.md.

Fresh output directories only. Calibration receives observations; truth-based
metrics, polynomial oracles and rank checks are explicitly evaluation-only.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
from itertools import product
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from . import pn_architectures as pa
    from .exp_cal06_joint_dither import metrics, waveform, write_json
except ImportError:
    import pn_architectures as pa
    from exp_cal06_joint_dither import metrics, waveform, write_json


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def pipeline_boundaries():
    """Exact Cartesian moments; not a claim of background convergence time."""
    a,g,k = 1/15,1.02,-.24
    grid = np.array(list(product(range(128),*[(-1,1)]*6)))
    phase = 2*np.pi*grid[:,0]/128
    p,pp = grid[:,1:4],grid[:,4:7]
    u,up = .25*np.sin(phase),.25*np.sin(phase-.861)
    d,dp = a*p.sum(axis=1),a*pp.sum(axis=1)
    v,vp = u+d,up+dp
    rows=[]
    for mode in ('before_amplifier','after_amplifier','fifth_order','nonlinear_memory'):
        r = g*v+k*v**3
        if mode == 'after_amplifier': r=g*u+k*u**3+d
        if mode == 'fifth_order': r += .2*v**5
        if mode == 'nonlinear_memory': r += .3*v*vp**2
        ss = r-d
        gamma3 = np.mean(ss*np.prod(p,axis=1))/(6*a**3)
        gain = 1+np.mean(ss*p[:,0])/a-(7*a*a+3*np.mean(ss*ss))*gamma3
        linear = r/gain
        corrected = linear-gamma3/gain*linear**3-d
        rows.append(dict(mode=mode,estimated_gain=gain,estimated_cubic=gamma3,
                         physical_gain=g,physical_cubic=k,probe_peak=float(max(abs(d))),
                         normalized_residue_headroom=float(.5-max(abs(r))),
                         nominal_error_rms=float(np.sqrt(np.mean((ss-u)**2))),
                         corrected_error_rms=float(np.sqrt(np.mean((corrected-u)**2))),
                         expected_cubic=(k+.2*(10*np.mean(u*u)+10*a*a)) if mode=='fifth_order' else
                             (0. if mode=='after_amplifier' else k)))
    return rows


def sar_case(seed=707, kind='sine', samples=2**19, valid_pair=True):
    size=8192; blocks=samples//size
    nominal=pa.nominal_sar_weights()
    initial=nominal.copy(); initial[:4] *= [1.004,.997,1.002,.998]
    final=nominal.copy(); final[:4] *= [1.008,.994,1.005,.996]
    tracker=pa.SARWeights(nominal)
    pn=pa.ThreePN(seed); rng=np.random.default_rng(seed+10000)
    clean,bins=waveform(np.arange(size),size,kind,peak=.7)
    rows=[]; gram=np.zeros((4,4)); moment=np.zeros(3)
    oracle_gram=np.zeros((2,2)); oracle_rhs=np.zeros(2)
    headroom=1.; rails=0
    for block in range(blocks):
        actual=initial if block<blocks//2 else final
        p,d=pn.draw(size)
        other=clean if valid_pair else np.roll(clean,-1)
        xp=clean+d+rng.normal(0,2e-5,size)
        xm=other-d+rng.normal(0,2e-5,size)
        headroom=min(headroom,float(actual.sum()-max(np.max(abs(xp)),np.max(abs(xm)))))
        bp,bm=pa.sar_decisions(xp,actual),pa.sar_decisions(xm,actual)
        rail=bool(np.any(np.all(bp==1,axis=1)|np.all(bp==-1,axis=1)|np.all(bm==1,axis=1)|np.all(bm==-1,axis=1)))
        rails += int(rail)
        y=tracker.process(bp,bm,d,learn=not rail)
        row=dict(block=block,held_pairs=(block+1)*size)
        for name,signal in [('nominal',(bp@nominal+bm@nominal)/2),('learned',y),('oracle_weights',(bp@actual+bm@actual)/2)]:
            row.update({name+'_'+key:value for key,value in metrics(signal,clean,bins).items()})
        for j in range(4):
            row[f'estimated_w{j}']=tracker.weights[j]; row[f'true_w{j}']=actual[j]
        rows.append(row)
        if block>=3*blocks//4:
            delta=(bp[:,:4].astype(float)-bm[:,:4])
            gram += delta.T@delta
            r=bp@nominal; ss=r-d
            moment += [np.sum(ss*p[:,0]),np.sum(ss*np.prod(p,axis=1)),np.sum(ss*ss)]
            # Evaluation-only best global cubic using true analog conversion input.
            features=np.column_stack([r,r**3])
            oracle_gram += features.T@features
            oracle_rhs += features.T@(clean+d)
    count=sum(size for b in range(blocks) if b>=3*blocks//4)
    m=moment/count; a=.2/3
    cubic=m[1]/(6*a**3); gain=1+m[0]/a-(7*a*a+3*m[2])*cubic
    # Diagnostic global cubic can never update the SAR weight learner.
    best_cubic=np.linalg.solve(oracle_gram,oracle_rhs)
    n=np.arange(32768)
    if kind=='sine':
        test_bins=[int(.179*len(n))|1]; test=.7*np.sin(2*np.pi*test_bins[0]*n/len(n)+.81)
    else:
        test_bins=[int(.071*len(n))|1,int(.193*len(n))|1]
        test=.4*np.sin(2*np.pi*test_bins[0]*n/len(n)+.8)+.3*np.sin(2*np.pi*test_bins[1]*n/len(n)+1.7)
    bits=pa.sar_decisions(test+rng.normal(0,2e-5,len(test)),final)
    raw=bits@nominal
    valid_hdc=bool(.75<gain<1.25 and abs(cubic/gain)<2)
    corrected=(raw/gain-cubic/gain*(raw/gain)**3) if valid_hdc else raw.copy()
    signals=dict(nominal=raw,learned=bits@tracker.weights,oracle_weights=bits@final,
                 misapplied_hdc=corrected,oracle_global_cubic_fit=best_cubic[0]*raw+best_cubic[1]*raw**3)
    heldout={name:metrics(y,test,test_bins) for name,y in signals.items()}
    eig=np.linalg.eigvalsh(gram/count)
    result=dict(seed=seed,kind=kind,valid_pair=valid_pair,held_pairs=samples,conversions=2*samples,
                probe_peak=.2,minimum_headroom=headroom,rail_blocks=rails,
                estimated_weights=tracker.weights.tolist(),true_weights=final.tolist(),
                leading_weight_relative_rmse=float(np.sqrt(np.mean(((tracker.weights[:4]-final[:4])/final[:4])**2))),
                excitation_minimum_eigenvalue=float(eig[0]),excitation_condition=float(eig[-1]/max(eig[0],1e-300)),
                updated=tracker.updated,rejected=tracker.rejected,unexcited=tracker.unexcited,
                misapplied_hdc_valid=valid_hdc,misapplied_hdc_gain=float(gain),misapplied_hdc_cubic=float(cubic),
                heldout_probe_off=heldout,
                final_training={name:{metric:float(np.mean([r[name+'_'+metric] for r in rows[3*blocks//4:]]))
                    for metric in ('spectral_ratio_db','spur_ratio_db','error_snr_db')} for name in ('nominal','learned','oracle_weights')})
    return rows,result


def ti_signal(n, size, kind, heldout=False):
    fs=[.179] if kind=='sine' and heldout else [.137] if kind=='sine' else ([.071,.193] if heldout else [.083,.217])
    bins=[int(f*size)|1 for f in fs]
    amplitudes=[.7] if len(bins)==1 else [.4,.3]
    return sum(a*np.sin(2*np.pi*b*n/size+.37+.81*i) for i,(a,b) in enumerate(zip(amplitudes,bins))),bins


def ti_case(seed=707,kind='sine',samples=2**22,placement='linear_before'):
    size=32768; pn=pa.ThreePN(seed); rng=np.random.default_rng(seed+20000)
    actual_gain=np.array([1.,1.02]); actual_skew=np.array([.01,.05])
    tracker=pa.TIMoments(); rows=[]; headroom=1.; overloads=0
    for start in range(0,samples,size):
        n=np.arange(start,start+size); labels=n%2
        _,d0,d1=pn.adjacent(size)
        signal,_=ti_signal(n+actual_skew[labels],size,kind)
        probe=(1-actual_skew[labels])*d0+actual_skew[labels]*d1 if placement=='linear_before' else d0
        analog=actual_gain[labels]*(signal+probe) if placement!='after_sampler' else actual_gain[labels]*signal+probe
        analog += rng.normal(0,1e-4,size)
        y,rail,overload=pa.quantize(analog,13)
        headroom=min(headroom,float(1-max(abs(analog)))); overloads+=overload
        if not rail: tracker.update(y,d0,d1,labels)
        estimate=tracker.estimate()
        rows.append(dict(samples=start+size,valid=estimate['valid'],**{f'gain{i}':float(estimate['gain'][i]) for i in range(2)},
                         **{f'skew{i}':float(estimate['skew'][i]) for i in range(2)}))
    estimate=tracker.estimate()
    gain=estimate['gain'] if estimate['valid'] else np.ones(2)
    trim=np.round(estimate['skew']*4096)/4096 if estimate['valid'] else np.zeros(2)
    n=np.arange(size); labels=n%2; clean,bins=ti_signal(n,size,kind,True)
    noise=rng.normal(0,1e-4,size)
    heldout={}
    for name,delay,scale in [('nominal',np.zeros(2),np.ones(2)),('learned',trim,gain),
                            ('oracle',actual_skew,actual_gain)]:
        x,_=ti_signal(n+actual_skew[labels]-delay[labels],size,kind,True)
        y,_,_=pa.quantize(actual_gain[labels]*x+noise,13)
        heldout[name]=metrics(y/scale[labels],clean,bins)
    result=dict(seed=seed,kind=kind,placement=placement,samples=samples,probe_peak=.2,
                minimum_headroom=headroom,overloads=overloads,estimate_valid=estimate['valid'],
                gain=estimate['gain'].tolist(),skew=estimate['skew'].tolist(),clock_trim=trim.tolist(),
                true_gain=actual_gain.tolist(),true_skew=actual_skew.tolist(),
                skew_rmse_samples=float(np.sqrt(np.mean((estimate['skew']-actual_skew)**2))),
                relative_skew_error_samples=float((estimate['skew'][1]-estimate['skew'][0])-.04),heldout_probe_off=heldout)
    return rows,result


def delta_sigma_cases(seed=707,size=2**18):
    warm=8192; n=np.arange(-warm,size)
    _,d=pa.ThreePN(seed).draw(len(n))
    rows=[]
    for kind,dc in [('dc_1_64',1/64),('dc_3_128',3/128),('sine',None)]:
        bins=[int(size/320)|1] if dc is None else []
        clean=.4*np.sin(2*np.pi*bins[0]*n/size+.37) if bins else np.full(len(n),dc)
        for location in ('off','input','quantizer'):
            probe=np.zeros(len(n)) if location=='off' else d
            at_quantizer=location!='input'
            y,e,state,maximum=pa.sigma_delta(clean,probe,at_quantizer)
            filtered=pa.difference(probe) if at_quantizer else probe
            identity=float(np.max(abs(y-clean-filtered-pa.difference(e))))
            outputs={'raw':y} if location=='off' else {'raw':y,'subtract_raw_d':y-probe,'subtract_matched_path':y-filtered}
            for method,z in outputs.items():
                row=dict(seed=seed,waveform=kind,location=location,method=method,samples=size,osr=32,
                    probe_peak=float(max(abs(probe))),maximum_state=maximum,transfer_identity_error=identity,
                    sndr_db=None,sfdr_db=None)
                row.update(pa.inband_metrics(z[warm:],clean[warm:],bins)); rows.append(row)
    return rows


def figures(sar,ti,ds,path):
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for k in range(4):
        axes[0].plot([r['held_pairs'] for r in sar],[(r[f'estimated_w{k}']/r[f'true_w{k}']-1)*100 for r in sar],label=f'w{k}')
    axes[0].set(title='SAR: causal weight tracking',xlabel='Held input pairs',ylabel='Relative error (%)'); axes[0].legend()
    for k in range(2):
        axes[1].plot([r['samples'] for r in ti],[r[f'skew{k}'] for r in ti],label=f'channel {k}')
        axes[1].axhline([.01,.05][k],color='k',ls='--',lw=.8)
    axes[1].set(title='TI: probe before sampler',xlabel='Samples',ylabel='Estimated skew / Ts'); axes[1].legend()
    selected=[r for r in ds if r['waveform']=='sine' and r['method'] in ('raw','subtract_matched_path')]
    labels=[r['location']+'\n'+r['method'].replace('subtract_matched_path','matched') for r in selected]
    axes[2].bar(labels,[r['sndr_db'] for r in selected])
    axes[2].set(title='Delta-sigma: in-band tone SNDR',ylabel='dB'); axes[2].tick_params(axis='x',labelsize=8)
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(path,dpi=150); plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[707,808,909])
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()): parser.error('Output directory is not empty')
    args.output.mkdir(parents=True,exist_ok=True)
    sources=['pn_architectures.py','exp_cal09_pn_architectures.py','hdc_fixed.py','dither_mechanisms.py','exp_cal06_joint_dither.py']
    here=Path(__file__).parent
    sizes=dict(sar=2**14 if args.quick else 2**19,ti=2**16 if args.quick else 2**22,ds=2**15 if args.quick else 2**18)
    write_json(args.output/'configuration.json',dict(seeds=args.seeds,quick=args.quick,samples=sizes,
        sources={p:hashlib.sha256((here/p).read_bytes()).hexdigest() for p in sources},
        python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
        pn=dict(lengths=[31,29,23],peak=.2,level_probabilities=[.125,.375,.375,.125]),
        sar=dict(decisions=14,radix=1.85,active_weights=4,rate=2**-10,input_noise_rms=2e-5),
        ti=dict(channels=2,backend_bits=13,noise_rms=1e-4,clock_trim_step=1/4096),
        ds=dict(order=1,osr=32,warmup=8192),
        acceptance='No overrange in valid cases; scalar update causality; observable-only estimators; expected invalid controls retained; spectral conclusions only for declared band/model'))
    t0=time.monotonic(); pipeline=pipeline_boundaries(); table(args.output/'pipeline.csv',pipeline)
    summaries=[]; spectra=[]; distribution=[]; sar_plot=ti_plot=ds_plot=None
    for seed in args.seeds:
        p,d=pa.ThreePN(seed).draw(65536)
        distribution.append(dict(seed=seed,peak=float(max(abs(d))),variance=float(np.var(d)),
            **{f'count_{v}':int(np.count_nonzero(p.sum(axis=1)==v)) for v in (-3,-1,1,3)}))
        for kind in ('sine','multitone'):
            for valid in (True,False):
                rows,result=sar_case(seed,kind,sizes['sar'],valid)
                tag=f'sar_{kind}_{seed}_{"held" if valid else "broken_pair"}'
                table(args.output/(tag+'.csv'),rows); summaries.append(dict(family='sar',**result))
                for method,value in result['heldout_probe_off'].items(): spectra.append(dict(family='sar',seed=seed,kind=kind,control='held' if valid else 'broken_pair',method=method,**value))
                if sar_plot is None and valid: sar_plot=rows
                print(tag,'SNDR',result['heldout_probe_off']['learned']['spectral_ratio_db'],flush=True)
            for placement in ('linear_before','flat_before','after_sampler'):
                rows,result=ti_case(seed,kind,sizes['ti'],placement)
                tag=f'ti_{kind}_{seed}_{placement}'
                table(args.output/(tag+'.csv'),rows); summaries.append(dict(family='ti',**result))
                for method,value in result['heldout_probe_off'].items(): spectra.append(dict(family='ti',seed=seed,kind=kind,control=placement,method=method,**value))
                if ti_plot is None and placement=='linear_before': ti_plot=rows
                print(tag,'SNDR',result['heldout_probe_off']['learned']['spectral_ratio_db'],flush=True)
        ds=delta_sigma_cases(seed,sizes['ds'])
        if ds_plot is None: ds_plot=ds
        table(args.output/f'delta_sigma_{seed}.csv',ds)
        summaries.append(dict(family='delta_sigma',seed=seed,rows=ds))
    table(args.output/'pn_distribution.csv',distribution)
    table(args.output/'heldout_spectra.csv',spectra)
    write_json(args.output/'summary.json',summaries)
    figures(sar_plot,ti_plot,ds_plot,args.output/'mechanisms.png')
    write_json(args.output/'completion.json',dict(elapsed_seconds=time.monotonic()-t0,cases=len(summaries),
        source_hashes_match=all(hashlib.sha256((here/p).read_bytes()).hexdigest()==
            __import__('json').loads((args.output/'configuration.json').read_text())['sources'][p] for p in sources)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
