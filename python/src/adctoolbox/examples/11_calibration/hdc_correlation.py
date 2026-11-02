"""Evaluator-only HDC correlation attribution; no calibration algorithm changes.

Internal analog values and error permutations are diagnostic controls, not
observable hardware inputs or a reproduction of the 2013 residue decorrelator.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
from adctoolbox.calibration import _dither_hdc
from itertools import product
from pathlib import Path
import platform
import time
import numpy as np
try:
    from .hdc_fixed import Device, FINAL
    from .dither_mechanisms import PRBS, quantize
    from .exp_cal06_joint_dither import save_csv, write_json
except ImportError:
    from hdc_fixed import Device, FINAL
    from dither_mechanisms import PRBS, quantize
    from exp_cal06_joint_dither import save_csv, write_json

CASES = [('large_sine', 0., .78), ('tiny_inner', .137, 1e-5),
         ('dc_bias_control', -.1846875, 0.)]
BITS = (0, 13, 24)  # 0 denotes an unquantized evaluator reference.


class DiagnosticPlant:
    """Same forward stage as Plant; exposes truth ONLY to the experiment."""
    def __init__(self, seed, truth=FINAL, noise=.0002, peak=.2):
        self.truth, self.noise, self.a = truth, noise, peak/3
        self.pn = [PRBS(seed,31), PRBS(seed+100,29), PRBS(seed+300,23)]
        self.switch = PRBS(seed+200,27)
        self.rng = np.random.default_rng(seed+10000)
        self.previous_v = 0.

    def capture(self, x):
        t = self.truth
        p = np.column_stack([pn.draw(len(x)) for pn in self.pn])
        s = self.switch.draw(len(x))
        q = np.where(x > .25, 1, np.where(x < -.25, -1, 0))
        d = self.a*p.sum(axis=1)
        u = x-(1+t.delta*s)*q/2
        v = u+d
        z = u+t.history/t.gain*np.r_[self.previous_v,v[:-1]]
        self.previous_v = v[-1]
        r0 = t.gain/2*(z+d)+t.cubic*(z+d)**3
        noise_r = self.rng.normal(0.,self.noise,len(x))/2
        return dict(p=p,d=d,z=z,r0=r0,noise_r=noise_r)


def block_moments(capture, truth, permutation, independent_reference):
    """Raw sums; algebraic partition remains exact for finite, correlated data."""
    p,d,z,r0,nr = (capture[k] for k in ('p','d','z','r0','noise_r'))
    p1,c3 = p[:,0], np.prod(p,axis=1)
    g,k = truth.gain/2,truth.cubic
    base = dict(n=len(d),c3=float(c3.sum()),p1=float(p1.sum()),
        p2=float(p[:,1].sum()),p3=float(p[:,2].sum()),
        p12=float(np.sum(p[:,0]*p[:,1])),p13=float(np.sum(p[:,0]*p[:,2])),
        p23=float(np.sum(p[:,1]*p[:,2])),z2=float(np.sum(z*z)),
        analog_m1=float(np.sum((r0-d)*p1)),analog_m3=float(np.sum((r0-d)*c3)),
        noise_m1=float(np.sum(nr*p1)),noise_m3=float(np.sum(nr*c3)),
        signal_term=float(np.sum((g*z+k*z**3)*c3)),
        linear_probe_term=float(np.sum(((g-1)*d+3*k*z*z*d)*c3)),
        quadratic_probe_term=float(np.sum(3*k*z*d*d*c3)),
        cubic_probe_term=float(np.sum(k*d**3*c3)),
        positive_count=int(np.count_nonzero(c3>0)))
    actual = r0+nr
    rows = {}
    for bits in BITS:
        r = actual if bits==0 else quantize(2*actual,bits)[0]/2
        e,ss = r-actual,r-d
        rows[bits] = dict(**base,m1=float(np.sum(ss*p1)),m3=float(np.sum(ss*c3)),
            eta=float(np.sum(ss*ss)),quant_m1=float(np.sum(e*p1)),
            quant_m3=float(np.sum(e*c3)),error_sum=float(np.sum(e)),error_square=float(np.sum(e*e)),
            error_positive=float(np.sum(e[c3>0])),error_negative=float(np.sum(e[c3<0])),
            shuffled_quant_m3=float(np.sum(e[permutation]*c3)),
            independent_m3=float(np.sum(ss*independent_reference)))
    return rows,dict(headroom=float(1-np.max(abs(2*actual))),overloads=int(np.sum(abs(2*actual)>=1)))


def summarize(sums, truth, a):
    n,g,k = sums['n'],truth.gain/2,truth.cubic
    m = {key:value/n for key,value in sums.items()}
    denominator = 6*a**3
    khat = m['m3']/denominator
    finite = m['analog_m3']/denominator-k
    noise = m['noise_m3']/denominator
    quant = m['quant_m3']/denominator
    gain_hat = 1+m['m1']/a-(7*a*a+3*m['eta'])*khat
    gamma1_finite = m['analog_m1']/a-(g-1+(7*a*a+3*m['z2'])*k)
    eta_bias = 3*k*(m['z2']-m['eta'])
    coupling = -(7*a*a+3*m['eta'])*(khat-k)
    centered = m['quant_m3']-m['error_sum']*m['c3']
    error_var = max(0.,m['error_square']-m['error_sum']**2)
    norm = np.sqrt(error_var*max(0.,1-m['c3']**2))
    parts = {key:m[key]/denominator for key in
             ('signal_term','linear_probe_term','quadratic_probe_term','cubic_probe_term')}
    parts['cubic_probe_term'] -= k
    return dict(samples=int(n),estimated_cubic=khat,cubic_error=khat-k,
        finite_cubic_error=finite,noise_cubic_error=noise,quant_cubic_error=quant,
        cubic_decomposition_residual=(khat-k)-(finite+noise+quant),
        finite_partition_residual=finite-sum(parts.values()),**parts,
        estimated_normalized_gain=gain_hat,gain_error=gain_hat-g,
        gamma1_finite_error=gamma1_finite,gamma1_noise_error=m['noise_m1']/a,
        gamma1_quant_error=m['quant_m1']/a,eta_gain_bias=eta_bias,cubic_gain_coupling=coupling,
        gain_decomposition_residual=(gain_hat-g)-(gamma1_finite+m['noise_m1']/a+m['quant_m1']/a+eta_bias+coupling),
        mean_p1=m['p1'],mean_p2=m['p2'],mean_p3=m['p3'],mean_c3=m['c3'],
        mean_p1p2=m['p12'],mean_p1p3=m['p13'],mean_p2p3=m['p23'],
        mean_quant_error=m['error_sum'],quant_error_rms=np.sqrt(m['error_square']),
        mean_error_times_c3=m['quant_m3'],centered_error_c3_covariance=centered,
        error_c3_correlation=None if norm==0 else centered/norm,
        mean_error_given_c3_positive=sums['error_positive']/sums['positive_count'],
        mean_error_given_c3_negative=sums['error_negative']/(n-sums['positive_count']),
        shuffled_estimated_cubic=(m['analog_m3']+m['noise_m3']+m['shuffled_quant_m3'])/denominator,
        shuffled_quant_cubic_error=m['shuffled_quant_m3']/denominator,
        independent_reference_estimated_cubic=m['independent_m3']/denominator)


def streaming(seed, mode, case, noise, checkpoints, peak=.2):
    name,offset,amplitude = case
    truth = FINAL if mode=='joint' else Device(FINAL.gain,FINAL.cubic,0.,0.)
    plant = DiagnosticPlant(seed,truth,noise,peak)
    control_rng = np.random.default_rng(seed+50000)
    reference = PRBS(seed+70000,19)
    sums={b:None for b in BITS}; rows=[]
    headroom,overloads = 1.,0
    # Prefixes are nested and share all physical samples. Shuffle is local to
    # each fixed-size block and never changes the measured error histogram.
    size = 1 << min(16,min(checkpoints))
    for start in range(0,1 << max(checkpoints),size):
        n = np.arange(start,start+size)
        x = offset+amplitude*np.sin(2*np.pi*4489*(n%32768)/32768+.37)
        capture = plant.capture(x)
        moments,diag = block_moments(capture,truth,control_rng.permutation(size),reference.draw(size))
        headroom,overloads = min(headroom,diag['headroom']),overloads+diag['overloads']
        for b,chunk in moments.items():
            if sums[b] is None: sums[b]=chunk
            else:
                for key,value in chunk.items(): sums[b][key]+=value
        if start+size in {1 << log2 for log2 in checkpoints}:
            for b in BITS:
                rows.append(dict(kind='stream',seed=seed,mode=mode,case=name,offset=offset,amplitude=amplitude,
                    noise=noise,backend_bits=b,**summarize(sums[b],truth,peak/3),
                    minimum_headroom=headroom,overloads=overloads))
    return rows


def balanced_controls(seed=919, repeats=(1,16,256), peak=.2):
    """Balanced PN/phase repeats are an expectation control, not a PN design."""
    truth=Device(FINAL.gain,FINAL.cubic,0.,0.)
    signs=np.array(list(product((-1.,1.),repeat=3)))
    rng=np.random.default_rng(seed); rows=[]
    for name,offset,amplitude in CASES:
        x=offset+amplitude*np.sin(2*np.pi*np.arange(256)/256+.37)
        q=np.where(x>.25,1,np.where(x<-.25,-1,0))
        for repeat in repeats:
            # Enumerate independent-reference +/-1 for EVERY phase/PN tuple.
            # A merely random finite reference retains DC leakage and would
            # obscure this balanced expectation control.
            p=np.tile(signs,(512*repeat,1)); z=np.tile(np.repeat(x-q/2,8),2*repeat)
            d=peak/3*p.sum(axis=1); r0=truth.gain/2*(z+d)+truth.cubic*(z+d)**3
            capture=dict(p=p,d=d,z=z,r0=r0,noise_r=np.zeros(len(d)))
            independent=np.r_[np.ones(len(d)//2),-np.ones(len(d)//2)]
            moments,diag=block_moments(capture,truth,rng.permutation(len(d)),independent)
            for b in BITS:
                rows.append(dict(kind='balanced',seed=seed,mode='isolated',case=name,offset=offset,
                    amplitude=amplitude,noise=0.,backend_bits=b,**summarize(moments[b],truth,peak/3),
                    minimum_headroom=diag['headroom'],overloads=diag['overloads']))
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[601,702,803])
    parser.add_argument('--checkpoints',type=int,nargs='+',default=[16,20,24])
    parser.add_argument('--noise-rms',type=float,nargs='+',default=[0.,.0002])
    parser.add_argument('--dither-peak',type=float,default=.2)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args(argv)
    if not args.checkpoints or any(not 8<=v<=26 for v in args.checkpoints): parser.error('checkpoints must be sample log2 values in [8,26]')
    if not np.isfinite(args.dither_peak) or not 0<args.dither_peak<1: parser.error('dither-peak must be in (0,1)')
    if any(not np.isfinite(v) or v<0 for v in args.noise_rms): parser.error('noise-rms must be finite and nonnegative')
    if args.quick: args.checkpoints=[12,14]
    if args.output.exists() and any(args.output.iterdir()): parser.error('Output directory is not empty')
    args.output.mkdir(parents=True,exist_ok=True)
    sources=['hdc_correlation.py','hdc_fixed.py','dither_mechanisms.py','exp_cal06_joint_dither.py']
    source_paths = {name: Path(__file__).with_name(name) for name in sources}
    source_paths['_dither_hdc.py'] = Path(_dither_hdc.__file__)
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name, path in source_paths.items()}
    write_json(args.output/'configuration.json',dict(arguments={**vars(args),'output':str(args.output)},
        sources=hashes,truth=asdict(FINAL),cases=CASES,backend_bits=BITS,python=platform.python_version(),numpy=np.__version__,
        scope='Diagnostic truth decomposition and nonphysical error permutation; no estimator changes; no 2013 RD implementation'))
    start=time.monotonic()
    balanced=balanced_controls(repeats=(1,4) if args.quick else (1,16,256),peak=args.dither_peak)
    save_csv(args.output/'balanced.csv',balanced)
    rows=[]
    for seed,mode,noise,case in product(args.seeds,('isolated','joint'),args.noise_rms,CASES):
        rows.extend(streaming(seed,mode,case,noise,args.checkpoints,args.dither_peak))
        print('completed',seed,mode,noise,case[0],flush=True)
    save_csv(args.output/'streaming.csv',rows)
    write_json(args.output/'completion.json',dict(elapsed_seconds=time.monotonic()-start,stream_rows=len(rows),balanced_rows=len(balanced),
        source_hashes_match=all(hashlib.sha256(source_paths[n].read_bytes()).hexdigest()==h for n,h in hashes.items()),
        maximum_cubic_identity_error=max(abs(r['cubic_decomposition_residual']) for r in rows+balanced),
        maximum_gain_identity_error=max(abs(r['gain_decomposition_residual']) for r in rows+balanced)))
    return 0


if __name__=='__main__': raise SystemExit(main())
