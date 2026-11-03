"""Analytical moment and worst-case range controls for exp_cal08; no training labels."""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
try:
    from .hdc_fixed import Plant, Tracker, FINAL, INITIAL
    from .exp_cal06_joint_dither import save_csv, write_json
except ImportError:
    from hdc_fixed import Plant, Tracker, FINAL, INITIAL
    from exp_cal06_joint_dither import save_csv, write_json


def balanced_moments():
    """Cartesian expectations, not a physical time trace or training procedure.

    Current/previous PN and element selection are enumerated independently of
    input phase. This removes finite-record PN correlation sampling error.
    """
    # Explicit Cartesian factors: phase, current switch, previous switch,
    # three current and three previous probe signs.
    grid = np.array(list(product(range(64), *[(-1,1)]*8)))
    phase, s, sp = grid[:,0]*2*np.pi/64, grid[:,1], grid[:,2]
    p, pp = grid[:,3:6], grid[:,6:9]
    x, xp = .8*np.sin(phase), .8*np.sin(phase-2*np.pi*.137)
    q = np.where(x > .25,1,np.where(x < -.25,-1,0))
    qp = np.where(xp > .25,1,np.where(xp < -.25,-1,0))
    a, rows = 1/15, []
    for k, h, delta, quantized in product((0.,-.04,-.24),(0.,FINAL.history/FINAL.gain),
                                          (0.,FINAL.delta),(False,True)):
        u, up = x-(1+delta*s)*q/2, xp-(1+delta*sp)*qp/2
        z = u+h*(up+a*pp.sum(axis=1))
        t = z+a*p.sum(axis=1)
        g = FINAL.gain/2
        r = g*t+k*t**3
        if quantized:
            # Same 13-bit midrise backend, expressed in normalized r=backend/2.
            r = (np.clip(np.floor((2*r+1)*4096),0,8191)+.5)/8192-.5
        ss = r-a*p.sum(axis=1)
        gamma1 = np.mean(ss*p[:,0])/a
        gamma3 = np.mean(ss*np.prod(p,axis=1))/(6*a**3)
        eta = np.mean(ss*ss)
        estimated_g = 1+gamma1-(7*a*a+3*eta)*gamma3
        rows.append(dict(cubic=k, history_ratio=h, delta=delta, backend_bits=13 if quantized else 0,
                         estimated_cubic=gamma3, cubic_error=gamma3-k,
                         gain_error=2*(estimated_g-g),
                         eta_gain_bias_prediction=6*k*(np.mean(z*z)-eta)))
    return rows


def headroom_controls():
    rows = []
    n = np.arange(65536)
    for name, x in [('dc_pos',np.full(len(n),.8)),('dc_neg',np.full(len(n),-.8)),
                    ('slow_sine',.8*np.sin(2*np.pi*67*n/len(n))),
                    ('dc_pos_derated',np.full(len(n),.78))]:
        plant = Plant(515)
        c, d = plant.convert(x,FINAL)
        tracker = Tracker(window_log2=16)
        tracker.process(c)
        rows.append(dict(signal=name, **d, held=tracker.held, accepted=tracker.updates,
                         cap_excited=tracker.events[0]['cap_excited']))
    bounds = []
    for name, truth in [('initial',INITIAL),('final',FINAL)]:
        def residue(peak):
            u = max(.25,peak-.5+abs(truth.delta)/2)
            t = (u+.2)*(1+abs(truth.history/truth.gain))
            return truth.gain*t+2*truth.cubic*t**3
        bounds.append(dict(device=name, conservative_peak_at_input_point8=residue(.8),
                           headroom_at_point8=1-residue(.8),
                           maximum_input_peak_no_noise=brentq(lambda x:residue(x)-1,.7,.9),
                           maximum_input_peak_six_sigma=brentq(
                               lambda x:residue(x)+6*.0002-1,.7,.9)))
    return rows, bounds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty')
    args.output.mkdir(parents=True,exist_ok=True)
    save_csv(args.output/'balanced_moments.csv',balanced_moments())
    rows, bounds = headroom_controls()
    save_csv(args.output/'headroom.csv',rows)
    write_json(args.output/'headroom_bounds.json',bounds)


if __name__ == '__main__':
    main()
