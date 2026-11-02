"""Explicit behavioral PN mechanisms; example helpers, not public ADC APIs.

See DITHER_GUIDE.md for charge assumptions, observability and hardware limits.
Only HDC exp_cal08 is fixed-point qualified; these helpers use floating words.
"""
from __future__ import annotations
import numpy as np
try:
    from .dither_mechanisms import PRBS, quantize
    from .hdc_fixed import njit
except ImportError:
    from dither_mechanisms import PRBS, quantize
    from hdc_fixed import njit


class ThreePN:
    """Three equal persistent PN paths; optional one-sample lookahead buffer."""
    def __init__(self, seed=101, peak=.2):
        if not 0 < peak < 1:
            raise ValueError('peak must be in (0,1)')
        self.peak = peak
        self.streams = [PRBS(seed,31), PRBS(seed+100,29), PRBS(seed+300,23)]
        self.pending = None

    def draw(self, size):
        if size < 1:
            raise ValueError('size must be positive')
        if self.pending is None:
            p = np.column_stack([s.draw(size) for s in self.streams]).astype(np.int8)
        else:
            tail = np.column_stack([s.draw(size-1) for s in self.streams]).astype(np.int8) if size > 1 else np.empty((0,3),np.int8)
            p = np.vstack([self.pending,tail]); self.pending = None
        return p, self.peak/3*p.sum(axis=1)

    def adjacent(self, size):
        p,d = self.draw(size)
        next_p,_ = self.draw(1)
        self.pending = next_p[0].copy()
        return p,d,np.r_[d[1:],self.peak/3*next_p[0].sum()]


def nominal_sar_weights(decisions=14, radix=1.85):
    if decisions < 4 or not 1 < radix <= 2:
        raise ValueError('require >=4 decisions and radix in (1,2]')
    weights = radix**(-np.arange(1,decisions+1,dtype=float))
    return weights/weights.sum()


@njit(cache=False)
def sar_decisions(x, weights):
    """Signed sub-binary successive decisions, actual weights in analog plant."""
    bits = np.empty((len(x),len(weights)),dtype=np.int8)
    for i in range(len(x)):
        residue = x[i]
        for j in range(len(weights)):
            bit = 1 if residue >= 0 else -1
            bits[i,j] = bit
            residue -= bit*weights[j]
    return bits


@njit(cache=False)
def sar_adapt(plus, minus, d, weights, nominal, rate, active, learn):
    """Output uses old weights; sign/zero NLMS update follows each held pair."""
    output = np.empty(len(d))
    updated = rejected = unexcited = 0
    for n in range(len(d)):
        yp = ym = 0.
        norm = 0.
        for j in range(len(weights)):
            yp += plus[n,j]*weights[j]
            ym += minus[n,j]*weights[j]
            if j < active:
                norm += (plus[n,j]-minus[n,j])**2
        output[n] = (yp+ym)/2
        if not learn:
            continue
        if norm == 0:
            unexcited += 1
            continue
        factor = rate*(2*d[n]-(yp-ym))/norm
        valid = True
        for j in range(active):
            candidate = weights[j]+factor*(plus[n,j]-minus[n,j])
            if not .9*nominal[j] < candidate < 1.1*nominal[j]:
                valid = False
        if valid:
            for j in range(active):
                weights[j] += factor*(plus[n,j]-minus[n,j])
            updated += 1
        else:
            rejected += 1
    return output,updated,rejected,unexcited


class SARWeights:
    def __init__(self, nominal=None, active=4, rate=2.**-10):
        self.nominal = nominal_sar_weights() if nominal is None else np.array(nominal,dtype=float)
        if not 0 < active <= len(self.nominal) or not 0 < rate <= 1:
            raise ValueError('invalid active weight count or rate')
        self.weights = self.nominal.copy()
        self.active,self.rate = active,rate
        self.updated = self.rejected = self.unexcited = 0

    def process(self, plus, minus, d, learn=True):
        if plus.shape != minus.shape or plus.shape != (len(d),len(self.weights)):
            raise ValueError('decision vectors and probe length disagree')
        y,u,r,z = sar_adapt(plus,minus,d,self.weights,self.nominal,self.rate,self.active,learn)
        self.updated += u; self.rejected += r; self.unexcited += z
        return y


class TIMoments:
    """Cumulative channel moments for the specified linear PN interpolation.

    No unknown x, slope of x, gain or skew is supplied to this estimator.
    A known analog probe variance defines the absolute gain reference.
    """
    def __init__(self, channels=2, peak=.2):
        self.channels,self.variance = channels,peak*peak/3
        self.sums = np.zeros((channels,2))
        self.count = np.zeros(channels,dtype=np.int64)

    def update(self, y, d0, d1, labels):
        for c in range(self.channels):
            mask = labels == c
            self.sums[c,0] += np.sum(y[mask]*d0[mask])
            self.sums[c,1] += np.sum(y[mask]*d1[mask])
            self.count[c] += np.count_nonzero(mask)

    def estimate(self):
        if np.any(self.count == 0):
            return dict(valid=False,gain=np.ones(self.channels),skew=np.zeros(self.channels))
        moments = self.sums/self.count[:,None]/self.variance
        gain = moments.sum(axis=1)
        if np.any(abs(gain) < 1e-12):
            return dict(valid=False,gain=gain,skew=np.zeros(self.channels))
        skew = moments[:,1]/gain
        valid = bool(np.all((gain>.8)&(gain<1.2)&(abs(skew)<.125)))
        return dict(valid=valid,gain=gain,skew=skew)


@njit(cache=False)
def sigma_delta(x,d,at_quantizer=True,state=0.):
    """First-order loop. e is evaluator-only; only y and d are observable.

    Quantizer injection: y=x+Delta(d)+Delta(e).
    Input injection:     y=x+d+Delta(e).
    """
    y = np.empty(len(x)); e = np.empty(len(x))
    maximum = abs(state)
    for n in range(len(x)):
        u = state+x[n]+(0. if at_quantizer else d[n])
        v = u+(d[n] if at_quantizer else 0.)
        y[n] = 1. if v >= 0 else -1.
        e[n] = y[n]-v
        state = u-y[n]
        maximum = max(maximum,abs(state))
    return y,e,state,maximum


def difference(x, previous=0.):
    return x-np.r_[previous,x[:-1]]


def inband_metrics(y, clean, bins, osr=32):
    """Coherent rectangular FFT; DC cases report dBFS noise/spur, not SNDR."""
    y = np.asarray(y)
    power = abs(np.fft.rfft(y-y.mean()))**2/len(y)**2
    power[1:-1] *= 2
    end = len(y)//(2*osr)
    power[end+1:] = 0
    signal = float(power[bins].sum()) if bins else 0.
    strongest = float(power[bins].max()) if bins else 0.
    if bins:
        power[bins] = 0.
    power[0] = 0.
    noise,spur = float(power.sum()),float(power.max())
    # 0 dBFS refers to the RMS power .5 of a peak-one sinusoid.
    result = dict(inband_noise_dbfs=float(10*np.log10(max(noise,1e-300)/.5)),
                  largest_inband_spur_dbfs=float(10*np.log10(max(spur,1e-300)/.5)),
                  dc_error=float(y.mean()-np.mean(clean)))
    if bins:
        result.update(sndr_db=float(10*np.log10(signal/max(noise,1e-300))),
                      sfdr_db=float(10*np.log10(strongest/max(spur,1e-300))))
    return result
