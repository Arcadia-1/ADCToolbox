"""Observable-only dither identification helpers for exp_cal02.

These are educational behavior models, not public API or transistor models.
See DITHER_COVERAGE.md for derivations, source attribution and limitations.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, max_len_seq


class PRBS:
    """Persistent maximal-length LFSR; seed selects a nonzero initial state."""

    def __init__(self, seed=1, bits=31):
        self.bits = bits
        rng = np.random.default_rng(seed)
        self.initial = rng.integers(0, 2, bits, dtype=np.int8)
        if not self.initial.any():
            self.initial[0] = 1
        self.reset()

    def reset(self):
        self.state = self.initial.copy()

    def draw(self, size):
        sequence, self.state = max_len_seq(self.bits, state=self.state, length=size)
        return 2.*sequence-1.


def quantize(value, bits=12):
    """Midrise backend; rail flags are observable, overload count is evaluator-only."""
    levels = 2**bits
    code = np.clip(np.floor((value+1)*levels/2), 0, levels-1).astype(np.int64)
    return ((code+.5)*2/levels-1, bool(np.any((code == 0) | (code == levels-1))),
            int(np.sum(abs(value) >= 1)))


def coarse4(value):
    return (np.searchsorted(np.arange(-5, 6, 2)/8, value)-3)/4


def covariance(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim == 1:
        left = left[:, None]
    if right.ndim == 1:
        right = right[:, None]
    return (left-left.mean(axis=0)).T @ (right-right.mean(axis=0))/len(left)


class MemoryPlant:
    """r[n] = G*v[n] + H*v[n-1], v = sampled_input - coarse + a*PN.

    Parameters may change at block boundaries. Analog history and random
    streams persist; quantization/noise are after the linear memory path.
    """

    def __init__(self, seed=1, amplitude=.0625, bits=12, noise=.0002):
        self.probe = PRBS(seed)
        self.rng = np.random.default_rng(seed+10000)
        self.amplitude, self.bits, self.noise = amplitude, bits, noise
        self.previous = 0.
        self.previous_probe = 0.

    def convert(self, x, parameters, probe_on=True):
        p = self.probe.draw(len(x))
        if not probe_on:
            p.fill(0.)
        q = coarse4(x)
        v = x-q+self.amplitude*p
        delayed = np.r_[self.previous, v[:-1]]
        pp = np.r_[self.previous_probe, p[:-1]]
        self.previous, self.previous_probe = v[-1], p[-1]
        analog = parameters[0]*v+parameters[1]*delayed
        analog += self.rng.normal(0., self.noise, len(x))
        r, rail, overload = quantize(analog, self.bits)
        return dict(q=q, r=r, p=p, pp=pp, rail=rail, overload=overload)


class MemoryCorrector:
    def __init__(self, amplitude=.0625):
        self.amplitude = amplitude
        self.previous = 0.

    def apply(self, capture, parameters):
        gain, history = parameters
        v, _ = lfilter([1/gain], [1, history/gain], capture['r'],
                       zi=[-history/gain*self.previous])
        self.previous = v[-1]
        return capture['q']+v-self.amplitude*capture['p']


class MemoryTracker:
    """Two-lag correlation with measured PN covariance, then EW averaging."""

    def __init__(self, amplitude=.0625, rate=.08, scalar=False):
        self.amplitude, self.rate, self.scalar = amplitude, rate, scalar
        self.parameters = np.array([4., 0.])

    def update(self, c):
        if c['rail']:
            return 'held_rail'
        p = c['p'][:, None] if self.scalar else np.column_stack([c['p'], c['pp']])
        gram = covariance(p, p)
        if np.linalg.eigvalsh(gram).min() < .1:
            return 'held_excitation'
        estimate = np.linalg.solve(gram, covariance(p, c['r'])).ravel()/self.amplitude
        if self.scalar:
            estimate = np.r_[estimate, 0.]
        if not (2 < estimate[0] < 6 and abs(estimate[1]) < .8):
            return 'held_bounds'
        self.parameters += self.rate*(estimate-self.parameters)
        return 'updated'


class CapacitorPlant:
    """Exact two-capacitor charge redistribution, 1.5-bit coarse stage.

    Cf=1-delta*p, Cs=1+delta*p, q in {-1,0,1}; nominal residue gain 2.
    r = (2*x-(1+delta*p)*q)/(1-delta*p). PRBS swaps the physical roles.
    This is related to CFCS, not an exact implementation of Sun et al.'s logic.
    """

    def __init__(self, seed=1, bits=13, noise=.0002):
        self.probe = PRBS(seed)
        self.rng = np.random.default_rng(seed+10000)
        self.bits, self.noise = bits, noise

    def convert(self, x, parameters, probe_on=True):
        p = self.probe.draw(len(x))
        if not probe_on:
            p.fill(1.)  # Fixed capacitor roles, not zero mismatch.
        q = np.where(x > .25, 1., np.where(x < -.25, -1., 0.))
        delta = parameters[0]
        analog = (2*x-(1+delta*p)*q)/(1-delta*p)
        analog += self.rng.normal(0., self.noise, len(x))
        r, rail, overload = quantize(analog, self.bits)
        return dict(q=q, r=r, p=p, y=(q+r)/2, rail=rail, overload=overload)


def capacitor_correct(c, parameters):
    return c['y']+parameters[0]*c['p']*(c['q']-c['y'])


class CapacitorTracker:
    """Instrument p*q distinguishes mismatch; p alone can cancel by symmetry.

    E[p*q*x]=0, x=y+delta*p*(q-y). PN is centered within each coarse
    code to remove finite-window constant nuisance terms. No true x is used.
    """

    def __init__(self, rate=.08, code_weighted=True):
        self.rate, self.code_weighted = rate, code_weighted
        self.parameters = np.zeros(1)
        self.numerator = 0.
        self.denominator = 0.
        self.observations = 0

    def update(self, c):
        if c['rail']:
            return 'held_rail'
        p, q, y = c['p'], c['q'], c['y']
        centered = p.copy()
        for code in (-1, 0, 1):
            selected = q == code
            if selected.any():
                centered[selected] -= p[selected].mean()
        instrument = centered*q if self.code_weighted else centered
        if np.mean(instrument**2) < .05:
            return 'held_excitation'
        weight = 1. if self.observations == 0 else self.rate
        self.numerator += weight*(np.mean(instrument*y)-self.numerator)
        self.denominator += weight*(np.mean(instrument*p*(q-y))-self.denominator)
        self.observations += 1
        if abs(self.denominator) < .01:
            return 'held_rank'
        estimate = -self.numerator/self.denominator
        if not abs(estimate) < .05:
            return 'held_bounds'
        self.parameters[:] = estimate
        return 'updated'


def compressed_amplifier(v, gain, beta):
    """Monotone compression defined by r+beta*r^3=gain*v, beta >= 0."""
    r = gain*v
    for _ in range(8):
        r -= (r+beta*r**3-gain*v)/(1+3*beta*r**2)
    return r


class NonlinearPlant:
    def __init__(self, seed=1, amplitude=.1, binary=False, bits=12, noise=.0002):
        self.probe = PRBS(seed)
        self.second = PRBS(seed+100, bits=29)
        self.rng = np.random.default_rng(seed+10000)
        self.amplitude, self.binary, self.bits, self.noise = amplitude, binary, bits, noise

    def convert(self, x, parameters, probe_on=True):
        first, second = self.probe.draw(len(x)), self.second.draw(len(x))
        p = first if self.binary else (2*first+second)/3
        if not probe_on:
            p.fill(0.)
        q = coarse4(x)
        # parameters are inverse coefficients [1/G, beta/G].
        gain, beta = 1/parameters[0], parameters[1]/parameters[0]
        analog = compressed_amplifier(x-q+self.amplitude*p, gain, beta)
        analog += self.rng.normal(0., self.noise, len(x))
        r, rail, overload = quantize(analog, self.bits)
        return dict(q=q, r=r, p=p, rail=rail, overload=overload)


def nonlinear_correct(c, parameters, amplitude=.1):
    return c['q']+parameters[0]*c['r']+parameters[1]*c['r']**3-amplitude*c['p']


class NonlinearTracker:
    """Two instrumental moments identify inverse polynomial coefficients.

    phi=[r,r^3], Z=[p, p^3-(41/45)*p], Cov(Z,phi)*c=Cov(Z,a*p).
    For equiprobable p in {-1,-1/3,1/3,1}, E[p^4]/E[p^2]=41/45.
    A binary PN makes the two instruments dependent and must be rejected.
    This is a derived educational estimator, not a reproduction of PCE.
    """

    def __init__(self, amplitude=.1, rate=.04, scalar=False):
        self.amplitude, self.rate, self.scalar = amplitude, rate, scalar
        self.parameters = np.array([.25, 0.])
        self.matrix = np.zeros((1, 1) if scalar else (2, 2))
        self.rhs = np.zeros((1, 1) if scalar else (2, 1))
        self.observations = 0
        self.condition = None

    def update(self, c):
        if c['rail']:
            return 'held_rail'
        p, r = c['p'], c['r']
        z = p[:, None] if self.scalar else np.column_stack([p, p**3-41/45*p])
        phi = r[:, None] if self.scalar else np.column_stack([r, r**3])
        if np.linalg.eigvalsh(covariance(z, z)).min() < 1e-6:
            return 'held_rank'
        weight = 1. if self.observations == 0 else self.rate
        self.matrix += weight*(covariance(z, phi)-self.matrix)
        self.rhs += weight*(covariance(z, self.amplitude*p)-self.rhs)
        self.observations += 1
        self.condition = float(np.linalg.cond(self.matrix))
        if self.condition > 1e5:
            return 'held_rank'
        estimate = np.linalg.solve(self.matrix, self.rhs).ravel()
        if self.scalar:
            estimate = np.r_[estimate, 0.]
        # Broad physical domain: positive inverse slope on the backend range.
        if not (.15 < estimate[0] < .4 and -.04 < estimate[1] < .15
                and estimate[0]+3*min(estimate[1], 0) > .1):
            return 'held_bounds'
        self.parameters[:] = estimate
        return 'updated'


def foreground_fit(plant, parameters, mechanism, size=32768, blocks=16):
    """Independent dedicated sine; estimate its amplitude/phase/DC jointly.

    No clean input labels or true parameters are passed to the LS estimator.
    parameters configure only the simulated device. Capacitor switching is
    retained during its dedicated foreground acquisition.
    """
    gram = rhs = None
    previous_q = 0.
    frequency = (int(round(.119*size)) | 1)/size
    for block in range(blocks):
        n = np.arange(block*size, (block+1)*size)
        signal = .81*np.sin(2*np.pi*frequency*n+.71)
        c = plant.convert(signal, parameters, probe_on=mechanism == 'capacitor')
        nuisance = np.column_stack([np.sin(2*np.pi*frequency*n), np.cos(2*np.pi*frequency*n),
                                    np.ones(size)])
        if mechanism in ('gain', 'memory'):
            features = np.column_stack([-c['q'], -np.r_[previous_q, c['q'][:-1]]])
            previous_q = c['q'][-1]
            target = c['r']
        elif mechanism == 'capacitor':
            features = (c['p']*(c['q']-c['y']))[:, None]
            nuisance = -nuisance
            target = -c['y']
        else:
            features = np.column_stack([c['r'], c['r']**3])
            nuisance = -nuisance
            target = -c['q']
        design = np.column_stack([features, nuisance])
        # Skip initial history mismatch between plant startup and sine model.
        if block == 0:
            design, target = design[1:], target[1:]
        if gram is None:
            gram = np.zeros((design.shape[1], design.shape[1]))
            rhs = np.zeros(design.shape[1])
        gram += design.T@design
        rhs += design.T@target
    solution = np.linalg.solve(gram, rhs)
    return solution[:features.shape[1]]
