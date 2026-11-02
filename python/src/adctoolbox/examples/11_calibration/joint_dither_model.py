"""Joint dither experiment helpers; educational example, not public API.

One 1.5-bit capacitor stage, followed by a two-tap linear path and a
memoryless compressive amplifier. See JOINT_DITHER.md for the charge equation,
injection convention, observable moments, and limits of this model.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter

try:
    from .dither_mechanisms import PRBS, compressed_amplifier, covariance, quantize
except ImportError:
    from dither_mechanisms import PRBS, compressed_amplifier, covariance, quantize


@dataclass(frozen=True)
class Parameters:
    gain: float = 2.
    history: float = 0.
    delta: float = 0.
    beta: float = 0.

    def __post_init__(self):
        values = [self.gain, self.history, self.delta, self.beta]
        if (not np.isfinite(values).all() or self.gain <= 0
                or abs(self.history) >= self.gain or abs(self.delta) >= .1
                or self.beta < 0):
            raise ValueError('Require positive gain, |history| < gain, |delta| < .1, beta >= 0')

    def inverse(self):
        return np.array([1/self.gain, self.beta/self.gain,
                         self.history/self.gain, self.delta])


INITIAL = Parameters(1.94, .06, .006, .16)
FINAL = Parameters(2.04, .12, .012, .24)
NOMINAL = Parameters()
COEFFICIENT_NAMES = ('c1', 'c3', 'h', 'delta')


def physical(coefficients):
    c1, c3, h, delta = coefficients
    return dict(gain=float(1/c1), history=float(h/c1),
                delta=float(delta), beta=float(c3/c1))


class JointPlant:
    """Charge redistribution -> additive probe -> memory -> compression -> ADC.

    w=(2*x-(1+delta*s)*q)/(1-delta*s), v=w/2+a*p,
    r+beta*r**3=gain*v+history*v_previous.
    q is the unperturbed 1.5-bit decision. s switches capacitor roles; p is
    a separate four-level excitation. a is known in w/2 coordinate units.
    Physical parameters and unquantized peaks never enter a tracker capture.
    """

    def __init__(self, seed=11, amplitude=.1, bits=13, noise=.0002, binary=False):
        if not np.isfinite(amplitude) or amplitude < 0:
            raise ValueError('amplitude must be finite and nonnegative')
        if not isinstance(bits, int) or not 2 <= bits <= 24:
            raise ValueError('bits must be an integer in [2, 24]')
        if not np.isfinite(noise) or noise < 0:
            raise ValueError('noise must be finite and nonnegative')
        self.amplitude, self.bits, self.noise, self.binary = amplitude, bits, noise, binary
        self.first, self.second = PRBS(seed, 31), PRBS(seed+100, 29)
        self.switch = PRBS(seed+200, 27)
        self.rng = np.random.default_rng(seed+10000)
        self.previous_v = self.previous_p = self.previous_q = 0.

    def convert(self, x, parameters=INITIAL, probe_on=True, switch_on=True):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError('x must be a nonempty finite 1D array')
        first, second = self.first.draw(len(x)), self.second.draw(len(x))
        p = first if self.binary else (2*first+second)/3
        s = self.switch.draw(len(x))
        if not probe_on:
            p.fill(0.)
        if not switch_on:
            s.fill(1.)
        q = np.where(x > .25, 1., np.where(x < -.25, -1., 0.))
        w = (2*x-(1+parameters.delta*s)*q)/(1-parameters.delta*s)
        v = w/2+self.amplitude*p
        vp = np.r_[self.previous_v, v[:-1]]
        pp = np.r_[self.previous_p, p[:-1]]
        qp = np.r_[self.previous_q, q[:-1]]
        self.previous_v, self.previous_p, self.previous_q = v[-1], p[-1], q[-1]
        drive = parameters.gain*v+parameters.history*vp
        compressed = compressed_amplifier(drive, 1., parameters.beta)
        analog = compressed+self.rng.normal(0., self.noise, len(x))
        r, rail, overload = quantize(analog, self.bits)
        capture = dict(q=q, qp=qp, r=r, p=p, pp=pp, s=s, rail=rail)
        # Separate evaluator-only diagnostics from estimator-accessible data.
        diagnostics = dict(cap_residue_peak=float(np.max(abs(w))),
            injected_v_peak=float(np.max(abs(v))), linear_drive_peak=float(np.max(abs(drive))),
            backend_peak=float(np.max(abs(analog))), headroom=float(1-np.max(abs(analog))),
            overloads=overload,
            rail_samples=int(np.sum(abs(r) >= 1-1/2**self.bits)))
        return capture, diagnostics


class JointCorrector:
    """Undo cubic, undo memory, subtract p, then undo capacitor mismatch."""

    def __init__(self, amplitude=.1):
        self.amplitude = amplitude
        self.previous_v = 0.

    def apply(self, capture, coefficients):
        c1, c3, h, delta = coefficients
        if (not np.isfinite(coefficients).all() or abs(h) >= 1 or abs(delta) >= 1
                or c1+3*min(c3, 0) <= 0):
            raise ValueError('Require a monotone cubic inverse and stable memory inverse')
        r = capture['r']
        v, _ = lfilter([1.], [1., h], c1*r+c3*r**3, zi=[-h*self.previous_v])
        self.previous_v = v[-1]
        y = capture['q']/2+v-self.amplitude*capture['p']
        return dict(signal=y+delta*capture['s']*(capture['q']-y), capacitor_raw=y)


class JointTracker:
    """Triangular identification: 3 electronics moments, then cap moment.

    Z=[p, p^3-41*p/45, p_previous]. With unknown business residues as nuisance,
    Cov(Z, [r,r^3,-a*p_previous]) @ [c1,c3,h] = Cov(Z,a*p).
    By default Z is centered within each observed (q,q_previous) pair to
    remove constant within-code nuisance terms, without a clean-input label.
    Capacitor identification uses y produced by the *start-of-block*
    electronics coefficients, so no estimate is applied retroactively.
    Neither ideal x, truth coefficients nor the evaluator diagnostics are used.
    """

    def __init__(self, amplitude=.1, rate=.04, cap_rate=.04, condition_codes=True):
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError('amplitude must be positive and finite')
        if not (0 < rate <= 1 and 0 < cap_rate <= 1):
            raise ValueError('averaging rates must lie in (0, 1]')
        self.amplitude, self.rate, self.cap_rate = amplitude, rate, cap_rate
        self.condition_codes = condition_codes
        self.coefficients = NOMINAL.inverse()
        self.matrix, self.rhs = np.zeros((3, 3)), np.zeros((3, 1))
        self.observations = self.cap_observations = 0
        self.numerator = self.denominator = 0.
        self.condition = None
        self.deembed = JointCorrector(amplitude)

    def update(self, capture):
        # Maintain the signal-path state even if estimation is held for rails.
        y = self.deembed.apply(capture, self.coefficients)['capacitor_raw']
        if capture['rail']:
            return dict(electronics='held_rail', capacitor='held_rail')
        p, pp, r = capture['p'], capture['pp'], capture['r']
        z = np.column_stack([p, p**3-41/45*p, pp])
        if self.condition_codes:
            groups = (capture['q']+1)*3+capture['qp']+1
            for group in np.unique(groups):
                selected = groups == group
                z[selected] -= z[selected].mean(axis=0)
        zg = covariance(z, z)
        if np.linalg.eigvalsh(zg).min() < 1e-6:
            # Without probe rank, the electronics are unknown: do not let a
            # biased nominal deembedding claim successful cap calibration.
            return dict(electronics='held_rank', capacitor='held_electronics')
        weight = 1. if self.observations == 0 else self.rate
        features = np.column_stack([r, r**3, -self.amplitude*pp])
        self.matrix += weight*(covariance(z, features)-self.matrix)
        self.rhs += weight*(covariance(z, self.amplitude*p)-self.rhs)
        self.observations += 1
        self.condition = float(np.linalg.cond(self.matrix))
        status = 'held_rank'
        if np.isfinite(self.condition) and self.condition < 1e5:
            estimate = np.linalg.solve(self.matrix, self.rhs).ravel()
            c1, c3, h = estimate
            if (.35 < c1 < .75 and -.08 < c3 < .3 and abs(h) < .2
                    and c1+3*min(c3, 0) > .2):
                self.coefficients[:3] = estimate
                status = 'updated'
            else:
                status = 'held_bounds'
        # The first block used nominal electronics, so skip its cap moment.
        if status != 'updated' or self.observations == 1:
            return dict(electronics=status, capacitor='held_electronics')
        s, q = capture['s'], capture['q']
        centered = s.copy()
        for code in (-1., 0., 1.):
            selected = q == code
            if selected.any():
                centered[selected] -= s[selected].mean()
        instrument = centered*q
        if np.mean(instrument**2) < .05:
            return dict(electronics=status, capacitor='held_excitation')
        weight = 1. if self.cap_observations == 0 else self.cap_rate
        self.numerator += weight*(np.mean(instrument*y)-self.numerator)
        self.denominator += weight*(np.mean(instrument*s*(q-y))-self.denominator)
        self.cap_observations += 1
        cap_status = 'held_rank'
        if abs(self.denominator) >= .01:
            delta = -self.numerator/self.denominator
            if abs(delta) < .05:
                self.coefficients[3] = delta
                cap_status = 'updated'
            else:
                cap_status = 'held_bounds'
        return dict(electronics=status, capacitor=cap_status)
