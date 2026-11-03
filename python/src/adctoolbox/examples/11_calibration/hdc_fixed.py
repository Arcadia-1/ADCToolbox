"""Three-PN HDC with bounded integer arithmetic; example plant and audited calibration kernel.

HDC follows Panigada/Galton JSSC 2009 equations (8)-(11), with additive
instead of subtractive probe polarity. A one-tap memory inverse and a
two-element DAC mismatch correlator are explicit extensions, not from HDC.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
# Datapath audits use the private integer kernel, which may change between
# releases; applications should use adctoolbox.DitherHDCTracker instead.
from adctoolbox.calibration._dither_hdc import (
    Tracker, njit, rounded_shift, saturate, mul, fixed_block, float_block,
    divide_round, serial_divide_round,
)
try:
    from .dither_mechanisms import PRBS, quantize
except ImportError:
    from dither_mechanisms import PRBS, quantize


@dataclass(frozen=True)
class Device:
    gain: float = 1.94
    cubic: float = -.16
    history: float = .06
    delta: float = .006

    def coefficients(self):
        g = self.gain/2
        return np.array([1/g, self.cubic/g, self.history/self.gain, self.delta])


INITIAL = Device()
FINAL = Device(2.04, -.24, .12, .012)


class Plant:
    """Fixed Cf; two DAC elements .5*(1+/-delta); three separate probe caps.

    u=x-(1+delta*s)*q/2, v=u+d, t=v+(H/G)*v_previous,
    backend=2*((G/2)*t+cubic*t**3). d=peak*(p1+p2+p3)/3.
    This is a forward cubic, NOT the previous exact-inverse-cubic plant.
    """
    def __init__(self, seed=101, peak=.2, bits=13, noise=.0002, probe_error=0.):
        if not np.isfinite(peak) or not 0 < peak < 1:
            raise ValueError('Dither peak must be finite and in (0,1) in residue-input units')
        self.peak, self.bits, self.noise, self.probe_error = peak, bits, noise, probe_error
        self.pn = [PRBS(seed, 31), PRBS(seed+100, 29), PRBS(seed+300, 23)]
        self.switch = PRBS(seed+200, 27)
        self.rng = np.random.default_rng(seed+10000)
        self.previous_v = 0.
        self.previous_p = 0

    def convert(self, x, truth=FINAL):
        x = np.asarray(x)
        p = np.column_stack([v.draw(len(x)) for v in self.pn]).astype(np.int64)
        s = self.switch.draw(len(x)).astype(np.int64)
        q = np.where(x > .25, 1, np.where(x < -.25, -1, 0)).astype(np.int64)
        d = self.peak/3*p.sum(axis=1)
        # Error on one of the three physical injection capacitors, diagnostic only.
        actual_d = d+self.peak/3*self.probe_error*p[:, 0]
        u = x-(1+truth.delta*s)*q/2
        v = u+actual_d
        vp = np.r_[self.previous_v, v[:-1]]
        pp = np.r_[self.previous_p, p[:-1, 0]].astype(np.int64)
        self.previous_v, self.previous_p = v[-1], int(p[-1, 0])
        t = v+truth.history/truth.gain*vp
        backend = truth.gain*t+2*truth.cubic*t**3
        analog = backend+self.rng.normal(0., self.noise, len(x))
        measured, rail, overload = quantize(analog, self.bits)
        capture = dict(r=measured/2, p=p, pp=pp, s=s, q=q, rail=rail,
                       rails=abs(measured) >= 1-2.**(-self.bits), dither_peak=self.peak)
        diagnostics = dict(headroom=float(1-np.max(abs(analog))), overloads=overload,
                           probe_peak=float(np.max(abs(actual_d))),
                           internal_peak=float(np.max(abs(t))),
                           minimum_slope=float(np.min(truth.gain/2+3*truth.cubic*t*t)))
        return capture, diagnostics


def resource_budget(fraction=24, bits=28, window_log2=24):
    """Explicit architecture budget, not a synthesized gate/netlist count.

    Six sample-rate multipliers + one shared slow-controller multiplier;
    One unsigned restoring divider sized for both scaled words and accumulators.
    Sign is handled outside the divider; six divisions per accepted update.
    No extra sample pipeline registers are included: timing is unvalidated.
    """
    W, L = bits, window_log2
    D, R = max(2*W, W+L+1), max(W, L+1)
    registers = dict(pn_lfsr_bits=31+29+23+27, delayed_p1_bits=1,
        correlation_accumulators_bits=5*(W+L), occupancy_counter_bits=L+1,
        snapshot_bits=5*(W+L)+L+1, window_counter_bits=L,
        active_coefficients_bits=4*W, shadow_coefficients_bits=4*W,
        previous_v_bits=W, output_bits=W,
        controller_temporaries_bits=8*W,
        controller_product_bits=2*W, divider_dividend_quotient_bits=D,
        divider_remainder_bits=R+1, divider_denominator_bits=R,
        divider_counter_bits=int(np.ceil(np.log2(D+1))),
        controller_cycle_counter_bits=9, controller_flags_bits=8)
    return dict(sample_rate_multipliers=6, slow_shared_multipliers=1,
                total_multipliers=7, multiplier_operand_width=W,
                correlation_accumulators=5, occupancy_counter=1,
                accumulator_width=W+L, snapshot_words=6,
                serial_dividers=1, controller_latency_samples=512,
                divider_width=D, division_iterations_per_update=6*D,
                registers=registers, total_register_bits=sum(registers.values()),
                scope='One calibrated stage; sample-rate unpipelined datapath; excludes backend, '
                      'IO, test counters and physical analog elements; no synthesis/PPA claim')
