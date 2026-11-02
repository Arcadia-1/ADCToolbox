"""Shared HDC arithmetic and scheduling kernel, independent of example plants.

Panigada/Galton JSSC 2009 equations (8)-(11), additive probe convention;
one-tap memory and two-element mismatch are the documented model extensions.
"""
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function


@njit(cache=False)
def rounded_shift(value, shift):
    if shift == 0:
        return value
    half = 1 << (shift-1)
    return (value+half) >> shift if value >= 0 else -((-value+half) >> shift)


@njit(cache=False)
def saturate(value, bits):
    return min(max(value, -(1 << (bits-1))), (1 << (bits-1))-1)


@njit(cache=False)
def mul(a, b, fraction, bits):
    raw = rounded_shift(a*b, fraction)
    result = saturate(raw, bits)
    return result, int(raw != result)


@njit(cache=False)
def fixed_block(r, p, pp, s, q, coeff, prev, a, fraction, bits, accumulator_bits, sums):
    """Six general products/sample: four HDC + memory + eta square.
    DAC correction and every correlation are sign/zero selections. Full product
    precision is retained until each explicit rounding operation.
    """
    scale = 1 << fraction
    output = np.empty(len(r), dtype=np.int64)
    saturations = accumulator_saturations = 0
    for i in range(len(r)):
        c1, k3, h, delta = coeff
        d = a*(p[i, 0]+p[i, 1]+p[i, 2])
        linear, ov = mul(c1, r[i], fraction, bits); saturations += ov
        square, ov = mul(linear, linear, fraction, bits); saturations += ov
        cube, ov = mul(square, linear, fraction, bits); saturations += ov
        nonlinear, ov = mul(k3, cube, fraction, bits); saturations += ov
        raw_w = linear-nonlinear
        w = saturate(raw_w, bits); saturations += int(w != raw_w)
        memory, ov = mul(h, prev, fraction, bits); saturations += ov
        raw_v = w-memory
        v = saturate(raw_v, bits); saturations += int(v != raw_v)
        prev = v
        y = q[i]*(scale//2)+v-d
        correction = rounded_shift(delta, 1)*s[i]*q[i]
        raw_x = y+correction
        output[i] = saturate(raw_x, bits); saturations += int(output[i] != raw_x)
        ss = r[i]-d
        eta, ov = mul(ss, ss, fraction, bits); saturations += ov
        values = (ss*p[i, 0], ss*p[i, 0]*p[i, 1]*p[i, 2], eta,
                  w*pp[i], s[i]*q[i]*y, q[i]*q[i])
        for j in range(6):
            raw = sums[j]+values[j]
            sums[j] = saturate(raw, accumulator_bits)
            accumulator_saturations += int(raw != sums[j])
    return output, prev, saturations, accumulator_saturations


@njit(cache=False)
def float_block(r, p, pp, s, q, coeff, prev, a, sums):
    output = np.empty(len(r))
    for i in range(len(r)):
        c1, k3, h, delta = coeff
        d = a*(p[i, 0]+p[i, 1]+p[i, 2])
        linear = c1*r[i]
        w = linear-k3*linear**3
        v = w-h*prev
        prev = v
        y = q[i]/2+v-d
        output[i] = y+delta/2*s[i]*q[i]
        ss = r[i]-d
        sums[0] += ss*p[i, 0]
        sums[1] += ss*p[i, 0]*p[i, 1]*p[i, 2]
        sums[2] += ss*ss
        sums[3] += w*pp[i]
        sums[4] += s[i]*q[i]*y
        sums[5] += q[i]*q[i]
    return output, prev


def divide_round(numerator, denominator):
    """Integer division, nearest, ties away from zero; no floating arithmetic."""
    if denominator == 0:
        raise ZeroDivisionError
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    return sign*((abs(int(numerator))+abs(int(denominator))//2)//abs(int(denominator)))


def serial_divide_round(numerator, denominator, width):
    """Restoring unsigned divider, one quotient bit/cycle plus final rounding.

    Sign conversion is outside the divider. Width is the dividend/quotient
    register width; overflow raises instead of silently using unbounded ints.
    """
    if denominator == 0:
        raise ZeroDivisionError
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    dividend, divisor = abs(int(numerator)), abs(int(denominator))
    if max(dividend, divisor) >= 1 << width:
        raise OverflowError('serial divider operand does not fit')
    remainder = quotient = 0
    for bit in range(width-1, -1, -1):
        remainder = (remainder << 1) | ((dividend >> bit) & 1)
        if remainder >= divisor:
            remainder -= divisor
            quotient |= 1 << bit
    quotient += int(2*remainder >= divisor)
    if quotient >= 1 << width:
        raise OverflowError('serial divider rounded quotient does not fit')
    return sign*quotient


class Tracker:
    """Average-and-dump HDC; no matrix solve, gradients, or truth labels.

    Every dump produces four shadow coefficients, committed atomically after
    the controller delay. Sampling and accumulation continue during the delay.
    """
    def __init__(self, fixed=True, fraction=24, bits=28, window_log2=20,
                 accumulator_bits=None, controller_latency=512, dither_peak=.2):
        if not 8 <= fraction <= 26 or not fraction+2 <= bits <= 30:
            raise ValueError('Require 8 <= fraction <= 26 and fraction+2 <= bits <= 30')
        if not 8 <= window_log2 <= 30:
            raise ValueError('window_log2 must be in [8,30]')
        if not 0 <= controller_latency < 2**window_log2:
            raise ValueError('controller latency must fit within one window')
        if not np.isfinite(dither_peak) or not 0 < dither_peak < 1:
            raise ValueError('Dither peak must be finite and in (0,1) in residue-input units')
        self.fixed, self.fraction, self.bits = fixed, fraction, bits
        self.window_log2, self.window = window_log2, 1 << window_log2
        self.accumulator_bits = accumulator_bits or bits+window_log2
        if not 8 <= self.accumulator_bits <= 62:
            raise ValueError('Accumulator width must be in [8,62]')
        self.scale = 1 << fraction
        self.divider_width = max(2*bits, self.accumulator_bits+1)
        self.dither_peak = float(dither_peak)
        self.a = int(np.floor(self.scale*dither_peak/3+.5)) if fixed else dither_peak/3
        if fixed:
            # Check the SAME staged rounding used by the controller. Testing
            # ideal A**3 or only A != 0 misses underflow at F=8,9,10 for peak=.2.
            a2, ov2 = mul(self.a, self.a, fraction, bits)
            a3, ov3 = mul(a2, self.a, fraction, bits)
            if self.a == 0 or a3 == 0 or ov2 or ov3:
                raise ValueError('Dither moment denominator is zero or saturated after '
                                 'fixed-point rounding; increase fraction or dither_peak')
        self.coefficients = np.array([self.scale, 0, 0, 0], dtype=np.int64) if fixed else np.array([1., 0., 0., 0.])
        self.sums = np.zeros(6, dtype=np.int64 if fixed else float)
        self.previous = 0 if fixed else 0.
        self.count = self.updates = self.held = self.samples = 0
        self.controller_latency = controller_latency
        self.pending = None
        self.commit_at = None
        self.window_bad = False
        self.saturations = self.accumulator_saturations = self.controller_saturations = 0
        self.events = []

    def values(self):
        return self.coefficients/self.scale if self.fixed else self.coefficients.copy()

    def physical(self):
        c1, k3, h, delta = self.values()
        return dict(gain=2/c1, cubic=k3/c1, history=2*h/c1, delta=delta)

    def _estimate(self):
        if self.fixed:
            f, b, S = self.fraction, self.bits, self.scale
            def qm(x, y):
                result, ov = mul(int(x), int(y), f, b)
                self.controller_saturations += ov
                return int(result)
            def qd(x, y):
                raw = serial_divide_round(int(x)*S, int(y), self.divider_width)
                result = int(saturate(raw, b))
                self.controller_saturations += int(raw != result)
                return result
            means = [int(rounded_shift(int(v), self.window_log2)) for v in self.sums[:5]]
            a2 = qm(self.a, self.a)
            a3 = qm(a2, self.a)
            gamma1 = qd(means[0], self.a)
            gamma3 = qd(means[1], 6*a3)
            gain = S+gamma1-qm(7*a2+3*means[2], gamma3)
            if not 3*S//4 < gain < 5*S//4:
                return None
            c1, k3 = qd(S, gain), qd(gamma3, gain)
            h = qd(means[3], self.a)
            occupancy = int(self.sums[5])
            delta = serial_divide_round(-2*int(self.sums[4]), occupancy, self.divider_width) if occupancy > self.window//20 else int(self.coefficients[3])
            result = np.array([c1, k3, h, delta], dtype=np.int64)
            valid = abs(k3) < 2*S and abs(h) < S//4 and abs(delta) < S//20
        else:
            means = self.sums/self.window
            gamma1, gamma3, eta = means[0]/self.a, means[1]/(6*self.a**3), means[2]
            gain = 1+gamma1-(7*self.a**2+3*eta)*gamma3
            if not .75 < gain < 1.25:
                return None
            delta = -2*means[4]/means[5] if means[5] > .05 else self.coefficients[3]
            result = np.array([1/gain, gamma3/gain, means[3]/self.a, delta])
            valid = abs(result[1]) < 2 and abs(result[2]) < .25 and abs(result[3]) < .05
        return result if valid else None

    def process(self, capture, learn=True):
        if 'dither_peak' in capture and capture['dither_peak'] != self.dither_peak:
            raise ValueError('Capture dither_peak must match the Tracker injection scale')
        outputs = []
        offset = 0
        while offset < len(capture['r']):
            if self.pending is not None and self.samples == self.commit_at:
                self.coefficients = self.pending
                self.pending = self.commit_at = None
            size = min(len(capture['r'])-offset, self.window-self.count)
            if self.pending is not None:
                size = min(size, self.commit_at-self.samples)
            sl = slice(offset, offset+size)
            p, pp, s, q = (capture[k][sl] for k in ('p', 'pp', 's', 'q'))
            if self.fixed:
                r = np.rint(capture['r'][sl]*self.scale).astype(np.int64)
                y, self.previous, ov, aov = fixed_block(r, p, pp, s, q,
                    self.coefficients, self.previous, self.a, self.fraction, self.bits,
                    self.accumulator_bits, self.sums)
                self.saturations += ov; self.accumulator_saturations += aov
                self.window_bad |= bool(ov or aov)
                outputs.append(y/self.scale)
            else:
                y, self.previous = float_block(capture['r'][sl], p, pp, s, q,
                    self.coefficients, self.previous, self.a, self.sums)
                outputs.append(y)
            self.window_bad |= bool(np.any(capture['rails'][sl])) if 'rails' in capture else bool(capture['rail'])
            self.count += size; self.samples += size; offset += size
            if self.count == self.window:
                old_overflow = self.controller_saturations
                estimate = self._estimate() if learn and not self.window_bad else None
                if self.controller_saturations != old_overflow:
                    estimate = None
                if learn:
                    self.events.append(dict(sample=self.samples, accepted=estimate is not None,
                        rails_or_overflow=self.window_bad, cap_excited=bool(self.sums[5] > self.window//20),
                        shadow_coefficients=None if estimate is None else
                            (estimate/self.scale if self.fixed else estimate).tolist()))
                    if estimate is not None:
                        self.pending = estimate
                        self.commit_at = self.samples+self.controller_latency
                        self.updates += 1
                    else:
                        self.held += 1
                self.sums[:] = 0; self.count = 0; self.window_bad = False
        return np.concatenate(outputs)
