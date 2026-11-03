"""Observable, code-stratified moments for the joint dither research example.

The original experiment is left unchanged so its saved source hashes remain
valid. This module keeps its plant, correction, guards and capacitor equation.
"""
from __future__ import annotations

import numpy as np

try:
    from . import joint_dither_model as jm
except ImportError:
    import joint_dither_model as jm


def grouped_moments(capture, amplitude, previous_coefficients):
    """Nine code pairs times three instruments; no waveform or truth labels.

    Weight covariance uses residuals from the start-of-block coefficients.
    Every moment is normalized by the total block length, including sparse
    groups. Empty groups contribute zero. This is feasible GMM, not an
    assertion of statistically optimal weights for a serial PRBS stream.
    """
    p, pp, r = (capture[k] for k in ('p', 'pp', 'r'))
    z = np.column_stack([p, p**3-41/45*p, pp])
    f = np.column_stack([r, r**3, -amplitude*pp])
    target = amplitude*p
    groups = ((capture['q']+1)*3+capture['qp']+1).astype(int)
    matrix, rhs, variance = np.zeros((9, 3, 3)), np.zeros((9, 3)), np.zeros((9, 3, 3))
    gram = np.zeros((3, 3))
    for group in np.unique(groups):
        selected = groups == group
        zg = z[selected]-z[selected].mean(axis=0)
        fg = f[selected]
        residual = fg @ previous_coefficients[:3]-target[selected]
        residual -= residual.mean()
        matrix[group] = zg.T @ fg/len(p)
        rhs[group] = zg.T @ target[selected]/len(p)
        score = zg*residual[:, None]
        variance[group] = score.T @ score/len(p)
        gram += zg.T @ zg/len(p)
    return matrix, rhs, variance, gram


def solve_grouped(matrix, rhs, variance):
    """Whiten moments, then solve by SVD; do not square the condition number."""
    eigenvalues, vectors = np.linalg.eigh(variance)
    cutoff = max(float(eigenvalues.max())*1e-8, 1e-15)
    weights = np.zeros_like(eigenvalues)
    valid = eigenvalues > cutoff
    weights[valid] = 1/np.sqrt(eigenvalues[valid])
    whitening = weights[:, :, None]*vectors.swapaxes(1, 2)
    design = (whitening @ matrix).reshape(-1, 3)
    target = (whitening @ rhs[:, :, None]).ravel()
    estimate, _, rank, singular = np.linalg.lstsq(design, target, rcond=1e-10)
    condition = float(singular[0]/singular[-1]) if rank == 3 else float('inf')
    return estimate, condition


class ConvergenceTracker:
    """Pooled or stratified electronics moments, EMA or cumulative averaging.

    rate=None retains all accepted blocks. No reset is triggered by device
    truth or by the evaluator's change point. Estimates apply next block.
    """

    def __init__(self, amplitude=.2, rate=.04, grouped=False):
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError('amplitude must be positive and finite')
        if rate is not None and not 0 < rate <= 1:
            raise ValueError('rate must be None or in (0, 1]')
        self.amplitude, self.rate, self.grouped = amplitude, rate, grouped
        self.coefficients = jm.NOMINAL.inverse()
        self.matrix = np.zeros((9, 3, 3))
        self.rhs = np.zeros((9, 3))
        self.variance = np.zeros((9, 3, 3))
        self.observations = self.cap_observations = 0
        self.numerator = self.denominator = 0.
        self.condition = None
        self.deembed = jm.JointCorrector(amplitude)

    def _weight(self, count):
        return 1/count if self.rate is None or count == 1 else self.rate

    def update(self, capture):
        y = self.deembed.apply(capture, self.coefficients)['capacitor_raw']
        if capture['rail']:
            return dict(electronics='held_rail', capacitor='held_rail')
        a, b, v, gram = grouped_moments(capture, self.amplitude, self.coefficients)
        if np.linalg.eigvalsh(gram).min() < 1e-6:
            return dict(electronics='held_rank', capacitor='held_electronics')
        self.observations += 1
        weight = self._weight(self.observations)
        self.matrix += weight*(a-self.matrix)
        self.rhs += weight*(b-self.rhs)
        self.variance += weight*(v-self.variance)
        if self.grouped:
            estimate, self.condition = solve_grouped(self.matrix, self.rhs, self.variance)
        else:
            matrix, rhs = self.matrix.sum(axis=0), self.rhs.sum(axis=0)
            self.condition = float(np.linalg.cond(matrix))
            estimate = np.linalg.solve(matrix, rhs) if self.condition < 1e5 else np.full(3, np.nan)
        status = 'held_rank'
        if self.condition < 1e5:
            c1, c3, h = estimate
            if (.35 < c1 < .75 and -.08 < c3 < .3 and abs(h) < .2
                    and c1+3*min(c3, 0) > .2):
                self.coefficients[:3] = estimate
                status = 'updated'
            else:
                status = 'held_bounds'
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
        self.cap_observations += 1
        weight = self._weight(self.cap_observations)
        self.numerator += weight*(np.mean(instrument*y)-self.numerator)
        self.denominator += weight*(np.mean(instrument*s*(q-y))-self.denominator)
        cap_status = 'held_rank'
        if abs(self.denominator) >= .01:
            delta = -self.numerator/self.denominator
            if abs(delta) < .05:
                self.coefficients[3] = delta
                cap_status = 'updated'
            else:
                cap_status = 'held_bounds'
        return dict(electronics=status, capacitor=cap_status)
