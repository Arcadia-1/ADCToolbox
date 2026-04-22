"""Monotonic digital-controlled variable delay line (VDL) and a TI-ADC model.

Simulation infrastructure for TI-ADC skew-calibration examples. Each channel
carries an independent VDL with its own random DNL (while staying monotonic),
so the algorithm under test has to deal with the same per-code nonlinearity a
real chip would expose.

Classes
-------
VariableDelayLine
    N-code monotonic delay line. ``code -> delay_sec`` via a lookup table
    built from positive random step sizes. Step sizes share a common mean
    (the "LSB") and a coefficient of variation that controls DNL.

TISARModel
    M-channel time-interleaved ADC with per-channel intrinsic skew + a
    per-channel VDL. The calibration algorithm only sees ``capture(...)``
    output; ``intrinsic_skew`` and ``effective_skew()`` are the "ground truth"
    the algorithm has to discover / cancel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class VariableDelayLine:
    """Monotonic digital-controlled delay line.

    Parameters
    ----------
    n_codes : int, default 128
        Number of codes, e.g. 2^7 = 128 for a 7-bit VDL.
    lsb_mean_sec : float, default 100e-15
        Mean step size (the nominal LSB).
    step_cv : float, default 0.15
        Coefficient of variation on the per-step size (relative std-dev).
        Step sizes are drawn once at construction and are always positive,
        so the resulting curve is guaranteed monotonically non-decreasing.
    offset_sec : float, optional
        Additional constant delay added to every code, modeling process
        offset that the VDL cannot cancel by itself.
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_codes: int = 128,
        lsb_mean_sec: float = 100e-15,
        step_cv: float = 0.15,
        offset_sec: float = 0.0,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        # Draw positive step sizes — floor at 5% of LSB to keep monotonicity
        # even under extreme CV.
        steps = lsb_mean_sec * np.maximum(
            1.0 + step_cv * rng.standard_normal(n_codes - 1),
            0.05,
        )
        delays = np.concatenate(([0.0], np.cumsum(steps)))
        # Center the curve so the midpoint code has zero delay.
        code_center = n_codes // 2
        delays -= delays[code_center]
        delays += offset_sec

        self.n_codes = n_codes
        self.delays = delays
        self.code_center = code_center
        self.code_min = 0
        self.code_max = n_codes - 1
        self.lsb_mean_sec = lsb_mean_sec

    def __call__(self, code):
        """Look up delay (seconds) for an integer code or array of codes."""
        code = np.clip(np.asarray(code, dtype=int), self.code_min, self.code_max)
        return self.delays[code]

    @property
    def total_range_sec(self) -> float:
        return float(self.delays[self.code_max] - self.delays[self.code_min])

    def nearest_code(self, target_delay_sec: float) -> int:
        """Return the code whose lookup delay is closest to the target."""
        return int(np.argmin(np.abs(self.delays - target_delay_sec)))


@dataclass
class TISARModel:
    """Synthetic TI-ADC with per-channel intrinsic skew and VDL trim.

    Parameters
    ----------
    M : int
        Sub-ADC count.
    fs : float
        Aggregate sample rate (Hz).
    intrinsic_skew_sec : array-like, shape (M,)
        Per-channel intrinsic sample-skew (ground truth, hidden from the
        calibration algorithm).
    vdls : list of VariableDelayLine, length M
        Per-channel programmable delay lines.
    trim_codes : array-like of int, optional
        Initial trim codes. Defaults to each VDL's ``code_center``.
    """

    M: int
    fs: float
    intrinsic_skew_sec: np.ndarray
    vdls: list
    trim_codes: np.ndarray = None

    def __post_init__(self) -> None:
        self.intrinsic_skew_sec = np.asarray(self.intrinsic_skew_sec, dtype=float)
        assert self.intrinsic_skew_sec.size == self.M
        assert len(self.vdls) == self.M
        if self.trim_codes is None:
            self.trim_codes = np.array([v.code_center for v in self.vdls], dtype=int)
        else:
            self.trim_codes = np.asarray(self.trim_codes, dtype=int).copy()

    def effective_skew(self) -> np.ndarray:
        """Total per-channel skew = intrinsic + VDL(code). Hidden from calibration."""
        vdl_delay = np.array([self.vdls[m](self.trim_codes[m]) for m in range(self.M)])
        return self.intrinsic_skew_sec + vdl_delay

    def capture(
        self,
        fin: float,
        amp: float,
        n_samples: int,
        noise_rms: float = 0.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return one interleaved capture with the current trim codes."""
        T = 1.0 / self.fs
        skew = self.effective_skew()
        n = np.arange(n_samples)
        m = n % self.M
        t = n * T + skew[m]
        x = amp * np.cos(2 * np.pi * fin * t)
        if noise_rms > 0:
            rng = np.random.default_rng(seed)
            x = x + rng.standard_normal(n_samples) * noise_rms
        return x
