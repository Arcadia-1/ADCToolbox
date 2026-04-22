"""Monotonic digital-controlled variable delay line (VDL) and a multi-sampler model.

Simulation infrastructure for time-interleaved skew-calibration examples.
The samplers are modeled in the **analog domain** — each "capture" produces
continuous-valued (infinite-precision) samples of a cos(·) waveform. There
is *no ADC quantization* here; the only non-ideality is per-channel sampling-
instant skew and the VDL's code-to-delay nonlinearity. This is enough to
reproduce the MAD / autocorrelation behavior that the calibration algorithm
feeds on.

Classes
-------
VariableDelayLine
    N-code monotonic delay line. ``code -> delay_sec`` via a lookup table
    built from positive random step sizes. Step sizes share a common mean
    (the "LSB") and a coefficient of variation that controls DNL.

TIMultiSampler
    M time-interleaved samplers, each with its own intrinsic skew and its
    own VDL. The calibration algorithm only sees ``capture(...)`` output;
    ``intrinsic_skew`` and ``effective_skew()`` are the "ground truth" the
    algorithm has to discover and cancel.
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
class TIMultiSampler:
    """M time-interleaved analog samplers with per-channel skew and VDL trim.

    This is NOT a quantizing ADC — each ``capture`` returns continuous-valued
    (``float64``) samples of a pure cos(·) waveform. The only non-idealities
    are per-sampler intrinsic skew plus whatever delay the VDL adds. That is
    enough to drive a MAD / autocorrelation-based skew-calibration algorithm.

    Parameters
    ----------
    M : int
        Number of interleaved samplers (they fire round-robin).
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
        hd3_dbc: float | None = None,
        seed: int | None = None,
        randomize_phase: bool = True,
    ) -> np.ndarray:
        """Return one interleaved capture with the current trim codes.

        ``randomize_phase`` (default True) picks a uniform random starting
        phase for the sinusoid each call. This is essential for MAD-based
        calibration driven by a sinusoidal input: without it every capture
        has the same per-channel phase pattern, which creates a systematic
        finite-sample bias in the MAD estimate that the calibration algorithm
        wrongly interprets as skew.

        ``hd3_dbc`` (default None) applies a static cubic non-linearity
        ``y = x + k3·x³`` sized so the 3rd-harmonic tone sits at
        ``hd3_dbc`` dBc relative to the fundamental. Models a realistic
        input source / buffer non-linearity and caps post-calibration SFDR
        at this level — its phase is coherent with the fundamental (as a
        real cubic distortion would be), same convention as
        ``ADC_Signal_Generator.apply_static_nonlinearity`` in the toolbox.
        """
        T = 1.0 / self.fs
        skew = self.effective_skew()
        rng = np.random.default_rng(seed)
        phase0 = rng.uniform(0.0, 2 * np.pi) if randomize_phase else 0.0
        n = np.arange(n_samples)
        m = n % self.M
        t = n * T + skew[m]
        x = amp * np.cos(2 * np.pi * fin * t + phase0)
        if hd3_dbc is not None:
            # cos³(θ) = ¾·cos(θ) + ¼·cos(3θ). For target HD3 ratio r vs fund,
            # k3·amp²/4 = r ⇒ k3 = 4·r/amp².
            hd3_ratio = 10 ** (hd3_dbc / 20.0)
            k3 = 4.0 * hd3_ratio / (amp ** 2)
            x = x + k3 * x ** 3
        if noise_rms > 0:
            x = x + rng.standard_normal(n_samples) * noise_rms
        return x
