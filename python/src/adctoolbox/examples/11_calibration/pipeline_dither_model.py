"""Behavioral first-stage gain calibration helpers for exp_cal01.

The plant and sine-fit baseline are example-only; gain tracking uses the public API. Voltages use
input and backend ranges [-1, 1]. A seven-level coarse stage has nominal gain
4 and an ideal quantized backend. A known PN voltage is injected AFTER the
coarse decision and BEFORE the residue amplifier:

    q = coarse_quantize(x)
    r = Q_backend(G * (x - q + a*p) + residue_noise)
    corrected = q + r/G_est - a*p

With independent PN, no overload, an accurate probe amplitude and a locally
constant gain, cov(p, r)/(a*var(p)) estimates G. Finite-record input/probe
correlation is estimator noise. Quantization need not be perfectly independent
of PN, so this is an approximate statistical estimator, not an exact identity.
The plant alone receives the true gain. The calibrator receives observable
codes/PN only. Its estimate is updated AFTER correction of each data block.
"""
from dataclasses import dataclass

import numpy as np

try:
    from .dither_gain import (
        DitherGainTracker, apply_gain_calibration,
        fit_pipeline_gain_sine as fit_foreground_gain,
    )
except ImportError:
    from dither_gain import (
        DitherGainTracker, apply_gain_calibration,
        fit_pipeline_gain_sine as fit_foreground_gain,
    )


@dataclass(frozen=True)
class PipelineConfig:
    nominal_gain: float = 4.0
    backend_bits: int = 12
    probe_amplitude: float = 0.0625
    input_noise_rms: float = 0.00028
    residue_noise_rms: float = 0.0002

    def __post_init__(self):
        if self.nominal_gain != 4.0:
            raise ValueError("This seven-level example uses nominal gain 4")
        if not isinstance(self.backend_bits, int) or not 2 <= self.backend_bits <= 24:
            raise ValueError("backend_bits must be an integer in [2, 24]")
        values = (self.probe_amplitude, self.input_noise_rms, self.residue_noise_rms)
        if not all(np.isfinite(v) for v in values) or self.probe_amplitude <= 0 or min(values[1:]) < 0:
            raise ValueError("Probe must be positive and noise RMS nonnegative, all finite")


class DitherPipeline:
    """Memoryless two-stage plant with persistent, independent RNG streams.

    The backend has 12 bits by default: nominally two effective coarse bits
    plus 12 backend bits give 14-bit ideal resolution. No DAC mismatch,
    nonlinear amplifier, aperture jitter or residue memory is modeled here.
    Call reset only for a new experiment, never between contiguous blocks.
    """

    def __init__(self, config=None, seed=1):
        self.config = config or PipelineConfig()
        self.seed = seed
        self.reset()

    def reset(self):
        streams = np.random.SeedSequence(self.seed).spawn(3)
        self._pn_rng, self._input_rng, self._residue_rng = [np.random.default_rng(s) for s in streams]
        self.sample_count = 0

    def convert(self, signal, gain, *, probe_enabled=True):
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1 or not signal.size or not np.isfinite(signal).all():
            raise ValueError("signal must be a finite nonempty one-dimensional array")
        gains = np.broadcast_to(np.asarray(gain, dtype=float), signal.shape)
        if not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("gain must be finite and positive")
        c = self.config
        # Always consume PN, including when disabled; independently seeded
        # probe-on/off plants then see exactly the same analog noise draws.
        pn = np.where(self._pn_rng.random(signal.size) < .5, -1., 1.)
        if not probe_enabled:
            pn.fill(0.)
        sampled = signal + self._input_rng.normal(0., c.input_noise_rms, signal.size)
        coarse_code = np.searchsorted(np.arange(-5, 6, 2)/8, sampled)
        coarse = (coarse_code - 3)/4
        analog_residue = gains * (sampled - coarse + c.probe_amplitude*pn)
        analog_residue += self._residue_rng.normal(0., c.residue_noise_rms, signal.size)
        levels = 2**c.backend_bits
        code = np.clip(np.floor((analog_residue+1)*levels/2), 0, levels-1).astype(np.int64)
        residue = (code+.5)*2/levels-1
        self.sample_count += signal.size
        return {"coarse": coarse, "backend": residue, "backend_code": code, "pn": pn,
                # True overload counters are evaluation diagnostics, not estimator inputs.
                "input_overloads": int(np.sum(abs(sampled) > 1)),
                "backend_overloads": int(np.sum(abs(analog_residue) >= 1))}


def reconstruct(coarse, backend, pn, gain_estimate, probe_amplitude):
    """Compatibility wrapper returning only the calibrated waveform."""
    return apply_gain_calibration(coarse, backend, pn, gain_estimate=gain_estimate,
                                  probe_amplitude=probe_amplitude)["calibrated_signal"]
