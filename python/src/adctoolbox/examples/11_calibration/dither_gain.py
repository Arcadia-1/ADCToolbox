"""Example-only Pipeline PN gain identification and correction.

The known probe must pass through the gain being identified, after the coarse
decision. This module does not inject a probe or model an ADC.
These helpers support the Dither experiments, not the public calibration API.
"""
import numpy as np


def apply_gain_calibration(coarse, backend, pn, *, gain_estimate, probe_amplitude):
    """Apply q + r/G_est - a*p using a frozen positive gain estimate.

    Inputs are aligned 1D observable arrays in consistent voltage/code units.
    ``pn`` is the normalized known probe; amplitude is in input-referred units.
    Returns ``calibrated_signal`` and ``gain_used``. The known probe must pass
    through the residue gain, after the coarse decision. This function only
    reconstructs the digital output; it does not inject a probe.
    """
    q, r, p = [np.asarray(a, float) for a in (coarse, backend, pn)]
    if q.ndim != 1 or r.shape != q.shape or p.shape != q.shape or not all(np.isfinite(a).all() for a in (q, r, p)):
        raise ValueError("coarse, backend and pn must be matching finite 1D arrays")
    if not np.isfinite(gain_estimate) or gain_estimate <= 0 or not np.isfinite(probe_amplitude) or probe_amplitude < 0:
        raise ValueError("gain_estimate must be positive and probe_amplitude nonnegative, both finite")
    return {"calibrated_signal": q+r/gain_estimate-probe_amplitude*p,
            "gain_used": float(gain_estimate)}


class DitherGainTracker:
    """Blockwise correlation estimate followed by exponential averaging.

    No ideal input, true gain, frequency or foreground label is accepted.
    backend_code is an observable saturation guard. Reject an entire block
    on a rail hit; selecting only non-clipped samples can bias PN correlation.
    The caller applies gain_estimate before calling update for that block.
    ``initial_gain`` may be a foreground estimate for hybrid startup. Unknown
    injection amplitude is inseparable from gain. Statistical bias/noise and
    excitation limitations remain architecture-dependent.
    """

    def __init__(self, probe_amplitude=.0625, initial_gain=4., learning_rate=.08,
                 backend_bits=12, gain_bounds=(2., 6.)):
        if not np.isfinite(probe_amplitude) or probe_amplitude <= 0:
            raise ValueError("probe_amplitude must be positive and finite")
        if not np.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not isinstance(backend_bits, int) or not 2 <= backend_bits <= 24:
            raise ValueError("backend_bits must be in [2, 24]")
        if len(gain_bounds) != 2 or not (0 < gain_bounds[0] <= initial_gain <= gain_bounds[1] < np.inf):
            raise ValueError("initial_gain must lie in finite positive gain_bounds")
        self.probe_amplitude = probe_amplitude
        self.initial_gain = initial_gain
        self.learning_rate = learning_rate
        self.backend_bits = backend_bits
        self.gain_bounds = gain_bounds
        self.reset()

    def reset(self):
        """Restore the configured initial gain and zero accepted updates."""
        self.gain_estimate = float(self.initial_gain)
        self.updates = 0

    def apply(self, coarse, backend, pn):
        """Correct with the current snapshot, without updating parameters."""
        return apply_gain_calibration(coarse, backend, pn,
                                      gain_estimate=self.gain_estimate,
                                      probe_amplitude=self.probe_amplitude)

    def update(self, backend, pn, backend_code):
        """Return gain_next, observed_gain and status; a hold preserves state.

        Each call is one averaging window. Changing block size changes the
        estimator dynamics even for the same total number of samples.
        """
        r, p, codes = np.asarray(backend, float), np.asarray(pn, float), np.asarray(backend_code)
        if r.ndim != 1 or r.shape != p.shape or r.shape != codes.shape or r.size < 2:
            raise ValueError("backend, pn and backend_code must have matching 1D shapes with >=2 samples")
        if not np.isfinite(r).all() or not np.isfinite(p).all() or not np.isin(p, [-1., 0., 1.]).all():
            raise ValueError("Finite backend and PN in {-1, 0, 1} required")
        if not np.isfinite(codes).all() or np.any(codes != np.floor(codes)) or np.any((codes < 0) | (codes >= 2**self.backend_bits)):
            raise ValueError("Invalid backend codes")
        status, observed = "updated", None
        centered = p-p.mean()
        energy = float(centered@centered)
        if np.any((codes == 0) | (codes == 2**self.backend_bits-1)):
            status = "held_backend_rail"
        elif energy/r.size < .25:
            status = "held_probe_excitation"
        else:
            observed = float(centered@(r-r.mean())/(self.probe_amplitude*energy))
            if not self.gain_bounds[0] <= observed <= self.gain_bounds[1]:
                status = "held_gain_out_of_range"
            else:
                self.gain_estimate += self.learning_rate*(observed-self.gain_estimate)
                self.updates += 1
        return {"gain_next": self.gain_estimate, "observed_gain": observed, "status": status}


def fit_pipeline_gain_sine(captures, frequency):
    """Dedicated sine calibration, without an ideal input or true gain.

    Probe is off. Fit q + c*r = A*sin(w*n) + B*cos(w*n) + DC, fixing the
    coarse-code coefficient to one. Captures are consecutive blocks of that
    dedicated sine; its amplitude and phase are NOT supplied to the fit.
    Return G_est=1/c, to be frozen during subsequent normal operation.
    The constant coarse DAC scale and known tone frequency are assumptions.
    """
    if not np.isfinite(frequency) or not 0 < frequency < .5:
        raise ValueError("frequency must be normalized in (0, .5)")
    triangular, count = np.empty((0, 5)), 0
    for capture in captures:
        q, r = np.asarray(capture["coarse"]), np.asarray(capture["backend"])
        if q.ndim != 1 or not q.size or q.shape != r.shape or not np.isfinite(q).all() or not np.isfinite(r).all():
            raise ValueError("Invalid foreground capture")
        pn = np.asarray(capture["pn"])
        if pn.shape != q.shape or np.any(pn != 0):
            raise ValueError("Foreground fit requires probe off")
        n = np.arange(count, count+len(q))
        design = np.column_stack([r, -np.sin(2*np.pi*frequency*n), -np.cos(2*np.pi*frequency*n), -np.ones(len(q))])
        # Streaming QR preserves least squares without squaring conditioning.
        triangular = np.linalg.qr(np.vstack([triangular, np.column_stack([design, -q])]), mode="r")
        count += len(q)
    coefficients, _, rank, singular = np.linalg.lstsq(triangular[:, :4], triangular[:, 4], rcond=1e-12)
    if count < 4 or rank < 4 or coefficients[0] <= 0:
        raise ValueError("Foreground record does not identify a positive gain")
    return {"gain_estimate": float(1/coefficients[0]), "samples": count,
            "condition_number": float(singular[0]/singular[-1]),
            "coefficients": coefficients.tolist(), "reference": "joint sine fit; frequency known, amplitude/phase/DC fitted"}
