"""Pipeline Dither calibration from aligned, observable captures.

No probe injection, plant simulation or workflow orchestration is performed here.
See the packaged DITHER_GUIDE.md and HDC_FIXED.md for theory and model limits.
"""
from numbers import Integral
import numpy as np

__all__ = ["DitherGainTracker", "apply_dither_gain", "DitherHDCTracker"]


def apply_dither_gain(coarse, backend, pn, *, gain_estimate, probe_amplitude):
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
        return apply_dither_gain(coarse, backend, pn,
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


class DitherHDCTracker:
    """Streaming three-PN Pipeline HDC, in fixed or floating arithmetic.

    The stage uses signed codes {-1, 0, 1}, nominal gain 2, DAC levels
    ``stage_code/2``, and a normalized backend range [-1, 1]. Three independent
    PN signs inject ``dither_peak/3 * sum(pn_signs)`` before the amplifier.
    ``selection`` identifies the two DAC-element roles. This is the model in
    HDC_FIXED.md, including its one-tap memory and mismatch extensions.

    Parameters are updated at averaging-window boundaries with an explicit
    controller delay. Arbitrary contiguous block partitions give the same
    result. ``learn=False`` freezes active coefficients, cancels a pending
    update and discards partial estimation windows, but retains signal history.
    Call :meth:`reset` for an independent capture.

    ``initial_coefficients`` is an optional dict with exactly ``linear``,
    ``cubic``, ``memory`` and ``dac_mismatch``. These are digital inverse
    coefficients c1, k3, h, delta, not physical device parameters. For backend
    r = backend/2, w = c1*r - k3*(c1*r)**3, v = w - h*v_previous,
    output = stage_code/2 + v - probe + delta*selection*stage_code/2.

    The estimator retains the documented HDC moment approximation; a public
    interface does not remove finite-record bias or expand physical validity.
    """

    _names = ("linear", "cubic", "memory", "dac_mismatch")

    def __init__(self, *, fixed=True, fraction=24, bits=28, window_log2=20,
                 accumulator_bits=None, controller_latency=512,
                 dither_peak=.2, initial_coefficients=None):
        if not isinstance(fixed, (bool, np.bool_)):
            raise ValueError("fixed must be boolean")
        for name, value in (("fraction", fraction), ("bits", bits),
                            ("window_log2", window_log2),
                            ("controller_latency", controller_latency)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if accumulator_bits is not None and (
            isinstance(accumulator_bits, (bool, np.bool_))
            or not isinstance(accumulator_bits, Integral)
            or not 8 <= accumulator_bits <= 62
        ):
            raise ValueError("accumulator_bits must be an integer in [8, 62]")
        self._configuration = dict(
            fixed=bool(fixed), fraction=int(fraction), bits=int(bits),
            window_log2=int(window_log2), accumulator_bits=(
                None if accumulator_bits is None else int(accumulator_bits)),
            controller_latency=int(controller_latency), dither_peak=dither_peak,
        )
        from ._dither_hdc import Tracker as _HDCTracker

        # Validate numeric format and moment denominators before scaling words.
        _HDCTracker(**self._configuration)
        self._initial = dict(zip(self._names, (1., 0., 0., 0.)))
        if initial_coefficients is not None:
            if not isinstance(initial_coefficients, dict) or set(initial_coefficients) != set(self._names):
                raise ValueError(f"initial_coefficients must contain exactly {self._names}")
            self._initial = dict(initial_coefficients)
        values = np.asarray([self._initial[name] for name in self._names], dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)) or values[0] <= 0:
            raise ValueError("initial coefficients must be finite scalars with positive linear gain")
        # Bound the recursive inverse and reject unrepresentable fixed words.
        if abs(values[2]) >= 1:
            raise ValueError("initial memory coefficient must have magnitude below one")
        if fixed:
            words = np.rint(values * (1 << int(fraction)))
            if np.any(words < -(1 << (int(bits)-1))) or np.any(words >= (1 << (int(bits)-1))) or words[0] <= 0:
                raise ValueError("initial coefficients do not fit the fixed-point format")
        self.reset()

    @property
    def coefficients(self):
        """A detached dict of active digital correction coefficients."""
        return dict(zip(self._names, map(float, self._tracker.values())))

    def reset(self):
        """Restore initial coefficients and clear all stream/estimator state."""
        from ._dither_hdc import Tracker as _HDCTracker

        self._tracker = _HDCTracker(**self._configuration)
        values = np.asarray([self._initial[name] for name in self._names], dtype=float)
        self._tracker.coefficients = (
            np.rint(values*self._tracker.scale).astype(np.int64)
            if self._tracker.fixed else values.copy()
        )
        self._previous_pn = 0
        self._learning = True

    def process(self, backend, pn_signs, stage_code, selection, backend_rails, *, learn=True):
        """Correct one contiguous block and optionally learn from its moments.

        Parameters
        ----------
        backend : array_like, shape (N,)
            Finite normalized backend samples in [-1, 1], not integer codes.
        pn_signs : array_like, shape (N, 3)
            Aligned signs {-1, +1} for the three injected PN streams.
        stage_code : array_like, shape (N,)
            Signed coarse decisions {-1, 0, 1}, before Dither injection.
        selection : array_like, shape (N,)
            Observable DAC role-selection tags {-1, +1}.
        backend_rails : array_like, shape (N,)
            Boolean rail flags; any flag rejects its entire estimation window.
        learn : bool
            False freezes coefficients while advancing correction history.

        Returns
        -------
        dict
            ``calibrated_signal`` in stage input units; active ``coefficients``
            after the block; ``updates`` (accepted, possibly pending) and
            ``held_updates`` cumulative counts; ``pending_update`` flag; and
            cumulative ``saturations`` for datapath, accumulator and controller.
            A block may span coefficient commits; the returned coefficients
            are not necessarily those used for every sample in that block.
        """
        if not isinstance(learn, (bool, np.bool_)):
            raise ValueError("learn must be boolean")
        r = np.asarray(backend, dtype=float)
        p, q, s, rails = map(np.asarray, (pn_signs, stage_code, selection, backend_rails))
        if r.ndim != 1 or not r.size or not np.isfinite(r).all() or np.any(abs(r) > 1):
            raise ValueError("backend must be a nonempty finite 1D array in [-1, 1]")
        if p.shape != (r.size, 3) or not np.isin(p, (-1, 1)).all():
            raise ValueError("pn_signs must have shape (N, 3) and values {-1, +1}")
        if q.shape != r.shape or not np.isin(q, (-1, 0, 1)).all():
            raise ValueError("stage_code must have shape (N,) and values {-1, 0, 1}")
        if s.shape != r.shape or not np.isin(s, (-1, 1)).all():
            raise ValueError("selection must have shape (N,) and values {-1, +1}")
        if rails.shape != r.shape or rails.dtype.kind != 'b':
            raise ValueError("backend_rails must be a boolean array of shape (N,)")
        p, q, s = (a.astype(np.int64) for a in (p, q, s))
        tracker = self._tracker
        if not learn or self._learning != learn:
            tracker.pending = tracker.commit_at = None
            tracker.sums[:] = 0
            tracker.count = 0
            tracker.window_bad = False
        previous_pn = np.r_[self._previous_pn, p[:-1, 0]]
        y = tracker.process(dict(r=r/2, p=p, pp=previous_pn, s=s, q=q, rails=rails), learn=learn)
        self._previous_pn = int(p[-1, 0])
        self._learning = bool(learn)
        # The public API exposes counters; avoid retaining an unbounded event log.
        tracker.events.clear()
        return dict(
            calibrated_signal=y, coefficients=self.coefficients,
            updates=tracker.updates, held_updates=tracker.held,
            pending_update=tracker.pending is not None,
            saturations=dict(datapath=tracker.saturations,
                             accumulator=tracker.accumulator_saturations,
                             controller=tracker.controller_saturations),
        )
