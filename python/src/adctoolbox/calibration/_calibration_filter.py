"""Finite-record lowpass objective for oversampled sine calibration."""

import numpy as np
from scipy.signal import fftconvolve, firwin


class _CalibrationFilter:
    """Apply one FIR to every model column, independently for each capture.

    Valid convolution discards both edge transients of the symmetric FIR. The
    remaining samples are aligned at the FIR centers; no records are joined
    across a boundary and no periodic extension or decimation is performed.
    """

    def __init__(self, osr, segment_lengths):
        self.osr = osr
        self.cutoff = 0.5 / osr
        self.segment_lengths = np.asarray(segment_lengths, dtype=int)
        numtaps = 16 * int(np.ceil(osr)) + 1
        if np.any(self.segment_lengths <= numtaps):
            raise ValueError(
                f"osr={osr:g} needs more than {numtaps} samples per capture "
                "for the calibration FIR; use longer captures or a lower osr."
            )
        self.taps = firwin(numtaps, self.cutoff, fs=1.0, window=("kaiser", 8.6))
        self.valid_lengths = self.segment_lengths - numtaps + 1

    def apply_record(self, values):
        values = np.asarray(values, dtype=float)
        kernel = self.taps if values.ndim == 1 else self.taps[:, None]
        return fftconvolve(values, kernel, mode="valid", axes=0)

    def apply(self, values):
        boundaries = np.cumsum(self.segment_lengths)[:-1]
        return np.concatenate([
            self.apply_record(record) for record in np.split(values, boundaries, axis=0)
        ], axis=0)

    def validate_frequencies(self, frequencies):
        frequencies = np.asarray(frequencies)
        if np.any(~np.isfinite(frequencies)) or np.any(
            (frequencies <= 0) | (frequencies >= self.cutoff)
        ):
            raise ValueError(
                "Oversampled calibration requires 0 < freq < 0.5/osr "
                f"({self.cutoff:g} cycles/sample), including frequency-search iterates."
            )

    def bound_frequencies(self, frequencies):
        """Keep estimated/search frequencies inside the requested signal band."""
        return np.clip(frequencies, np.finfo(float).eps,
                       np.nextafter(self.cutoff, 0.0))

    def metadata(self):
        return {
            "kind": "fir_lowpass",
            "osr": self.osr,
            "cutoff_normalized": self.cutoff,
            "numtaps": len(self.taps),
            "valid_samples": self.valid_lengths.copy(),
        }


def _make_calibration_filter(osr, segment_lengths):
    osr = float(osr)
    if not np.isfinite(osr) or osr < 1:
        raise ValueError("osr must be finite and >= 1")
    return None if osr == 1 else _CalibrationFilter(osr, segment_lengths)
