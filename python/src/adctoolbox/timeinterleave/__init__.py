"""Time-interleaved ADC (TI-ADC) analysis and calibration.

This submodule covers the four canonical mismatches between sub-ADCs in a
time-interleaved converter:

- **Offset mismatch**: per-channel DC offset -> spurs at k * fs / M
- **Gain mismatch**:   per-channel gain error -> spurs at fin +/- k * fs / M
- **Timing mismatch (sample skew)**: per-channel sampling-instant error ->
  gain-like spurs whose amplitude scales with the input frequency
- **Bandwidth mismatch**: per-channel frequency-response differences; same spur
  placement as timing, requires multi-tone/swept input to separate (not covered
  yet — see RFC)

Sample-indexing convention: channel ``m`` contains samples ``x[m::M]``.

Public API
----------
- :func:`deinterleave`            / :func:`interleave`              — ingress/egress
- :func:`extract_mismatch_sine`                                     — measure mismatches
- :func:`predict_spurs`                                             — predict TI spurs from params
- :func:`fractional_delay_fft`    / :func:`fractional_delay_farrow` — DSP primitives
- :func:`calibrate_foreground`                                      — apply offset/gain/skew correction

Planned: :func:`analyze_ti_spectrum`.
"""

from adctoolbox.timeinterleave.deinterleave import deinterleave, interleave
from adctoolbox.timeinterleave.extract_mismatch_sine import extract_mismatch_sine
from adctoolbox.timeinterleave.predict_spurs import predict_spurs
from adctoolbox.timeinterleave.fractional_delay import (
    fractional_delay_fft,
    fractional_delay_farrow,
)
from adctoolbox.timeinterleave.calibrate_foreground import calibrate_foreground

__all__ = [
    "deinterleave",
    "interleave",
    "extract_mismatch_sine",
    "predict_spurs",
    "fractional_delay_fft",
    "fractional_delay_farrow",
    "calibrate_foreground",
]
