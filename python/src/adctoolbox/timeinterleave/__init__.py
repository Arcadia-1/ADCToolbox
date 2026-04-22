"""Time-interleaved ADC (TI-ADC) analysis and calibration.

This submodule adds tools for characterizing and correcting the four canonical
mismatches between sub-ADCs in a time-interleaved converter:

- **Offset mismatch**: per-channel DC offset -> spurs at k * fs / M
- **Gain mismatch**:   per-channel gain error -> spurs at fin +/- k * fs / M
- **Timing mismatch (sample skew)**: per-channel sampling-instant error -> gain-like
  spurs whose amplitude scales with the input frequency
- **Bandwidth mismatch**: per-channel frequency-response differences -> similar spur
  placement to timing, requires multi-tone/swept input to separate (not covered yet)

All routines take the *interleaved* time series ``x`` of length ``N`` and the channel
count ``M``; ``N`` must be a multiple of ``M`` and channel ``m`` contains samples
``x[m::M]``.

Public API
----------
- :func:`deinterleave`   / :func:`interleave`      — ingress / egress helpers
- :func:`extract_mismatch_sine`                    — per-channel sine fit -> dict
- :func:`predict_spurs`                            — from dict -> spur list

Planned (see RFC):
- ``analyze_ti_spectrum(x, M, fs, fin=None, mark_spurs=True)``
- ``calibrate_foreground(x, M, params)``
"""

from adctoolbox.timeinterleave.deinterleave import deinterleave, interleave
from adctoolbox.timeinterleave.extract_mismatch_sine import extract_mismatch_sine
from adctoolbox.timeinterleave.predict_spurs import predict_spurs

__all__ = [
    "deinterleave",
    "interleave",
    "extract_mismatch_sine",
    "predict_spurs",
]
