"""Time-interleaved ADC (TI-ADC) analysis and calibration.

This submodule adds tools for characterizing and correcting the four canonical
mismatches between sub-ADCs in a time-interleaved converter:

- **Offset mismatch**: per-channel DC offset -> spurs at k * fs / M
- **Gain mismatch**:   per-channel gain error -> spurs at fin +/- k * fs / M
- **Timing mismatch (sample skew)**: per-channel sampling instant error -> gain-like spurs
  with frequency-dependent amplitude
- **Bandwidth mismatch**: per-channel frequency-response differences -> similar placement,
  full spectrum shape required to separate from timing

All routines take the *interleaved* time series `x` of length N and the channel
count `M`; N must be a multiple of M and channel 0 corresponds to `x[0::M]`.

Public API (planned)
--------------------
- deinterleave(x, M) -> (M, N//M) channel array
- interleave(channels) -> 1D array
- extract_mismatch_sine(x, M, fs, fin=None) -> dict of per-channel params
- predict_spurs(channel_params, fs, fin, M) -> DataFrame(freq, kind, dbc)
- analyze_ti_spectrum(x, M, fs, fin=None, mark_spurs=True, create_plot=True) -> dict
- calibrate_foreground(x, M, channel_params) -> corrected x
"""

# Scaffolding — functions land as they are implemented.
from .deinterleave import deinterleave, interleave

__all__ = [
    "deinterleave",
    "interleave",
]
