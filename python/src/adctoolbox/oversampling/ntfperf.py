"""MATLAB-compatible NTF performance analyzer."""

from __future__ import annotations

from adctoolbox.oversampling.ntf_analyzer import ntf_analyzer


def ntfperf(ntf, fl: float, fh: float, disp: bool | int = False) -> float:
    """Analyze NTF in-band noise suppression.

    This is the Python counterpart of MATLAB ``ntfperf(ntf, fl, fh, disp)``.
    It returns the SNR improvement in dB over a flat ``NTF = 1`` baseline.
    """

    return ntf_analyzer(ntf, fl, fh, is_plot=int(bool(disp)))

