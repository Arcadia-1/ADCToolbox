"""Oversampling and noise transfer function analysis tools.

This subpackage contains MATLAB-compatible oversampling helpers:

- ``ifilter``: ideal FFT brickwall filtering
- ``perfosr``: performance metrics versus OSR
- ``ntfperf`` / ``ntf_analyzer``: NTF in-band noise suppression
"""

from .ifilter import ifilter
from .ntf_analyzer import ntf_analyzer
from .ntfperf import ntfperf
from .perfosr import perfosr

__all__ = [
    'ifilter',
    'ntf_analyzer',
    'ntfperf',
    'perfosr',
]
