Oversampling Analysis (oversampling)
=====================================

The ``oversampling`` module provides MATLAB-compatible tools for oversampling
and Delta-Sigma ADC analysis.

.. currentmodule:: adctoolbox.oversampling

MATLAB-Compatible Entry Points
------------------------------

.. autofunction:: ifilter
.. autofunction:: perfosr
.. autofunction:: ntfperf
.. autofunction:: ntf_analyzer

Workflow Notes
--------------

``ifilter``
    Ideal FFT brickwall filtering. Use ``ifilter(x, [[0, 0.5 / OSR]])`` to
    extract the in-band waveform before downstream narrow-band analysis.

``perfosr``
    MATLAB-style four-output wrapper around the Python OSR sweep engine:
    ``osr, sndr, sfdr, enob = perfosr(x, osr=[2, 4, 8, 16])``.

``ntfperf`` / ``ntf_analyzer``
    Analyze theoretical NTF in-band noise suppression. These functions do not
    generate a noise-shaped waveform; they evaluate an NTF response over a
    signal band.
