"""Backward-compatible common utility namespace.

Historically several tests and examples imported small helpers from
``adctoolbox.common``.  The implementations now live in ``fundamentals``; this
module keeps the old import path working without duplicating logic.
"""

from adctoolbox.fundamentals import (
    convert_cap_to_weight,
    estimate_frequency,
    fit_sine_4param,
    fold_frequency_to_nyquist,
    nsd_to_snr as _nsd_to_snr,
    snr_to_nsd as _snr_to_nsd,
)


def fit_sine(*args, **kwargs):
    """Compatibility alias for :func:`fit_sine_4param`."""
    return fit_sine_4param(*args, **kwargs)


def snr_to_nsd(*args, signal_pwr_dbfs=None, **kwargs):
    """Compatibility wrapper accepting the historical ``signal_pwr_dbfs`` name."""
    if signal_pwr_dbfs is not None:
        kwargs["psignal_dbfs"] = signal_pwr_dbfs
    return _snr_to_nsd(*args, **kwargs)


def nsd_to_snr(*args, signal_pwr_dbfs=None, **kwargs):
    """Compatibility wrapper accepting the historical ``signal_pwr_dbfs`` name."""
    if signal_pwr_dbfs is not None:
        kwargs["psignal_dbfs"] = signal_pwr_dbfs
    return _nsd_to_snr(*args, **kwargs)


__all__ = [
    "convert_cap_to_weight",
    "estimate_frequency",
    "fit_sine",
    "fit_sine_4param",
    "fold_frequency_to_nyquist",
    "nsd_to_snr",
    "snr_to_nsd",
]
