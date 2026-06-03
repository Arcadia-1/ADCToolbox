"""Digital output (code-level) analysis tools."""

from adctoolbox.dout.analyze_overflow import analyze_overflow
from adctoolbox.dout.analyze_bit_activity import analyze_bit_activity
from adctoolbox.dout.analyze_enob_sweep import analyze_enob_sweep
from adctoolbox.dout.analyze_weight_radix import analyze_weight_radix
from adctoolbox.dout.plot_residual_scatter import plot_residual_scatter
from adctoolbox.calibration import calibrate_weight_sine as _calibrate_weight_sine


def check_bit_activity(*args, **kwargs):
    """Compatibility alias for :func:`analyze_bit_activity`."""
    return analyze_bit_activity(*args, **kwargs)


def check_overflow(*args, disp=None, **kwargs):
    """Compatibility wrapper accepting the historical ``disp`` plot flag."""
    if disp is not None and "create_plot" not in kwargs:
        kwargs["create_plot"] = bool(disp)
    return analyze_overflow(*args, **kwargs)


def plot_weight_radix(*args, **kwargs):
    """Compatibility wrapper returning only the historical radix vector."""
    return analyze_weight_radix(*args, **kwargs)['radix']


def calibrate_weight_sine(*args, order=None, **kwargs):
    """Compatibility wrapper returning the historical calibration tuple."""
    if order is not None and "harmonic_order" not in kwargs:
        kwargs["harmonic_order"] = order
    result = _calibrate_weight_sine(*args, **kwargs)
    return (
        result['weight'],
        result['offset'],
        result['calibrated_signal'],
        result['ideal'],
        result['error'],
        result['refined_frequency'],
    )

__all__ = [
    'analyze_overflow',
    'analyze_bit_activity',
    'analyze_enob_sweep',
    'analyze_weight_radix',
    'plot_residual_scatter',
    'calibrate_weight_sine',
    'check_bit_activity',
    'check_overflow',
    'plot_weight_radix',
]
