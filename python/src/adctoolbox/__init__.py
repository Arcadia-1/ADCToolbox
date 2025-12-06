"""
ADCToolbox: A comprehensive toolbox for ADC testing and characterization.

This package provides tools for analyzing both analog and digital aspects of
Analog-to-Digital Converters, including spectrum analysis, error characterization,
calibration algorithms, and more.

Usage:
------
>>> from adctoolbox import analyze_spectrum, sine_fit, fg_cal_sine
>>> from adctoolbox import alias, find_bin, err_hist_sine
"""

__version__ = '0.2.0'

# Import all public functions from submodules
from .common import (
    alias,
    find_bin,
    find_fin_coherent,
    find_fin,
    sine_fit,
    bit_in_band,
    cap2weight,
    find_vinpp,
)

from .aout import (
    analyze_spectrum,
    analyze_phase_spectrum,
    analyze_two_tone_spectrum,
    plot_envelope_spectrum,
    plot_error_autocorr,
    err_hist_sine,
    plot_error_pdf,
    decompose_harmonics,
    calc_inl_sine,
    fit_static_nonlin,
)

from .dout import (
    cal_weight_sine,
    cal_weight_sine_os,
    cal_weight_sine_2freq,
    overflow_chk,
    bit_activity,
    sweep_bit_enob,
    weight_scaling,
)

from .oversampling import (
    ntf_analyzer,
)

from .utils import (
    generate_multimodal_report,
    calculate_jitter,
)

from .data_generation import (
    generate_jitter_signal,
)

# Keep submodules accessible for those who prefer explicit imports
from . import common
from . import aout
from . import dout
from . import oversampling
from . import utils
from . import data_generation

__all__ = [
    # Version
    '__version__',

    # Common functions
    'alias',
    'find_bin',
    'find_fin_coherent',
    'find_fin',
    'sine_fit',
    'bit_in_band',
    'cap2weight',
    'find_vinpp',

    # Analog output (aout) functions
    'analyze_spectrum',
    'analyze_phase_spectrum',
    'analyze_two_tone_spectrum',
    'plot_envelope_spectrum',
    'plot_error_autocorr',
    'err_hist_sine',
    'plot_error_pdf',
    'decompose_harmonics',
    'calc_inl_sine',
    'fit_static_nonlin',

    # Digital output (dout) functions
    'cal_weight_sine',
    'cal_weight_sine_os',
    'cal_weight_sine_2freq',
    'overflow_chk',
    'bit_activity',
    'sweep_bit_enob',
    'weight_scaling',

    # Oversampling functions
    'ntf_analyzer',

    # Utility functions
    'generate_multimodal_report',
    'calculate_jitter',

    # Data generation functions
    'generate_jitter_signal',

    # Submodules (for explicit imports)
    'common',
    'aout',
    'dout',
    'oversampling',
    'utils',
    'data_generation',
]
