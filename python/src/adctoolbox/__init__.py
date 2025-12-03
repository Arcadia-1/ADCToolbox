"""
ADCToolbox: A comprehensive toolbox for ADC testing and characterization.

This package provides tools for analyzing both analog and digital aspects of
Analog-to-Digital Converters, including spectrum analysis, error characterization,
calibration algorithms, and more.

Usage:
------
>>> from adctoolbox import spec_plot, sine_fit, fg_cal_sine
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
    spec_plot,
    spec_plot_phase,
    spec_plot_2tone,
    err_envelope_spectrum,
    err_auto_correlation,
    err_hist_sine,
    err_pdf,
    tom_decomp,
    inl_sine,
    fit_static_nol,
)

from .dout import (
    fg_cal_sine,
    fg_cal_sine_os,
    fg_cal_sine_2freq,
    overflow_chk,
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
    'spec_plot',
    'spec_plot_phase',
    'spec_plot_2tone',
    'err_envelope_spectrum',
    'err_auto_correlation',
    'err_hist_sine',
    'err_pdf',
    'tom_decomp',
    'inl_sine',
    'fit_static_nol',

    # Digital output (dout) functions
    'fg_cal_sine',
    'fg_cal_sine_os',
    'fg_cal_sine_2freq',
    'overflow_chk',

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
