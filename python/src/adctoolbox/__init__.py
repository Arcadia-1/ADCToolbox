"""
ADCToolbox: A comprehensive toolbox for ADC testing and characterization.

This package provides tools for analyzing both analog and digital aspects of
Analog-to-Digital Converters, including spectrum analysis, error characterization,
calibration algorithms, and more.

Usage:
------
>>> from adctoolbox import analyze_spectrum, sine_fit, cal_weight_sine
>>> from adctoolbox import alias, find_bin, plot_error_hist_phase
"""

__version__ = '0.2.4'

# Import all public functions from submodules
from .common import (
    calc_aliased_freq,
    calc_coherent_freq,
    estimate_frequency,
    fit_sine,
    extract_freq_components,
    convert_cap_to_weight,
    calc_target_vpp,
)

from .aout import (
    analyze_spectrum,
    analyze_phase_spectrum,
    analyze_two_tone_spectrum,
    plot_envelope_spectrum,
    plot_error_autocorr,
    plot_error_hist_code,
    plot_error_hist_phase,
    plot_error_pdf,
    decompose_harmonics,
    calc_inl_sine,
    fit_static_nonlin,
)

from .dout import (
    calibrate_weight_sine,
    calibrate_weight_sine_osr,
    calibrate_weight_two_tone,
    check_overflow,
    check_bit_activity,
    analyze_enob_sweep,
    plot_weight_radix,
    generate_dout_dashboard,
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
    'calc_aliased_freq',
    'calc_coherent_freq',
    'estimate_frequency',
    'fit_sine',
    'extract_freq_components',
    'convert_cap_to_weight',
    'calc_target_vpp',

    # Analog output (aout) functions
    'analyze_spectrum',
    'analyze_phase_spectrum',
    'analyze_two_tone_spectrum',
    'plot_envelope_spectrum',
    'plot_error_autocorr',
    'plot_error_hist_code',
    'plot_error_hist_phase',
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
    'generate_dout_dashboard',

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
