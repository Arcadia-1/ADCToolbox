"""
ADCToolbox: A comprehensive toolbox for ADC testing and characterization.

This package provides tools for analyzing both analog and digital aspects of
Analog-to-Digital Converters, including spectrum analysis, error characterization,
calibration algorithms, and more.

Submodules:
-----------
- common: Common utility functions (alias, findBin, findFin, sineFit, etc.)
- aout: Analog output / time-domain analysis (spectrum, error analysis, etc.)
- dout: Digital output / code-level analysis (calibration, overflow detection, etc.)
- os: Oversampling and noise transfer function tools
- utils: Utility functions and reporting tools
- data_generation: Test data generation utilities

Usage:
------
>>> from adctoolbox.aout import spec_plot
>>> from adctoolbox.common import sine_fit
>>> from adctoolbox.dout import FGCalSine
"""

__version__ = '0.1.0'

# Expose submodules for convenience
from . import common
from . import aout
from . import dout
from . import os
from . import utils
from . import data_generation

__all__ = [
    'common',
    'aout',
    'dout',
    'os',
    'utils',
    'data_generation',
    '__version__',
]
