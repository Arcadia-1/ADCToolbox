"""
Mapping between MATLAB and Python test folder names.

MATLAB uses camelCase (e.g., test_bitActivity)
Python uses snake_case (e.g., test_bit_activity)

This mapping allows comparison scripts to find corresponding folders.
"""

# MATLAB folder name -> Python folder name
MATLAB_TO_PYTHON = {
    'test_bitActivity': 'test_bit_activity',
    'test_weightScaling': 'test_weight_scaling',
    'test_ENoB_bitSweep': 'test_enob_bit_sweep',
    'test_sineFit': 'test_sine_fit',
    'test_FGCalSine': 'test_fg_cal_sine',
    'test_FGCalSine_overflowChk': 'test_fg_cal_sine_overflow_chk',
    'test_INLsine': 'test_inl_sine',
    'test_tomDecomp': 'test_tom_decomp',
    'test_errHistSine': 'test_err_hist_sine',
    'test_errHistSine_phase': 'test_err_hist_sine_phase',
    'test_errHistSine_code': 'test_err_hist_sine_code',
    'test_errPDF': 'test_err_pdf',
    'test_errAutoCorrelation': 'test_err_auto_correlation',
    'test_errSpectrum': 'test_err_spectrum',
    'test_errEnvelopeSpectrum': 'test_err_envelope_spectrum',
    'test_specPlot': 'test_spec_plot',
    'test_specPlotPhase': 'test_spec_plot_phase',
    'test_alias': 'test_alias',
    'test_jitter_load': 'test_jitter_load',
}

# Python folder name -> MATLAB folder name
PYTHON_TO_MATLAB = {v: k for k, v in MATLAB_TO_PYTHON.items()}


def get_python_folder(matlab_folder_name):
    """Convert MATLAB folder name to Python folder name."""
    return MATLAB_TO_PYTHON.get(matlab_folder_name, matlab_folder_name)


def get_matlab_folder(python_folder_name):
    """Convert Python folder name to MATLAB folder name."""
    return PYTHON_TO_MATLAB.get(python_folder_name, python_folder_name)
