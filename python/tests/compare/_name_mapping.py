"""
Mapping between MATLAB and Python test folder names.

MATLAB uses abbreviated lowercase names (e.g., test_bitact, test_plotspec)
Python uses descriptive snake_case (e.g., test_bit_activity, test_spec_plot)

This mapping allows comparison scripts to find corresponding folders.
"""

# MATLAB folder name -> Python folder name
MATLAB_TO_PYTHON = {
    # DOUT tests (digital output)
    'test_bitact': 'test_bit_activity',
    'test_wscaling': 'test_weight_scaling',
    'test_bitsweep': 'test_enob_bit_sweep',
    'test_wcalsine': 'test_fg_cal_sine',
    'test_ovfchk': 'test_fg_cal_sine_overflow_chk',

    # AOUT tests (analog output)
    'test_inlsine': 'test_inl_sine',
    'test_tomdec': 'test_tom_decomp',
    'test_errsin_phase': 'test_err_hist_sine_phase',
    'test_errsin_code': 'test_err_hist_sine_code',
    'test_errpdf': 'test_err_pdf',
    'test_errac': 'test_err_auto_correlation',
    'test_errspec': 'test_err_spectrum',
    'test_errevspec': 'test_err_envelope_spectrum',
    'test_plotspec': 'test_spec_plot',
    'test_plotphase': 'test_spec_plot_phase',

    # COMMON tests (identical names)
    'test_sine_fit': 'test_sine_fit',
    'test_alias': 'test_alias',
    'test_jitter_load': 'test_jitter_load',
    'test_basic': 'test_basic',
}

# Python folder name -> MATLAB folder name
PYTHON_TO_MATLAB = {v: k for k, v in MATLAB_TO_PYTHON.items()}


def get_python_folder(matlab_folder_name):
    """Convert MATLAB folder name to Python folder name."""
    return MATLAB_TO_PYTHON.get(matlab_folder_name, matlab_folder_name)


def get_matlab_folder(python_folder_name):
    """Convert Python folder name to MATLAB folder name."""
    return PYTHON_TO_MATLAB.get(python_folder_name, python_folder_name)
