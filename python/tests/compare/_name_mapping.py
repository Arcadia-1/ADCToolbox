"""
Mapping between MATLAB and Python test folder names.

MATLAB integration tests use 'run_*' prefix (e.g., run_plotspec, run_errpdf)
Python integration tests use 'test_*' prefix (e.g., test_spec_plot, test_err_pdf)

This mapping allows comparison scripts to find corresponding folders.
"""

# MATLAB folder name -> Python folder name
MATLAB_TO_PYTHON = {
    # DOUT tests (digital output) - MATLAB uses test_* for digital
    'test_bitact': 'test_bit_activity',
    'test_wscaling': 'test_weight_scaling',
    'test_bitsweep': 'test_enob_bit_sweep',
    'test_wcalsine': 'test_fg_cal_sine',
    'test_ovfchk': 'test_overflow_chk',

    # AOUT tests (analog output) - MATLAB uses run_* for analog integration tests
    'run_inlsine': 'test_inl_sine',
    'run_tomdec': 'test_decompose_harmonics',
    'run_errsin_phase': 'test_err_hist_sine_phase',
    'run_errsin_code': 'test_err_hist_sine_code',
    'run_errpdf': 'test_err_pdf',
    'run_errac': 'test_err_auto_correlation',
    'run_errspec': 'test_err_spectrum',
    'run_errevspec': 'test_err_envelope_spectrum',
    'run_plotspec': 'test_analyze_spectrum',
    'run_plotphase': 'test_analyze_phase_spectrum',
    'run_fitstaticnl': 'test_fit_static_nonlin',

    # Legacy test_* names (for backward compatibility)
    'test_inlsine': 'test_inl_sine',
    'test_tomdec': 'test_decompose_harmonics',
    'test_errsin_phase': 'test_err_hist_sine_phase',
    'test_errsin_code': 'test_err_hist_sine_code',
    'test_errpdf': 'test_err_pdf',
    'test_errac': 'test_err_auto_correlation',
    'test_errspec': 'test_err_spectrum',
    'test_errevspec': 'test_err_envelope_spectrum',
    'test_plotspec': 'test_analyze_spectrum',
    'test_plotphase': 'test_analyze_phase_spectrum',
    'test_fitstaticnl': 'test_fit_static_nonlin',

    # COMMON tests
    'test_sinfit': 'test_sine_fit',
    'run_sinfit': 'test_sine_fit',
    'test_alias': 'test_alias',
    'run_alias': 'test_alias',
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
