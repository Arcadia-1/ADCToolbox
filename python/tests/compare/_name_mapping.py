"""
Mapping between MATLAB and Python test folder names.

MATLAB integration tests use 'run_*' prefix (e.g., run_plotspec, run_errpdf)
Python integration tests use 'test_*' prefix (e.g., test_spec_plot, test_err_pdf)

This mapping allows comparison scripts to find corresponding folders.
"""

# MATLAB folder name -> Python folder name
MATLAB_TO_PYTHON = {
    # DOUT tests (digital output) - MATLAB uses test_* for digital
    'test_bitact': 'test_check_bit_activity',
    'test_wscaling': 'test_plot_weight_radix',
    'test_bitsweep': 'test_analyze_enob_sweep',
    'test_wcalsine': 'test_calibrate_weight_sine',
    'test_ovfchk': 'test_check_overflow',
    'test_ovfchk_wcalsine': 'test_calibrate_weight_sine_check_overflow',

    # AOUT tests (analog output) - MATLAB uses run_* for analog integration tests
    'run_inlsine': 'test_calc_inl_sine',
    'run_tomdec': 'test_decompose_harmonics',
    'run_errsin_phase': 'test_plot_error_hist_phase',
    'run_errsin_code': 'test_plot_error_hist_code',
    'run_errpdf': 'test_plot_error_pdf',
    'run_errac': 'test_plot_error_autocorr',
    'run_errspec': 'test_err_spectrum',
    'run_errevspec': 'test_plot_envelope_spectrum',
    'run_plotspec': 'test_analyze_spectrum',
    'run_plotphase': 'test_analyze_phase_spectrum',
    'run_plotphase_fft': 'test_analyze_phase_spectrum_fft',
    'run_plotphase_lms': 'test_analyze_phase_spectrum_lms',
    'run_fitstaticnl': 'test_fit_static_nonlin',

    # Legacy test_* names (for backward compatibility)
    'test_inlsine': 'test_calc_inl_sine',
    'test_tomdec': 'test_decompose_harmonics',
    'test_errsin_phase': 'test_plot_error_hist_phase',
    'test_errsin_code': 'test_plot_error_hist_code',
    'test_errpdf': 'test_plot_error_pdf',
    'test_errac': 'test_plot_error_autocorr',
    'test_errspec': 'test_err_spectrum',
    'test_errevspec': 'test_plot_envelope_spectrum',
    'test_plotspec': 'test_analyze_spectrum',
    'test_plotphase': 'test_analyze_phase_spectrum',
    'test_fitstaticnl': 'test_fit_static_nonlin',

    # COMMON tests
    'test_sinfit': 'test_fit_sine',
    'run_sinfit': 'test_fit_sine',
    'test_alias': 'test_verify_alias',
    'run_alias': 'test_verify_alias',
    'test_cap2weight': 'test_cap2weight',
    'test_spec_plot': 'test_verify_spec_plot',
    'test_spec_plot_phase': 'test_verify_spec_plot_phase',
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
