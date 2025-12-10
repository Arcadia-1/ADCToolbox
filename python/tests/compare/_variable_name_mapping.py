"""
Two-way mapping between Pythonic and MATLAB variable names.

Purpose:
- Python functions return Pythonic names (signal_power, noise_floor, frequency)
- Tests automatically save with MATLAB-compatible names (sigpwr, noi, freq)
- Enables seamless comparison between Python and MATLAB test outputs

Note: It's fine for different functions to use the same Pythonic names
(e.g., 'frequency' in both sine_fit and find_fin). Each function's
return values are independent.
"""

# Master mapping: Pythonic name -> MATLAB name
VARIABLE_MAPPINGS = {
    # spec_plot / plotspec
    'enob': 'enob',
    'sndr': 'sndr',
    'sfdr': 'sfdr',
    'snr': 'snr',
    'thd': 'thd',
    'signal_power': 'sigpwr',
    'noise_floor': 'noi',
    'noise_spectral_density': 'nsd',
    'plot_line': 'h',

    # err_envelope_spectrum / errEnvelopeSpectrum (uses capitalized names)
    'ENoB': 'ENoB',        # Alternate capital form used by errEnvelopeSpectrum
    'SNDR': 'SNDR',        # Alternate capital form
    'SFDR': 'SFDR',        # Alternate capital form
    'SNR': 'SNR',          # Alternate capital form
    'THD': 'THD',          # Alternate capital form
    'pwr': 'pwr',          # Alternate form used by errEnvelopeSpectrum
    'NF': 'NF',            # Alternate form (Noise Floor)

    # sine_fit / sinfit
    'fitted_signal': 'data_fit',  # MATLAB test uses data_fit variable name
    'frequency': 'freq',
    'amplitude': 'mag',
    'dc_offset': 'dc',
    'phase': 'phi',

    # tom_decomp / tomdec
    'fundamental_signal': 'sine',
    'total_error': 'error',
    'harmonic_error': 'harmic',
    'residual_error': 'others',

    # err_hist_sine (phase mode)
    'phase_code': 'xx',           # MATLAB uses xx for phase axis
    'error_mean': 'emean',        # Future: use error_mean instead of emean
    'error_rms': 'erms',          # Future: use error_rms instead of erms
    'amplitude_noise': 'anoi',    # Future: use amplitude_noise instead of anoi
    'phase_noise': 'pnoi',        # Future: use phase_noise instead of pnoi

    # inl_sine
    'INL': 'INL',          # Domain convention - keep uppercase
    'DNL': 'DNL',          # Domain convention - keep uppercase
    'code': 'code',

    # err_spectrum
    'err_data': 'err_data',

    # Other functions (names already match)
    'bit_usage': 'bit_usage',
    'acf': 'acf',
    'lags': 'lags',
    'code_axis': 'code_axis',
    'emean_code': 'emean_code',
    'erms_code': 'erms_code',
    'emean': 'emean',
    'erms': 'erms',
    'anoi': 'anoi',
    'pnoi': 'pnoi',
}


def get_matlab_name(pythonic_name):
    """
    Convert Pythonic variable name to MATLAB equivalent for test compatibility.

    Args:
        pythonic_name: Pythonic variable name (e.g., 'signal_power')

    Returns:
        str: MATLAB variable name (e.g., 'sigpwr')

    Examples:
        >>> get_matlab_name('signal_power')
        'sigpwr'
        >>> get_matlab_name('noise_floor')
        'noi'
        >>> get_matlab_name('enob')
        'enob'
    """
    return VARIABLE_MAPPINGS.get(pythonic_name, pythonic_name)


def pythonic_to_matlab(name):
    """
    Alias for get_matlab_name().

    This is the primary function used in save_variable() for test compatibility.
    """
    return get_matlab_name(name)


def matlab_to_pythonic(matlab_name):
    """
    Convert MATLAB name to Pythonic name.

    Args:
        matlab_name: MATLAB variable name (e.g., 'sigpwr')

    Returns:
        str: Pythonic variable name (e.g., 'signal_power')

    Examples:
        >>> matlab_to_pythonic('sigpwr')
        'signal_power'
        >>> matlab_to_pythonic('noi')
        'noise_floor'
    """
    reverse_map = {v: k for k, v in VARIABLE_MAPPINGS.items()}
    return reverse_map.get(matlab_name, matlab_name)


__all__ = [
    'VARIABLE_MAPPINGS',
    'get_matlab_name',
    'pythonic_to_matlab',
    'matlab_to_pythonic',
]
