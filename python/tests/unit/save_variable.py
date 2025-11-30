"""save_variable.py - Save individual variables to CSV files (matches MATLAB format)

This module provides a utility function to save variables to CSV files,
matching the behavior of MATLAB's saveVariable.m function.
"""

import numpy as np
from pathlib import Path


def save_variable(folder, var, var_name, verbose=False):
    """
    Save a single variable to CSV with auto truncation.

    Matches MATLAB's saveVariable.m behavior:
    - Each variable saved to separate CSV file
    - Filename format: {var_name}_python.csv
    - Truncates to max 1000 rows or columns to save space

    Parameters
    ----------
    folder : Path or str
        Output folder path
    var : array_like
        Variable to save (scalar, vector, or array)
    var_name : str
        Name for the variable (used in filename)
    verbose : bool, optional
        Print save confirmation (default: False)

    Examples
    --------
    >>> save_variable(output_dir, mu, 'mu')
    >>> save_variable(output_dir, freq_est, 'freq_est', verbose=True)
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Convert to numpy array
    var = np.atleast_1d(var)

    # Truncate to max 1000 elements (matching MATLAB behavior)
    if var.ndim == 1:
        # Vector: truncate to 1000 elements
        var_truncated = var[:1000] if len(var) > 1000 else var
    else:
        # Array: truncate along longest dimension
        if var.shape[0] > var.shape[1]:
            # More rows than columns
            var_truncated = var[:1000, :] if var.shape[0] > 1000 else var
        else:
            # More columns than rows
            var_truncated = var[:, :1000] if var.shape[1] > 1000 else var

    # Save to CSV
    file_name = f'{var_name}_python.csv'
    file_path = folder / file_name

    # Use appropriate format based on data type
    if np.issubdtype(var_truncated.dtype, np.integer):
        fmt = '%d'
    else:
        fmt = '%.16f'

    np.savetxt(file_path, var_truncated, delimiter=',', fmt=fmt)

    if verbose:
        print(f"  [save_variable] -> [{file_path}]")

    return file_path
