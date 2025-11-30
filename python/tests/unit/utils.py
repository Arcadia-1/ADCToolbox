"""Test utilities for saving figures and variables (matches MATLAB format)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_fig(folder, png_filename, verbose=False, dpi=150, close_fig=True):
    """Save current figure to PNG file (matches MATLAB saveFig.m)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / png_filename

    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')

    if verbose:
        print(f"  [save_fig] -> [{file_path}]")
    if close_fig:
        plt.close()

    return file_path


def save_variable(folder, var, var_name, verbose=False):
    """Save variable to CSV with auto truncation (matches MATLAB saveVariable.m)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    var = np.atleast_1d(var)

    # Truncate to max 1000 elements
    if var.ndim == 1:
        var = var[:1000]
    elif var.shape[0] > var.shape[1]:
        var = var[:1000, :]
    else:
        var = var[:, :1000]

    file_path = folder / f'{var_name}_python.csv'
    fmt = '%d' if np.issubdtype(var.dtype, np.integer) else '%.16f'
    np.savetxt(file_path, var, delimiter=',', fmt=fmt)

    if verbose:
        print(f"  [save_variable] -> [{file_path}]")

    return file_path
