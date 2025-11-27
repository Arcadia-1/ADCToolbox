"""save_fig.py - Save matplotlib figures to PNG files (matches MATLAB format)

This module provides a utility function to save figures to PNG files,
matching the behavior of MATLAB's saveFig.m function.
"""

import matplotlib.pyplot as plt
from pathlib import Path


def save_fig(folder, png_filename, verbose=False, dpi=150, close_fig=True):
    """
    Save the current figure to a PNG file.

    Matches MATLAB's saveFig.m behavior:
    - Creates folder if it doesn't exist
    - Saves current figure to PNG
    - Optionally prints save confirmation
    - Closes figure after saving

    Parameters
    ----------
    folder : Path or str
        Output folder path
    png_filename : str
        PNG filename (e.g., 'errPDF_python.png')
    verbose : bool, optional
        Print save confirmation (default: False)
    dpi : int, optional
        DPI for saved figure (default: 150)
    close_fig : bool, optional
        Close figure after saving (default: True)

    Returns
    -------
    file_path : Path
        Path to saved PNG file

    Examples
    --------
    >>> save_fig(output_dir, 'spectrum_python.png')
    >>> save_fig(output_dir, 'errPDF_python.png', verbose=True, dpi=200)
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    file_path = folder / png_filename

    # Save current figure
    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')

    if verbose:
        print(f"  [save_fig] -> [{file_path}]")

    if close_fig:
        plt.close()

    return file_path
