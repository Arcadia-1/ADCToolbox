"""Test utilities for saving figures and variables (matches MATLAB format)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union

def auto_search_files(file_list: List[str], input_dir: Union[str, Path], *patterns: str) -> List[str]:
    """
    Auto-search for files if the input list is empty.
    
    Equivalent to MATLAB's autoSearchFiles.
    
    Args:
        file_list (list): Existing list of filenames. If not empty, returned as-is.
        input_dir (Path or str): Directory to search in.
        *patterns (str): Search patterns (e.g., 'sinewave_*.csv', '*.txt').
        
    Returns:
        List[str]: The original list (if not empty) or the discovered filenames.
        
    Raises:
        FileNotFoundError: If no files are found after search.
    """
    # 1. Early return: User manually specified files
    if file_list:
        return file_list

    # 2. Setup directory
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # 3. Perform Search
    discovered_files = []
    # Default pattern if none provided
    search_patterns = patterns if patterns else ["*.csv"]
    
    for pattern in search_patterns:
        # Use sorted() for deterministic order (essential for reproducibility)
        found = sorted([f.name for f in input_path.glob(pattern)])
        discovered_files.extend(found)

    # Remove duplicates if any (e.g. if patterns overlap)
    # Using dict.fromkeys to preserve order (set() loses order)
    discovered_files = list(dict.fromkeys(discovered_files))

    # 4. Logging and Validation
    patterns_str = ", ".join(search_patterns)
    print(f"[auto_search_files] Discovered [{len(discovered_files)}] files in [{input_path}] matching [{patterns_str}]")

    if not discovered_files:
        raise FileNotFoundError(f"No test files found in {input_path} matching {search_patterns}")
        
    return discovered_files



def save_fig(folder, png_filename, verbose=True, dpi=150, close_fig=True):
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


def save_variable(folder, var, var_name, verbose=True):
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
