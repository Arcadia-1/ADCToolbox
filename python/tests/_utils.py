"""Test utilities for saving figures and variables (matches MATLAB format)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union, Optional, Callable


def _glob_search(
    search_dir: Union[str, Path],
    pattern: str = "*",
    filter_func: Optional[Callable[[Path], bool]] = None,
    extract_func: Optional[Callable[[Path], str]] = None
) -> List[str]:
    """
    Generic glob search utility.

    Args:
        search_dir: Directory to search in
        pattern: Glob pattern (default: "*")
        filter_func: Optional filter function for found items
        extract_func: Optional function to extract desired string from Path

    Returns:
        List[str]: Sorted list of results
    """
    search_path = Path(search_dir)

    # Find all matching items
    found = search_path.glob(pattern)

    # Apply filter if provided
    if filter_func:
        found = [p for p in found if filter_func(p)]
    else:
        found = list(found)

    # Extract desired string from each Path
    if extract_func:
        results = [extract_func(p) for p in found]
    else:
        results = [p.name for p in found]

    return sorted(results)


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
    # Early return: User manually specified files
    if file_list:
        return file_list

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    # Use default pattern if none provided
    search_patterns = patterns if patterns else ["*.csv"]

    # Search for all patterns
    discovered_files = []
    for pattern in search_patterns:
        found = _glob_search(input_path, pattern, filter_func=lambda p: p.is_file())
        discovered_files.extend(found)

    # Remove duplicates while preserving order
    discovered_files = list(dict.fromkeys(discovered_files))

    # Logging and validation
    patterns_str = ", ".join(search_patterns)
    print(f"[auto_search_files] [{len(discovered_files)}] files in [{input_path}] matching [{patterns_str}]")

    if not discovered_files:
        raise FileNotFoundError(f"No test files found in {input_path} matching {search_patterns}")

    return discovered_files


def discover_test_datasets(reference_dir: Union[str, Path], subfolder: str = "test_sineFit") -> List[str]:
    """
    Auto-discover test datasets by looking for subdirectories with specified test folder.

    Used by comparison tests to find all datasets with golden references.

    Args:
        reference_dir: Path to test_reference directory
        subfolder: Name of subfolder to look for (default: "test_sineFit")

    Returns:
        List[str]: Dataset names (sorted)
    """
    def has_subfolder(p: Path) -> bool:
        return p.is_dir() and (p / subfolder).exists()

    datasets = _glob_search(reference_dir, pattern="*", filter_func=has_subfolder)

    print(f"[discover_test_datasets] Found [{len(datasets)}] dataset(s) with [{subfolder}] subfolder")

    return datasets


def discover_test_variables(test_dir: Union[str, Path], pattern: str = "*_matlab.csv") -> List[str]:
    """
    Auto-discover test variables by searching for files matching pattern.

    Used by comparison tests to find all variables in a dataset.

    Args:
        test_dir: Path to dataset's test folder directory
        pattern: Glob pattern to search for (default: "*_matlab.csv")

    Returns:
        List[str]: Variable names (sorted)
    """
    # Extract variable name by removing suffix
    suffix = pattern.replace("*", "")  # "_matlab.csv"
    suffix_base = suffix.replace(".csv", "")  # "_matlab"

    def extract_variable_name(p: Path) -> str:
        return p.stem.replace(suffix_base, "")

    variables = _glob_search(
        test_dir,
        pattern=pattern,
        filter_func=lambda p: p.is_file(),
        extract_func=extract_variable_name
    )

    # Get dataset name from parent directory for logging
    print(f"[discover_test_variables] Found [{len(variables)}] variable(s): {', '.join(variables)}")

    return variables


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
    """
    Save variable to CSV with MATLAB-compatible naming.

    Automatically converts Pythonic names to MATLAB names for test compatibility.
    For example: 'signal_power' → 'sigpwr_python.csv'

    Args:
        folder: Output directory path
        var: Variable data to save
        var_name: Pythonic variable name
        verbose: Print save confirmation (default: True)

    Returns:
        Path: Path to saved CSV file
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    var = np.atleast_1d(var)

    # Truncate to max 100 elements
    if var.ndim == 1:
        var = var[:100]
        # MATLAB saves 1D arrays as row vectors (1 row, N columns)
        # But scalars (length 1) remain as scalars
        if len(var) > 1:
            var = var.reshape(1, -1)
    elif var.shape[0] > var.shape[1]:
        var = var[:100, :]
    else:
        var = var[:, :100]

    # Convert Pythonic name → MATLAB name for test compatibility
    from tests.compare._variable_name_mapping import pythonic_to_matlab
    matlab_name = pythonic_to_matlab(var_name)

    file_path = folder / f'{matlab_name}_python.csv'
    fmt = '%d' if np.issubdtype(var.dtype, np.integer) else '%.16f'
    np.savetxt(file_path, var, delimiter=',', fmt=fmt)

    if verbose:
        if matlab_name != var_name:
            print(f"  [save_variable] {var_name} → {matlab_name}_python.csv")
        else:
            print(f"  [save_variable] → {matlab_name}_python.csv")

    return file_path
