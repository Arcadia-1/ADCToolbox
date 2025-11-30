"""Example data access utilities."""
import sys
from pathlib import Path

def get_example_data_path(filename):
    """
    Get full path to an example data file.
    Works in both development and pip-installed environments.

    Parameters
    ----------
    filename : str
        Name of example data file

    Returns
    -------
    Path
        Full path to data file
    """
    # For Python 3.9+, use importlib.resources
    if sys.version_info >= (3, 9):
        try:
            from importlib.resources import files
            data_dir = files('adctoolbox.examples.data')
            return data_dir.joinpath(filename)
        except (ImportError, TypeError):
            pass

    # Fallback: use __file__
    data_dir = Path(__file__).parent
    filepath = data_dir / filename

    if not filepath.is_file():
        raise FileNotFoundError(
            f"Example data file not found: {filename}\n"
            f"Available files: {', '.join(list_example_data())}"
        )

    return filepath

def list_example_data():
    """List all available example data files."""
    return [
        'sinewave_jitter_400fs.csv',
        'sinewave_noise_270uV.csv',
        'sinewave_gain_error_0P98.csv',
        'sinewave_clipping_0P012.csv',
        'dout_SAR_12b_weight_1.csv'
    ]
