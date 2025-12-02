"""
Test configuration - all test paths in one place.
Just edit the paths below to change where tests look for data.
"""

# Configuration for different test types
AOUT = {
    'input_path': 'reference_dataset',
    'file_pattern': 'sinewave_*.csv',
}

DOUT = {
    'input_path': 'reference_dataset',
    'file_pattern': 'dout_*.csv',
}

JITTER = {
    'input_path': 'test_dataset/jitter_sweep',
    'file_pattern': 'jitter_sweep_*.csv',
}


# ============================================================================
# Alternative configurations (uncomment to use):
# ============================================================================

# Use subdirectories:
# AOUT = {'input_path': 'dataset/aout', 'file_pattern': '*.csv'}
# DOUT = {'input_path': 'dataset/dout', 'file_pattern': '*.csv'}

# Use reference datasets:
# AOUT = {'input_path': 'reference_dataset/sinewave', 'file_pattern': '*.csv'}
# DOUT = {'input_path': 'reference_dataset/dout', 'file_pattern': '*.csv'}
