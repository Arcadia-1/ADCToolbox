"""Test bitActivity function."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from python.src.adctoolbox.bit_activity import bit_activity

# Configuration
verbose = False
input_dir = Path('dataset')
output_dir = Path('test_output')

# Get list of files
files_list = sorted(input_dir.glob('dout_*.csv'))
if not files_list:
    print('No dout files found')
    sys.exit(0)

# Test Loop
for k, filepath in enumerate(files_list, 1):
    print(f'[test_bit_activity] [{k}/{len(files_list)}] [{filepath.name}]')

    bits = np.loadtxt(filepath, delimiter=',')

    fig = plt.figure(figsize=(10, 7.5))
    bit_usage = bit_activity(bits, annotate_extremes=True)
    plt.gca().tick_params(labelsize=16)

    dataset_name = filepath.stem
    sub_folder = output_dir / dataset_name / 'test_bit_activity'
    sub_folder.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig_path = sub_folder / 'bitActivity.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    if not verbose:
        plt.close(fig)
    print(f'  [save] -> [{fig_path}]')

    # Save bit_usage data
    csv_path = sub_folder / 'bit_usage_python.csv'
    np.savetxt(csv_path, bit_usage, delimiter=',', fmt='%.6f')
    print(f'  [save] -> [{csv_path}]')

print('\n=== Test complete ===')
