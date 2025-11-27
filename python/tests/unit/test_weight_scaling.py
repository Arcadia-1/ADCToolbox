"""Test weightScaling function."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from python.src.adctoolbox.weight_scaling import weight_scaling
from python.src.adctoolbox.dout.fg_cal_sine import fg_cal_sine

# Configuration
verbose = False
input_dir = Path('dataset')
output_dir = Path('test_output')
order = 5

# Get list of files
files_list = sorted(input_dir.glob('dout_*.csv'))
if not files_list:
    print('No dout files found')
    sys.exit(0)

# Test Loop
for k, filepath in enumerate(files_list, 1):
    print(f'[test_weight_scaling] [{k}/{len(files_list)}] [{filepath.name}]')

    bits = np.loadtxt(filepath, delimiter=',')

    weight_cal, offset, k_static, residual, cost, freq_cal = fg_cal_sine(
        bits, freq=0, order=order)

    # Run weightScaling tool
    fig = plt.figure(figsize=(10, 7.5))
    radix = weight_scaling(weight_cal)
    plt.gca().tick_params(labelsize=16)

    dataset_name = filepath.stem
    sub_folder = output_dir / dataset_name / 'test_weightScaling'
    sub_folder.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig_path = sub_folder / 'weightScaling.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    if not verbose:
        plt.close(fig)
    print(f'  [save] -> [{fig_path}]')

    # Save radix data
    csv_path = sub_folder / 'radix_python.csv'
    np.savetxt(csv_path, radix, delimiter=',', fmt='%.6f')
    print(f'  [save] -> [{csv_path}]')

    # Save weight_cal data
    csv_path = sub_folder / 'weight_cal_python.csv'
    np.savetxt(csv_path, weight_cal, delimiter=',', fmt='%.6f')
    print(f'  [save] -> [{csv_path}]')

print('\n=== Test complete ===')
