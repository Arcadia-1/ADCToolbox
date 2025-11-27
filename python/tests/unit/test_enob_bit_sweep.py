"""Test ENoB_bitSweep function."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from python.src.adctoolbox.enob_bit_sweep import enob_bit_sweep

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
    print(f'[test_enob_bit_sweep] [{k}/{len(files_list)}] [{filepath.name}]')

    read_data = np.loadtxt(filepath, delimiter=',')

    fig = plt.figure(figsize=(10, 7.5))
    enob_sweep, n_bits_vec = enob_bit_sweep(
        read_data, freq=0, order=5, harmonic=5, osr=1, win_type=4, plot=True)

    dataset_name = filepath.stem
    sub_folder = output_dir / dataset_name / 'test_enob_bit_sweep'
    sub_folder.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig_path = sub_folder / 'ENoB_bitSweep.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    if not verbose:
        plt.close(fig)
    print(f'  [save] -> [{fig_path}]')

    # Save ENoB_sweep data
    csv_path = sub_folder / 'ENoB_sweep_python.csv'
    np.savetxt(csv_path, enob_sweep, delimiter=',', fmt='%.6f')
    print(f'  [save] -> [{csv_path}]')

    # Save nBits_vec data
    csv_path = sub_folder / 'nBits_vec_python.csv'
    np.savetxt(csv_path, n_bits_vec, delimiter=',', fmt='%d')
    print(f'  [save] -> [{csv_path}]')

print('\n=== Test complete ===')
