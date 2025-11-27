"""Compare ENoB_bitSweep results between MATLAB and Python."""

import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def compare_results():
    """Compare MATLAB and Python ENoB_bitSweep results."""

    output_dir = Path('test_output')

    print('[compare_enob_bit_sweep]')
    print(f'  [search] -> [{output_dir}]')
    print()

    # Find all datasets with both MATLAB and Python results
    datasets = []

    print('[Searching for datasets]')
    for test_folder in output_dir.glob('*/test_ENoB_bitSweep'):
        dataset_name = test_folder.parent.name

        matlab_file = test_folder / 'ENoB_sweep_matlab.csv'
        python_file = test_folder / 'ENoB_sweep_python.csv'

        print(f'  [dataset] [{dataset_name}]')
        print(f'    [MATLAB] -> [{matlab_file}]', end='')
        if matlab_file.exists():
            print(' OK')
        else:
            print(' NOT FOUND')

        print(f'    [Python] -> [{python_file}]', end='')
        if python_file.exists():
            print(' OK')
        else:
            print(' NOT FOUND')

        if matlab_file.exists() and python_file.exists():
            datasets.append(dataset_name)
            print(f'    [status] MATCHED')
        else:
            print(f'    [status] INCOMPLETE')
    print()

    if not datasets:
        print('[FAIL] No matching datasets found.')
        print()
        print('To fix this:')
        print('  1. Run MATLAB test:')
        print('     >> cd matlab/tests/unit')
        print('     >> test_ENoB_bitSweep')
        print('  2. Run Python test:')
        print('     >> cd d:\\ADCToolbox')
        print('     >> python python/tests/unit/test_enob_bit_sweep.py')
        print('  3. Re-run this comparison script')
        return

    print('='*80)
    print('Comparing ENoB_bitSweep Results: MATLAB vs Python')
    print('='*80)
    print()

    all_match = True

    for dataset in sorted(datasets):
        print(f'[{dataset}]')

        test_folder = output_dir / dataset / 'test_ENoB_bitSweep'
        matlab_enob = test_folder / 'ENoB_sweep_matlab.csv'
        python_enob = test_folder / 'ENoB_sweep_python.csv'

        if not matlab_enob.exists():
            print(f'  [SKIP] MATLAB file not found')
            continue

        if not python_enob.exists():
            print(f'  [SKIP] Python file not found')
            continue

        # Load data
        matlab_data = np.loadtxt(matlab_enob, delimiter=',')
        python_data = np.loadtxt(python_enob, delimiter=',')

        # Compare (allowing for some NaN differences)
        if matlab_data.shape != python_data.shape:
            print(f'  [FAIL] Shape mismatch: MATLAB {matlab_data.shape} vs Python {python_data.shape}')
            all_match = False
            continue

        # Mask out NaN values
        valid_mask = ~(np.isnan(matlab_data) | np.isnan(python_data))

        if not np.any(valid_mask):
            print(f'  [SKIP] All values are NaN')
            continue

        diff = np.abs(matlab_data[valid_mask] - python_data[valid_mask])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = max_diff / (np.max(np.abs(matlab_data[valid_mask])) + 1e-10)

        if max_diff < 0.01:  # ENoB tolerance: 0.01 bits
            print(f'  [PASS] Max diff: {max_diff:.4f} bits, Mean diff: {mean_diff:.4f} bits')
        elif max_diff < 0.1:
            print(f'  [WARN] Max diff: {max_diff:.4f} bits, Mean diff: {mean_diff:.4f} bits')
        else:
            print(f'  [FAIL] Max diff: {max_diff:.4f} bits, Mean diff: {mean_diff:.4f} bits, Rel: {rel_error:.2e}')
            all_match = False

    print()
    print('='*80)
    if all_match:
        print('[PASS] All datasets match!')
    else:
        print('[FAIL] Some datasets have mismatches')
    print('='*80)

if __name__ == '__main__':
    compare_results()
