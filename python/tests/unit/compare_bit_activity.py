"""Compare bit_activity results between MATLAB and Python."""

import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def compare_results():
    """Compare MATLAB and Python bit_activity results."""

    output_dir = Path('test_output')

    # Find all datasets with both MATLAB and Python results
    datasets = []
    for matlab_folder in output_dir.glob('*/test_bitActivity'):
        dataset_name = matlab_folder.parent.name
        python_folder = output_dir / dataset_name / 'test_bit_activity'
        if python_folder.exists():
            datasets.append(dataset_name)

    if not datasets:
        print('[WARNING] No matching datasets found. Run both MATLAB and Python tests first.')
        return

    print('='*80)
    print('Comparing bit_activity Results: MATLAB vs Python')
    print('='*80)
    print()

    all_match = True

    for dataset in sorted(datasets):
        print(f'[{dataset}]')

        matlab_file = output_dir / dataset / 'test_bitActivity' / 'bit_usage_matlab.csv'
        python_file = output_dir / dataset / 'test_bit_activity' / 'bit_usage_python.csv'

        if not matlab_file.exists():
            print(f'  [SKIP] MATLAB file not found: {matlab_file}')
            continue

        if not python_file.exists():
            print(f'  [SKIP] Python file not found: {python_file}')
            continue

        # Load data
        matlab_data = np.loadtxt(matlab_file, delimiter=',')
        python_data = np.loadtxt(python_file, delimiter=',')

        # Compare
        if matlab_data.shape != python_data.shape:
            print(f'  [FAIL] Shape mismatch: MATLAB {matlab_data.shape} vs Python {python_data.shape}')
            all_match = False
            continue

        max_diff = np.max(np.abs(matlab_data - python_data))
        rel_error = max_diff / (np.max(np.abs(matlab_data)) + 1e-10)

        if max_diff < 1e-6:
            print(f'  [PASS] Max difference: {max_diff:.2e}')
        elif max_diff < 1e-3:
            print(f'  [WARN] Max difference: {max_diff:.2e} (small difference)')
        else:
            print(f'  [FAIL] Max difference: {max_diff:.2e}, Relative error: {rel_error:.2e}')
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
