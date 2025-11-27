"""Compare weight_scaling results between MATLAB and Python."""

import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

def compare_results():
    """Compare MATLAB and Python weight_scaling results."""

    output_dir = Path('test_output')

    # Find all datasets with both MATLAB and Python results
    datasets = []
    for matlab_folder in output_dir.glob('*/test_weightScaling'):
        dataset_name = matlab_folder.parent.name
        python_folder = output_dir / dataset_name / 'test_weight_scaling'
        if python_folder.exists():
            datasets.append(dataset_name)

    if not datasets:
        print('[WARNING] No matching datasets found. Run both MATLAB and Python tests first.')
        return

    print('='*80)
    print('Comparing weight_scaling Results: MATLAB vs Python')
    print('='*80)
    print()

    all_match = True

    for dataset in sorted(datasets):
        print(f'[{dataset}]')

        matlab_radix = output_dir / dataset / 'test_weightScaling' / 'radix_matlab.csv'
        python_radix = output_dir / dataset / 'test_weight_scaling' / 'radix_python.csv'

        matlab_weight = output_dir / dataset / 'test_weightScaling' / 'weight_cal_matlab.csv'
        python_weight = output_dir / dataset / 'test_weight_scaling' / 'weight_cal_python.csv'

        # Compare radix
        if matlab_radix.exists() and python_radix.exists():
            matlab_data = np.loadtxt(matlab_radix, delimiter=',')
            python_data = np.loadtxt(python_radix, delimiter=',')

            # Skip NaN values (first element)
            valid_mask = ~(np.isnan(matlab_data) | np.isnan(python_data))

            if np.any(valid_mask):
                max_diff = np.max(np.abs(matlab_data[valid_mask] - python_data[valid_mask]))

                if max_diff < 1e-6:
                    print(f'  [PASS] radix - Max difference: {max_diff:.2e}')
                elif max_diff < 1e-3:
                    print(f'  [WARN] radix - Max difference: {max_diff:.2e}')
                else:
                    print(f'  [FAIL] radix - Max difference: {max_diff:.2e}')
                    all_match = False
            else:
                print(f'  [SKIP] radix - All NaN values')

        # Compare weight_cal
        if matlab_weight.exists() and python_weight.exists():
            matlab_data = np.loadtxt(matlab_weight, delimiter=',')
            python_data = np.loadtxt(python_weight, delimiter=',')

            max_diff = np.max(np.abs(matlab_data - python_data))
            rel_error = max_diff / (np.max(np.abs(matlab_data)) + 1e-10)

            if max_diff < 1e-6:
                print(f'  [PASS] weight_cal - Max difference: {max_diff:.2e}')
            elif max_diff < 1e-3:
                print(f'  [WARN] weight_cal - Max difference: {max_diff:.2e}, Relative: {rel_error:.2e}')
            else:
                print(f'  [FAIL] weight_cal - Max difference: {max_diff:.2e}, Relative: {rel_error:.2e}')
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
