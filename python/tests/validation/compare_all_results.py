"""Compare all test results between MATLAB and Python implementations.

Modes:
    1. Parity test: Compare MATLAB vs Python in test_output/ (default)
    2. Regression test: Compare Python test_output/ vs test_reference/

Usage:
    python compare_all_results.py                 # Parity test (MATLAB vs Python)
    python compare_all_results.py --regression    # Regression test (vs golden)
"""

import numpy as np
from pathlib import Path
import sys
import argparse

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def compare_csv_files(file1, file2, tolerance=1e-6, name='', label1='File1', label2='File2'):
    """
    Compare two CSV files with numerical data.

    Parameters
    ----------
    file1 : Path
        Path to first CSV file (e.g., MATLAB or golden reference)
    file2 : Path
        Path to second CSV file (e.g., Python or test output)
    tolerance : float
        Acceptable difference threshold
    name : str
        Name for reporting
    label1 : str
        Label for first file (default: 'File1')
    label2 : str
        Label for second file (default: 'File2')

    Returns
    -------
    status : str
        'PASS', 'WARN', 'FAIL', or 'SKIP'
    message : str
        Description of result
    """
    if not file1.exists():
        return 'SKIP', f'{label1} file not found'

    if not file2.exists():
        return 'SKIP', f'{label2} file not found'

    try:
        # Load data (handle headers if present)
        try:
            data1 = np.loadtxt(file1, delimiter=',')
        except ValueError:
            # Has headers, skip first row and flatten
            data1 = np.loadtxt(file1, delimiter=',', skiprows=1)
            data1 = data1.flatten()

        try:
            data2 = np.loadtxt(file2, delimiter=',')
        except ValueError:
            data2 = np.loadtxt(file2, delimiter=',', skiprows=1)
            data2 = data2.flatten()

        # Handle scalar vs array
        data1 = np.atleast_1d(data1)
        data2 = np.atleast_1d(data2)

        # Check shape
        if data1.shape != data2.shape:
            return 'FAIL', f'Shape mismatch: {label1} {data1.shape} vs {label2} {data2.shape}'

        # Mask NaN values
        valid_mask = ~(np.isnan(data1) | np.isnan(data2))

        if not np.any(valid_mask):
            return 'SKIP', 'All values are NaN'

        # Calculate differences
        diff = np.abs(data1[valid_mask] - data2[valid_mask])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = max_diff / (np.max(np.abs(data1[valid_mask])) + 1e-10)

        # Determine status
        if max_diff < tolerance:
            return 'PASS', f'Max diff: {max_diff:.2e}'
        elif max_diff < tolerance * 1000:
            return 'WARN', f'Max diff: {max_diff:.2e}, Rel: {rel_error:.2e}'
        else:
            return 'FAIL', f'Max diff: {max_diff:.2e}, Rel: {rel_error:.2e}'

    except Exception as e:
        return 'FAIL', f'Error: {str(e)}'


def compare_test_results(mode='parity'):
    """Compare all test results.

    Parameters
    ----------
    mode : str
        'parity' - Compare MATLAB vs Python in test_output/ (default)
        'regression' - Compare Python test_output/ vs test_reference/
    """
    if mode == 'regression':
        # Regression mode: compare test_output vs test_reference
        test_output_dir = Path('test_output')
        golden_ref_dir = Path('test_reference')
        label1 = 'Golden'
        label2 = 'Current'
        compare_suffix = 'python'  # Only compare Python files
        title = f'Regression Test: Python output vs Golden reference'
    else:
        # Parity mode: compare MATLAB vs Python
        test_output_dir = Path('test_output')
        golden_ref_dir = test_output_dir  # Both in same dir
        label1 = 'MATLAB'
        label2 = 'Python'
        compare_suffix = None  # Compare MATLAB vs Python
        title = f'Parity Test: MATLAB vs Python'

    print('[compare_all_results]')
    print(f'  Mode: {mode}')
    if mode == 'regression':
        print(f'  Reference: {golden_ref_dir}')
        print(f'  Test output: {test_output_dir}')
    else:
        print(f'  Search: {test_output_dir}')
    print()

    # Define test mappings (test folder -> files to compare)
    test_configs = {
        'bit_activity': {
            'test_folder': 'test_bitActivity',
            'files': [('bit_usage_matlab.csv', 'bit_usage_python.csv', 'bit_usage', 1e-6)],
        },
        'weight_scaling': {
            'test_folder': 'test_weightScaling',
            'files': [
                ('radix_matlab.csv', 'radix_python.csv', 'radix', 1e-6),
                ('weight_cal_matlab.csv', 'weight_cal_python.csv', 'weight_cal', 1e-6),
            ],
        },
        'enob_bit_sweep': {
            'test_folder': 'test_ENoB_bitSweep',
            'files': [('ENoB_sweep_matlab.csv', 'ENoB_sweep_python.csv', 'ENoB_sweep', 0.01)],
        },
        'sine_fit': {
            'test_folder': 'test_sineFit',
            'files': [
                ('freq_matlab.csv', 'freq_python.csv', 'freq', 1e-6),
                ('mag_matlab.csv', 'mag_python.csv', 'mag', 1e-6),
                ('dc_matlab.csv', 'dc_python.csv', 'dc', 1e-6),
                ('phi_matlab.csv', 'phi_python.csv', 'phi', 1e-6),
            ],
        },
        'fg_cal_sine': {
            'test_folder': 'test_FGCalSine',
            'files': [
                ('weight_matlab.csv', 'weight_python.csv', 'weight', 1e-6),
                ('offset_matlab.csv', 'offset_python.csv', 'offset', 1e-6),
                ('freqCal_matlab.csv', 'freqCal_python.csv', 'freqCal', 1e-6),
            ],
        },
    }

    print('='*90)
    print(title)
    print('='*90)
    print()

    overall_stats = {'PASS': 0, 'WARN': 0, 'FAIL': 0, 'SKIP': 0}

    for test_name, config in sorted(test_configs.items()):
        print(f'\n[{test_name.upper()}]')
        print('-'*90)

        # Find datasets in test_output
        test_folders = list(test_output_dir.glob(f'*/{config["test_folder"]}'))

        if not test_folders:
            print(f'  No datasets found for {test_name}')
            continue

        for test_folder in sorted(test_folders):
            dataset_name = test_folder.parent.name

            print(f'\n  [{dataset_name}]')

            for file1_name, file2_name, var_name, tolerance in config['files']:
                if mode == 'regression':
                    # Regression: compare test_reference/*_python.csv vs test_output/*_python.csv
                    if compare_suffix and f'_{compare_suffix}' not in file2_name:
                        continue  # Skip non-Python files
                    file1_path = golden_ref_dir / dataset_name / config['test_folder'] / file2_name  # golden
                    file2_path = test_folder / file2_name  # current output
                else:
                    # Parity: compare MATLAB vs Python in test_output
                    file1_path = test_folder / file1_name  # MATLAB
                    file2_path = test_folder / file2_name  # Python

                # Show file paths
                print(f'    [{label1}] -> [{file1_path}]', end=' ')
                if file1_path.exists():
                    print('OK', end='')
                else:
                    print('NOT FOUND', end='')

                print(f' | [{label2}] -> [{file2_path}]', end=' ')
                if file2_path.exists():
                    print('OK')
                else:
                    print('NOT FOUND')

                status, message = compare_csv_files(file1_path, file2_path, tolerance, var_name, label1, label2)
                overall_stats[status] += 1

                status_symbol = {
                    'PASS': 'OK',
                    'WARN': 'WARN',
                    'FAIL': 'FAIL',
                    'SKIP': 'SKIP'
                }[status]

                print(f'    [{status_symbol}] {var_name:<15} {message}')

    # Summary
    print()
    print('='*90)
    print('SUMMARY')
    print('='*90)
    total = sum(overall_stats.values())
    print(f'Total comparisons: {total}')
    print(f'  PASS: {overall_stats["PASS"]} ({overall_stats["PASS"]/max(total,1)*100:.1f}%)')
    print(f'  WARN: {overall_stats["WARN"]} ({overall_stats["WARN"]/max(total,1)*100:.1f}%)')
    print(f'  FAIL: {overall_stats["FAIL"]} ({overall_stats["FAIL"]/max(total,1)*100:.1f}%)')
    print(f'  SKIP: {overall_stats["SKIP"]} ({overall_stats["SKIP"]/max(total,1)*100:.1f}%)')
    print()

    if overall_stats['FAIL'] == 0:
        print('[PASS] All comparisons passed!')
        return 0
    else:
        print('[FAIL] Some comparisons failed!')
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python compare_all_results.py              # Parity test (MATLAB vs Python)
  python compare_all_results.py --regression # Regression test (vs golden)
  python compare_all_results.py --parity     # Parity test (explicit)
        '''
    )
    parser.add_argument('--regression', action='store_true',
                        help='Run regression test (compare against golden reference)')
    parser.add_argument('--parity', action='store_true',
                        help='Run parity test (compare MATLAB vs Python)')

    args = parser.parse_args()

    # Determine mode
    if args.regression:
        mode = 'regression'
    else:
        mode = 'parity'  # default

    sys.exit(compare_test_results(mode=mode))
