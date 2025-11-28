"""Compare all test results between MATLAB and Python implementations."""

import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def compare_csv_files(matlab_file, python_file, tolerance=1e-6, name=''):
    """
    Compare two CSV files with numerical data.

    Parameters
    ----------
    matlab_file : Path
        Path to MATLAB CSV file
    python_file : Path
        Path to Python CSV file
    tolerance : float
        Acceptable difference threshold
    name : str
        Name for reporting

    Returns
    -------
    status : str
        'PASS', 'WARN', 'FAIL', or 'SKIP'
    message : str
        Description of result
    """
    if not matlab_file.exists():
        return 'SKIP', f'MATLAB file not found'

    if not python_file.exists():
        return 'SKIP', f'Python file not found'

    try:
        # MATLAB CSV: has headers + horizontal layout (var_1,var_2,...)
        # Python CSV: no headers + vertical layout
        try:
            matlab_data = np.loadtxt(matlab_file, delimiter=',')
        except ValueError:
            # Has headers, skip first row and flatten
            matlab_data = np.loadtxt(matlab_file, delimiter=',', skiprows=1)
            matlab_data = matlab_data.flatten()

        python_data = np.loadtxt(python_file, delimiter=',')

        # Handle scalar vs array
        matlab_data = np.atleast_1d(matlab_data)
        python_data = np.atleast_1d(python_data)

        # Check shape
        if matlab_data.shape != python_data.shape:
            return 'FAIL', f'Shape mismatch: MATLAB {matlab_data.shape} vs Python {python_data.shape}'

        # Mask NaN values
        valid_mask = ~(np.isnan(matlab_data) | np.isnan(python_data))

        if not np.any(valid_mask):
            return 'SKIP', 'All values are NaN'

        # Calculate differences
        diff = np.abs(matlab_data[valid_mask] - python_data[valid_mask])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        rel_error = max_diff / (np.max(np.abs(matlab_data[valid_mask])) + 1e-10)

        # Determine status
        if max_diff < tolerance:
            return 'PASS', f'Max diff: {max_diff:.2e}'
        elif max_diff < tolerance * 1000:
            return 'WARN', f'Max diff: {max_diff:.2e}, Rel: {rel_error:.2e}'
        else:
            return 'FAIL', f'Max diff: {max_diff:.2e}, Rel: {rel_error:.2e}'

    except Exception as e:
        return 'FAIL', f'Error: {str(e)}'


def compare_test_results():
    """Compare all test results."""

    output_dir = Path('test_output')

    print('[compare_all_results]')
    print(f'  [search] -> [{output_dir}]')
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
    print('Comparing All Test Results: MATLAB vs Python')
    print('='*90)
    print()

    overall_stats = {'PASS': 0, 'WARN': 0, 'FAIL': 0, 'SKIP': 0}

    for test_name, config in sorted(test_configs.items()):
        print(f'\n[{test_name.upper()}]')
        print('-'*90)

        # Find datasets
        test_folders = list(output_dir.glob(f'*/{config["test_folder"]}'))

        if not test_folders:
            print(f'  No datasets found for {test_name}')
            continue

        for test_folder in sorted(test_folders):
            dataset_name = test_folder.parent.name

            print(f'\n  [{dataset_name}]')

            for matlab_file, python_file, var_name, tolerance in config['files']:
                matlab_path = test_folder / matlab_file
                python_path = test_folder / python_file

                # Show file paths
                print(f'    [MATLAB] -> [{matlab_path}]', end=' ')
                if matlab_path.exists():
                    print('OK', end='')
                else:
                    print('NOT FOUND', end='')

                print(f' | [Python] -> [{python_path}]', end=' ')
                if python_path.exists():
                    print('OK')
                else:
                    print('NOT FOUND')

                status, message = compare_csv_files(matlab_path, python_path, tolerance, var_name)
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
    sys.exit(compare_test_results())
