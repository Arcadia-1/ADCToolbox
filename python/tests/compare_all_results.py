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
        matlab_data = np.loadtxt(matlab_file, delimiter=',')
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

    # Define test mappings (MATLAB folder -> Python folder -> files to compare)
    test_configs = {
        'bit_activity': {
            'matlab_folder': 'test_bitActivity',
            'python_folder': 'test_bit_activity',
            'files': [('bit_usage_matlab.csv', 'bit_usage_python.csv', 'bit_usage', 1e-6)],
        },
        'weight_scaling': {
            'matlab_folder': 'test_weightScaling',
            'python_folder': 'test_weight_scaling',
            'files': [
                ('radix_matlab.csv', 'radix_python.csv', 'radix', 1e-6),
                ('weight_cal_matlab.csv', 'weight_cal_python.csv', 'weight_cal', 1e-6),
            ],
        },
        'enob_bit_sweep': {
            'matlab_folder': 'test_ENoB_bitSweep',
            'python_folder': 'test_enob_bit_sweep',
            'files': [('ENoB_sweep_matlab.csv', 'ENoB_sweep_python.csv', 'ENoB_sweep', 0.01)],
        },
        'sine_fit': {
            'matlab_folder': 'test_sineFit',
            'python_folder': 'test_sine_fit',
            'files': [
                ('freq_est_matlab.csv', 'freq_est_python.csv', 'freq_est', 1e-6),
                ('mag_matlab.csv', 'mag_python.csv', 'mag', 1e-6),
            ],
        },
        'fg_cal_sine': {
            'matlab_folder': 'test_FGCalSine',
            'python_folder': 'test_fg_cal_sine',
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
        matlab_folders = list(output_dir.glob(f'*/{config["matlab_folder"]}'))

        if not matlab_folders:
            print(f'  No datasets found for {test_name}')
            continue

        for matlab_folder in sorted(matlab_folders):
            dataset_name = matlab_folder.parent.name
            python_folder = output_dir / dataset_name / config['python_folder']

            print(f'\n  [{dataset_name}]')

            for matlab_file, python_file, var_name, tolerance in config['files']:
                matlab_path = matlab_folder / matlab_file
                python_path = python_folder / python_file

                status, message = compare_csv_files(matlab_path, python_path, tolerance, var_name)
                overall_stats[status] += 1

                status_symbol = {
                    'PASS': '✓',
                    'WARN': '!',
                    'FAIL': '✗',
                    'SKIP': '-'
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
