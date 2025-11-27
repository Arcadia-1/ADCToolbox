"""Test DOUT toolset on ADC digital output."""

import numpy as np
from pathlib import Path
import sys
import os

# Add project root to sys.path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from python.src.adctoolbox.toolset_dout import toolset_dout

# Configuration
verbose = False
input_dir = Path('dataset')
output_dir = Path('test_output')
order = 5

# Get list of files
files_list = sorted(input_dir.glob('dout_*.csv'))
if not files_list:
    print('No dout files found')
    sys.exit(1)

# Test only first file for quick testing
files_list = files_list[:1]

# Test Loop
for k, filepath in enumerate(files_list, 1):
    print(f'[test_toolset_dout] [{k}/{len(files_list)}] [{filepath.name}]')

    bits = np.loadtxt(filepath, delimiter=',')
    dataset_name = filepath.stem
    sub_folder = output_dir / dataset_name / 'test_toolset_dout'

    status = toolset_dout(bits, sub_folder, visible=verbose, order=order)

    if status['success']:
        print('[All tools completed successfully]')
    else:
        print('[Some tools failed!]')
        for error in status['errors']:
            print(f'    - {error}')

print('\n=== Test complete ===')
