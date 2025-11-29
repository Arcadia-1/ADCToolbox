"""Run tests to generate golden reference data

Run golden tests - only processes files from golden_data_list.txt
Outputs are saved to test_output/
"""

import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
unit_tests_dir = current_dir.parent / 'unit'
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(unit_tests_dir))

# Import and run golden tests
from golden_sine_fit import golden_sineFit
from golden_alias import golden_alias

print('Running golden tests...\n')

# Run golden tests (these use golden_data_list.txt)
golden_sineFit()
golden_alias()

print('\nDone! Outputs saved to test_reference/')
print('Next: python ../../tests/validation/compare_parity.py')
