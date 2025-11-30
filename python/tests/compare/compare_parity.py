"""Compare MATLAB vs Python outputs in test_output/

Validates that Python implementation matches MATLAB reference.
"""

import sys
from python.tests.compare.compare_all_results import compare_test_results

if __name__ == '__main__':
    sys.exit(compare_test_results(mode='parity'))
