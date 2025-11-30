"""Compare Python outputs against golden references

Regression test: detects unintended changes in Python code.
Compares test_output/ against test_reference/
"""

import sys
from python.tests.compare.compare_all_results import compare_test_results

if __name__ == '__main__':
    sys.exit(compare_test_results(mode='regression'))
