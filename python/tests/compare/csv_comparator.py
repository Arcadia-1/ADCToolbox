import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

class CSVComparator:
    """
    Strict CSV Comparator.
    
    Purpose:
        Verifies that Python output matches MATLAB output within machine precision.
        Uses Absolute Difference only. No percentage calculations.
    """
    # Tolerance threshold for floating point comparison
    # 1e-10 is standard for double-precision verification
    THRESHOLD = 1e-10

    def compare_pair(self, py_path: Union[str, Path], mat_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Compares two CSV files as numeric arrays.
        Returns 'PERFECT' if max absolute difference <= THRESHOLD, else 'FAIL'.
        """
        try:
            # 1. Load Data (single-column numeric files, no delimiter)
            # Use numpy.loadtxt for simple numeric files
            arr_py = np.loadtxt(py_path, delimiter=',', ndmin=1)
            arr_mat = np.loadtxt(mat_path, delimiter=',', ndmin=1)

            # 2. Shape Check
            if arr_py.shape != arr_mat.shape:
                return self._build_result(
                    'ERROR', 
                    f"Shape mismatch: Py{arr_py.shape} vs Mat{arr_mat.shape}"
                )

            # 3. Fast Vectorized Calculation
            # Absolute difference: |A - B|
            diff_abs = np.abs(arr_py - arr_mat)
            max_diff_abs = np.max(diff_abs)

            # 4. Strict Pass/Fail Logic
            if max_diff_abs <= self.THRESHOLD:
                status = 'PERFECT'
            else:
                status = 'FAIL'

            return {
                'status': status,
                'max_diff_abs': max_diff_abs,
                'shape': arr_py.shape,
                'msg': None
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'max_diff_abs': np.nan,
                'msg': f"Crash: {str(e)}"
            }