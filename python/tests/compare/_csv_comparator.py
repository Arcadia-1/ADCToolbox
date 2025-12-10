import numpy as np
from pathlib import Path
from typing import Dict, Any, Union


class CSVComparator:
    """
    Strict CSV Comparator with both absolute and relative error tracking.

    Purpose:
        Verifies that Python output matches MATLAB output within machine precision.
        Calculates both absolute and relative differences for analysis.
    """
    # Tolerance threshold for floating point comparison
    # 1e-6 allows for numerical differences between MATLAB and NumPy implementations
    THRESHOLD = 1e-6

    def compare_pair(self, py_path: Union[str, Path], mat_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Compares two CSV files as numeric arrays.
        Returns 'PERFECT' if max absolute difference <= THRESHOLD, else 'FAIL'.

        Returns dict with:
            - status: 'PERFECT', 'FAIL', or 'ERROR'
            - max_diff_abs: Maximum absolute difference
            - max_diff_rel: Maximum relative difference (%)
            - shape: Array shape
            - msg: Error message if any
        """
        try:
            # 1. Load Data (single-column numeric files, no delimiter)
            # Use numpy.loadtxt for simple numeric files
            arr_py = np.loadtxt(py_path, delimiter=',', ndmin=1)
            arr_mat = np.loadtxt(mat_path, delimiter=',', ndmin=1)
            arr_py = np.squeeze(arr_py)
            arr_mat = np.squeeze(arr_mat)
            
            # 2. Shape Check
            if arr_py.shape != arr_mat.shape:
                return self._build_result(
                    'ERROR', 
                    f"Shape mismatch: Py{arr_py.shape} vs Mat{arr_mat.shape}"
                )

            # 3. Fast Vectorized Calculation
            # Absolute difference: |A - B|
            # Relative difference: |A - B| / |B| * 100%
            diff_abs = np.abs(arr_py - arr_mat)

            # Calculate relative error (as percentage)
            # Avoid division by zero: use absolute value of MATLAB as denominator
            # For values near zero, relative error is not meaningful
            with np.errstate(divide='ignore', invalid='ignore'):
                diff_rel = np.abs(diff_abs / (np.abs(arr_mat) + 1e-100)) * 100  # Add tiny epsilon to avoid division by zero

            # Handle scalar case differently (0-D arrays can't be indexed)
            if diff_abs.ndim == 0:
                if np.isnan(arr_py) and np.isnan(arr_mat):
                    max_diff_abs = 0.0
                    max_diff_rel = 0.0
                elif np.isnan(arr_py) or np.isnan(arr_mat):
                    max_diff_abs = np.inf
                    max_diff_rel = np.inf
                else:
                    max_diff_abs = float(diff_abs)
                    max_diff_rel = float(diff_rel)
            else:
                # Create masks for NaN positions
                nan_mask_py = np.isnan(arr_py)
                nan_mask_mat = np.isnan(arr_mat)

                # Both have NaN at same positions: set diff to 0
                both_nan = nan_mask_py & nan_mask_mat
                diff_abs[both_nan] = 0
                diff_rel[both_nan] = 0

                # One has NaN but not the other: set diff to inf (will fail)
                only_one_nan = nan_mask_py ^ nan_mask_mat
                diff_abs[only_one_nan] = np.inf
                diff_rel[only_one_nan] = np.inf

                max_diff_abs = np.max(diff_abs)
                max_diff_rel = np.max(diff_rel)

            # 4. Strict Pass/Fail Logic
            if max_diff_abs <= self.THRESHOLD:
                status = 'PERFECT'
            else:
                status = 'FAIL'

            return {
                'status': status,
                'max_diff_abs': max_diff_abs,
                'max_diff_rel': max_diff_rel,
                'shape': arr_py.shape,
                'msg': None
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'max_diff_abs': np.nan,
                'max_diff_rel': np.nan,
                'msg': f"Crash: {str(e)}"
            }