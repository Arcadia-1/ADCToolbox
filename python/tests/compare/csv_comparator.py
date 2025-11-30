"""CSV Comparator Utility

Universal CSV file comparator for MATLAB vs Python validation.
Supports both metrics-style and array-style CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class CSVComparator:
    """Universal CSV file comparator."""

    # Thresholds for classification
    THRESHOLD_PERFECT = 1e-10      # < 1e-10% diff = PERFECT
    THRESHOLD_EXCELLENT = 0.01     # < 0.01% diff = EXCELLENT
    THRESHOLD_GOOD = 0.1           # < 0.1% diff = GOOD
    THRESHOLD_ACCEPTABLE = 1.0     # < 1.0% diff = ACCEPTABLE
    # Above 1.0% = NEEDS REVIEW

    # Absolute tolerance for machine precision (treat as perfect match)
    ABS_TOLERANCE = 1e-10          # Differences < 1e-10 treated as zero

    # Small value threshold - ignore percentage when values are this small
    SMALL_VALUE_THRESHOLD = 1e-2   # When |value| < 0.01, use absolute metric only
    SMALL_VALUE_ABS_TOLERANCE = 1e-3  # For small values, accept diff < 0.001 as EXCELLENT

    # Near-zero threshold - even more permissive for histogram bins near zero
    NEAR_ZERO_THRESHOLD = 1e-3     # When |value| < 0.001, very permissive
    NEAR_ZERO_ABS_TOLERANCE = 1e-4  # For near-zero, accept diff < 0.0001 as EXCELLENT

    def __init__(self, test_output_dir: str = 'test_output'):
        self.test_output_dir = Path(test_output_dir)
        self.results = []

    def find_csv_pairs(self) -> List[Dict]:
        """Find all *_python.csv and *_matlab.csv pairs."""
        if not self.test_output_dir.exists():
            print(f"[ERROR] Output directory not found: {self.test_output_dir}")
            return []

        # Find all python CSV files
        python_files = list(self.test_output_dir.glob('**/*_python.csv'))

        pairs = []
        for python_file in python_files:
            # Construct corresponding matlab file path
            matlab_file = python_file.parent / python_file.name.replace('_python.csv', '_matlab.csv')

            if matlab_file.exists():
                # Extract metadata from path
                parts = python_file.relative_to(self.test_output_dir).parts
                dataset_name = parts[0] if len(parts) > 0 else 'unknown'
                test_type = parts[1] if len(parts) > 1 else 'unknown'
                csv_name = python_file.stem.replace('_python', '')

                pairs.append({
                    'python_file': python_file,
                    'matlab_file': matlab_file,
                    'dataset': dataset_name,
                    'test_type': test_type,
                    'csv_name': csv_name,
                    'pair_name': f"{dataset_name}/{test_type}/{csv_name}"
                })

        return pairs

    def compare_pair(self, python_file: Path, matlab_file: Path) -> Dict:
        """Compare a single CSV pair."""
        try:
            df_py = pd.read_csv(python_file)
            df_mat = pd.read_csv(matlab_file)

            # Detect file type and compare accordingly
            if 'Metric' in df_py.columns or 'metric' in df_py.columns:
                return self._compare_metrics_style(df_py, df_mat, python_file, matlab_file)
            else:
                return self._compare_numeric_array(df_py, df_mat, python_file, matlab_file)

        except Exception as e:
            return {
                'status': 'ERROR',
                'max_diff_pct': np.nan,
                'max_diff_abs': np.nan,
                'worst_field': 'N/A',
                'num_mismatches': 0,
                'total_values': 0,
                'note': None,
                'error_msg': str(e)
            }

    def _compare_metrics_style(self, df_py: pd.DataFrame, df_mat: pd.DataFrame,
                               py_file: Path, mat_file: Path) -> Dict:
        """Compare metrics-style CSV (columns: Metric, Value)."""
        try:
            # Detect column names (case-insensitive)
            metric_col_py = next((c for c in df_py.columns if c.lower() == 'metric'), None)
            metric_col_mat = next((c for c in df_mat.columns if c.lower() == 'metric'), None)

            if not metric_col_py or not metric_col_mat:
                # Fall back to numeric comparison
                return self._compare_numeric_array(df_py, df_mat, py_file, mat_file)

            # Find value columns
            value_cols_py = [c for c in df_py.columns if c != metric_col_py]
            value_cols_mat = [c for c in df_mat.columns if c != metric_col_mat]

            # Merge on metric column
            merged = df_py.merge(df_mat, left_on=metric_col_py, right_on=metric_col_mat,
                                how='outer', suffixes=('_py', '_mat'))

            # Compare all numeric value columns
            diffs = []
            for col_py in value_cols_py:
                col_mat = col_py if col_py in value_cols_mat else col_py + '_mat'
                if col_mat not in merged.columns:
                    continue

                diff = np.abs(merged[col_py] - merged[col_mat])
                diff_pct = np.abs(diff / (merged[col_mat] + 1e-100) * 100)

                diffs.extend(diff_pct.dropna().values)

            if not diffs:
                return {
                    'status': 'ERROR',
                    'max_diff_pct': np.nan,
                    'max_diff_abs': np.nan,
                    'worst_field': 'N/A',
                    'num_mismatches': 0,
                    'total_values': 0,
                    'error_msg': 'No numeric columns to compare'
                }

            max_diff_pct = np.max(diffs)
            max_diff_abs = np.max(np.abs(merged[value_cols_py[0]] - merged[value_cols_mat[0]]))

            # Find worst metric
            worst_idx = np.argmax(diffs)
            worst_field = merged[metric_col_py].iloc[worst_idx] if worst_idx < len(merged) else 'N/A'

            # Count mismatches
            num_mismatches = np.sum(np.array(diffs) > self.THRESHOLD_GOOD)
            total_values = len(diffs)

            status = self._classify_status(max_diff_pct)

            return {
                'status': status,
                'max_diff_pct': max_diff_pct,
                'max_diff_abs': max_diff_abs,
                'worst_field': worst_field,
                'num_mismatches': num_mismatches,
                'total_values': total_values,
                'error_msg': None
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'max_diff_pct': np.nan,
                'max_diff_abs': np.nan,
                'worst_field': 'N/A',
                'num_mismatches': 0,
                'total_values': 0,
                'note': None,
                'error_msg': f'Metrics comparison failed: {str(e)}'
            }

    def _compare_numeric_array(self, df_py: pd.DataFrame, df_mat: pd.DataFrame,
                               py_file: Path, mat_file: Path) -> Dict:
        """Compare numeric array CSV files."""
        try:
            # Get all numeric columns
            py_numeric = df_py.select_dtypes(include=[np.number])
            mat_numeric = df_mat.select_dtypes(include=[np.number])

            if py_numeric.empty or mat_numeric.empty:
                return {
                    'status': 'ERROR',
                    'max_diff_pct': np.nan,
                    'max_diff_abs': np.nan,
                    'worst_field': 'N/A',
                    'num_mismatches': 0,
                    'total_values': 0,
                    'error_msg': 'No numeric data found'
                }

            # Check for phase and magnitude columns
            phase_columns = [col for col in py_numeric.columns
                           if 'phase' in col.lower() or 'angle' in col.lower()]
            mag_columns = [col for col in py_numeric.columns
                          if 'mag' in col.lower() and 'imag' not in col.lower()]

            # Columns that should use absolute tolerance only (typically near-zero values)
            abs_only_columns = [col for col in py_numeric.columns
                              if col.lower() in ['mu', 'dc', 'anoi', 'pnoi',
                                                'rms_error', 'rms_indep', 'rms_dep',
                                                'pwr', 'offset']]

            # Compare column by column to handle phase angles specially
            all_diffs_pct = []
            col_max_diffs = []

            for col in py_numeric.columns:
                if col not in mat_numeric.columns:
                    continue

                py_col = py_numeric[col].values
                mat_col = mat_numeric[col].values

                if len(py_col) != len(mat_col):
                    return {
                        'status': 'ERROR',
                        'max_diff_pct': np.nan,
                        'max_diff_abs': np.nan,
                        'worst_field': 'N/A',
                        'num_mismatches': 0,
                        'total_values': 0,
                        'error_msg': f'Length mismatch in column {col}: Python={len(py_col)}, MATLAB={len(mat_col)}'
                    }

                # Calculate difference
                diff = py_col - mat_col

                # For phase columns, use angle wrapping and ignore when magnitude is zero
                if col in phase_columns:
                    # Wrap difference to [-π, π]
                    diff = np.angle(np.exp(1j * diff))
                    diff_abs = np.abs(diff)

                    # Find corresponding magnitude column to ignore phase when mag ≈ 0
                    if mag_columns:
                        # Use first magnitude column found
                        mag_col = mag_columns[0]
                        if mag_col in mat_numeric.columns:
                            mag_vals = mat_numeric[mag_col].values
                            # Ignore phase at noise floor (< 1.0 dB for normalized data)
                            # For raw magnitudes use 1e-10, for dB-normalized use 1.0
                            threshold = 1.0 if np.max(np.abs(mag_vals)) < 1000 else 1e-10
                            zero_mag_mask = np.abs(mag_vals) < threshold
                            diff_abs[zero_mag_mask] = 0.0  # Set diff to 0 for noise floor points
                else:
                    diff_abs = np.abs(diff)

                # Flag if this column should use absolute-only comparison
                is_abs_only = col in abs_only_columns

                # Apply absolute tolerance - treat tiny differences as zero
                tiny_diff_mask = diff_abs < self.ABS_TOLERANCE
                diff_abs[tiny_diff_mask] = 0.0

                # Percentage difference (avoid division by zero)
                with np.errstate(divide='ignore', invalid='ignore'):
                    # For absolute-only columns (mu, dc, etc.), use absolute difference directly
                    if is_abs_only:
                        # Convert absolute difference to percentage-equivalent for grading
                        # Use fixed thresholds for absolute difference
                        col_diff_pct = np.zeros_like(diff_abs)
                        col_diff_pct[diff_abs < 1e-10] = 0.0001   # PERFECT
                        col_diff_pct[(diff_abs >= 1e-10) & (diff_abs < 1e-6)] = 0.005   # EXCELLENT
                        col_diff_pct[(diff_abs >= 1e-6) & (diff_abs < 1e-4)] = 0.05    # GOOD
                        col_diff_pct[(diff_abs >= 1e-4) & (diff_abs < 1e-3)] = 0.5     # ACCEPTABLE
                        col_diff_pct[diff_abs >= 1e-3] = 5.0      # NEEDS REVIEW
                    else:
                        col_diff_pct = diff_abs / (np.abs(mat_col) + 1e-100) * 100
                        # Zero out percentages where absolute difference is negligible
                        col_diff_pct[tiny_diff_mask] = 0.0

                        # For small values, cap percentage based on absolute difference
                        # Three-tier approach:
                        # 1. Near-zero values (< 1e-4): Very permissive, diff < 1e-6 = EXCELLENT
                        # 2. Small values (< 1e-3): Permissive, diff < 1e-5 = EXCELLENT
                        # 3. Normal values (≥ 1e-3): Use percentage as usual

                        # Tier 1: Near-zero values
                        near_zero_mask = np.abs(mat_col) < self.NEAR_ZERO_THRESHOLD
                        near_zero_diff_mask = diff_abs < self.NEAR_ZERO_ABS_TOLERANCE
                        near_zero_and_close = near_zero_mask & near_zero_diff_mask
                        col_diff_pct[near_zero_and_close] = 0.001  # Treat as EXCELLENT

                        # Tier 2: Small values (but not near-zero)
                        small_value_mask = (np.abs(mat_col) >= self.NEAR_ZERO_THRESHOLD) & \
                                          (np.abs(mat_col) < self.SMALL_VALUE_THRESHOLD)
                        small_diff_mask = diff_abs < self.SMALL_VALUE_ABS_TOLERANCE
                        small_and_close = small_value_mask & small_diff_mask
                        col_diff_pct[small_and_close] = 0.005  # Treat as EXCELLENT

                    col_diff_pct = col_diff_pct[np.isfinite(col_diff_pct)]

                if len(col_diff_pct) > 0:
                    all_diffs_pct.extend(col_diff_pct)
                    col_max_diffs.append((col, col_diff_pct.max(), diff_abs.max()))

            if not all_diffs_pct:
                return {
                    'status': 'ERROR',
                    'max_diff_pct': np.nan,
                    'max_diff_abs': np.nan,
                    'worst_field': 'N/A',
                    'num_mismatches': 0,
                    'total_values': 0,
                    'error_msg': 'No numeric columns to compare'
                }

            # Find worst column
            worst_col, max_diff_pct, max_diff_abs = max(col_max_diffs, key=lambda x: x[1])

            # Count mismatches
            num_mismatches = np.sum(np.array(all_diffs_pct) > self.THRESHOLD_GOOD)
            total_values = len(all_diffs_pct)

            status = self._classify_status(max_diff_pct)

            # Add note if large percentage is on a near-zero value
            note = None
            if status in ['NEEDS REVIEW', 'ACCEPTABLE'] and max_diff_abs < 1e-5:
                note = f"Large % on near-zero value (abs_diff={max_diff_abs:.2e})"

            return {
                'status': status,
                'max_diff_pct': max_diff_pct,
                'max_diff_abs': max_diff_abs,
                'worst_field': worst_col,
                'num_mismatches': num_mismatches,
                'total_values': total_values,
                'note': note,
                'error_msg': None
            }

        except Exception as e:
            return {
                'status': 'ERROR',
                'max_diff_pct': np.nan,
                'max_diff_abs': np.nan,
                'worst_field': 'N/A',
                'num_mismatches': 0,
                'total_values': 0,
                'note': None,
                'error_msg': f'Numeric comparison failed: {str(e)}'
            }

    def _classify_status(self, max_diff_pct: float) -> str:
        """Classify comparison status based on max difference."""
        if np.isnan(max_diff_pct):
            return 'ERROR'
        elif max_diff_pct < self.THRESHOLD_PERFECT:
            return 'PERFECT'
        elif max_diff_pct < self.THRESHOLD_EXCELLENT:
            return 'EXCELLENT'
        elif max_diff_pct < self.THRESHOLD_GOOD:
            return 'GOOD'
        elif max_diff_pct < self.THRESHOLD_ACCEPTABLE:
            return 'ACCEPTABLE'
        else:
            return 'NEEDS REVIEW'

    def compare_all(self, test_types: List[str] = None, datasets: List[str] = None) -> pd.DataFrame:
        """Compare all CSV pairs and return results DataFrame."""
        # Find all pairs
        pairs = self.find_csv_pairs()

        if not pairs:
            print("[No CSV pairs found]")
            return pd.DataFrame()

        # Apply filters
        if test_types:
            pairs = [p for p in pairs if p['test_type'] in test_types]

        if datasets:
            pairs = [p for p in pairs if p['dataset'] in datasets]

        if not pairs:
            print("[No pairs match filters]")
            return pd.DataFrame()

        print(f"[Found] {len(pairs)} CSV pairs to compare")
        print()

        # Compare all pairs
        results = []
        total = len(pairs)
        for i, pair in enumerate(pairs, 1):
            comparison = self.compare_pair(pair['python_file'], pair['matlab_file'])
            results.append({
                **pair,
                **comparison
            })

            # Print formatted progress with fixed-width status
            status = comparison['status']
            status_padded = f"{status:15s}"  # Fixed width of 15 chars
            max_diff_pct = comparison['max_diff_pct']
            csv_name = f"{pair['pair_name']}.csv"

            print(f"[{i:3d}/{total:3d}][{status_padded}] (diff: {max_diff_pct:8.4f}%) {csv_name}")

        self.results = results
        return pd.DataFrame(results)

    def print_summary(self, df: pd.DataFrame):
        """Print comprehensive summary of comparisons."""
        if df.empty:
            print("[No results to summarize]")
            return

        print("=" * 80)
        print("MATLAB vs Python CSV Comparison Summary")
        print("=" * 80)
        print()

        # Overall statistics
        print(f"Total pairs compared: {len(df)}")
        print()

        # Group by test type
        print("By Test Type:")
        for test_type, group in df.groupby('test_type'):
            print(f"  {test_type:30s}: {len(group):3d} pairs")
        print()

        # Status distribution
        print("By Status:")
        for status in ['PERFECT', 'EXCELLENT', 'GOOD', 'ACCEPTABLE', 'NEEDS REVIEW', 'ERROR']:
            count = (df['status'] == status).sum()
            pct = 100 * count / len(df)
            symbol = {
                'PERFECT': '[PERFECT]',
                'EXCELLENT': '[EXCELLENT]',
                'GOOD': '[GOOD]',
                'ACCEPTABLE': '[WARNING]',
                'NEEDS REVIEW': '[FAIL]',
                'ERROR': '[ERROR]'
            }.get(status, '?')
            print(f"  {status:15s}: {count:3d} ({pct:5.1f}%) {symbol}")
        print()

        # Best matches
        perfect_count = (df['status'] == 'PERFECT').sum()
        excellent_count = (df['status'] == 'EXCELLENT').sum()
        good_count = (df['status'] == 'GOOD').sum()

        if perfect_count + excellent_count + good_count > 0:
            print("=" * 80)
            print(f"High Quality Matches: {perfect_count + excellent_count + good_count}/{len(df)}")
            print("=" * 80)
            print()

        # Files needing review
        needs_review = df[df['status'] == 'NEEDS REVIEW']
        if not needs_review.empty:
            print("=" * 80)
            print(f"Files Needing Review ({len(needs_review)})")
            print("=" * 80)
            print()

            for idx, row in needs_review.iterrows():
                print(f"[FAIL] {row['pair_name']}")
                print(f"   Max diff: {row['max_diff_pct']:.4f}%")
                print(f"   Worst field: {row['worst_field']}")
                print(f"   Mismatches: {row['num_mismatches']}/{row['total_values']}")
                if row['error_msg']:
                    print(f"   Error: {row['error_msg']}")
                print()

        # Errors
        errors = df[df['status'] == 'ERROR']
        if not errors.empty:
            print("=" * 80)
            print(f"Errors ({len(errors)})")
            print("=" * 80)
            print()

            for idx, row in errors.iterrows():
                print(f"[ERROR] {row['pair_name']}")
                print(f"   Error: {row['error_msg']}")
                print()
