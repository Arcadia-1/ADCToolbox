"""
Run all comparison tests and generate a single consolidated log file.
"""
from datetime import datetime
from pathlib import Path
from tests.compare._runner import run_comparison_suite


def run_all_comparisons(project_root):
    """
    Run all comparison tests and generate a single log file.
    """
    # Setup consolidated log
    log_lines = []

    # Header
    log_lines.append("=" * 80)
    log_lines.append("ADCToolbox: Python vs MATLAB Comparison Test Results")
    log_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 80)
    log_lines.append("")

    # List of all comparison tests
    # Format: (matlab_test_name, structure_type)
    tests = [
        # Basic tests
        ("test_basic", "flat"),
        ("test_sinfit", "nested"),
        ("test_alias", "nested"),

        # AOUT tests (analog output)
        ("test_errsin_phase", "nested"),
        ("test_errsin_code", "nested"),
        ("test_plotspec", "nested"),
        ("test_plotphase", "nested"),
        ("test_fitstaticnl", "nested"),
        ("test_errpdf", "nested"),
        ("test_errspec", "nested"),
        ("test_errac", "nested"),
        ("test_errevspec", "nested"),
        ("test_inlsine", "nested"),
        ("test_tomdec", "nested"),

        # DOUT tests (digital output)
        ("test_bitact", "nested"),
        ("test_wscaling", "nested"),
        ("test_bitsweep", "nested"),
        ("test_wcalsine", "nested"),
        ("test_ovfchk", "nested"),
    ]

    total_tests = len(tests)
    passed_tests = 0
    failed_tests = []

    for idx, (matlab_test_name, structure) in enumerate(tests, 1):
        try:
            log_lines.append("")
            log_lines.append(f"[{idx}/{total_tests}] Running comparison: {matlab_test_name}")
            log_lines.append("-" * 80)

            # Run comparison and append logs
            run_comparison_suite(
                project_root=project_root,
                matlab_test_name=matlab_test_name,
                structure=structure,
                enable_logging=False,  # Don't create individual log files
                log_lines_output=log_lines  # Append to consolidated log
            )

            passed_tests += 1

        except AssertionError as e:
            failed_tests.append((matlab_test_name, str(e)))
            log_lines.append(f"  -> [FAILED] {str(e)}")
        except Exception as e:
            failed_tests.append((matlab_test_name, f"Error: {str(e)}"))
            log_lines.append(f"  -> [ERROR] {str(e)}")

    # Final summary
    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append("FINAL SUMMARY")
    log_lines.append("=" * 80)
    log_lines.append(f"Total Tests:  {total_tests}")
    log_lines.append(f"Passed:       {passed_tests}")
    log_lines.append(f"Failed:       {len(failed_tests)}")

    if failed_tests:
        log_lines.append("")
        log_lines.append("Failed Tests:")
        for test_name, error_msg in failed_tests:
            log_lines.append(f"  - {test_name}: {error_msg}")

    # Save consolidated log file
    log_dir = project_root / "test_comparison_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"all_comparisons_{timestamp}.txt"

    log_lines.append("")
    log_lines.append("=" * 80)
    log_lines.append(f"Log saved to: {log_file}")
    log_lines.append("=" * 80)

    # Write to file
    log_file.write_text('\n'.join(log_lines), encoding='utf-8')

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON TEST SUITE COMPLETED")
    print("=" * 80)
    print(f"Total Tests:  {total_tests}")
    print(f"Passed:       {passed_tests}")
    print(f"Failed:       {len(failed_tests)}")
    print(f"\nLog saved to: {log_file}")
    print("=" * 80 + "\n")

    # Raise error if any test failed
    if failed_tests:
        raise AssertionError(f"{len(failed_tests)} comparison test(s) failed. See log for details.")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Get project root (3 levels up from this script)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent

    run_all_comparisons(project_root)
