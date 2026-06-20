from tests.compare._runner import run_comparison_suite


def test_compare_overflow_chk(project_root, comparison_output_root):

    run_comparison_suite(project_root, matlab_test_name="test_ovfchk",
                         out_folder=comparison_output_root, structure="nested")
