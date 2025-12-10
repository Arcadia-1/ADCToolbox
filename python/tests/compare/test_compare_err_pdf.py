from tests.compare._runner import run_comparison_suite


def test_compare_err_pdf(project_root):

    run_comparison_suite(project_root, matlab_test_name="run_errpdf",
                         ref_folder="reference_output", out_folder="test_output", structure="nested")
