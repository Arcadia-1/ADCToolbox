from tests.compare._runner import run_comparison_suite


def test_compare_spec_plot(project_root):

    run_comparison_suite(project_root, matlab_test_name="test_specPlot",
                         ref_folder="test_reference", out_folder="test_output", structure="nested")
