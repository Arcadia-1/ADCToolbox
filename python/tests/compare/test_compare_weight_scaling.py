from tests.compare._runner import run_comparison_suite


def test_compare_weight_scaling(project_root, comparison_output_root):

    run_comparison_suite(project_root, matlab_test_name="test_wscaling",
                         out_folder=comparison_output_root, structure="nested")
