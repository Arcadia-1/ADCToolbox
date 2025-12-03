from tests.compare._runner import run_comparison_suite


def test_compare_weight_scaling(project_root):

    run_comparison_suite(project_root, matlab_test_name="test_wscaling",
                         structure="nested")
