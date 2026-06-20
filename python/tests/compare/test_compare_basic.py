from tests.compare._runner import run_comparison_suite


def test_compare_basic(project_root, comparison_output_root):

    run_comparison_suite(project_root, matlab_test_name="test_basic",
                         ref_folder="reference_output", out_folder=comparison_output_root,
                         structure="flat", variables=["sinewave"], threshold=1e-5)
