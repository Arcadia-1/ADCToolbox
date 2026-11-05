import pytest
from tests.compare._runner import run_comparison_suite


@pytest.mark.skip(reason="run_inlsine MATLAB references were saved at 10 bits with the old inlsin "
                         "code axis; regenerate them with MATLAB before re-enabling.")
def test_compare_inl_sine(project_root, comparison_output_root):

    run_comparison_suite(project_root, matlab_test_name="run_inlsine",
                         ref_folder="reference_output", out_folder=comparison_output_root, structure="nested")
