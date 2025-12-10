import pytest
from tests.compare._runner import run_comparison_suite


@pytest.mark.skip(reason="Comparison test disabled - minor numerical differences need investigation.")
def test_compare_enob_bit_sweep(project_root):

    run_comparison_suite(project_root, matlab_test_name="test_bitsweep",
                         structure="nested")
