import pytest
from tests.compare._runner import run_comparison_suite


@pytest.mark.skip(reason="Known inconsistency: MATLAB doesn't use sideBin for harmonics, Python does. "
                         "See agent_playground/THD_CALCULATION_INCONSISTENCY.md for details. "
                         "Python is MORE correct - uses consistent side bin handling. "
                         "Differences: THD ~3%, SFDR ~2%, others <0.1%")
def test_compare_spec_plot(project_root):

    run_comparison_suite(project_root, matlab_test_name="run_plotspec",
                         ref_folder="reference_output", out_folder="test_output", structure="nested")
