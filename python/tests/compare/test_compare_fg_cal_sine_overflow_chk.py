from tests.compare._runner import run_comparison_suite


def test_compare_fg_cal_sine_overflow_chk(project_root):

    run_comparison_suite(project_root, matlab_test_name="test_FGCalSine_overflowChk",
                         ref_folder="test_output", out_folder="test_output", structure="nested")
