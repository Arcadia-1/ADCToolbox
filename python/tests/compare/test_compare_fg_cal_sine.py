from tests.compare._runner import run_comparison_suite


def test_compare_fg_cal_sine(project_root, comparison_output_root):

    # Compare the calibration solve itself. Spectrum metrics from pre/post
    # waveforms are covered by spectrum parity tests and can differ for reasons
    # unrelated to wcalsin/calibrate_weight_sine numerical agreement.
    run_comparison_suite(project_root, matlab_test_name="test_wcalsine",
                         out_folder=comparison_output_root, structure="nested",
                         variables=[
                             "weight",
                             "offset",
                             "postCal",
                             "ideal",
                             "err",
                             "freqCal",
                             "preCal",
                         ])
