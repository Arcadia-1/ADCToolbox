from tests.compare._runner import run_comparison_suite


def test_compare_err_envelope_spectrum(project_root, comparison_output_root):

    # The fixture now matches MATLAB errevspec by analyzing a precomputed
    # residual. A small tolerance remains because Python fit_sine_4param and
    # MATLAB sinfit generate that residual with slightly different solvers.
    run_comparison_suite(project_root, matlab_test_name="run_errevspec",
                         ref_folder="reference_output", out_folder=comparison_output_root,
                         structure="nested", threshold=1e-4)
