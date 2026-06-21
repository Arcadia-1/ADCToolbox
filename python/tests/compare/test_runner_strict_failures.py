import pytest

from tests.conftest import _resolve_comparison_output_root
from tests.compare._runner import run_comparison_suite


def test_flat_reference_folder_missing_fails(tmp_path):
    with pytest.raises(AssertionError, match="Flat reference folder not found"):
        run_comparison_suite(
            tmp_path,
            matlab_test_name="test_basic",
            ref_folder="reference_output",
            out_folder="test_output",
            structure="flat",
        )


def test_nested_reference_dataset_missing_fails(tmp_path):
    (tmp_path / "reference_output").mkdir()

    with pytest.raises(AssertionError, match="No reference datasets found"):
        run_comparison_suite(
            tmp_path,
            matlab_test_name="run_plotspec",
            ref_folder="reference_output",
            out_folder="test_output",
            structure="nested",
        )


def test_missing_python_output_directory_fails(tmp_path):
    mat_dir = tmp_path / "reference_output" / "run_plotspec"
    mat_dir.mkdir(parents=True)
    (mat_dir / "enob_matlab.csv").write_text("1.0\n")

    with pytest.raises(AssertionError, match="Python directory not found"):
        run_comparison_suite(
            tmp_path,
            matlab_test_name="run_plotspec",
            ref_folder="reference_output",
            out_folder="test_output",
            structure="flat",
        )


def test_missing_matlab_reference_for_python_output_fails(tmp_path):
    mat_dir = tmp_path / "reference_output" / "run_plotspec"
    py_dir = tmp_path / "test_output" / "test_analyze_spectrum"
    mat_dir.mkdir(parents=True)
    py_dir.mkdir(parents=True)
    (py_dir / "enob_python.csv").write_text("1.0\n")

    with pytest.raises(AssertionError, match="No MATLAB reference"):
        run_comparison_suite(
            tmp_path,
            matlab_test_name="run_plotspec",
            ref_folder="reference_output",
            out_folder="test_output",
            structure="flat",
        )


def test_matching_csv_pair_passes(tmp_path):
    mat_dir = tmp_path / "reference_output" / "run_plotspec"
    py_dir = tmp_path / "test_output" / "test_analyze_spectrum"
    mat_dir.mkdir(parents=True)
    py_dir.mkdir(parents=True)
    (mat_dir / "enob_matlab.csv").write_text("1.0\n2.0\n")
    (py_dir / "enob_python.csv").write_text("1.0\n2.0\n")

    run_comparison_suite(
        tmp_path,
        matlab_test_name="run_plotspec",
        ref_folder="reference_output",
        out_folder="test_output",
        structure="flat",
    )


def test_absolute_output_root_passes(tmp_path):
    mat_dir = tmp_path / "reference_output" / "run_plotspec"
    out_root = tmp_path / "generated" / "test_output"
    py_dir = out_root / "test_analyze_spectrum"
    mat_dir.mkdir(parents=True)
    py_dir.mkdir(parents=True)
    (mat_dir / "enob_matlab.csv").write_text("1.0\n2.0\n")
    (py_dir / "enob_python.csv").write_text("1.0\n2.0\n")

    run_comparison_suite(
        tmp_path,
        matlab_test_name="run_plotspec",
        ref_folder="reference_output",
        out_folder=out_root,
        structure="flat",
    )


def test_explicit_variables_ignore_extra_python_outputs(tmp_path):
    mat_dir = tmp_path / "reference_output" / "test_basic"
    py_dir = tmp_path / "test_output" / "test_basic"
    mat_dir.mkdir(parents=True)
    py_dir.mkdir(parents=True)
    (mat_dir / "sinewave_matlab.csv").write_text("1.0\n2.0\n")
    (py_dir / "sinewave_python.csv").write_text("1.0\n2.0\n")
    (py_dir / "test_scalar_python.csv").write_text("1.5\n")

    run_comparison_suite(
        tmp_path,
        matlab_test_name="test_basic",
        ref_folder="reference_output",
        out_folder="test_output",
        structure="flat",
        variables=["sinewave"],
    )


def test_comparison_output_root_accepts_artifact_root(tmp_path):
    artifact_root = tmp_path / "artifacts"
    test_output = artifact_root / "test_output"
    test_output.mkdir(parents=True)

    assert _resolve_comparison_output_root(artifact_root) == test_output
