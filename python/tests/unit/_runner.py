import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tests._utils import auto_search_files


@dataclass(frozen=True)
class BatchTestResult:
    """Summary returned by run_unit_test_batch for integration assertions."""

    files: list[str]
    success_count: int
    output_dir: Path
    figures_dir: Path
    output_folders: list[Path]


def run_unit_test_batch(
    project_root,
    input_subpath,
    test_module_name,
    file_pattern,
    process_callback,
    output_subpath="test_output",
    flatten=True,
    artifact_root=None,
    figures_subpath="test_plots",
):
    """
    Generic batch runner for unit tests.
    Executes process_callback(raw_data, output_folder, dataset_name, figures_folder, test_name) for each file.
    Raises AssertionError if any file fails processing.

    :param output_subpath: Relative path for CSV output (default: "test_output")
    :param flatten: Whether to flatten data to 1D (default: True for aout, False for dout)
    :param artifact_root: Root for generated CSV/PNG artifacts. Defaults to project_root
                          for compatibility with explicit golden workflows.
    :param figures_subpath: Relative path for generated plots.

    Output structure:
    - CSV data: test_output/{dataset_name}/{test_module_name}/
    - Plots: test_plots/
    """
    input_dir = project_root / input_subpath
    artifact_base = Path(artifact_root) if artifact_root is not None else project_root
    output_dir = artifact_base / output_subpath  # CSV data output
    figures_dir = artifact_base / figures_subpath  # Plots output

    files_list = []
    files_list = auto_search_files(files_list, input_dir, file_pattern)

    success_count = 0
    failures = []
    output_folders = []

    for k, current_filename in enumerate(files_list, 1):
        try:
            data_file_path = input_dir / current_filename
            print(f"[{k}/{len(files_list)}] Processing [{current_filename}]")

            raw_data = np.loadtxt(data_file_path, delimiter=',')
            if flatten:
                raw_data = raw_data.flatten()

            dataset_name = data_file_path.stem
            sub_folder = output_dir / dataset_name / test_module_name
            sub_folder.mkdir(parents=True, exist_ok=True)

            process_callback(raw_data, sub_folder, dataset_name, figures_dir, test_module_name)
            if not any(sub_folder.glob("*_python.csv")):
                raise AssertionError(f"No Python CSV outputs were generated in {sub_folder}.")

            success_count += 1
            output_folders.append(sub_folder)

        except Exception as e:
            error_msg = f"{current_filename}: {str(e)}"
            print(f"   -> [ERROR] Failed processing {current_filename}")
            print(f"   -> {str(e)}")
            failures.append(error_msg)

    print("-" * 60)
    print(f"[{test_module_name}] Done [{success_count}/{len(files_list)}]")

    if failures:
        raise AssertionError(
            f"Test failed for {len(failures)} file(s):\n" + "\n".join(f"  - {f}" for f in failures)
        )

    if success_count == 0:
        raise AssertionError(f"No files were successfully processed for {test_module_name}.")

    if success_count != len(files_list):
        raise AssertionError(
            f"Processed {success_count}/{len(files_list)} file(s) for {test_module_name}."
        )

    return BatchTestResult(
        files=files_list,
        success_count=success_count,
        output_dir=output_dir,
        figures_dir=figures_dir,
        output_folders=output_folders,
    )
