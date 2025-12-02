import numpy as np
from pathlib import Path
from tests._utils import auto_search_files

def run_unit_test_batch(project_root, input_subpath, test_module_name, file_pattern, process_callback, output_subpath="test_data", flatten=True):
    """
    Generic batch runner for unit tests.
    Executes process_callback(raw_data, output_folder, dataset_name, figures_folder, test_name) for each file.
    Raises AssertionError if any file fails processing.

    :param output_subpath: Relative path for output (default: "test_data")
    :param flatten: Whether to flatten data to 1D (default: True for aout, False for dout)
    """
    input_dir = project_root / input_subpath
    output_dir = project_root / output_subpath  # Now configurable
    figures_dir = project_root / "test_plots"  # Flat figures directory

    files_list = []
    files_list = auto_search_files(files_list, input_dir, file_pattern)

    success_count = 0
    failures = []

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

            success_count += 1

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