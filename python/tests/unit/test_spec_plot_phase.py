import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from adctoolbox.aout import spec_plot_phase
from tests._utils import auto_search_files, save_variable

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def test_spec_plot_phase(project_root):
    """
    Batch runner for spec_plot_phase (Single Channel Version).
    """
    input_dir = project_root / "reference_dataset" / "sinewave"
    output_dir = project_root / "test_output"

    files_list = []
    files_list = auto_search_files(files_list, input_dir, 'sinewave_*.csv')

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for k, current_filename in enumerate(files_list, 1):
        try:
            data_file_path = input_dir / current_filename
            print(f"[{k}/{len(files_list)}] Processing: [{current_filename}]")

            raw_data = np.loadtxt(data_file_path, delimiter=',').flatten()

            dataset_name = data_file_path.stem
            sub_folder = output_dir / dataset_name / "test_spec_plot_phase"
            sub_folder.mkdir(parents=True, exist_ok=True)

            phase_plot_path = sub_folder / 'phase_python.png'
            result = spec_plot_phase(raw_data, save_path=str(phase_plot_path))

            spec = result['spec']
            freq_bins = result['freq_bins']

            freq_bins_no_dc = freq_bins[1:]
            spec_no_dc = spec[1:len(freq_bins)]
            phi_no_dc = result['phi'][1:len(freq_bins)]

            save_variable(sub_folder, freq_bins_no_dc, 'freq_bin')
            save_variable(sub_folder, np.real(spec_no_dc), 'spec_real')
            save_variable(sub_folder, np.imag(spec_no_dc), 'spec_imag')
            save_variable(sub_folder, np.abs(spec_no_dc), 'spec_mag')
            save_variable(sub_folder, np.angle(spec_no_dc), 'spec_phase')
            save_variable(sub_folder, np.real(phi_no_dc), 'phi_real')
            save_variable(sub_folder, np.imag(phi_no_dc), 'phi_imag')

            success_count += 1

        except Exception as e:
            print(f"      -> [ERROR] Failed in processing [{current_filename}]")
            print(f"      -> {str(e)}\n")

    print("-" * 60)
    print(f"[DONE] Generation complete. Success: {success_count}/{len(files_list)}")
    plt.close('all')

