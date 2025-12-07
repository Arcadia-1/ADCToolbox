import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from adctoolbox.aout import analyze_phase_spectrum
from tests._utils import auto_search_files, save_variable
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def test_analyze_phase_spectrum_lms(project_root):
    """
    Batch runner for analyze_phase_spectrum - LMS Mode (extracting harmonic info).
    """
    input_dir = project_root / config.AOUT['input_path']
    output_dir = project_root / "test_output"
    figures_dir = project_root / "test_plots"

    files_list = []
    files_list = auto_search_files(files_list, input_dir, config.AOUT['file_pattern'])

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for k, current_filename in enumerate(files_list, 1):
        try:
            data_file_path = input_dir / current_filename
            print(f"[{k}/{len(files_list)}] Processing: [{current_filename}]")

            raw_data = np.loadtxt(data_file_path, delimiter=',').flatten()

            dataset_name = data_file_path.stem
            sub_folder = output_dir / dataset_name / "test_plotphase_lms"
            sub_folder.mkdir(parents=True, exist_ok=True)

            figure_name = f"test_plotphase_lms_{dataset_name}_python.png"
            phase_plot_path = figures_dir / figure_name
            result = analyze_phase_spectrum(raw_data, harmonic=10, mode='LMS', save_path=str(phase_plot_path))

            # Extract LMS mode outputs
            harm_phase = result['harm_phase']
            harm_mag = result['harm_mag']
            freq = result['freq']
            noise_dB = result['noise_dB']

            save_variable(sub_folder, harm_phase, 'harm_phase')
            save_variable(sub_folder, harm_mag, 'harm_mag')
            save_variable(sub_folder, freq, 'freq')
            save_variable(sub_folder, noise_dB, 'noise_dB')

            success_count += 1

        except Exception as e:
            print(f"      -> [ERROR] Failed in processing [{current_filename}]")
            print(f"      -> {str(e)}\n")

    print("-" * 60)
    print(f"[DONE] LMS mode complete. Success: {success_count}/{len(files_list)}")
    plt.close('all')
