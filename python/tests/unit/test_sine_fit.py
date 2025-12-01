import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from adctoolbox.common import sine_fit
from tests._utils import auto_search_files, save_variable, save_fig

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def test_sine_fit(project_root):
    """
    Batch runner for sine_fit (Single Channel Version).
    """
    input_dir = project_root / "dataset" / "aout" / "sinewave"
    output_dir = project_root / "test_output" 

    files_list = []
    files_list = auto_search_files(files_list, input_dir, 'sinewave_*.csv')

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for k, current_filename in enumerate(files_list, 1):
        try:
            data_file_path = input_dir / current_filename
            print(f"[{k}/{len(files_list)}] Processing: [{current_filename}]")

            raw_data = pd.read_csv(data_file_path, header=None).values.flatten()

            dataset_name = data_file_path.stem
            sub_folder = output_dir / dataset_name / "test_sine_fit"
            sub_folder.mkdir(parents=True, exist_ok=True)

            data_fit, freq, mag, dc, phi = sine_fit(raw_data)

            save_variable(sub_folder, freq, 'freq')
            save_variable(sub_folder, mag, 'mag')
            save_variable(sub_folder, dc, 'dc')
            save_variable(sub_folder, phi, 'phi')            
            save_variable(sub_folder, data_fit, 'data_fit')

            period_samples = int(round(1.0 / freq)) if freq > 0 else len(raw_data) # freq=0 means DC signal
            n_plot = min(max(period_samples, 20), len(raw_data))

            fig = plt.figure(figsize=(8, 6))

            t_data = np.arange(n_plot)
            plt.plot(t_data, raw_data[:n_plot], 'bo-', linewidth=2, markersize=6, label='Original')

            t_dense = np.linspace(0, n_plot - 1, n_plot * 50)
            fitted_sine = mag * np.cos(2 * np.pi * freq * t_dense + phi) + dc
            plt.plot(t_dense, fitted_sine, 'r--', linewidth=2, label='Fitted Sine')

            plt.title(f'Sine Fit: {dataset_name}')
            plt.ylim([np.min(fitted_sine) - 0.1, np.max(fitted_sine) + 0.2])
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.legend(loc='upper left')
            plt.tight_layout()

            save_fig(sub_folder, "sineFit_python.png")
            plt.close(fig)
            success_count += 1

        except Exception as e:
            print(f"      -> [ERROR] Failed in processing [{current_filename}]")
            print(f"      -> {str(e)}\n")

    print("-" * 60)
    print(f"[DONE] Generation complete. Success: {success_count}/{len(files_list)}")