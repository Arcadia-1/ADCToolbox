import numpy as np
import matplotlib.pyplot as plt

from adctoolbox.spectrum import analyze_spectrum_polar
from tests._utils import auto_search_files
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def test_analyze_phase_spectrum_fft(project_root):
    """
    Batch runner for analyze_phase_spectrum - FFT Mode.
    """
    input_dir = project_root / config.AOUT['input_path']
    figures_dir = project_root / "test_plots"

    files_list = []
    files_list = auto_search_files(files_list, input_dir, config.AOUT['file_pattern'])

    figures_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    failures = []

    for k, current_filename in enumerate(files_list, 1):
        try:
            data_file_path = input_dir / current_filename
            print(f"[{k}/{len(files_list)}] Processing: [{current_filename}]")

            raw_data = np.loadtxt(data_file_path, delimiter=',').flatten()
            dataset_name = data_file_path.stem

            figure_name = f"test_plotphase_fft_{dataset_name}_python.png"
            phase_plot_path = figures_dir / figure_name
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "polar"})
            coherent_result = analyze_spectrum_polar(raw_data, harmonic=10, ax=ax)
            fig.savefig(phase_plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            expected_keys = {"enob", "sndr_dbc", "sfdr_dbc", "snr_dbc", "thd_dbc"}
            assert expected_keys <= set(coherent_result)
            assert np.isfinite(coherent_result["enob"])
            assert np.isfinite(coherent_result["sndr_dbc"])
            assert phase_plot_path.exists()
            assert phase_plot_path.stat().st_size > 0

            success_count += 1

        except Exception as e:
            print(f"      -> [ERROR] Failed in processing [{current_filename}]")
            print(f"      -> {str(e)}\n")
            failures.append(f"{current_filename}: {e}")

    print("-" * 60)
    print(f"[DONE] FFT mode complete. Success: {success_count}/{len(files_list)}")
    plt.close('all')

    if failures:
        raise AssertionError(
            f"FFT phase spectrum failed for {len(failures)} file(s):\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
    assert success_count == len(files_list) > 0
