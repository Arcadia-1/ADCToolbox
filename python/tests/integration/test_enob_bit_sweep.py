import matplotlib.pyplot as plt

from adctoolbox.dout import analyze_enob_sweep
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch
from tests import config

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_analyze_enob_sweep(raw_data, sub_folder, dataset_name, figures_folder, test_name):
    """
    Callback function to process a single file:
    1. Run ENOB bit sweep analysis
    2. Save ENOB sweep vector
    3. Save plot
    """
    # Create figure and run analyze_enob_sweep
    # Note: Using 'hamming' (Pythonic) instead of win_type=4 (legacy)
    fig = plt.figure(figsize=(10, 7.5))
    enob_sweep, n_bits_vec = analyze_enob_sweep(
        raw_data, freq=0, order=5, harmonic=5, osr=1, win_type='hamming', plot=True)

    # Save figure
    figure_name = f"{test_name}_{dataset_name}_python.png"
    save_fig(figures_folder, figure_name, dpi=150, close_fig=False)
    plt.close(fig)

    # Save variables (only enob_sweep, matching MATLAB)
    save_variable(sub_folder, enob_sweep, 'ENoB_sweep')

def test_analyze_enob_sweep(project_root):
    """
    Batch runner for ENOB bit sweep analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath=config.DOUT['input_path'], test_module_name="test_analyze_enob_sweep", file_pattern=config.DOUT['file_pattern'],        process_callback=_process_analyze_enob_sweep,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
