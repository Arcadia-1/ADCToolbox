import matplotlib.pyplot as plt

from adctoolbox.dout.enob_bit_sweep import enob_bit_sweep
from tests._utils import save_variable, save_fig
from tests.unit._runner import run_unit_test_batch

plt.rcParams['font.size'] = 14
plt.rcParams['axes.grid'] = True

def _process_enob_bit_sweep(raw_data, sub_folder, dataset_name):
    """
    Callback function to process a single file:
    1. Run ENOB bit sweep analysis
    2. Save ENOB sweep and nBits vectors
    3. Save plot
    """
    # Create figure and run enob_bit_sweep
    fig = plt.figure(figsize=(10, 7.5))
    enob_sweep, n_bits_vec = enob_bit_sweep(
        raw_data, freq=0, order=5, harmonic=5, osr=1, win_type=4, plot=True)

    # Save figure
    save_fig(sub_folder, 'ENoB_bitSweep.png', dpi=150)

    # Save variables
    save_variable(sub_folder, enob_sweep, 'ENoB_sweep')
    save_variable(sub_folder, n_bits_vec, 'nBits_vec')

def test_enob_bit_sweep(project_root):
    """
    Batch runner for ENOB bit sweep analysis.
    """
    run_unit_test_batch(
        project_root=project_root,
        input_subpath="dataset/dout",
        test_module_name="test_enob_bit_sweep",
        file_pattern="dout_*.csv",
        output_subpath="test_output",
        process_callback=_process_enob_bit_sweep,
        flatten=False  # Digital output data is 2D (N samples x M bits)
    )
