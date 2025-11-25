"""visual_comparison.py - Create side-by-side visual comparisons

Creates visual comparisons of MATLAB vs Python plots for selected datasets.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from glob import glob
from pathlib import Path


def create_side_by_side_comparison(dataset_name, output_dir="test_output", project_root=None):
    """Create side-by-side comparison plot."""
    # Paths to MATLAB and Python images
    matlab_img_path = os.path.join(output_dir, dataset_name, "test_specPlot", "spectrum_matlab.png")
    python_img_path = os.path.join(output_dir, dataset_name, "test_specPlot", "spectrum_python.png")

    if not os.path.exists(matlab_img_path):
        print(f"MATLAB image not found: {matlab_img_path}")
        return False

    if not os.path.exists(python_img_path):
        print(f"Python image not found: {python_img_path}")
        return False

    # Read images
    matlab_img = mpimg.imread(matlab_img_path)
    python_img = mpimg.imread(python_img_path)

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.imshow(matlab_img)
    ax1.set_title(f'MATLAB: {dataset_name}', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2.imshow(python_img)
    ax2.set_title(f'Python: {dataset_name}', fontsize=14, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()

    # Save comparison
    comparison_path = os.path.join(output_dir, dataset_name, "test_specPlot", "comparison_visual.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {comparison_path}")
    plt.close()

    return True


def main():
    """Create visual comparisons for representative datasets."""
    # Find project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parents[2]  # adctoolbox/test/unit -> adctoolbox -> ADCToolbox

    # Change to project root directory
    original_dir = os.getcwd()
    os.chdir(project_root)

    print(f"Working directory: {project_root}")
    print()

    output_dir = "test_output"

    # Select representative datasets
    datasets = [
        "sinewave_jitter_1fs",           # EXCELLENT - clean signal
        "batch_sinewave_Nrun_16",         # EXCELLENT - batch processing
        "sinewave_Zone2_Tj_100fs",        # GOOD - jitter zone
        "sinewave_HD2_n80dB_HD3_n70dB",   # NEEDS REVIEW - harmonics
        "sinewave_clipping_0P060",        # NEEDS REVIEW - clipping
    ]

    print("=" * 80)
    print("Creating Visual Comparisons")
    print("=" * 80)
    print()

    for dataset in datasets:
        dataset_path = os.path.join(output_dir, dataset)
        if os.path.exists(dataset_path):
            print(f"Processing: {dataset}")
            create_side_by_side_comparison(dataset, output_dir)
            print()
        else:
            print(f"Skipping: {dataset} (not found)")
            print()

    print("=" * 80)
    print("Visual comparison complete!")
    print("Check test_output/<dataset>/test_specPlot/comparison_visual.png for each dataset")
    print("=" * 80)

    # Restore original directory
    os.chdir(original_dir)


if __name__ == "__main__":
    main()
