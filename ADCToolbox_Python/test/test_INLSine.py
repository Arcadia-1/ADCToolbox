"""Test INLSine.py against MATLAB golden reference."""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from ADC_Toolbox_Python.INLSine import INLsine


def run_inl_tests():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    matlab_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "matlab_reference")
    output_dir = os.path.join(current_dir, "output", "INL_FIRAS_chip2")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Test: INLSine (INL/DNL from Sine Wave)")
    print("=" * 60)

    # Load CSV file
    csv_file = os.path.join(matlab_dir, "FIRAS_chip2_SingleTone_20240408162613.csv")

    if not os.path.exists(csv_file):
        print(f"ERROR: {csv_file} not found")
        return False

    try:
        # Load data from CSV
        d_out_dec_INL = np.loadtxt(csv_file, delimiter=',').flatten()

        print(f"[Data] {len(d_out_dec_INL)} samples")

        # Calculate INL and DNL
        INL, DNL, code = INLsine(d_out_dec_INL, clip=0.01)

        print(f"[INL] range: [{np.min(INL):.2f}, {np.max(INL):.2f}] LSB")
        print(f"[DNL] range: [{np.min(DNL):.2f}, {np.max(DNL):.2f}] LSB")
        print(f"[Code] range: {code[0]} to {code[-1]}")

        # Plot INL
        plt.figure(figsize=(10, 6))
        plt.plot(code, INL, linewidth=0.5)
        plt.xlabel('Code')
        plt.ylabel('INL (LSB)')
        inl_min, inl_max = np.min(INL), np.max(INL)
        plt.title(f'Integral Nonlinearity: INL=[{inl_min:+.2f}, {inl_max:+.2f}] LSB')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "INL.png"), dpi=150)
        plt.close()

        # Plot DNL
        plt.figure(figsize=(10, 6))
        plt.plot(code, DNL, linewidth=0.5)
        plt.xlabel('Code')
        plt.ylabel('DNL (LSB)')
        dnl_min, dnl_max = np.min(DNL), np.max(DNL)
        plt.title(f'Differential Nonlinearity: DNL=[{dnl_min:+.2f}, {dnl_max:+.2f}] LSB')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "DNL.png"), dpi=150)
        plt.close()

        print(f"[OK] Plots saved to: {output_dir}")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if run_inl_tests() else 1)
