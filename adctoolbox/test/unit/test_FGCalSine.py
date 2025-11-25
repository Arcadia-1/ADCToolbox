"""Test FGCalSine.py - Foreground calibration for SAR ADC data.

Configuration - assumes running from project root d:\ADCToolbox

Tests FGCalSine function with SAR ADC digital code datasets.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

from adctoolbox.dout import FGCalSine
from adctoolbox.aout import spec_plot


def run_fgcal_tests():
    """Test FGCalSine function on SAR ADC datasets."""

    # Configuration - assumes running from project root d:\ADCToolbox
    input_dir = "test_data"
    output_dir = "test_output"

    file_list = [
        "dout_SAR_12b_weight_1.csv",
        "dout_SAR_12b_weight_2.csv",
        "dout_SAR_12b_weight_3.csv",
    ]

    print("=" * 60)
    print("Test: FGCalSine (Foreground Calibration)")
    print("=" * 60)

    results = []

    for filename in file_list:
        name = filename.replace(".csv", "")
        filepath = os.path.join(input_dir, filename)
        case_dir = os.path.join(output_dir, name, "test_FGCalSine")
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n[{name}]")

        if not os.path.exists(filepath):
            print(f"  ERROR: File not found")
            results.append((name, False, "File not found"))
            continue

        try:
            # Read and calibrate
            bits = np.loadtxt(filepath, delimiter=',')
            n_bits = bits.shape[1]
            weight_cal, offset, _, _, _, freq_cal = FGCalSine(bits)

            # Calibrated signal
            cal_signal = bits @ weight_cal
            ENoB, SNDR, SFDR, _, _, _, _, _ = spec_plot(cal_signal, Fs=1.0, harmonic=7, label=1, OSR=1)
            plt.savefig(os.path.join(case_dir, "specPlot_FGCalSine.png"), dpi=150)
            plt.close()

            # Uncalibrated signal
            nom_weights = 2.0 ** np.arange(n_bits - 1, -1, -1)
            nom_weights /= np.sum(nom_weights)
            uncal_signal = bits @ nom_weights
            ENoB_uncal, SNDR_uncal, _, _, _, _, _, _ = spec_plot(uncal_signal, Fs=1.0, harmonic=7, label=1, OSR=1)
            plt.savefig(os.path.join(case_dir, "specPlot_uncalibrated.png"), dpi=150)
            plt.close()

            print(f"  Uncal: ENoB={ENoB_uncal:.2f}, SNDR={SNDR_uncal:.2f}dB")
            print(f"  Cal:   ENoB={ENoB:.2f}, SNDR={SNDR:.2f}dB")
            print(f"  Gain:  +{ENoB - ENoB_uncal:.2f} bits, +{SNDR - SNDR_uncal:.2f}dB")

            results.append((name, True, f"ENoB={ENoB:.2f}"))

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    ok = sum(1 for _, passed, _ in results if passed)
    for name, passed, info in results:
        print(f"  {'OK' if passed else 'FAIL'} {name}: {info}")
    print(f"Total: {ok}/{len(results)} passed")

    return ok == len(results)


if __name__ == "__main__":
    sys.exit(0 if run_fgcal_tests() else 1)
