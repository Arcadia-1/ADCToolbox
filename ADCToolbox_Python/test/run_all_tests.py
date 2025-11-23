"""
Test runner for ADC analysis tools.

Edit the 'tools' list at the bottom to select which tools to test.
Set tools = None to run all tools.

Available tools: specPlot, specPlotPhase, tomDecomp, errHistSine
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# Add SpecMind directory to path (so ADC_Toolbox_Python can be imported as a package)
# This allows running the test from any directory
current_file = os.path.abspath(__file__)
tests_dir = os.path.dirname(current_file)  # tests/
adc_toolbox_dir = os.path.dirname(tests_dir)  # ADC_Toolbox_Python/
specmind_dir = os.path.dirname(adc_toolbox_dir)  # SpecMind/
sys.path.insert(0, specmind_dir)


from ADC_Toolbox_Python.spec_plot import spec_plot
from ADC_Toolbox_Python.specPlotPhase import spec_plot_phase
from ADC_Toolbox_Python.sineFit import sine_fit
from ADC_Toolbox_Python.findBin import find_bin
from ADC_Toolbox_Python.tomDecomp import tomDecomp
from ADC_Toolbox_Python.errHistSine import errHistPhase, errHistCode

ALL_TOOLS = ['specPlot', 'specPlotPhase', 'tomDecomp', 'errHistSine']

FILE_LIST = [
    "Sine_wave_10_70_bit_nonlinearity.csv", "Sine_wave_13_69_bit.csv",
    "gain_error_0P95.csv", "gain_error_0P99.csv", "gain_error_1P01.csv", "gain_error_1P05.csv",
    "clip_0P06.csv", "clip_0P07.csv",
    "jitter_0P001.csv", "jitter_0P002.csv", "jitter_0P0002.csv",
    "kickback_0P07.csv", "kickback_0P007.csv",
]


def run_tests(tools=None):
    """Run tests for selected tools."""
    tools = tools or ALL_TOOLS
    data_dir = os.path.join(tests_dir, "reference_data")
    output_dir = os.path.join(tests_dir, "output")

    print("=" * 70)
    print(f"ADC Tools Test Runner\nTools: {', '.join(tools)}\nCases: {len(FILE_LIST)}")
    print("=" * 70)

    results = []

    for idx, filename in enumerate(FILE_LIST, 1):
        case_name = filename.replace(".csv", "")
        filepath = os.path.join(data_dir, filename)
        case_dir = os.path.join(output_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n[{idx}/{len(FILE_LIST)}] {case_name}")

        if not os.path.exists(filepath):
            print("  [ERROR] File not found")
            results.append(False)
            continue

        try:
            data = np.loadtxt(filepath, delimiter=',').flatten()
            N = len(data)
            _, freq_est, _, _, _ = sine_fit(data)
            fin = find_bin(1, freq_est, N) / N

            if 'specPlot' in tools:
                ENoB, SNDR, *_ = spec_plot(data, Fs=1.0, harmonic=0, label=1, OSR=1)
                plt.savefig(os.path.join(case_dir, "specPlot.png"), dpi=150)
                plt.close()
                print(f"  specPlot: ENoB={ENoB:.2f}, SNDR={SNDR:.2f}dB")

            if 'specPlotPhase' in tools:
                result = spec_plot_phase(data, harmonic=50,
                    save_path=os.path.join(case_dir, "specPlotPhase.png"))
                print(f"  specPlotPhase: freq_bin={result['freq_bin']}")

            if 'tomDecomp' in tools:
                signal, error, indep, dep, phi = tomDecomp(data, fin, 50, 0)
                xlim = min(max(int(1.5 / fin), 100), N)

                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(data[:xlim], 'kx', markersize=3, alpha=0.5, label='data')
                ax1.plot(signal[:xlim], '-', color='gray', linewidth=1.5, label='signal')
                ax1.set_xlim([0, xlim])
                ax1.set_ylabel('Signal')

                ax2 = ax1.twinx()
                ax2.plot(dep[:xlim], 'r-', label='dep', linewidth=1.5)
                ax2.plot(indep[:xlim], 'b-', label='indep', linewidth=1)
                ax2.set_ylabel('Error')

                ax1.legend(loc='upper left')
                ax2.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(case_dir, "tomDecomp.png"), dpi=150)
                plt.close()
                print(f"  tomDecomp: indep_rms={np.sqrt(np.mean(indep**2)):.6f}")

            if 'errHistSine' in tools:
                scaled = data * (2**12)
                errHistCode(scaled, bin_count=1000, fin=fin,
                    save_path=os.path.join(case_dir, "errHistSine_code.png"))
                errHistPhase(scaled, bin_count=1000, fin=fin,
                    save_path=os.path.join(case_dir, "errHistSine_phase.png"))
                print("  errHistSine: saved code and phase plots")

            results.append(True)

        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print(f"Results: {sum(results)}/{len(results)} OK")
    print("=" * 70)

    return all(results)


if __name__ == "__main__":
    # ========== SELECT TOOLS TO TEST ==========
    tools = [
        'specPlot',
        'specPlotPhase',
        'tomDecomp',
        'errHistSine',
    ]
    # tools = None  # uncomment to run all tools
    # ==========================================

    sys.exit(0 if run_tests(tools) else 1)
