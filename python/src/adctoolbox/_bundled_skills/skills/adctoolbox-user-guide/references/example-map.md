# ADCToolbox Example Map

Use this file when you want a real script to adapt. The paths below match the
current packaged examples under `python/src/adctoolbox/examples/`.

## Start Here

- Environment / smoke test:
  `01_basic/exp_b01_environment_check.py`
- Coherent vs non-coherent sampling:
  `01_basic/exp_b02_coherent_vs_non_coherent.py`
- Simplest FFT metrics:
  `02_spectrum/exp_s01_analyze_spectrum_simplest.py`
- Interactive spectrum exploration:
  `02_spectrum/exp_s02_analyze_spectrum_interactive.py`
- Save or reuse figures:
  `02_spectrum/exp_s03_analyze_spectrum_savefig.py`

## Example Families

- Spectrum analysis:
  start in `02_spectrum/`
- Signal generation and impairment injection:
  start in `03_generate_signals/`
- Analog debug:
  start in `04_debug_analog/`
- Digital calibration and DOUT debug:
  start in `05_debug_digital/`
- Toolset dashboards:
  start in `06_use_toolsets/`
- Metric and frequency conversions:
  start in `07_conversions/`

## High-Value Files

- Signal generation:
  `03_generate_signals/exp_g01_generate_signal_demo.py`
- Sine fitting:
  `04_debug_analog/exp_a01_fit_sine_4param.py`
- Error by phase:
  `04_debug_analog/exp_a03_analyze_error_by_phase.py`
- Full weight calibration:
  `05_debug_digital/exp_d02_cal_weight_sine.py`
- AOUT dashboard:
  `06_use_toolsets/exp_t01_aout_dashboard_single.py`
- DOUT dashboard:
  `06_use_toolsets/exp_t03_dout_dashboard_single.py`

## Practical Notes

- The packaged examples are what `adctoolbox-get-examples` copies into a user workspace.
- Some older docs refer to outdated filenames. Prefer the paths in this file.
- Keep the example frequency conventions:
  `fs` and `Fin` are in Hz for spectrum and signal generation.
- Calibration examples usually use normalized `freq=Fin/Fs`.
- The toolset examples are the fastest way to see how multiple helpers are
  composed together.
