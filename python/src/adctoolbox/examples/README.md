# ADCToolbox Examples

This directory contains 64 runnable ADCToolbox examples. They are copied to a
user workspace by:

```bash
adctoolbox-get-examples
```

Most examples save figures into a local `output/` directory and avoid blocking
on GUI display.

## Quick Start

```bash
cd adctoolbox_examples
python 01_basic/exp_b01_environment_check.py
python 02_spectrum/exp_s01_analyze_spectrum_simplest.py
python 05_debug_digital/exp_d02_cal_weight_sine.py
```

## Example Map

### `01_basic/` - fundamentals

| File | Topic |
|---|---|
| `exp_b01_environment_check.py` | Verify the install and save a test figure |
| `exp_b02_coherent_vs_non_coherent.py` | Coherent versus non-coherent sampling behavior |

### `02_spectrum/` - FFT and spectrum analysis

| File | Topic |
|---|---|
| `exp_s00_fft_fundamentals.py` | FFT-bin fundamentals |
| `exp_s01_analyze_spectrum_simplest.py` | Minimal single-tone spectrum analysis |
| `exp_s02_analyze_spectrum_interactive.py` | Interactive spectrum plotting |
| `exp_s03_analyze_spectrum_savefig.py` | Save spectrum figures |
| `exp_s04_sweep_dynamic_range.py` | Dynamic-range sweep |
| `exp_s05_annotating_spur.py` | Spur annotation |
| `exp_s06_sweeping_fft_and_osr.py` | FFT length and OSR sweep |
| `exp_s07_spectrum_averaging.py` | Spectrum averaging |
| `exp_s08_windowing_deep_dive.py` | Windowing comparison |
| `exp_s09_sar_fft_length_near_nyquist.py` | SAR spectrum near Nyquist |
| `exp_s10_cartesian_and_polar_plot.py` | Cartesian and polar spectrum views |
| `exp_s11_polar_memory_effect.py` | Memory-effect visualization in polar spectrum |
| `exp_s12_polar_coherent_averaging.py` | Coherent averaging with polar plots |
| `exp_s13_fft_length_mc_sfdr_sndr.py` | FFT-length Monte Carlo sweep for SFDR/SNDR |

### `03_generate_signals/` - synthetic ADC stimulus

| File | Topic |
|---|---|
| `exp_g01_generate_signal_demo.py` | Basic generated signal with thermal noise |
| `exp_g03_sweep_quant_bits.py` | Quantization-noise sweep |
| `exp_g04_sweep_jitter_fin.py` | Jitter versus input frequency |
| `exp_g05_sweep_static_nonlin.py` | Static HD2/HD3 nonlinearity cases |
| `exp_g06_sweep_dynamic_nonlin.py` | Dynamic nonlinear effects |
| `exp_g07_sweep_interferences.py` | Interference and clipping cases |

### `04_debug_analog/` - analog-output diagnostics

| File | Topic |
|---|---|
| `exp_a01_fit_sine_4param.py` | Four-parameter sine fitting |
| `exp_a02_analyze_error_by_value.py` | Error versus value/code |
| `exp_a03_analyze_error_by_phase.py` | Error versus sine phase |
| `exp_a04_jitter_calculation.py` | Jitter calculation |
| `exp_a11_decompose_harmonics.py` | Time-domain harmonic decomposition |
| `exp_a12_decompose_harmonics_polar.py` | Polar harmonic decomposition |
| `exp_a21_analyze_error_pdf.py` | Error PDF comparison |
| `exp_a22_analyze_error_spectrum.py` | Error spectrum |
| `exp_a23_analyze_error_autocorrelation.py` | Error autocorrelation |
| `exp_a24_analyze_error_envelope_spectrum.py` | Error-envelope spectrum |
| `exp_a25_spectra.py` | Multi-case spectrum comparison |
| `exp_a31_fit_static_nonlin.py` | Static nonlinearity fitting |
| `exp_a32_inl_from_sine_sweep_length.py` | INL/DNL versus record length |
| `exp_a33_inl_from_ramp.py` | Ramp-histogram INL/DNL from output codes |
| `exp_a41_analyze_phase_plane.py` | Phase-plane analysis |
| `exp_a42_analyze_error_phase_plane.py` | Error phase-plane analysis |

### `05_debug_digital/` - digital-code diagnostics and calibration

| File | Topic |
|---|---|
| `exp_d01_cal_weight_sine_lite.py` | Lightweight sine-weight calibration |
| `exp_d02_cal_weight_sine.py` | Full sine-weight calibration |
| `exp_d03_redundancy_comparison.py` | Pipeline redundancy comparison |
| `exp_d11_bit_activity.py` | Bit activity |
| `exp_d12_sweep_bit_enob.py` | ENOB versus active bit count |
| `exp_d13_weight_scaling.py` | Weight scaling |
| `exp_d14_overflow_check.py` | Overflow checking |
| `exp_d15_sar_unit_cap_mismatch_uncal_spectra.py` | SAR unit-cap mismatch without calibration |
| `exp_d16_sar_unit_cap_mismatch_mc.py` | SAR unit-cap mismatch Monte Carlo calibration |
| `exp_d17_sar_msb_error_binary_vs_repeat_calibration.py` | MSB error: binary versus redundant SAR |
| `exp_d18_sar_redundant_mismatch_training_length_sweep.py` | SAR calibration versus training length |

### `06_use_toolsets/` - dashboard workflows

| File | Topic |
|---|---|
| `exp_t01_aout_dashboard_single.py` | Single AOUT dashboard |
| `exp_t02_aout_dashboard_batch.py` | Batch AOUT dashboard |
| `exp_t03_dout_dashboard_single.py` | Single DOUT dashboard |
| `exp_t04_dout_dashboard_batch.py` | Batch DOUT dashboard |

### `07_conversions/` - conversions and metrics

| File | Topic |
|---|---|
| `exp_c01_aliasing_nyquist_zones.py` | Aliasing and Nyquist zones |
| `exp_c02_unit_conversions.py` | Unit conversions |
| `exp_c03_calculate_fom.py` | ADC FoM calculations |
| `exp_c04_amplitudes_to_snr.py` | Amplitude-to-SNR conversion |
| `exp_c05_convert_nsd_snr.py` | NSD/SNR conversion |

### `08_time_interleave/` - TI-ADC analysis

| File | Topic |
|---|---|
| `exp_ti01_compare_skew_methods.py` | Timing-skew extraction methods |
| `exp_ti02_autocorr_background_skew_calibration.py` | Autocorrelation-based background skew calibration |

### `09_downsample/` - subsample debug output

| File | Topic |
|---|---|
| `exp_d00_subsample_aliasing.py` | Subsample-only debug output and aliasing behavior |

### `10_oversampling/` - oversampling and noise shaping

| File | Topic |
|---|---|
| `exp_o01_noise_shaping_spectrum.py` | Noise-shaped quantization spectra |
| `exp_o02_ifilter_band_analysis.py` | In-band extraction with MATLAB-compatible `ifilter` |
| `exp_o03_ntfperf_perfosr.py` | NTF theory and performance-vs-OSR sweep |
