# Python vs MATLAB Parity Tracker

Last updated: 2026-03-10

This document tracks feature parity between the Python (`adctoolbox` v0.5.0) and MATLAB (`ADCToolbox` v1.30) implementations.

---

## Function Mapping

### Shared (present in both)

| Category | MATLAB | Python | Notes |
|---|---|---|---|
| Spectrum | `plotspec` | `analyze_spectrum` | Python returns dict; MATLAB returns individual values |
| Polar Spectrum | `plotphase` | `analyze_spectrum_polar` | |
| Perf vs OSR | `perfosr` | *(in-example only)* | Python has no dedicated function; done in `exp_s06` example script |
| Sine Fit | `sinfit` | `fit_sine_4param` | |
| Find Coherent Bin | `findbin` | `find_coherent_frequency` | Python returns frequency, MATLAB returns bin index |
| Estimate Frequency | `findfreq` | `estimate_frequency` | |
| Alias Folding | `alias` | `fold_frequency_to_nyquist` | Python also has `fold_bin_to_nyquist` |
| Error by Phase/Value | `errsin` | `analyze_error_by_phase`, `analyze_error_by_value` | MATLAB uses `xaxis` param to switch; Python split into two functions |
| Error by Value (shortcut) | `errsinv` | `analyze_error_by_value` | Python absorbed the shortcut |
| Time Decomposition | `tomdec` | `analyze_decomposition_time`, `analyze_decomposition_polar` | Python split into time and polar variants |
| INL from Sine | `inlsin` | `analyze_inl_from_sine` | |
| Overflow Check | `bitchk` | `analyze_overflow` | |
| Weight Calibration | `wcalsin` | `calibrate_weight_sine` | |
| Weight/Radix Plot | `plotwgt` | `analyze_weight_radix` | |
| NTF Analysis | `ntfperf` | `ntf_analyzer` | |
| Dashboard | `adcpanel` | `generate_aout_dashboard`, `generate_dout_dashboard` | Python split analog/digital into separate dashboards |

### MATLAB-only (not yet ported to Python)

| MATLAB Function | Description | Priority |
|---|---|---|
| `cdacwgt` | Capacitive DAC weight calculation from capacitor values (cd, cb, cp) | Low — niche, design-phase utility |
| `ifilter` | FFT-based ideal (brickwall) bandpass filter | Medium — useful general utility |
| `plotres` | Partial-sum residual plots for bit-matrix correlation analysis | Medium — useful debug visualization |
| `plotressin` | Shortcut: `wcalsin` + `plotres` combined | Low — follows once `plotres` is ported |
| `perfosr` (dedicated fn) | Sweep and plot SNDR/SFDR/ENOB vs OSR | Medium — exists as example script but not a reusable function |

### Python-only (not in MATLAB)

| Python Function | Description | Notes |
|---|---|---|
| `analyze_two_tone_spectrum` | Two-tone spectrum with IMD analysis | New feature in Python |
| `analyze_error_pdf` | Error probability density function | New analysis |
| `analyze_error_spectrum` | FFT of error signal | New analysis |
| `analyze_error_autocorr` | Error autocorrelation | New analysis |
| `analyze_error_envelope_spectrum` | Envelope spectrum of error | New analysis |
| `fit_static_nonlin` | Fit static nonlinearity polynomial | New analysis |
| `analyze_bit_activity` | Bit toggle activity analysis | MATLAB has legacy `bitact.m` only |
| `analyze_enob_sweep` | ENOB per bit sweep | MATLAB has legacy `bitsweep.m` only |
| `calibrate_weight_sine_lite` | Lightweight calibration variant | Internal (not exported), no MATLAB equivalent |
| `analyze_phase_plane` | Phase-space anomaly detection | Internal (not exported), no MATLAB equivalent |
| `siggen.nonidealities` | Signal generation with jitter, thermal noise, quantization, settling, nonlinearity | Entire module, no MATLAB equivalent |
| Unit conversions (16 functions) | `db_to_mag`, `snr_to_enob`, `lsb_to_volts`, `dbm_to_vrms`, etc. | Utility library, no MATLAB equivalent |
| FOM calculators (4 functions) | `calculate_walden_fom`, `calculate_schreier_fom`, `calculate_thermal_noise_limit`, `calculate_jitter_limit` | Utility library, no MATLAB equivalent |
| `vpp_for_target_dbfs` | Calculate Vpp for target dBFS | Internal utility, no MATLAB equivalent |

---

## Structural / Behavioral Differences

| Aspect | MATLAB | Python |
|---|---|---|
| **Naming** | camelCase (`plotspec`, `sinfit`, `wcalsin`) | snake_case (`analyze_spectrum`, `fit_sine_4param`, `calibrate_weight_sine`) |
| **Return type** | Multiple return values (`[enob,sndr,sfdr,...] = plotspec(...)`) | Single dictionary (`result = analyze_spectrum(...)`) |
| **Plot control** | `disp` parameter (0/1) | `create_plot` boolean; separate `plot_*` functions |
| **API granularity** | One function does multiple things (e.g., `errsin` handles both phase and value binning) | Split into focused functions (`analyze_error_by_phase` vs `analyze_error_by_value`) |
| **Dashboard** | Single `adcpanel` handles both analog and digital | Separate `generate_aout_dashboard` and `generate_dout_dashboard` |
| **Legacy functions** | 15 deprecated functions in `legacy/` | No legacy layer; MATLAB names were never shipped |
| **Shortcut wrappers** | `shortcut/` directory with convenience wrappers | Absorbed into main API (no separate layer) |

---

## Recommendations for Convergence

1. **Port to Python**: `ifilter`, `plotres`, `perfosr` (as dedicated functions)
2. **Port to MATLAB**: `analyze_two_tone_spectrum`, error analysis suite (PDF, spectrum, autocorr, envelope), `fit_static_nonlin`, signal generation module, unit conversion utilities, FOM calculators
3. **Promote in Python**: `analyze_phase_plane` and `calibrate_weight_sine_lite` are implemented but not exported — decide whether to make them public
4. **Decide on `cdacwgt`**: Very MATLAB-specific (CDAC design). May not be needed in Python if the toolbox focuses on measurement/characterization rather than circuit design
