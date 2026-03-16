# ADCToolbox API Quick Reference

> All functions are flat-exported: `from adctoolbox import <function_name>`
> The signal generator class uses a submodule import: `from adctoolbox.siggen import ADC_Signal_Generator`

---

## Spectrum Analysis

### `analyze_spectrum`
```python
analyze_spectrum(data, fs=1.0, osr=1, max_scale_range=None, win_type='hann',
                 side_bin=None, max_harmonic=5, nf_method=2,
                 assumed_sig_pwr_dbfs=np.nan, coherent_averaging=False,
                 create_plot=True, show_title=True, show_label=True,
                 plot_harmonics_up_to=3, ax=None)
```
**Returns dict:** `enob`, `sndr_dbc`, `sfdr_dbc`, `snr_dbc`, `thd_dbc`, `sig_pwr_dbfs`, `noise_floor_dbfs`, `nsd_dbfs_hz`
```python
result = analyze_spectrum(data, fs=800e6, create_plot=False)
print(f"ENOB={result['enob']:.2f}, SNDR={result['sndr_dbc']:.1f} dBc")
```

### `analyze_spectrum_polar`
```python
analyze_spectrum_polar(data, max_code=None, harmonic=5, osr=1,
                       cutoff_freq=0, fs=1.0, win_type='boxcar',
                       create_plot=True, ax=None, fixed_radial_range=None)
```
**Returns dict:** Same structure as `compute_spectrum()` results.
```python
result = analyze_spectrum_polar(data, fs=800e6, harmonic=5)
```

---

## Frequency Utilities

### `find_coherent_frequency`
```python
find_coherent_frequency(fs, fin_target, n_fft, force_odd=True, search_radius=200)
```
**Returns tuple:** `(fin_actual, best_bin)` — coherent frequency (Hz) and FFT bin index.
```python
fin, bin_idx = find_coherent_frequency(800e6, 100e6, 8192)
```

### `estimate_frequency`
```python
estimate_frequency(data, frequency_estimate=None, fs=1.0)
```
**Returns float:** Estimated frequency in Hz.
```python
freq_hz = estimate_frequency(signal, fs=800e6)
```

### `fold_frequency_to_nyquist`
```python
fold_frequency_to_nyquist(fin, fs)
```
**Returns float:** Aliased frequency in [0, Fs/2].
```python
f_alias = fold_frequency_to_nyquist(900e6, 800e6)  # → 100 MHz
```

### `fold_bin_to_nyquist`
```python
fold_bin_to_nyquist(bin_idx, n_fft)
```
**Returns float:** Aliased bin index in [0, n_fft/2].
```python
aliased_bin = fold_bin_to_nyquist(5000, 8192)  # → 3192.0
```

### `bin_to_freq`
```python
bin_to_freq(bin_idx, fs, n_fft)
```
**Returns float:** Frequency in Hz.

### `freq_to_bin`
```python
freq_to_bin(freq, fs, n_fft)
```
**Returns int:** Nearest FFT bin index.

---

## Sine Fitting

### `fit_sine_4param`
```python
fit_sine_4param(data, frequency_estimate=None, max_iterations=1, tolerance=1e-9)
```
**Returns dict:** `fitted_signal`, `residuals`, `frequency` (normalized 0–0.5), `amplitude`, `phase` (rad), `dc_offset`, `rmse`
```python
result = fit_sine_4param(data)
print(f"Freq={result['frequency']:.6f}, Amp={result['amplitude']:.4f}")
```

---

## Unit Conversions

| Function | Signature | Description |
|----------|-----------|-------------|
| `db_to_mag` | `(db)` | dB → magnitude: 10^(x/20) |
| `mag_to_db` | `(mag)` | magnitude → dB: 20·log10(x) |
| `db_to_power` | `(db)` | dB → power ratio: 10^(x/10) |
| `power_to_db` | `(power)` | power → dB: 10·log10(x) |
| `snr_to_enob` | `(snr_db)` | SNR → ENOB: (SNR − 1.76) / 6.02 |
| `enob_to_snr` | `(enob)` | ENOB → SNR: ENOB × 6.02 + 1.76 |
| `snr_to_nsd` | `(snr_db, fs, signal_pwr_dbfs=0, osr=1)` | SNR → NSD (dBFS/Hz) |
| `nsd_to_snr` | `(nsd_dbfs_hz, fs, signal_pwr_dbfs=0, osr=1)` | NSD → SNR (dB) |
| `lsb_to_volts` | `(lsb_count, vref, n_bits)` | LSB count → voltage |
| `volts_to_lsb` | `(volts, vref, n_bits)` | voltage → LSB count |
| `bin_to_freq` | `(bin_idx, fs, n_fft)` | FFT bin → frequency (Hz) |
| `freq_to_bin` | `(freq, fs, n_fft)` | frequency → FFT bin index |
| `dbm_to_vrms` | `(dbm, z_load=50)` | dBm → Vrms |
| `vrms_to_dbm` | `(vrms, z_load=50)` | Vrms → dBm |
| `dbm_to_mw` | `(dbm)` | dBm → milliwatts |
| `mw_to_dbm` | `(mw)` | milliwatts → dBm |
| `sine_amplitude_to_power` | `(amplitude, z_load=50)` | peak amplitude → power (W) |
| `amplitudes_to_snr` | `(signal_amp, noise_amp)` | signal/noise amplitudes → SNR (dB) |

---

## FOM / Theoretical Limits

### `calculate_walden_fom`
```python
calculate_walden_fom(power, fs, enob)
```
**Returns float:** FoM_w in J/conv-step. Lower is better.
```python
fom_w = calculate_walden_fom(power=1e-3, fs=100e6, enob=10.5)
```

### `calculate_schreier_fom`
```python
calculate_schreier_fom(power, sndr_db, bw)
```
**Returns float:** FoM_s in dB. Higher is better.
```python
fom_s = calculate_schreier_fom(power=1e-3, sndr_db=65, bw=50e6)
```

### `calculate_thermal_noise_limit`
```python
calculate_thermal_noise_limit(cap_pf, v_fs=1.0)
```
**Returns float:** Max SNR (dB) limited by kT/C noise.
```python
snr_max = calculate_thermal_noise_limit(cap_pf=2.0, v_fs=1.0)
```

### `calculate_jitter_limit`
```python
calculate_jitter_limit(freq, jitter_rms_sec)
```
**Returns float:** Max SNR (dB) limited by aperture jitter.
```python
snr_max = calculate_jitter_limit(freq=100e6, jitter_rms_sec=0.5e-12)
```

---

## Analog Output (AOUT) Error Analysis

### `analyze_inl_from_sine`
```python
analyze_inl_from_sine(data, num_bits=None, full_scale=None, clip_percent=0.01,
                      create_plot=True, show_title=True, col_title=None, ax=None)
```
**Returns dict:** `inl`, `dnl`, `code`

### `analyze_decomposition_time`
```python
analyze_decomposition_time(signal, harmonic=5, n_cycles=5.0,
                           create_plot=True, ax=None, title=None)
```
**Returns dict:** `magnitudes`, `phases`, `magnitudes_db`, `residual_rms`, `noise_db`, `fundamental_freq`, `noise_residual`, `reconstructed_signal`, `fundamental_signal`, `harmonic_signal`

### `analyze_decomposition_polar`
```python
analyze_decomposition_polar(signal, harmonic=5, create_plot=True, ax=None, title=None)
```
**Returns dict:** Same keys as `analyze_decomposition_time`.

### `analyze_error_by_value`
```python
analyze_error_by_value(signal, norm_freq=None, n_bins=100, clip_percent=0.01,
                       value_range=None, create_plot=True, axes=None, ax=None, title=None)
```
**Returns dict:** `error_mean`, `error_rms`, `bin_centers`, `bin_indices`, `error`, plus sine fit keys.

### `analyze_error_by_phase`
```python
analyze_error_by_phase(signal, norm_freq=None, n_bins=100, include_base_noise=True,
                       create_plot=True, axes=None, ax=None, title=None)
```
**Returns dict:** `am_noise_rms_v`, `pm_noise_rms_v`, `pm_noise_rms_rad`, `base_noise_rms_v`, `total_rms_v`, `r_squared_raw`, `r_squared_binned`, `bin_error_rms_v`, `bin_error_mean_v`, `phase_bin_centers_rad`, `amplitude`, `dc_offset`, `norm_freq`, `fitted_signal`, `error`, `phase`

### `analyze_error_pdf`
```python
analyze_error_pdf(signal, resolution=12, full_scale=None, frequency=None,
                  create_plot=True, ax=None, title=None)
```
**Returns dict:** `err_lsb`, `mu`, `sigma`, `kl_divergence`, `x`, `pdf`, `gauss_pdf`

### `analyze_error_spectrum`
```python
analyze_error_spectrum(signal, fs=1, frequency=None, create_plot=True, ax=None, title=None)
```
**Returns dict:** `enob`, `sndr_db`, `sfdr_db`, `snr_db`, `thd_db`, `sig_pwr_dbfs`, `noise_floor_dbfs`, `error_signal`

### `analyze_error_autocorr`
```python
analyze_error_autocorr(signal, frequency=None, max_lag=50, normalize=True,
                       create_plot=True, ax=None, title=None)
```
**Returns dict:** `acf`, `lags`, `error_signal`

### `analyze_error_envelope_spectrum`
```python
analyze_error_envelope_spectrum(signal, fs=1, frequency=None, create_plot=True,
                                ax=None, title=None)
```
**Returns dict:** `enob`, `sndr_db`, `sfdr_db`, `snr_db`, `thd_db`, `sig_pwr_dbfs`, `noise_floor_dbfs`, `error_signal`, `envelope`

### `fit_static_nonlin`
```python
fit_static_nonlin(sig_distorted, order)
```
**Returns tuple:** `(k2_extracted, k3_extracted, fitted_sine, fitted_transfer)` — Note: returns a tuple, not a dict.

---

## Digital Output (DOUT) Analysis

### `analyze_bit_activity`
```python
analyze_bit_activity(bits, create_plot=True, ax=None, title=None)
```
**Returns ndarray:** Percentage of 1's for each bit column.
```python
activity = analyze_bit_activity(bit_matrix, create_plot=False)
```

### `analyze_overflow`
```python
analyze_overflow(raw_code, weight, ofb=None, create_plot=True, ax=None, title=None)
```
**Returns tuple:** `(range_min, range_max, ovf_percent_zero, ovf_percent_one)` — per-bit residue ranges and overflow percentages.

### `analyze_weight_radix`
```python
analyze_weight_radix(weights, create_plot=True, ax=None, title=None)
```
**Returns ndarray:** Radix between consecutive bits (weight[i-1]/weight[i]).

### `analyze_enob_sweep`
```python
analyze_enob_sweep(bits, freq=None, harmonic_order=1, osr=1, win_type='hamming',
                   create_plot=True, ax=None, title=None, verbose=False)
```
**Returns tuple:** `(enob_sweep, n_bits_vec)` — ENOB array and corresponding bit counts.

---

## Calibration

### `calibrate_weight_sine`
```python
calibrate_weight_sine(bits, freq=None, force_search=False,
                      nominal_weights=None, harmonic_order=1,
                      learning_rate=0.5, reltol=1e-12,
                      max_iter=100, verbose=0)
```
**Returns dict:** `weight`, `offset`, `calibrated_signal`, `ideal`, `error`, `refined_frequency`

**Important:** `freq` is **normalized** as Fin/Fs (not Hz). Pass `None` for auto-detection.
```python
result = calibrate_weight_sine(bit_matrix, freq=fin/fs)
calibrated = result['calibrated_signal']
weights = result['weight']
```

---

## Signal Generation (`from adctoolbox.siggen import ADC_Signal_Generator`)

### `ADC_Signal_Generator(N, Fs, Fin, A, DC)`
Constructor parameters: sample count, sampling freq (Hz), input freq (Hz), amplitude, DC offset.

| Method | Key Parameters | Description |
|--------|----------------|-------------|
| `get_clean_signal()` | — | Clean sine: A·sin(2π·Fin·t) + DC |
| `apply_thermal_noise()` | `input_signal=None, noise_rms=50e-6` | Additive white noise |
| `apply_quantization_noise()` | `input_signal=None, n_bits=10, quant_range=(0.0, 1.0)` | ADC quantization |
| `apply_jitter()` | `input_signal=None, jitter_rms=2e-12` | Sampling jitter |
| `apply_static_nonlinearity()` | `input_signal=None, k2=0, k3=0, k4=0, k5=0` | Polynomial distortion |
| `apply_static_nonlinearity_hd()` | `input_signal=None, hd2_dB=None, hd3_dB=None, hd4_dB=None, hd5_dB=None` | HD-specified distortion |
| `apply_memory_effect()` | `input_signal=None, memory_strength=0.009` | MSB charge injection |
| `apply_incomplete_sampling()` | `input_signal=None, T_track=None, tau_nom=40e-12, coeff_k=0.15` | Settling error |
| `apply_ra_gain_error()` | `input_signal=None, relative_gain=0.99, msb_bits=4, lsb_bits=12` | Static interstage gain |
| `apply_ra_gain_error_dynamic()` | `input_signal=None, relative_gain=1, coeff_3=0.15, msb_bits=4, lsb_bits=12` | Dynamic gain error |
| `apply_reference_error()` | `input_signal=None, settling_tau=2.0, droop_strength=0.01` | Vref settling |
| `apply_am_noise()` | `input_signal=None, strength=0.01` | Multiplicative noise |
| `apply_am_tone()` | `input_signal=None, am_tone_freq=500e3, am_tone_depth=0.05` | AM spur |
| `apply_clipping()` | `input_signal=None, percentile_clip=1.0` | Hard clipping |
| `apply_drift()` | `input_signal=None, drift_scale=5e-5` | Low-freq random walk |
| `apply_glitch()` | `input_signal=None, glitch_prob=0.00015, glitch_amplitude=0.1` | Random glitches |
| `apply_noise_shaping()` | `input_signal=None, n_bits=10, quant_range=(0.0, 1.0), order=1` | Delta-sigma NTF (order 1–5) |

**Chaining pattern:** Each `apply_*` method accepts `input_signal` — pass `None` to start from clean signal, or pass the output of a previous method to chain effects.
```python
gen = ADC_Signal_Generator(N=8192, Fs=800e6, Fin=fin, A=0.49, DC=0.5)
sig = gen.apply_thermal_noise(noise_rms=100e-6)
sig = gen.apply_jitter(input_signal=sig, jitter_rms=1e-12)
sig = gen.apply_quantization_noise(input_signal=sig, n_bits=12)
```

---

## Oversampling

### `ntf_analyzer`
```python
ntf_analyzer(ntf, flow, fhigh, is_plot=None)
```
**Returns float:** Integrated noise suppression in dB within the signal band [flow, fhigh].
- `ntf`: Noise Transfer Function (z-domain transfer function)
- `flow`, `fhigh`: Normalized frequency bounds (0–0.5)
