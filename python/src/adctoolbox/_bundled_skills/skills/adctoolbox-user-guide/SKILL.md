---
name: adctoolbox-user-guide
description: >
  Router skill for using ADCToolbox from Python. Trigger when a task
  involves: computing or plotting spectra (SNDR, SFDR, ENOB, THD) from
  ADC output, fitting a sine to measured aout, calibrating SAR weights
  (weight_sine / weight_sine_lite), generating synthetic ADC
  stimulus/output, or validating aout/dout buffer shapes. For deeper
  debug (dashboards, phase-plane, bit-level, error decomposition,
  static nonlinearity, cap-to-weight), open
  references/advanced-debug.md.
  NOT for analog topology selection, transistor sizing, Spectre
  simulation, or layout/parasitic review — those belong to the
  analog-agents skills (analog-design, analog-verify, analog-audit).
  NOT for editing ADCToolbox source code — use
  adctoolbox-contributor-guide instead.
---

# ADCToolbox Usage Guide

Router, not a full manual. Keep the basic tier resident; open
`references/*.md` only when you need more.

## 1. When to use (and not to use)

Use for:
- Writing, fixing, or reviewing Python that calls ADCToolbox APIs
- Picking the right spectrum / calibration helper
- Getting from a raw `dout` / `aout` buffer to SNDR / SFDR / ENOB
- Generating synthetic ADC stimulus for a testbench
- **Forward-modeling an ADC architecture in Python** — for SAR, use
  `adctoolbox.models.sar_convert` / `sar_reconstruct` / `sar_ideal_weights`
  / `sar_apply_mismatch` (binary or sub-radix-2, with optional cap mismatch
  + sampling noise + comparator noise; vectorized). Convention: `vin` is
  interpreted relative to `quant_range=(v_min, v_max)`, default `(0, 1)`.
  SAR weights are still explicit and normalized by `sum(bit_weights) + 1 LSB`
  (for example `[8, 4, 2, 1] / 16`, or redundant `[8, 4, 4, 2, 1] / 20`).
  For differential SAR, pass `VIP - VIN` with a differential `quant_range`
  such as `(-VDD, VDD)`. Keep analog CDAC weights and digital reconstruction
  weights explicit; they match unless modeling mismatch or calibration.

Do NOT use for:
- Analog topology / transistor design → `analog-design`, `analog-explore`
- Spectre simulation, pre/post-layout audit → `analog-verify`, `analog-audit`
- Editing ADCToolbox's own source → `adctoolbox-contributor-guide`

## 2. Critical conventions (read first — these are the common bug sources)

### Names

- **`bits`** is the per-sample binary decision matrix shape `(N_samples, N_bits)`,
  values in `{0, 1}` — used by every digital-calibration / bit-level helper.
- **`aout`** is the analog output (1D `float` array) — used by spectrum and
  error-analysis helpers.
- ADC integer codes are *not* `bits`; convert separately if your data is
  packed as integers.

### Frequency units

- `fs`, `Fin`, plotting frequencies: **Hz**.
- `fit_sine_4param(...)["frequency"]`: **normalized** `Fin/Fs` (range 0–0.5),
  **not** Hz.
- `calibrate_weight_sine`, `calibrate_weight_sine_lite`, `analyze_enob_sweep`,
  `generate_dout_dashboard`: `freq` parameter is **normalized** `Fin/Fs`.
- `generate_aout_dashboard`: `freq` is in **Hz** (it normalizes internally).
- `analyze_spectrum` does NOT take `Fin` — it auto-detects the fundamental
  from the FFT.

### Return shapes

Most analysis functions return `dict`. Notable exceptions and dict-key gotchas:

| Function | Return |
|---|---|
| `analyze_spectrum`, `analyze_spectrum_polar`, `analyze_spectrum_virtuoso` | `dict` — keys: `enob`, `sndr_dbc`, `sfdr_dbc`, `snr_dbc`, `thd_dbc`, `sig_pwr_dbfs`, `noise_floor_dbfs`, `nsd_dbfs_hz`, `harmonics_dbc` |
| `quick_sndr` | `dict` — minimal: only `sndr_dbc`, `enob`. No SFDR/THD/HD/NSD breakdown. |
| `compute_spectrum` | `dict` — top-level keys `metrics` (same as above) and `plot_data` (`freq`, `power_spectrum_db_plot`, `complex_spectrum`, `fundamental_bin`, …) |
| `fit_sine_4param` | `dict` — `frequency` (normalized), `amplitude`, `phase`, `dc_offset`, `rmse`, `fitted_signal`, `residuals` |
| `find_coherent_frequency` | `tuple (fin_actual_hz, best_bin)` |
| `calibrate_weight_sine` | `dict` — `weight`, `offset`, `calibrated_signal`, `ideal`, `error`, `refined_frequency` |
| `calibrate_weight_sine_lite` | `ndarray` (weights only) |
| `analyze_bit_activity` | `ndarray` (% of 1's per bit, length = N_bits) |
| `analyze_overflow` | `tuple` of 4 ndarrays `(range_min, range_max, ovf_pct_zero, ovf_pct_one)` |
| `analyze_enob_sweep` | `tuple (enob_sweep, n_bits_vec)` |
| `analyze_weight_radix` | `dict` — `radix`, `wgtsca`, `effres` (weight-list resolution estimate) |
| `fit_static_nonlin` | `tuple (k2, k3, fitted_sine, fitted_transfer)` |
| `convert_cap_to_weight` | `tuple (weights, c_total)` |

`analyze_weight_radix(weights)["effres"]` is computed from significant
absolute weights as `log2(sum(abs_w_sig) / min(abs_w_sig) + 1)`. It estimates
the theoretical span of the supplied SAR/DAC weight list; it is not a
missing-code, DNL/INL, or SAR-reachability check.

When docs conflict, trust the current `__init__.py` exports + the
`tests/integration/test_user_guide_skill_examples.py` smoke tests.

## 3. Basic workflow — spectrum

```python
from adctoolbox import (
    analyze_spectrum, analyze_spectrum_polar,
    find_coherent_frequency, fit_sine_4param,
)
from adctoolbox.fundamentals import validate_aout_data

validate_aout_data(aout)
metrics = analyze_spectrum(aout, fs=fs, create_plot=False)
print(metrics["sndr_dbc"], metrics["sfdr_dbc"], metrics["enob"])
```

For subplot layouts, pass the target Matplotlib axes directly. Do this before
falling back to custom FFT plotting:

```python
fig, axes = plt.subplots(3, 1)
for ax, trace in zip(axes, traces):
    metrics = analyze_spectrum(trace, fs=fs, create_plot=True, ax=ax)
```

When `side_bin=None`, spectrum tools use the selected window's coherent
main-lobe width. If the capture is intentionally non-coherent, pass a larger
`side_bin` explicitly; the analyzer does not infer that from the waveform.

To set up a coherent capture *upstream* (where you control the stimulus
frequency), snap `Fin` to an FFT bin first:

```python
fin_hz, k_bin = find_coherent_frequency(fs, fin_target_hz, n_fft=len(aout))
# now drive the test with fin_hz
```

Pick the variant by output:
- `analyze_spectrum` — magnitude spectrum + SNDR/SFDR/ENOB/THD metrics dict
  (default — annotated white-bg plot, Hann window)
- `analyze_spectrum_virtuoso` — same metrics, but Cadence Virtuoso /
  ADE-Explorer dark-theme stem plot. Defaults to rectangular window
  (one stem = one bin, no main-lobe smearing).
- `analyze_spectrum_polar` — phase-aware (I/Q or mixer contexts); same keys
- `quick_sndr` — **lean** SNDR + ENOB only. Use in optimization loops,
  parameter sweeps, spec gates. No plot, no SFDR/THD/HD/NSD breakdown.
  Returns just `{sndr_dbc, enob}`.
- `compute_spectrum` (from `adctoolbox.spectrum`) — both metrics and plot-ready
  data (access via `result["plot_data"]["freq"]` etc.)
- `find_coherent_frequency` — pre-step at *signal generation* time, not analysis
- `fit_sine_4param` — pre-step for nonlinearity work; remember its
  `"frequency"` key is normalized `Fin/Fs`

For the lean path:

```python
from adctoolbox import quick_sndr
m = quick_sndr(aout, fs=fs)
print(m["sndr_dbc"], m["enob"])
# Override the window when the upstream stimulus is coherent and
# you want a clean rectangular FFT instead of Hann:
m = quick_sndr(aout, fs=fs, win_type='rectangular')
```

## 4. Basic workflow — digital calibration

```python
from adctoolbox import calibrate_weight_sine
from adctoolbox.calibration import calibrate_weight_sine_lite
from adctoolbox.fundamentals import validate_dout_data

validate_dout_data(bits)            # bits: (N_samples, N_bits) in {0, 1}

freq_norm = fin_hz / fs             # normalized — not Hz
result = calibrate_weight_sine(bits, freq=freq_norm)
weights = result["weight"]
calibrated = result["calibrated_signal"]

weights_fast = calibrate_weight_sine_lite(bits, freq_norm)   # ndarray, no dict
```

`calibrate_weight_sine` returns a dict with `weight`, `offset`,
`calibrated_signal`, `ideal`, `error`, `refined_frequency`. The `_lite` variant
returns just the weights ndarray and is positional (no `freq=` kw).

## 5. Import rules (compressed)

| Kind | Use |
|---|---|
| Anything re-exported by `adctoolbox.__init__` | `from adctoolbox import X` |
| Submodule-only public tool (`siggen`, `toolset`, `aout`, `calibration`, `fundamentals`, `spectrum`) | `from adctoolbox.<submodule> import X` |

If a flat import fails, check the submodule's `__init__.py` before
concluding the tool is gone. Common submodule-only names:
`ADC_Signal_Generator` (siggen), `compute_spectrum` (spectrum),
`calibrate_weight_sine_lite` (calibration), `validate_aout_data` /
`validate_dout_data` / `convert_cap_to_weight` (fundamentals),
`analyze_phase_plane` / `analyze_error_phase_plane` (aout),
`generate_aout_dashboard` / `generate_dout_dashboard` (toolset).

## 6. Going further

- Dashboards, phase-plane, bit-level, error decomposition, static
  nonlinearity, cap-to-weight → **`references/advanced-debug.md`**
- Function signatures / return keys → `references/api-quickref.md`
- Ready-to-adapt example files → `references/example-map.md`

**Highly Recommended Baseline:** For the simplest end-to-end analysis
+ plot template, adapt `02_spectrum/exp_s03_analyze_spectrum_savefig.py`
(see `references/example-map.md` for the path). The packaged CLI
`adctoolbox-get-examples [dest]` dumps the full example tree.

Every code block in this file (and in `references/advanced-debug.md`) is
exercised by `python/tests/integration/test_user_guide_skill_examples.py`
— if a future edit breaks one, that test fails.
