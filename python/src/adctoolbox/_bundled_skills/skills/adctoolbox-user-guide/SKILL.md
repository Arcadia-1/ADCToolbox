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

Do NOT use for:
- Analog topology / transistor design → `analog-design`, `analog-explore`
- Spectre simulation, pre/post-layout audit → `analog-verify`, `analog-audit`
- Editing ADCToolbox's own source → `adctoolbox-contributor-guide`

## 2. Critical conventions (read first — these are the common bug sources)

### Frequency units

- `fs`, `Fin`, and plotting frequencies are in **Hz**.
- `fit_sine_4param(...)['frequency']` returns **normalized `Fin/Fs`**, not Hz.
- `calibrate_weight_sine`, `calibrate_weight_sine_lite`, and most
  `dout` helpers expect **normalized `freq = Fin/Fs`**.

### Return shapes are not uniform

Most analysis functions return `dict`. Exceptions:

| Function | Return |
|---|---|
| `find_coherent_frequency` | `tuple (fin_hz, bin_idx)` |
| `analyze_bit_activity` | `ndarray` |
| `analyze_overflow` | `tuple` |
| `analyze_enob_sweep` | `tuple (enob_sweep, n_bits_vec)` |
| `fit_static_nonlin` | `tuple` |
| `calibrate_weight_sine_lite` | `ndarray` |
| `convert_cap_to_weight` | `tuple (weights, c_total)` |
| `analyze_weight_radix` | `dict` (was `ndarray` in old versions) |
| `compute_spectrum` | both metrics and plot data |

When docs conflict, trust the current `__init__.py` exports and
packaged examples over older README text.

## 3. Basic workflow — spectrum

```python
from adctoolbox import (
    analyze_spectrum, analyze_spectrum_polar,
    find_coherent_frequency, fit_sine_4param,
)
from adctoolbox.fundamentals import validate_aout_data, validate_dout_data

validate_dout_data(dout)
fin_hz, k = find_coherent_frequency(fs=fs, n=len(dout), fin_target=fin_target_hz)
metrics = analyze_spectrum(dout, fs=fs, Fin=fin_hz, n_bits=N)
print(metrics["SNDR"], metrics["SFDR"], metrics["ENOB"])
```

Pick the variant by output:
- `analyze_spectrum` — standard magnitude spectrum + SNDR/SFDR/ENOB/THD
- `analyze_spectrum_polar` — complex/phase-aware spectrum (I/Q or mixer contexts)
- `compute_spectrum` — both metrics and plot-ready data (use when you
  want to customize plotting)
- `find_coherent_frequency` — pre-step to align `Fin` to an FFT bin
- `fit_sine_4param` — pre-step for nonlinearity work; remember its
  `'frequency'` key is normalized

## 4. Basic workflow — digital calibration

```python
from adctoolbox import calibrate_weight_sine
from adctoolbox.calibration import calibrate_weight_sine_lite

freq_norm = fin_hz / fs   # normalized — not Hz
weights_full = calibrate_weight_sine(dout, freq=freq_norm, n_bits=N)
weights_fast = calibrate_weight_sine_lite(dout, freq=freq_norm, n_bits=N)
```

Pick `_lite` when you need a fast estimate; use the full variant when
you need convergence quality or diagnostic fields.

## 5. Import rules (compressed)

| Kind | Use |
|---|---|
| Anything re-exported by `adctoolbox.__init__` | `from adctoolbox import X` |
| Submodule-only public tool (e.g. `siggen`, `toolset`, `aout`, `calibration`, `fundamentals`) | `from adctoolbox.<submodule> import X` |

If a flat import fails, check the submodule's `__init__.py` before
concluding the tool is gone.

## 6. Going further

- Dashboards, phase-plane, bit-level, error decomposition, static
  nonlinearity, cap-to-weight → **`references/advanced-debug.md`**
- Function signatures / return keys → `references/api-quickref.md`
- Ready-to-adapt example files → `references/example-map.md`

**Highly Recommended Baseline:** For the simplest end-to-end analysis
+ plot template, adapt `02_spectrum/exp_s03_analyze_spectrum_savefig.py`
(see `references/example-map.md` for the path). The packaged CLI
`adctoolbox-get-examples [dest]` dumps the full example tree.
