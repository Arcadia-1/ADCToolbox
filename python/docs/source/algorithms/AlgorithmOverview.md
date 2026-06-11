# ADCToolbox Architecture Overview

**Last Updated:** 2026-05-26

ADCToolbox is organized around Python modules that map to common ADC analysis
workflows: generate or import data, analyze analog or digital output, calibrate
weights when needed, and build plots or dashboards from the results.

## Python Package Layout

```text
python/src/adctoolbox/
├── fundamentals/      # sine fitting, coherent bins, unit conversions, FOMs
├── spectrum/          # FFT metrics, spectrum plotting, polar plots, OSR sweeps
├── aout/              # analog-output residual, INL, PDF, ACF, phase-plane tools
├── dout/              # bit activity, overflow, weight radix, ENOB sweeps
├── calibration/       # sine-based ADC weight calibration
├── models/            # SAR behavioral conversion and mismatch models
├── siggen/            # synthetic signal generation with non-idealities
├── timeinterleave/    # TI-ADC splitting, mismatch extraction, spur prediction
├── oversampling/      # NTF and oversampling utilities
└── toolset/           # analog and digital dashboard generators
```

The top-level package exports the most common functions through
`adctoolbox.__init__`, while submodules remain available for explicit imports
such as `from adctoolbox.models import sar_convert`.

## Common Data Paths

### Analog Output Analysis

```text
sampled waveform
    -> fit_sine_4param / estimate_frequency
    -> analyze_spectrum or analyze_spectrum_polar
    -> aout residual tools
    -> optional generate_aout_dashboard
```

This path is used for SNDR/SNR/SFDR/ENOB measurement, residual statistics,
static nonlinearity fitting, phase-plane views, and dashboard generation.

### Digital Output Analysis

```text
bit decision matrix
    -> calibrate_weight_sine or calibrate_weight_sine_lite
    -> reconstruct with calibrated weights
    -> spectrum / bit activity / overflow / radix / ENOB sweep
    -> optional generate_dout_dashboard
```

This path is used for SAR and bit-weighted ADC debug. Calibration and
application are separate steps: calibration estimates the true bit weights,
and normal operation reconstructs output with those calibrated weights.

### SAR Behavioral Modeling

```text
input waveform
    -> sar_ideal_weights or user weights
    -> optional sar_apply_cap_mismatch
    -> sar_convert
    -> sar_reconstruct
    -> spectrum or digital debug tools
```

The SAR model supports binary or redundant weights, explicit quantization
ranges, sampling noise, comparator noise, and capacitor mismatch.

### Time-Interleaved ADC Analysis

```text
interleaved sample stream
    -> deinterleave
    -> extract_mismatch_sine
    -> predict_spurs
    -> calibrate_foreground
    -> interleave
```

The TI-ADC utilities focus on offset, gain, and timing-skew mismatch for sine
captures. Fractional-delay helpers are provided for correction workflows.

## Examples And Documentation

The packaged examples are the canonical runnable workflows. Copy them with:

```bash
adctoolbox-get-examples
```

The example tree currently contains 63 scripts across spectrum analysis,
signal generation, analog debug, digital/SAR calibration, toolset dashboards,
conversions, time-interleaving, downsampling, and oversampling.

## Maintenance Notes

- Public flat exports live in `python/src/adctoolbox/__init__.py`.
- User-facing examples live under `python/src/adctoolbox/examples/`.
- Generated example outputs live in `output/` directories and are not package
  source files.
- Bundled Codex skills live under
  `python/src/adctoolbox/_bundled_skills/skills/`.
