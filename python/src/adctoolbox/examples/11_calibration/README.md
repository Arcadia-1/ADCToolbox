# Dither calibration simulation examples

Start with [DITHER_GUIDE.md](DITHER_GUIDE.md): it maps
ADC architecture to physical injection, observable moments, correction and
failure conditions, and separates published mechanisms from our model extensions.

Six experiments are available (historical numbering is preserved):

- `exp_cal01_dither_background.py`: minimal gain-only calibration, continuous
  drift, step tracking and probe-amplitude diagnostics (documented below).
- `exp_cal02_dither_coverage.py`: subtractive randomization, gain, linear memory,
  capacitor mismatch and nonlinear moment identification, with failure controls.
  See [DITHER_COVERAGE.md](DITHER_COVERAGE.md) for equations, sources and limits.
- `exp_cal06_joint_dither.py`: gain, memory, capacitor mismatch and nonlinearity
  in one conversion path, with joint background identification, correction
  ablations, parameter convergence bands and injection headroom checks.
  See [JOINT_DITHER.md](JOINT_DITHER.md) for the model and statistical assumptions.
- `exp_cal07_joint_convergence.py`: fixed-amplitude sample-budget, averaging,
  coupling and bias controls, plus code-stratified joint moment identification.
  See [JOINT_CONVERGENCE.md](JOINT_CONVERGENCE.md).
- `exp_cal08_hdc_fixed.py`: three equal PN probes, HDC sign correlations,
  fixed-point gain/cubic correction and joint memory/DAC mismatch tests.
  See [HDC_FIXED.md](HDC_FIXED.md) for the charge model, fixed arithmetic,
  range limitations, long averaging windows and explicit hardware budget.
  `hdc_small_signal.py` adds DC-offset/amplitude scans and large-signal training
  followed by continued versus frozen small-signal operation. HDC exposes
  `--dither-peak` consistently across injection, identification and subtraction.
  `hdc_correlation.py` separates finite-record, quantization, analog-noise and
  gain/cubic coupling terms, with balanced and error-permutation controls.
- `exp_cal09_pn_architectures.py`: common three-PN construction across Pipeline
  placement controls, redundant SAR paired weight calibration, TI timing
  observability and Delta-sigma transfer-aware subtraction. See DITHER_GUIDE.md.

`exp_cal01_dither_background.py` models foreground and background interstage
gain calibration entirely in software. It is a minimal two-stage Pipeline
model, not a reproduction of the earlier five-stage GCN paper or a transistor
implementation. The plant in `pipeline_dither_model.py` remains an example
interface. Gain fitting/tracking uses `dither_gain.py` in this directory;
these helpers are example-only and are not exported as public package APIs.
The frozen sine-fit baseline is retained as a control for Dither tracking.

## Run

From the copied examples directory:

```bash
python 11_calibration/exp_cal01_dither_background.py
```

The default run executes three scenarios, two waveforms and three seeds.
For a short execution/plotting check, not a convergence qualification:

```bash
python 11_calibration/exp_cal01_dither_background.py --blocks 64 --block-size 2048 --foreground-blocks 2 --seeds 11 --scenarios constant --waveforms sine --output output/dither_smoke
```

Figures, per-block CSV trajectories, configuration/source hashes, foreground
fits and aggregate statistics are saved under `output/dither_background/`.
Use a fresh `--output` directory when changing parameters or source versions.
No GUI window, neural-network library or external measurement file is needed.

## Physical model and observable signals

The input and backend ranges are both [-1, 1] V. A seven-level coarse stage
uses thresholds [-5, -3, -1, 1, 3, 5]/8 V and DAC values [-3, ..., 3]/4 V.
Its nominal gain is 4. The ideal 12-bit backend makes the nominal combined
resolution 14 bits, although noise and calibration errors reduce actual ENOB.

For sampled input `u`, coarse DAC output `q`, known PN sign `p` and injection
amplitude `a`, the conversion equations are:

```text
q = coarse_quantize(u)
r = backend_quantize(G * (u - q + a*p) + residue_noise)
y = q + r/G_est - a*p
```

PN is injected after the coarse decision and before the residue amplifier.
It therefore experiences the actual gain error, while the coarse decision is
independent of PN. Adding random numbers after final digital reconstruction
would not implement this identification mechanism.

The plant includes input noise, residue-amplifier noise, backend quantization
and clipping. It intentionally isolates interstage gain error: there is no
DAC mismatch, amplifier nonlinearity, aperture jitter or residue memory.
Probe amplitude and coarse DAC scale are assumed accurate. The amplitude
assumption matters: correlation identifies the product `G*a`, not `G` and
`a` separately. Unknown probe amplitude error biases the recovered gain.

True gain is passed only to the ADC plant and evaluator. The tracker accepts
backend output, known PN signs and backend integer codes, with no ideal input,
true gain, input frequency or fitted sine labels.

## Background estimator

Let `z = u-q`. Under independent, zero-mean PN and without clipping:

```text
cov(p, r) = G*a*var(p) + G*cov(p, z) + cov(p, noise + quantization_error)
G_observed = cov(p, r) / (a*var(p))
G_est_next = (1-learning_rate)*G_est + learning_rate*G_observed
```

The last two covariance terms vanish in the ideal statistical limit, subject
to the quantization-error approximation. Finite-window signal/PN correlation
causes estimator fluctuations and limits the tracking/noise tradeoff.
The estimate is an exponentially averaged block correlation, not an exact
deterministic gain measurement and not a generic nonlinear-error calibrator.

Each block is corrected using the estimate available at its start. Only then
is the estimate updated for the next block. No history-best SFDR selection,
truth-based stopping, same-block refitting or test-set optimization is used.
Correlation histories and RNG states persist across blocks; phase and input
time also advance continuously. The plant itself has no analog memory state.

A block is held if PN excitation is insufficient, any observable backend
code hits a rail, or the observed gain falls outside broad fixed bounds
[2, 6]. Rail-hit rejection is conservative: selecting only unclipped samples
could bias the PN correlation. Frequent holds indicate the configured probe
and input headroom are unsuitable; freezing is protection, not recovery.

The gain bounds are model assumptions, not bounds inferred from true gain.
Saturation-free independent PN does not ensure identifiability of arbitrary
ADC errors. No claim is made about calibration of nonlinear or memory errors.

## Foreground comparison

Before normal operation, acquire a separate dedicated sine with probe off
and initial device gain fixed. With known frequency, jointly fit:

```text
q + c*r = A*sin(w*n) + B*cos(w*n) + DC
G_foreground = 1/c
```

The coarse coefficient is fixed to one, establishing scale. Sine amplitude,
phase and DC are estimated, not supplied from the simulator. The resulting
gain is frozen throughout the normal-operation trace. This is a specialized
foreground subcode/sine fit, not a call to `calibrate_weight_sine` on bit data.

Primary comparison methods all use the same probe-on capture:

- `uncalibrated`: nominal gain 4, subtract PN at its nominal input amplitude;
- `foreground_frozen`: use the dedicated initial fit;
- `background`: start at nominal gain and update causally;
- `oracle`: evaluator-only true instantaneous gain, a diagnostic bound.

Additional probe-off uncalibrated, frozen-foreground and oracle controls use
the same continuous input and identical analog-noise draws. They make the
comparison to ordinary operation without injected PN explicit. The tracker
does not see this counterfactual probe-off path.

## Default experiments

| Item | Value |
| --- | --- |
| Sampling frequency | 100 MS/s, behavioral timing convention |
| Foreground acquisition | 32 blocks x 32768 samples, about 10.49 ms |
| Normal operation | 512 blocks x 32768 samples, about 167.77 ms |
| Background averaging factor | 0.08 per block; one update every 327.68 us |
| Probe amplitude | 0.0625 V before nominal gain 4 |
| Input / residue noise RMS | 0.00028 V / 0.0002 V |
| Input waveforms | 0.85 V peak single tone; two tones of 0.5 V and 0.35 V |
| Initial actual gain | 3.88, or -3% relative to nominal |
| Constant scenario | Actual gain stays 3.88 |
| Ramp scenario | Per-sample linear drift 3.88 to 4.12 over middle half |
| Step scenario | Gain changes 3.88 to 4.08 halfway through |
| Random seeds | 11, 22, 33 |

The drift rates are deliberately strong tracking stress cases, not a measured
temperature-to-gain model or a claimed PVT drift rate. Input frequencies are
coherent at each evaluation block; the stream is never rephased per block.
The foreground fit uses different noise/PN RNG streams from normal operation.

Single-tone metrics are SNDR and SFDR. For two tones the spectral ratio is
total desired-tone power / noise+distortion, and the spur ratio uses the
strongest desired tone. Do not call the two-tone ratio single-tone SFDR.
`error_snr_db` compares output with the simulator clean input without gain,
phase or offset fitting; it is evaluation-only. DC is excluded from FFT ratios.

Startup and step recovery are reported as the first completed 16-block window
with RMS relative gain error below 0.2%. These truth-based diagnostics are not
used to stop or select the estimator. A short smoke run can correctly report
that no such window was reached. Window statistics average per-block dB values.
In run summaries, `updates` counts attempted block updates; `held_updates`
counts rejected attempts. Their difference is the number of accepted updates.

## Validation

`tests/unit/test_dither_background_example.py` verifies the injection path,
ideal quantization bound, stream partition invariance, independent-correlation
identification, freeze conditions, foreground gain recovery, causality, drift
tracking benefit, metric agreement with ADCToolbox, and CLI artifact output.
The probe-amplitude identifiability limitation has an explicit regression test.

Physical dither implementations are architecture-dependent. Related literature:
[A dither-based background calibration circuit for pipelined ADCs](https://www.jstage.jst.go.jp/article/elex/22/5/22_22.20240726/_article).
The equations here define this simplified behavioral experiment; the example
does not claim to reproduce that paper's injection circuit or measured results.
