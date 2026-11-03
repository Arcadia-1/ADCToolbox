# Supervised foreground calibration workflows

The public foreground interfaces are in `adctoolbox.calibration` and also exported
from `adctoolbox`. Existing sine-weight and TI calibration calls are unchanged.
The examples in `12_foreground` run independently of the Dither examples.

Read packaged `FOREGROUND_THEORY.md` for a short explanation, mathematical
derivation, symbol definitions, physical evidence and applicability limits.
The supervised fitters learn digital post-correction coefficients, not internal
ADC component parameters. Normal operation evaluates the frozen model after the
ADC without reference labels or adaptation.

For a frozen SNDR/SFDR comparison with bit-weight calibration on both the smooth
front end and binary/redundant SAR mismatch, see packaged `FOREGROUND_BITWEIGHTS.md`
and `exp_cal10_foreground_bitweights.py`.

## Public interfaces

| Interface | Purpose and result |
| --- | --- |
| `build_calibration_features(samples, history=..., state=...)` | Build aligned current/previous sample features; returns features, local sample indices and continuation state |
| `fit_calibration_polynomial(features, reference, degree=..., ridge=...)` | Direct supervised correction including cross terms; returns model and fitting diagnostics |
| `fit_calibration_mlp(features, reference, validation_features=..., validation_reference=...)` | Tanh residual network with a frozen affine baseline; returns validation-selected model, diagnostics and loss history |
| `apply_calibration(features, model)` | Frozen application; returns corrected signal and marginal training-range flags |

Models returned by the polynomial/MLP fitters are JSON-compatible dictionaries.
Save the `model` field using `json.dump` and load it with `json.load`. Normalization,
coefficients and input training ranges travel with the model. Application does not
refit, renormalize against a new capture, or alter model parameters. Keep channel
order, history length, sample alignment and units the same as during fitting.
Feature state belongs to the stream, not to the saved model.

### Alignment and reference contract

For each independent training record, call `build_calibration_features` with
`state=None`. Use its `sample_indices` to select the corresponding reference
samples. It discards startup rows lacking history. For a continuous stream, pass
the returned `state` into the next block. Do not connect independent records.

The reference must be an aligned measurement or a justified signal model. The
fitters do not know ADC truth and do not manufacture reference labels. A reference
channel's gain, bandwidth, timing and distortion errors become calibration errors
unless separately characterized. Fitting a sine to a distorted ADC output removes
harmonics from the label, but preserves that output's fundamental gain and phase;
it cannot establish absolute input accuracy by itself.

The polynomial basis includes every monomial up to the requested total degree.
Training-only feature standardization and ridge regularization improve numerical
behavior, but cannot supply missing excitation. Review `design_rank`,
`rank_deficient` and `design_condition_number`. The condition number describes
the standardized design before regularization; it is None for numerical rank
deficiency, including underdetermined fits.
Parameter count grows combinatorially; more than 4096 terms is
rejected. An outside-training-range mask detects marginal range violations only,
not joint feature coverage, identifiability or monotonicity.

The MLP uses a single tanh hidden layer, a linear output, and a fixed affine
baseline fitted on the training set. Its residual target is normalized using
training data. SciPy L-BFGS trains all network weights; independent validation
reference MSE selects an initial/accepted/final iterate. A maximum-iteration
termination is reported as such, not silently called optimizer convergence.
The iteration budget is not a count of mini-batch epochs. No PyTorch dependency
is needed. Held-out test data must never be supplied as validation data.

## exp_cal03: nonlinear foreground benchmark

See [the repository-model rerun](FOREGROUND_REPOSITORY_RESULTS.md) for current
SNDR/SFDR results and their limits. Previous custom-FIR tables are historical.

From the copied examples directory:

```bash
python 12_foreground/exp_cal03_foreground_nonlinear.py
```

The default is 32 training and 8 validation records per case, 8192 points each,
two plants, and three seeds. The larger acceptance experiment uses:

```bash
python 12_foreground/exp_cal03_foreground_nonlinear.py --records 400 --validation-records 40 --output output/foreground_acceptance
```

The fixed fitting stride is 16. Thus the larger run captures 3,276,800 training
points per case and fits on 204,800 rows; validation uses 20,480 selected rows.
Reference sine fits use the full captures. Every test waveform is evaluated at
all 8192 samples. This is separate from the earlier GCN paper reproduction.

### Device and observations

Both plants call the existing `ADC_Signal_Generator` APIs. The static plant uses
`apply_static_nonlinearity(x, k2=.015, k3=.04)`, hence
`u = x + .015*x^2 + .04*x^3`. The memory plant additionally calls
`apply_memory_effect(u, memory_strength=.04)`. That original API implements
`q=floor(16*u)/16`, `l=floor(4096*(u-q))/4096`, then `q+l+.04*q_previous`.
It includes coarse/fine quantization; it is not a linear FIR on the analog input.
The static plant bypasses the memory API entirely (calling it with zero strength
would still quantize). Both use seeded 10 microvolt RMS Gaussian noise and the
original 14-bit `apply_quantization_noise` over [-1,1], with half an LSB added
to express the returned lower-edge code at its midpoint. The original quantizer
also produces the 16-bit monitor observations.

Captures include 64 warm-up samples plus three history samples before sample
zero. The memory API's `np.roll` wraps its first sample; this sample is discarded
and never enters training or evaluation. The helper processes independent whole
captures, not stateful converter blocks. Calibration sees only final quantized
output and up to three preceding outputs. These are repository behavioral models,
not measured hardware or a full multistage Pipeline ADC.

The previous custom `NonlinearFrontend` (gain .99, cubic, linear FIR) was removed.
New manifests carry `repository_siggen_static_coarse_memory_v1`; old manifests
are rejected instead of silently applying their fitted parameters to a new plant.
Old results and snapshots are historical evidence only. Rerun exp_cal03, exp_cal05
and the frozen transfer check together in new output directories. Model parameters
are fixed across seeds; this is not device-mismatch Monte Carlo.

A separate 16-bit monitor captures the training input with 200 microvolt RMS
noise and nominal unity gain. A known-frequency least-squares projection extracts
its fundamental and DC; amplitude and phase are fitted. This observable reference
is the main label. Simulator clean input is reserved for test evaluation. A
separate baseline learns a sine fit of the distorted ADC output instead, exposing
reference bias. It is not mixed into the primary reference-channel results.

### Data splits and methods

Training and validation use separate seeds, phases, amplitudes and distinct tone
bins. Test tone bins are excluded from both. Amplitudes span [0.05, 0.90]; training
frequencies span approximately [0.02, 0.45] of sample rate. Tests at 0.467 also
probe frequency extrapolation, so the combined single-tone table is broader than
strict in-band interpolation. Single-tone training does not guarantee coverage
of arbitrary multitone histories; multi-tone and chirp tests explicitly check it.

All methods use the same captures and reference channel:

- Raw output; static affine; affine with three-sample history.
- Degree-three and degree-five polynomials, each with and without history.
- A 16-hidden-unit MLP with the same four input columns as the history polynomials.
- Degree-five history polynomial using the output-fitted reference (bias control).

The affine baseline is a fit of final output samples, not the bit-weight solver:
binary internal decisions are not available in this model. Method hyperparameters
are fixed before test evaluation. MLP validation selects the saved iterate;
polynomial degrees are reported separately, not chosen by test performance.

Tests cover 15 single-tone amplitude/frequency combinations, two- and three-tone
signals, a chirp, amplitude 0.98 overload stress, and a changed device (gain +2%,
cubic coefficient +50%). The last two are separate stress groups, not pooled
into ordinary performance. Report clipping counts and domain flags alongside
performance; clipped information is not recoverable by a general inverse map.

Single-tone ratios are SNDR/SFDR. Multi-tone ratios use total desired-tone power
and the strongest desired tone respectively. Chirp quality uses time-domain
error only. Absolute error SNR and RMSE compare with evaluator truth without gain,
phase or offset fitting. Summary standard deviations are across seed-wise means,
after averaging conditions within each seed.

Artifacts include configuration, model identity and source snapshots/hashes, per-record split manifests,
JSON models, MLP training/validation history, per-test metrics, aggregate CSV/JSON,
and a comparison figure. A nonempty output directory is rejected.
Single-tone CSV fields explicitly include `sndr_db` and `sfdr_db`; these are null
for multi-tone and chirp captures. Single-tone results are cross-checked against
the public spectrum analyzer when its detected fundamental matches the intended
tone. Failed corrections with a stronger spur remain in the results.

## exp_cal05: joint excitation coverage control

The single-tone-trained history polynomial in `exp_cal03` can perform well on
held-out single tones yet fail badly on multi-tone inputs. Numerical full rank
and marginal feature-range coverage are insufficient evidence of generalization.
For an ideal zero-DC sine, four consecutive samples satisfy
`x0*x2 + x2**2 - x1**2 - x1*x3 = 0`, even as amplitude, phase and frequency vary.
Distortion and noise perturb this relation, but adding many single-tone records
does not automatically provide independent excitation of all nonlinear history
features. Multi-tone captures probe combinations absent from that training set.

Run the fixed-model control after `exp_cal03`:

```bash
python 12_foreground/exp_cal05_foreground_excitation.py --baseline output/foreground_acceptance --output output/excitation_acceptance
```

The experiment retains record count, samples per record, fitting stride,
polynomial degrees and regularization, and generates a new training dataset.
Half the new captures are single tones,
half are two tones, with summed peak amplitude budget in [0.05, 0.90]. Their
individual tone bins exclude the baseline validation/test frequencies. It refits
affine history, static degree-five, and history degree-three/five models. It does
not retrain the MLP by default: the original single-tone-trained MLP is explicitly
labeled as a reference. Model selection still does not use test performance.

To also train the MLP under mixed excitation, run:

```bash
python 12_foreground/exp_cal05_foreground_excitation.py --baseline output/foreground_acceptance --output output/excitation_mlp --retrain-mlp
```

This retains the baseline MLP's hidden-unit count, seed, L2 and optimizer budget,
and gives both model families the same mixed training rows. Separate mixed
validation records retain the baseline validation row budget; their tone bins
exclude the mixed training and original test bins. The baseline's single-tone
validation bins also remain excluded from mixed training. The experiment exports
the new validation manifest, MLP loss history and convergence diagnostics, and
preserves the original single-tone results. Validation selection uses reference
MSE only. This changes the training/validation distribution, not just one paired
waveform; it is a fixed-budget comparison, not a proof of universal model ranking.
The original tests are reused as a diagnostic benchmark, not a newly blind test
after the earlier failure was observed. A production claim needs a fresh test set.

The saved models are evaluated on exactly the original test captures. Report
single-tone and multi-tone quality separately. This is an additional excitation
diagnostic, not a replacement that hides the original single-tone-training failure.

### Fresh frozen-model transfer check

After the mixed MLP run, evaluate saved models without fitting or selection:

```bash
python 12_foreground/audit_foreground_transfer.py --baseline output/foreground_acceptance --mixed output/excitation_mlp --output output/foreground_transfer
```

For each plant/seed, this generates four records each of random two, three and
five tones (72 captures for the default two plants and three seeds). Frequencies
exclude all saved training, validation and original test bins for that case.
Peak amplitude budgets span [0.20, 0.85], split randomly among tones. It compares
raw output and the frozen static polynomial, history polynomial and MLP from
both training distributions on identical captures. No test reference is passed
to the models. The clean simulated input is used only to measure error.

Artifacts retain the new capture manifest, source snapshots, input model hashes,
per-record metrics, overload/range flags and seed-wise summaries. This checks
transfer within the same behavioral plants; it does not establish arbitrary
waveform performance, measured ADC accuracy or robustness to PVT changes.


## Audit completed runs

Run `audit_repository_foreground.py --baseline DIR --mixed DIR --fresh DIR --output NEW.json` after exp_cal03, exp_cal05 and audit_foreground_transfer. The audit replays frozen models, checks source hashes and split separation, and independently recomputes full-FFT ratios. Historical source hashes are not rewritten after example extraction.
