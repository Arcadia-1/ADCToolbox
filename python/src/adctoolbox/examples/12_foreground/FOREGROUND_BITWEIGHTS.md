# Spectral comparison of foreground correction models

`exp_cal10_foreground_bitweights.py` compares uncorrected output, bit weights,
static/history fifth-degree polynomials and a history MLP on identical test
captures. Its two device families answer different questions; do not pool them
into one model ranking. All results are behavioral, not measured silicon.

For both smooth nonlinearity and bit-weight mismatch in the same converter path,
with standalone versus cascaded correction, see packaged `COMBINED_FOREGROUND.md`
and `exp_cal11_combined_foreground.py`.

## Run

From the copied examples directory, the standalone SAR comparison is:

```bash
python 12_foreground/exp_cal10_foreground_bitweights.py --output output/bitweight_spectral
```

To also compare the previously saved smooth-front-end models:

```bash
python 12_foreground/exp_cal10_foreground_bitweights.py --smooth-baseline output/foreground_acceptance --smooth-mixed output/excitation_mlp --output output/bitweight_spectral_all
```

The smooth mixed run must include `--retrain-mlp`; see `FOREGROUND_WORKFLOWS.md`.
Nonempty output directories are rejected. Capture manifests, saved models,
diagnostics, per-condition metrics, summaries, example spectra, source snapshots
and input hashes are saved. Test evaluation reloads the frozen JSON models.

Replay and verify the completed run:

```bash
python 12_foreground/audit_foreground_bitweights.py --directory output/bitweight_spectral_all --output output/bitweight_spectral_audit.json
```

The audit checks hashes, split separation, saved validation selection and frozen
test outputs. It independently recalculates spectral ratios using a full complex
FFT and recomputes aggregate statistics.
For smooth-front-end comparisons, keep the original baseline and mixed-model
directories at their original paths: the audit verifies those inputs using the
absolute paths saved in `configuration.json`.

## Models and observation contracts

| Method | Training observations | Runtime computation |
| --- | --- | --- |
| `raw` | None | Nominal output reconstruction |
| `bits_reference` | Binary columns and the same aligned monitor-reference rows as polynomial/MLP | Offset plus weighted sum of current bit columns |
| `bits_sine` | One full training sine record; known frequency; no external reference | Frozen offset plus weighted sum of current bit columns |
| `poly5_static` | Current scalar output and monitor reference | Six-coefficient polynomial |
| `poly5_history` | Current output plus three previous outputs and monitor reference | Full fifth-degree basis, 126 coefficients |
| `mlp_history` | Same four scalar features/reference, independent validation | Frozen affine baseline plus 16-unit tanh residual network |
| `known_weights_control` | Evaluator-only actual SAR weights | Diagnostic digital reconstruction, never a fitted candidate |

The **matched-data comparison** is `bits_reference` versus polynomial/MLP.
`bits_reference` is a degree-one fit of the binary feature matrix, with intercept,
training standardization and ridge=1e-10. In original bit units it is exactly
`offset + sum(w_j * b_j[n])`; no powers or history of the bits are used.
It is mathematically the affine bit-weight model, although a fitted weight is
not automatically an independently identifiable physical capacitance.

`bits_sine` calls the public `calibrate_weight_sine`, with explicit normalized
frequency, `force_search=False`, `harmonic_order=1`. It uses the largest-amplitude
single-tone record already present in training (chosen by input amplitude, not
test performance), at its full sample rate. `scale_calibration_output` fixes
the weight sum to nominal; the fitted DC is removed. This scaling does not affect
SNDR/SFDR, but does not establish absolute gain accuracy. It is a practical
separate calibration protocol, **not a matched training-data/label budget**.
The solver's fitted-residual `snr_db` is never used as test SNDR.

## Device families

**Signal-generator front end:** reuse the original static nonlinearity and
coarse-quantized previous-sample feedback from current `exp_cal03/05`, along with
their test records. Polynomial/MLP models are frozen within each replay.
The old custom cubic-plus-FIR runs are historical and must not be replayed against
the new device. Both input runs must carry the current model identity; rerun them
together before using `--smooth-baseline` / `--smooth-mixed`.
The new supervised bit fit uses exactly the saved mixed training records,
stride and monitor reference. Fourteen binary columns are decoded from the
observable final midrise code. These do not introduce internal subcodes or
identify capacitor mismatch in this model; they are a restricted final-code
correction. In particular, current-bit weights cannot generally invert memory.

**SAR MSB mismatch:** use public `sar_convert`, with 14 binary weights, or
15 decisions obtained by repeating the third largest weight. Normalize by the
sum plus one LSB, then increase the actual MSB weight by 1%. Nominal digital
weights remain unchanged. Input range is [-1,1]; sample noise is 1e-5 RMS,
with no comparator noise or other analog nonlinearity/memory. Nominal scalar
reconstruction includes a half-LSB midpoint. Bit methods receive internal
decision columns; polynomial/MLP receive only the nominally reconstructed scalar
and its history. Redundant decisions can collapse to the same nominal scalar,
so the observation advantage of bit methods is part of this comparison.

Strict binary mismatch can create conversion gaps. Even reconstruction using
actual weights cannot generally restore all lost resolution. The known-weight
control shows this limit; it is not a universal optimum or noiseless-input result.
Repeated-weight redundancy supplies correction margin in this specific model.
The same deterministic 1% error is used for all seeds; seed variation is stimulus,
noise and training initialization, not a chip-mismatch Monte Carlo.

Default SAR budgets: 64 training and eight validation records, 8192 samples each,
stride eight, giving 65,536 training and 8192 validation rows. Half the records
are single tones, half two tones. Training/validation/test tone bins are disjoint.
Peak budgets span 0.05–0.90, frequency approximately 0.02–0.45 fs. MLP L2=1e-6,
16 hidden units, maximum 1000 L-BFGS-B iterations; independent reference validation
MSE selects an observed iterate. Report termination status rather than equating
the iteration limit with convergence. The saved smooth MLP retains its original
200-iteration budget. Cross-device scores are not a controlled iteration sweep.

## SNDR, SFDR and multi-tone metrics

Each case tests 15 coherent single tones: amplitude 0.1/0.5/0.85 at approximately
0.031/0.113/0.239/0.371/0.467 fs. The last frequency is outside the training range.
Two- and three-tone tests are reported separately. All test records have 8192
usable samples at defaults, after warmup and history alignment; no test refitting.

For a single tone, remove DC, use a rectangular window, and calculate:

```text
SNDR = 10 log10(fundamental power / all other non-DC spectral power)
SFDR = 10 log10(fundamental power / largest unwanted spectral-bin power)
```

Use the full Nyquist band, include all harmonics in SNDR's denominator, and do
not double the Nyquist bin in one-sided power. Coherent tones need no neighboring
signal bins. Single-tone values are cross-checked with public `analyze_spectrum`
using `win_type='boxcar', side_bin=0`. If a failed correction makes a spur larger
than the known input fundamental, retain the known-bin result and flag that the
automatic peak finder would select another line. Do not silently reinterpret it.

For multi-tone input, `spectral_ratio_db` uses total desired-tone power over
remaining non-DC power. `spur_ratio_db` uses the strongest desired tone over the
largest unwanted bin. `sndr_db` and `sfdr_db` are left empty for these rows to
avoid confusing this definition with a single-tone test. Intermodulation products
outside desired bins remain in noise/distortion; any product coinciding with a
desired bin cannot be separated by this FFT measurement.

`error_snr_db` remains available: it compares output with evaluator-only clean
input without gain, phase or DC adjustment. Good spectral ratios alone do not
prove absolute accuracy, correct phase, or successful inversion of memory.
Summaries first average conditions within each seed, then report mean and sample
SD across seeds. Inspect per-amplitude/frequency rows as well as aggregate means.

The mathematical derivation and physical-literature boundaries for polynomial
and MLP correction are in packaged `FOREGROUND_THEORY.md`. SAR decision and
reconstruction conventions are defined by `adctoolbox.models.sar` and the existing
`05_debug_digital/exp_d17_sar_msb_error_binary_vs_repeat_calibration.py` example.
