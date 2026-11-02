# Supervised foreground correction

Read packaged `12_foreground/FOREGROUND_WORKFLOWS.md` for model assumptions,
reference construction, experiment commands and output interpretation.
Read packaged `12_foreground/FOREGROUND_THEORY.md` for the short explanation,
mathematics, symbol definitions and measured-hardware literature. Polynomial/MLP
weights are digital correction parameters, not ADC capacitor weights or gains.
Training needs references; frozen normal operation needs only ADC features.
For SNDR/SFDR and bit-weight comparisons, read packaged
`12_foreground/FOREGROUND_BITWEIGHTS.md` and run `exp_cal10_foreground_bitweights.py`.
It distinguishes matched-reference bit fitting from separate sine calibration,
and final-code features from internal redundant SAR decision columns.
For both errors in one device and bit-weight plus polynomial/MLP correction,
use `exp_cal11_combined_foreground.py` and packaged `COMBINED_FOREGROUND.md`.
The primary cascade requires dedicated calibration injection at the SAR input;
it does not assume internal weights can be identified from final codes alone.

Use a supplied, aligned reference for supervised correction. Training-reference
error becomes calibration error. `fit_static_nonlin` characterizes forward
distortion; use `fit_calibration_polynomial` for a trained direct correction.

For independent training, validation and test records:

```python
from adctoolbox import (
    build_calibration_features, fit_calibration_polynomial,
    fit_calibration_mlp, apply_calibration,
)

train = build_calibration_features(training_output, history=3)
valid = build_calibration_features(validation_output, history=3)
test = build_calibration_features(test_output, history=3)
train_reference = training_reference[train["sample_indices"]]
valid_reference = validation_reference[valid["sample_indices"]]

poly = fit_calibration_polynomial(train["features"], train_reference, degree=3)
mlp = fit_calibration_mlp(
    train["features"], train_reference,
    validation_features=valid["features"], validation_reference=valid_reference,
    hidden_units=16, max_iterations=200, seed=11,
)
corrected = apply_calibration(test["features"], poly["model"])
corrected_mlp = apply_calibration(test["features"], mlp["model"])
```

`corrected["calibrated_signal"]` is in the reference's units and corresponds to
`test["sample_indices"]`. The model dictionary can be JSON-serialized directly.
`outside_training_range` is a per-row marginal-range flag, not a guarantee about
joint feature coverage. Features are current-first, then lag-1, lag-2, etc.
Polynomial diagnostics include `design_condition_number` for the standardized,
unregularized design (None for numerical rank deficiency). A finite value is not
a generalization guarantee. For a mixed-excitation model comparison, use
`exp_cal05_foreground_excitation.py --retrain-mlp`; the default retains the old
single-tone MLP as a reference without retraining it.
These benchmarks use the original signal-generator static nonlinearity and
coarse-code feedback memory APIs. The custom FIR front end was removed; rerun
exp_cal03/05 together for current results. Old model manifests are rejected.
For consecutive stream blocks, pass the returned feature `state` to the next
call. Reset to `state=None` for each independent waveform. Do not concatenate
records before making history features.


The supervised snippet is exercised by `tests/integration/test_user_guide_skill_examples.py`.
