# Dither experiment results and evidence

The examples cover PN gain tracking, isolated error mechanisms, joint identification,
fixed-point three-PN HDC, and architecture-specific observation controls. Their
equations and reproduction commands ship in
[DITHER_GUIDE.md](../../python/src/adctoolbox/examples/11_calibration/DITHER_GUIDE.md).
The public `DitherGainTracker`, `apply_dither_gain` and `DitherHDCTracker` reuse
the gain and HDC algorithms measured here.

## Measured HDC result

The historical joint HDC run uses a three-PN peak of 0.2, a 13-bit backend and
28-bit signed arithmetic with 24 fractional bits. Gain, cubic nonlinearity,
one-tap memory and a two-cell DAC mismatch are present together. The main run
uses three seeds, 2^27 samples per case and a simultaneous parameter step;
the additional constant-parameter run uses seed 404 and 2^30 samples.

| Single-tone correction | Main SNDR / SFDR (dB) | Additional long-run SNDR / SFDR (dB) |
|---|---:|---:|
| Uncalibrated, nominal probe removed | 33.33 / 39.99 | 33.32 / 39.98 |
| Fixed point, 2^20-sample window | 51.04 / 61.33 | 50.85 / 61.58 |
| Fixed point, 2^24 / 2^28-sample window | 55.76 / 66.18 | 56.93 / 67.12 |
| Floating point, same long windows | 55.76 / 66.18 | 56.93 / 67.12 |
| True parameters, same cubic approximate inverse | 56.79 / 66.49 | 56.79 / 66.49 |
| True parameters, exact (Newton) inverse, evaluation only | 74.36 / 106.29 | 74.36 / 106.26 |

Metrics use coherent 32768-point FFTs and averages over the last quarter of
each record; the main column also averages three seeds. These results support
longer averaging at unchanged Dither amplitude. They do not establish general
convergence or a bound on all inputs.

The final device has k/g of about -0.235, so the cubic approximate inverse caps
SNDR near 57 dB even with true parameters, about 17 dB below the exact inverse.
That gap comes from the inverse order, not from noise or estimation. The
spectral metrics therefore do not show parameter convergence: with 2^24-sample
windows the cubic coefficient's relative RMSE over the last quarter was
11.81%, 14.60% and 2.54% for the three seeds (gain 0.38%, 0.54% and 0.17%), so
two seeds exceed the registered 10% band while mean SNDR stays within about
1 dB of the true-parameter cubic inverse. The single-seed 2^28-window run
reached 1.50% cubic and 0.084% gain error. Each update uses one window only;
the public `DitherHDCTracker` therefore defaults to 2^24-sample windows, and
shorter windows trade accuracy for tracking speed.

The main sine/two-tone runs have no overloads, but their minimum backend
headroom is only 0.005647 / 0.011304 of full-scale amplitude. DC and low-frequency
inputs with peak 0.8 can overload. The corresponding DC 0.78 control has
positive headroom; this is a range check, not a full spectral qualification.

The 2^28-window resource model uses six sample-rate multipliers, one shared
slow multiplier, five 56-bit correlation accumulators, one serial divider and
1456 register bits including accumulator state. This is an arithmetic budget,
not synthesis/PPA evidence. Full assumptions and register accounting are in
[HDC_FIXED.md](../../python/src/adctoolbox/examples/11_calibration/HDC_FIXED.md).

## Physical and statistical limits

- The [model basis](hdc-model-basis.md) and
  [state/phase derivation](hdc-physical-state-and-phases.md) come with
  executable algebra checks. The fixed-feedback capacitor
  connection supports the static charge equation; the one-tap memory model is
  conditional and PN need not observe every physical memory state.
- Finite-record correlation variance, quantization-correlated bias and
  gain/cubic estimator coupling are distinct. Increasing a window cannot
  remove every systematic bias. Small-signal diagnostics include offset and
  amplitude sweeps and continued-update versus frozen-coefficient controls.
- The cross-architecture controls have separate assumptions: held-sample
  paired SAR conversions, a known analog probe slope for TI timing, and
  transfer-aware subtraction for Delta-sigma. A shared PN distribution does
  not make their identification mechanisms interchangeable.

## Detailed evidence archive

Detailed CSV/JSON outputs, trajectories, figures and dated reports are retained
on [`codex/dither-evidence-20261102`](https://github.com/chenzc24/ADCToolbox/tree/codex/dither-evidence-20261102).
The links below pin the complete snapshot to commit
`daa4e4a28425265b94a5dc97c182e37da7b8d4c3`, so later branch changes cannot alter
the cited evidence. Large generated outputs are not stored in this repository.

| Evidence | Archived report |
|---|---|
| Gain tracking and isolated mechanisms | [Gain](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-10-31-dither-background.md), [mechanisms](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-01-dither-coverage.md) |
| Joint model and convergence diagnostics | [Joint](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-01-joint-dither.md), [convergence](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-01-joint-convergence.md) |
| HDC spectra, headroom, parameters and resources | [Fixed-point HDC](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-02-hdc-fixed.md) |
| Offset/amplitude scans and correlation controls | [Small signal](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-02-hdc-small-signal.md), [correlation](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-02-hdc-correlation.md) |
| Pipeline, SAR, TI and Delta-sigma controls | [Architectures](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-02-pn-architectures.md) |

Reports link to their data and original configuration/source SHA-256 records.
The archived full runs predate the current code: the frozen gain-fit helper has
since changed, and the HDC tracker now holds only the mismatch coefficient when
that estimate is unobservable or out of range, instead of rejecting the whole
update. Their source hashes identify the archived files, not every file in the
current tree. Current checks are recorded in [Dither validation](dither-validation.md).
