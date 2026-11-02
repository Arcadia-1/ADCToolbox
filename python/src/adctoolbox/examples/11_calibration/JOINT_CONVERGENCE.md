# Fixed-amplitude joint convergence study

**Hardware scope:** Keep this estimator as a software diagnostic reference.
The target project requires a simple hardware implementation; the grouped
covariance weighting, eigendecomposition and SVD have not been validated in
fixed-point RTL, synthesis, transistor-level simulation or silicon. Floating-
point spectral improvements are not hardware performance claims. Subsequent
implementation work must start from an injection circuit and a bounded digital
datapath, with correlation accumulators and scalar updates preferred where
their identifiability and stability have actually been verified.

`exp_cal07_joint_convergence.py` diagnoses the gain/cubic fluctuations in
[the joint experiment](JOINT_DITHER.md) (exp_cal06) without modifying it.
This is an example, not a public API.

```bash
python 11_calibration/exp_cal07_joint_convergence.py --output output/joint_convergence
```

The default uses independent validation seeds 101, 202, 303; only seed 44 was
used in the small exploratory comparison. Probe peak is fixed at **0.2**,
input peak at 0.8, block size at 32768. All methods process identical captures.
The probe is the samplewise weighted sum `(2*p31+p29)/3`, not concatenated
time segments, Gaussian noise, or a single binary PN. Its actual injected
levels are `{-0.2,-0.2/3,+0.2/3,+0.2}` in the normalized `w/2` coordinate.
The separate binary PN `s` controls capacitor roles. Backend Gaussian noise
with sigma 0.0002 is a distinct physical noise source, not the dither.

## Experiments and controls

The five estimators are pooled moments with EMA weights 0.04 and 0.01,
cumulative pooled moments, and code-stratified moments with weights 0.04 and
0.01. Each starts from nominal coefficients and updates after correcting the
current block. Neither signal frequency, clean input, device truth nor the
known evaluator change point enters an estimator. No estimator is reset at
the step. The capacitor equation and acceptance guards are unchanged.

For each seed, both sine and multitone inputs run for 256 blocks, with either
constant FINAL parameters or INITIAL-to-FINAL parameters at block 128. The
constant case uses FINAL here to match the long-record bias controls;
exp_cal06 uses INITIAL for its constant case. Do not compare their
constant-case averages without accounting for that difference.

An additional 1024-block sine record compares the default joint plant with a
24-bit, zero-noise joint plant and a 24-bit, zero-noise plant with H=delta=0.
Their input and PN streams match. Saved block moment matrices permit replay
of averaging policies without regenerating signals. An **oracle-only**
diagnostic fixes inverse cubic coefficient c3 to truth and fits c1,h; it is
never used to report corrected-output SNDR/SFDR. QR reparameterization of the
exactly identified equation checks whether numerical conditioning alone
changes the solution.

Cartesian-product controls enumerate independent input/probe/switch/noise
states, removing accidental finite PN/input correlation. They check exact
identification, quantization error and independent symmetric backend-noise
bias. The two-point noise integrates cubic noise moments exactly; it is a
diagnostic distribution, while streaming experiments use Gaussian noise.

## Why the original averaging can fluctuate indefinitely

The update is an EMA of correlation matrices followed by a direct solve,
not gradient descent. With block-independent equal-variance moments, the
asymptotic variance-equivalent record length is

```text
N_eff = block_size * (2-alpha)/alpha
alpha=0.04: 49 blocks = 1,605,632 samples
alpha=0.01: 199 blocks = 6,520,832 samples
```

These are idealized variance-equivalent counts, not exact independence claims
for a finite PRBS stream. The first block has weight one, so startup behavior
also matters. After a parameter step, old moments decay as `(1-alpha)^k`:
at 128 blocks about 0.54% remains for 0.04, but 27.6% for 0.01. Cumulative
averaging reduces stationary variance but keeps stale pre-step observations.

## Preserve valid conditional moments instead of pooling them

The original estimator centers three instruments within each of nine observed
`(q[n],q[n-1])` pairs, then pools them into three equations. The new estimator
retains a separate equation for each occupied group. With

```text
z = [p, p^3-41*p/45, p_previous]
f = [r, r^3, -a*p_previous]
theta = [c1, c3, h] = [1/G, beta/G, H/G]
A_g = mean(1_group * centered(z) * f^T)
b_g = mean(1_group * centered(z) * a*p)
```

it fits up to 27 moments to three unknowns. Conditional exogeneity is valid
in this plant because q is determined **before** the probe and independently
of the capacitor-switch sequence. Business residue remains a nuisance, but
different coarse-code pairs retain different cubic response sensitivities.
Both schemes center within the same groups; the change is retaining these
moments separately, not adding new analog excitation or assuming a sine fit.

Weight matrices come from the within-group covariance of `z*residual`, using
start-of-block coefficients for residuals, and are averaged at the same rate
as A and b. Eigenvalue truncation removes empty/numerically unsupported
directions. A whitened SVD solve avoids squaring the condition number. This
is feasible generalized method of moments (GMM); the weights do not model
all temporal correlation and are not claimed universally optimal. Finite
records and estimated weights can still introduce bias and variation.

An invertible basis change to the original three equations cannot supply
additional statistical information. The improvement being tested comes from
additional valid conditional moments; whitening is numerical implementation.

## Bias boundary

For independent symmetric additive backend noise e with variance sigma^2,
`E[(r+e)^3 | r]=r^3+3*sigma^2*r`. With no quantization, this shifts the fitted
inverse linear coefficient by `-3*sigma^2*c3`. It does not make the physical
gain/cubic ratios exactly unbiased. The balanced controls quantify the shift
and separate it from finite excitation variance. Unknown probe amplitude,
wrong delay, clipping or a missing physical nonlinear term can create much
larger structural errors; this study retains the original modeled assumptions.

## Outputs and interpretation

Per-block CSV files retain physical parameter errors, statuses, SNDR/SFDR,
headroom and a causality check. Summary windows are the second quarter, first
quarter after midpoint, and final quarter. Multitone spectral quantities are
desired-tone to residual-power and largest desired-tone to spur ratios, not
single-tone SNDR/SFDR. Parameter bands remain gain 1%, H/delta/beta 10%.

`long_summary.csv` reports the last quarter of each acquired-record length.
Its diagnostic solves are post-update estimates without deployment bounds,
whereas main-run parameters and spectra use accepted start-of-block estimates.
Those overlapping windows are not independent Monte Carlo trials. Aggregate
uncertainty should be evaluated across seeds, not by treating all time blocks
as independent. Three seeds and two stationary input spectra are evidence for
this example, not a universal convergence guarantee.

`configuration.json` saves source hashes and versions; `audit.json` records
sample counts, clipping and causal application. Output directories must be new
or empty. `fixed_probe_convergence.png` and `sample_budget.png` provide figures.

The archived [2026-11-01 results](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-01-joint-convergence.md)
record the fixed-seed measurements and remaining multitone limitations.
