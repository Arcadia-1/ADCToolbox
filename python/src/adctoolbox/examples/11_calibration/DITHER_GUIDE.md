# Dither: architecture, injection, identification and correction

This is the entry point for the repository's Dither material. **A common PN
generator does not imply a common injection node, estimator, or correction.**
Parameter calibration, quantization randomization, DAC element selection and
timing excitation are separate mechanisms. Random thermal noise and random
aperture jitter are not deterministic parameters removed by these algorithms.

Public Python entry points are `DitherGainTracker`, `apply_dither_gain` and
`DitherHDCTracker`, imported from `adctoolbox` or `adctoolbox.calibration`.
The first two cover the gain-only model; HDC covers the three-PN nominal-gain-2
model below. Pass aligned observables, retain the tracker across contiguous
blocks, and reset for independent captures. Plants and architecture controls
remain examples; their simulation truth is not an estimator input.

## Shared excitation contract

New comparisons use the HDC construction

```
p1,p2,p3 in {-1,+1}, separate persistent 31/29/23-bit LFSRs
d[n] = (0.2/3)*(p1[n]+p2[n]+p3[n])
```

The four levels are {-0.2,-1/15,1/15,0.2}; ideal probabilities are 1:3:3:1,
variance is 0.2^2/3. PN streams are deterministic and finite-record correlations
are not identically zero. The triple product p1*p2*p3 is a **digital correlation
reference**, not an analog cubic operation. Injection peak is 0.2 in each
model's declared node coordinates; this does not equate circuit voltage/power
or ADC performance across architectures.

Earlier examples intentionally retain their original stimuli and source
hashes: exp_cal01 uses binary PN, exp_cal02/06/07 include a two-PN weighted
four-level probe, and the older randomization study uses continuous uniform
dither. They are historical controls, not implementations of this shared
three-PN contract. Changing their distributions without re-deriving their
moments would invalidate the estimators.

HDC's `--dither-peak` now configures the plant, tracker and subtraction together;
the default and all published acceptance results remain 0.2. See the exposed
parameter table in [HDC_FIXED.md](HDC_FIXED.md). A nonzero cubic denominator,
adequate analog headroom and convergence are separate requirements. That
guide also documents the offset/amplitude scan and large-to-small input
continue/freeze controls in `hdc_small_signal.py`.

## Architecture and injection map

| Architecture / objective | Physical node or action | Observable information | Identification / correction |
|---|---|---|---|
| Pipeline gain/cubic HDC | Stage DAC/reference charge before residue amplifier | Digitized residue, stage code, aligned PN | PN and triple-PN moments; gain and cubic approximate inverse |
| Pipeline memory | Probe must traverse the path with memory | Current and delayed probe, residue | Lagged correlations; dynamic inverse with state |
| Pipeline DAC mismatch | Random physical element/role selection | Selection tags, stage codes, residue | Code/element-conditioned moments; per-element or code correction |
| SAR DAC weights | Offset charge at held-sample/CDAC comparator node, or redundant decision selection | Individual SAR decisions and known offset/selection | Paired conversion residual or bit-level correlations; weighted bit reconstruction |
| TI timing | Common analog path **before** the aperture error, with known local probe slope | Channel labels, PN level/slope references, output | Timing-sensitive correlations; clock trim or digital resampling |
| TI gain/offset | Probe through the relevant gain path; output moments | Channel labels and probe | Gain/offset statistics; channel scaling/offset correction |
| Delta-sigma randomization | Input summing node or quantizer input; different transfer functions | Output stream and known probe | Dither decorrelates quantization; subtraction must match its transfer function |
| Flash / generic static quantizer | Analog input or threshold network, depending on target | Output codes; threshold tags if explicitly modulated | Randomization alone does not estimate every comparator threshold |
| Pipeline-SAR | Separate residue-amplifier and CDAC nodes | Stage residue plus SAR decision bits | Combine architecture-specific methods; no universal cubic correction |

The map states physical requirements, not a claim that every row is a fully
implemented ADC. A probe added downstream cannot characterize an upstream
error it never experiences. A digital-only addition cannot excite an analog
block. An unknown injection gain is confounded with the gain being measured.

## Evidence and scope

**Published HDC:** [Panigada/Galton 2006](https://ispg.ucsd.edu/uploads/2017/05/2006-TCASI-A.-Panigada-Digital-Background-Correction-of-Harmonic-Distortion-in-Pipelined-ADCs.pdf)
derives residue-amplifier polynomial calibration. The
[2009 chip](https://ispg.ucsd.edu/uploads/2017/05/2009-JSSC-A.-Panigada-A-130-mW-100-MSps-Pipelined-ADC-With-69-dB-SNDR-Enabled-by-Digital-Harmonic-Distortion-Correction.pdf)
implements gain/cubic HDC and separate DAC noise cancellation. Its small-signal
measurements use coefficients trained at larger input then frozen. The
[2013 follow-up](https://ispg.ucsd.edu/wordpress/wp-content/uploads/2017/05/2013-TCASI-N.-Rakuljic-Suppression-of-Quantization-Induced-Convergence-Error-in-Pipelined-ADCs-With-Harmonic-Distortion-Correction.pdf)
addresses quantization-correlated estimation errors. Our three-PN example
does not implement that follow-up's complete remedy.

**Published SAR mechanisms:** [Chiu et al., SampTA 2013](https://www.eurasip.org/Proceedings/Ext/SampTA2013/papers/p544-chiu.pdf)
describes offset double conversion (ODC), PN-based bit-weight calibration and
redundancy dithering. ODC compares two conversions of the same held input with
opposite offsets; a single SAR incurs twice the conversions. Our new SAR
example adapts that residual equation to a varying three-PN offset and a
declared sub-binary model. It is not a reproduction of the paper's silicon
or its ICA/internal-redundancy algorithms.

**Published TI placement:** the authors' entry for
[ISSCC 2024, 22.4](https://icas.tsinghua-sz.edu.cn/WebPublications/PaperDetail/1477?unit=1002)
describes a dithered reference clock injected at an input buffer and discusses
path-mismatch limitations of other injection locations. Our linearly
interpolated PN waveform is an analytic timing-observability model, not that
circuit. No analog ramp generator or clock actuator is transistor-verified.

**Delta-sigma foundation:** [Galton 2002](https://ispg.ucsd.edu/wordpress/wp-content/uploads/2017/05/2002-TMTT-I.-Galton-Delta-Sigma-Data-Conversion-in-Wireless-Transceivers.pdf)
explains signal/noise transfer functions, granular quantization and overload
limitations. Our first-order loop below is separately derived. A four-level
PN sum does not satisfy every theorem for continuous random dither and does
not guarantee removal of all idle tones.

Source retrieval: 2026-11-02. Full HDC/SAR/Delta-sigma papers were inspected;
the TI reference supports placement at author-abstract level only.

## Equations used in the new architecture experiments

### Pipeline: the probe must pass through the amplifier

For an independent held residue u and d=A*sum(p), let F(v)=g*v+k*v^3.

```
before: r=F(u+d) -> E[(r-d)*p1*p2*p3]/(6*A^3) = k
after:  r=F(u)+d -> E[(r-d)*p1*p2*p3] = 0
```

Thus identical PN values at different nodes carry different information.
Balanced Cartesian enumeration removes sample variance for this boundary
check; it is explicitly an analysis bench, not a background acquisition
protocol. Additional controls include a fifth-order term and history-dependent
nonlinearity: the fitted cubic coefficient then mixes unmodeled terms.
Streaming convergence, fixed arithmetic and all four joint errors are already
covered by [HDC_FIXED.md](HDC_FIXED.md), including DC/low-frequency overload.

### SAR: redundant decisions and paired residual

The example uses 14 signed decisions and radix 1.85, not a binary 14-bit ADC.
Weights nominally sum to one; four leading weights are unknown, the lower
ten are held at their nominal values. Greedy decisions subtract +/-w_j from
the current comparator residue. This assumes an ideal comparator apart from
specified noise, known offset scale and sufficient architectural redundancy.

For two conversions of the **same held input**, with injected offsets +d,-d:

```
Delta_b = b_plus-b_minus
error = 2*d - sum(Delta_b[j]*w_hat[j])
w_hat[j] <- w_hat[j] + mu*Delta_b[j]*error / sum(Delta_b[0:4]^2)
x_hat = sum((b_plus+b_minus)*w_hat_used)/2
```

Only the first four weights update, after producing the current output. A zero
denominator holds the update; weights have nominal +/-10% guards. Signed bit
products and denominator values permit muxes/shifts and a small reciprocal
table. This is floating arithmetic intended to expose the mechanism; only
exp_cal08 has a fixed-point hardware budget. SVD/rank diagnostics are
evaluation-only. The learner never sees the true input or true weights.

Validation includes a new frequency with coefficients frozen and dither off,
and an intentionally invalid pair using different held inputs. It also
compares correct bit-weight reconstruction with inappropriate global cubic
HDC on the same DAC-mismatch problem. A non-redundant SAR can lose information
before digital correction; bit weighting cannot restore arbitrary gaps.

### TI: voltage PN needs a known time slope

Let d_n be the same three-PN sum and take nonnegative skew 0<=tau<1 sample.
A specified linear interpolation before the sampler gives

```
d(n+tau)=(1-tau)*d_n+tau*d_(n+1)
y=g*[x(n+tau)+(1-tau)*d_n+tau*d_(n+1)]
m0=E[y*d_n]/Var(d); m1=E[y*d_(n+1)]/Var(d)
g_hat=m0+m1; tau_hat=m1/g_hat
```

This requires PN/input independence and the specified analog waveform, not
just its four digital levels. A flat zero-order-held probe away from its edges
has no skew information, as does a probe inserted after the sampler. Two
matched failure controls use those placements. Streaming lookahead retains
the same d_(n+1) at block boundaries. Evaluation applies estimated gain and a
quantized ideal clock trim on an independent dither-off record. Real delay
range, jitter, generator bandwidth and channel-path mismatch remain unmodeled.
The linear ramp takes intermediate analog levels between the four digital
endpoints while retaining peak 0.2. In the `after_sampler` control it also
bypasses the modeled channel gain, so neither that gain nor skew is observable
from its PN path. The next PN value is precomputed, not an unknown future input.

### Delta-sigma: subtract the correct transfer function

Our first-order loop is

```
u[n]=state[n-1]+x[n]; y[n]=sign(u[n]+d[n]); state[n]=u[n]-y[n]
e[n]=y[n]-(u[n]+d[n])
```

Algebra gives `y=x+(1-z^-1)d+(1-z^-1)e` (with consistent initial state).
Injection at the loop input instead gives `y=x+d+(1-z^-1)e`. Therefore
quantizer-input dither requires subtracting **d[n]-d[n-1]**, not simply d[n].
This removes the explicit probe path, while e still depends on the randomized
quantizer. It does not reconstruct the undithered bitstream or identify an
analog error coefficient. Tests report in-band noise/spur levels for DC,
in-band SNDR/SFDR for tones, and internal-state bounds; DC has no SNDR label.
Subtraction is output postprocessing; the analog loop continues to feed back
the unmodified quantizer output.

## Experiment and documentation navigation

| Entry | Role and evidence |
|---|---|
| exp_cal01 | Binary-PN gain tracking, ramp/step and frozen foreground controls |
| [DITHER_COVERAGE.md](DITHER_COVERAGE.md), exp_cal02 | Isolated mechanisms; distinct legacy plants/probes |
| [JOINT_DITHER.md](JOINT_DITHER.md), exp_cal06 | Derived joint inverse-cubic behavioral model |
| [JOINT_CONVERGENCE.md](JOINT_CONVERGENCE.md), exp_cal07 | Statistical diagnosis; grouped matrix estimator is software-only |
| [HDC_FIXED.md](HDC_FIXED.md), exp_cal08 | Published-form gain/cubic HDC, fixed arithmetic, extended memory/DAC model |
| exp_cal09_pn_architectures | Shared-PN placement, SAR paired weight calibration, TI timing visibility, Delta-sigma subtraction |

The examples are research/teaching models, not new public calibration APIs.
The gain tracker and frozen sine-fit baseline are local example helpers.
Full multi-stage silicon, general SAR weight calibration,
nonlinear memory, arbitrary Delta-sigma topologies and complete TI PVT/jitter
tracking are not implied by passing these tests.

## Reproduction and acceptance

From `python/`, choose a new output directory:

```powershell
uv run --no-sync --with numba python src/adctoolbox/examples/11_calibration/exp_cal09_pn_architectures.py --output ../agent_playground/pn-architectures
```

Numba is optional acceleration; output directories are never overwritten.
The runner records arguments, source hashes, environment, PN distribution,
per-case measurements and figures. Use `--quick` only for command/artifact
smoke checks, not convergence qualification.

Acceptance checks: no truth leakage into estimators; causal coefficient use;
stream partition invariance; identical captures for paired method comparisons;
probe peak/range accounting; independent-seed/held-out evaluation; expected
failures retained; finite-record spectral metrics labeled by waveform/band;
fixed-point claims limited to tested arithmetic; shipped examples run outside
the repository. Dated experiment reports are snapshots, not user guides.
