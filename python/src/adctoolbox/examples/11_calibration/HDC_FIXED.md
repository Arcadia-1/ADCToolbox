# Three-PN HDC with fixed-point correlation accumulators

`exp_cal08_hdc_fixed.py` implements a bounded-integer arithmetic architecture
for one calibrated pipeline stage. It follows equations (8)–(11) of
[Panigada and Galton, JSSC 2009](https://ispg.ucsd.edu/uploads/2017/05/2009-JSSC-A.-Panigada-A-130-mW-100-MSps-Pipelined-ADC-With-69-dB-SNDR-Enabled-by-Digital-Harmonic-Distortion-Correction.pdf).
The paper supplies the gain/cubic HDC method and an implemented ADC precedent.
The one-tap memory inverse and two-element DAC mismatch correlator below are
our extensions. This repository version is an arithmetic model and an explicit
architecture budget, not RTL, a synthesized netlist, or a transistor simulation.

The repository-side physical review derives a fixed-feedback switch-phase
candidate that supports the static DAC equation. Actual reset memory can include
previous output, DAC and PN states; the one-tap residue model here is a conditional
approximation. A delayed-PN correlation need not observe all physical memory.
See `docs/experiments/2026-11-02-hdc-physical-state-and-phases.md` in the repository.

The public `adctoolbox.DitherHDCTracker` uses the same kernel, by default with
floating arithmetic and 2^24-sample windows; `fixed=True` selects the
bit-accurate datapath described here. Its `estimate_memory` and
`estimate_dac_mismatch` switches hold an extension's coefficient, and its
`selection` input must be balanced random DAC role tags.

## Run

From `python/`, with a new output directory each time:

```powershell
uv run --no-sync --with numba python src/adctoolbox/examples/11_calibration/exp_cal08_hdc_fixed.py --output ../agent_playground/hdc-main
uv run --no-sync --with numba python src/adctoolbox/examples/11_calibration/exp_cal08_hdc_fixed.py --output ../agent_playground/hdc-long --total-log2 30 --window-log2 28 --seeds 404 --kinds sine --scenario constant
uv run --no-sync --with numba python src/adctoolbox/examples/11_calibration/hdc_fixed_controls.py --output ../agent_playground/hdc-controls
uv run --no-sync --with numba python src/adctoolbox/examples/11_calibration/hdc_small_signal.py --output ../agent_playground/hdc-small
```

Numba (the optional `accel` extra) only accelerates the scalar reference loops. Without it, the same Python
integer operations run, but billion-sample experiments are slow. No matrix
inversion, SVD, GMM, sine fit, or simulator truth enters the tracker. The
Newton inverse in `oracle()` is strictly an evaluation control.

### Exposed parameters

| Parameter | CLI / Python interface | Default and meaning |
|---|---|---|
| Physical Dither peak | `--dither-peak`; `Plant(peak=...)`, `Tracker(dither_peak=...)` | 0.2, in residue-input/reference units; one PN amplitude is peak/3 |
| Signal amplitude | `--input-peak`; `run_case(peak=...)` | 0.8, distinct from Dither amplitude |
| Backend resolution | `--backend-bits`; `Plant(bits=...)` | 13 bits |
| Backend noise RMS | `--noise-rms`; `Plant(noise=...)` | 0.0002, before backend quantization |
| Record/window size | `--total-log2`, `--window-log2` | 27 and 24; samples and correlations per dump |
| PN seed and input scenario | `--seeds`, `--kinds`, `--scenario` | Recorded in each run configuration |
| Fixed word format | `Tracker(fraction=..., bits=...)` | F=24, W=28; Python interface, not a main CLI option |
| Accumulator/delay | `Tracker(accumulator_bits=..., controller_latency=...)` | W+L and 512 clocks; Python interface |

The CLI routes the same Dither scale through physical injection, every
tracker, nominal subtraction and evaluation oracles. `Plant` captures carry
their nominal scale, and a mismatched tracker rejects them. Captures without
that metadata are accepted; their scale is then the caller's responsibility.
Changing the physical scale requires the appropriate injection
capacitor/reference in a circuit, not just a digital coefficient.
These are run/design-time settings, not an automatic amplitude controller.
The resource inventory assumes fixed constants; runtime-programmable Dither
amplitude would also need configuration registers and a coordinated analog
switching/update protocol, neither of which is budgeted here.

For example, from the copied `11_calibration` directory:

```python
from hdc_fixed import Plant, Tracker
plant = Plant(peak=0.15)
tracker = Tracker(dither_peak=0.15, fraction=24, bits=28, window_log2=24)
```

Peak must be finite and in (0,1), and its actual fixed-point cubic denominator
must be nonzero without saturation. This is an arithmetic validity check,
not a headroom or convergence guarantee. At peak=0.2, F=8/9/10 are rejected
at initialization because staged rounding makes A^3 zero; F>=11 passes this
particular check, but low precision can still be inaccurate. The main CLI's
first case also runs F=16/20 controls, so its chosen peak must pass F=16 too.
The archived acceptance runs and the small-signal study use peak 0.2.

## Physical charge convention and model scope

All voltages are normalized to the reference magnitude; the backend range is
[-1, 1]. The input-referred probe is

```
d = A*(p1+p2+p3), A = dither_peak/3;  p_i in {-1,+1}
```

At the default peak 0.2, A=1/15 and the levels are {-0.2, -1/15, 1/15, 0.2}, with
ideal probabilities {1/8, 3/8, 3/8, 1/8}. These are three equal probes;
exp_cal02/06/07 instead use the two-PN weighted probe `(2*p1+p2)/3` with equal
level probabilities, so their moments differ.
Three separate persistent LFSRs have lengths 31, 29, and 23. Their product is
an XOR/sign operation. A separate 27-bit LFSR selects the DAC element; it is
not a fourth analog dither injection.

A charge-consistent ideal switched-capacitor interpretation at the default peak is fixed feedback
`Cf=15 Cu`, input sampling `Cs=30 Cu`, two selectable DAC capacitors
`CD+/-=15(1+/-delta) Cu`, and three injection capacitors of `2 Cu` each.
With appropriate reset/reference phases, charge conservation gives

```
ideal_backend = 2*x - (1+delta*s)*q + (2/15)*sum(p)
             = 2*(u+d)
q = -1,0,+1 with thresholds -0.25,+0.25
u = x - (1+delta*s)*q/2
```

Thus 0.2 refers to the normalized input/residue coordinate; the ideal backend
probe peak before other errors is 0.4. The two DAC elements have constant
total capacitance; one is selected for the reference charge. Injection
capacitor ratios and the reference are assumed known. Their mismatch,
parasitics, charge injection, amplifier settling, and timing require circuit
verification. A different injection scale is inseparable from gain without
an additional reference or calibration mechanism.

The simulated nonideal stage is

```
v[n] = u[n] + d[n]
t[n] = v[n] + h*v[n-1], h=H/G
r[n] = (G/2)*t[n] + k*t[n]^3 + backend_noise/2
```

The backend quantizes `2*r` to 13 bits. Noise RMS at the backend is 0.0002.
Initial parameters are `(G,k,H,delta)=(1.94,-0.16,0.06,0.006)`; final parameters
are `(2.04,-0.24,0.12,0.012)`. In step tests all four change at the midpoint;
analog/digital history and random streams continue without resets.

This forward-cubic model matches the HDC derivation. The joint-identification
examples (exp_cal06/07) instead use an exact inverse-cubic plant with
coefficient beta and exchange sampling/feedback capacitors; beta and this k
are not interchangeable. This experiment covers one
DAC mismatch mode and one linear memory tap, not an arbitrary multi-bit DAC
error map or nonlinear memory kernel.

## Correlation equations and correction

The paper subtracts its probe; we add it and reverse the corresponding signs.
For `g=G/2`, form `s0=r-d`, then accumulate

```
M1   = mean(s0*p1)
M3   = mean(s0*p1*p2*p3)
eta  = mean(s0*s0)
Mh   = mean(w*p1_previous)
Mcap = mean(s*q*y)
Q    = mean(q*q)
```

The HDC part is

```
gamma1 = M1/A
gamma3 = M3/(6*A^3)
g_hat  = 1 + gamma1 - (7*A^2 + 3*eta)*gamma3
c1     = 1/g_hat
k3     = gamma3/g_hat
linear = c1*r
w      = linear - k3*linear^3
```

The extensions are

```
h_hat = Mh/A
v_hat[n] = w[n] - h_hat*v_hat[n-1]
y = q/2 + v_hat - d
delta_hat = -2*Mcap/Q
x_hat = y + delta_hat*s*q/2
```

For exact cubic inversion and independent current/delayed probe, `Mh/A=h`.
For exact gain/memory correction, `y=x-delta*s*q/2`, giving the mismatch
correlator. Approximate HDC inversion and finite correlations couple these
extensions; they are tested jointly rather than assumed exact.

For independent probes, letting `z=u+h*v_previous`, the exact unquantized
moments are `gamma3=k` and `gamma1=g-1+(7*A^2+3*E[z^2])*k`. The paper replaces
`E[z^2]` by `eta=E[(r-d)^2]`. Consequently, even with infinite samples,
`g_hat-g=3*k*(E[z^2]-eta)`. Balanced Cartesian controls separate this bias
from sample error. Quantizer/probe correlation can add another bias.

The cubic inverse is a weak-nonlinearity approximation. True coefficients
inserted into that same inverse are a useful residual reference, not a
mathematical upper bound on every possible coefficient choice. The exact
Newton oracle measures the noise/quantization reference separately.

## Fixed-point contract and update timing

* Main words: signed 28 bits with 24 fractional bits, range [-8,8).
* Products: full 56-bit signed product, then round to nearest with ties away
  from zero, then saturate. The 13-bit backend midpoint values are exactly
  representable at this fractional precision.
  Add/subtract expressions retain widened combinational intermediates before
  saturation or bounds checks (up to W+2 bits in the word datapath); a rejected
  gain candidate must not wrap into the valid interval when stored.
* Five correlation accumulators: signed `W+L` bits for `N=2^L`; 52 bits for
  L=24 or 56 bits for L=28. `sum(q^2)` is an unsigned L+1-bit occupancy counter.
* Average-and-dump divides by N using rounding shifts. `A`, `A^2`, `A^3` follow
  the declared fixed arithmetic; small denominator rounding is observable in
  the 16-/20-fractional-bit comparisons.
* A signed wrapper around a restoring unsigned divider supplies six divisions
  per valid update. It performs one quotient bit per iteration and rounds
  the final quotient. Dividend width is `max(2W,W+L+1)`.
* Each dump snapshots the six sums and immediately starts the next window.
  Coefficients commit atomically 512 sample clocks later; old coefficients
  remain active during that interval. A fixed 512-slot controller schedule
  can allocate six 64-slot divider operations plus 128 setup/check/multiply
  slots, decoded from a 9-bit counter. Timing closure is not verified.
* Gain bounds are 0.75<g<1.25, |k3|<2, |h|<0.25, |delta|<0.05. Digital or
  accumulator overflow, controller saturation, or any backend rail sample
  invalidates the window. If Q<=0.05, or the delta estimate is outside its
  bound (as with constant DAC roles), only delta is held and the gain, cubic
  and memory update proceeds. Either extension can also be disabled
  (`estimate_memory`, `estimate_dac_mismatch`), holding its coefficient. No
  truth-based stop criterion or coefficient reset is used.

This is a dump estimator, so there is no LMS update step size to tune. Window
length sets its variance/tracking tradeoff. A rejected window can retain
previous estimates; it cannot reconstruct already clipped samples.

## Small-signal acceptance and quantization-correlated bias

DC/low-frequency headroom tests do not establish unbiased parameter learning.
[Rakuljic and Galton, TCAS-I 2013](https://doi.org/10.1109/TCSI.2012.2215754)
identifies small variations around particular DC levels as a failure condition
for earlier HDC. The four-level probe here does not implement that paper's
modified correlations and residue-decorrelation circuitry.

`hdc_small_signal.py` separates three questions:

1. An offset/amplitude scan enumerates all eight current PN states, with phase
   quadrature, and compares unquantized, 13-bit and 24-bit backends. Memory,
   DAC mismatch and analog noise are disabled only in this diagnostic. The
   cubic coefficient bias is exactly `E[e*p1*p2*p3]/(6*A^3)`, where e is the
   backend quantization error in r units. Repeating the table cannot reduce it.
2. Continuous joint-model runs retain gain/cubic/memory/DAC errors and compare
   zero versus default backend noise. Large-signal training is followed by
   the same small-input captures for float/fixed continued adaptation, a
   shorter fixed window, frozen active coefficients and cold fixed startup.
   Freeze snapshots discard pending estimates and partial old-input windows;
   filter state is retained. No true coefficient enters the learners.
3. Final active coefficients are tested on an independent larger sine with
   updates disabled. This exposes coefficient damage hidden by a nearly-DC
   input. DC itself gets bias/RMS metrics, not SNDR. Small tones may be below
   the noise floor; poor small-tone SNDR is not alone proof of bad calibration.

The scan uses a simplified single-backend quantizer, not a reproduction of the
2013 multi-stage ADC. More samples reduce sampling fluctuations but cannot
remove its deterministic correlation bias. Added analog noise may weaken the
correlation while increasing output noise. Freezing protects the learned
coefficients only while the analog parameters remain stable; it cannot track
later drift. Low coarse-code occupancy also leaves the DAC mismatch mode
unobservable, so holding its value is not evidence of convergence.

The small-signal runner exposes `--train-log2`, `--test-log2`, `--window-log2`,
`--short-log2`, `--seeds`, `--noise-rms`, and `--dither-peak`. Its declared
offset/amplitude matrix and the balanced-scan grid are recorded with source
hashes. See the [archived small-signal report](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-02-hdc-small-signal.md) for the dated
results and acceptance limits.

## Correlation attribution experiments

`hdc_correlation.py` adds evaluator-only controls without changing the HDC
tracker or PN amplitude. Low-amplitude/DC inputs are an operating-boundary
study. The repository's foreground bit-weight amplitude sweep tests a
different mechanism.

For `r0=g*(z+d)+k*(z+d)^3`, let n be additive noise in r units and
`e=Q(r0+n)-(r0+n)` the quantization error. The experiment records the exact
finite-record identity, using `c3=p1*p2*p3` and angle brackets for averages:

```
k_hat-k = [<(r0-d)*c3>/(6A^3)-k] + <n*c3>/(6A^3) + <e*c3>/(6A^3)
```

The first bracket is further divided into input leakage, terms linear and
quadratic in d, and imbalance of the cubic-PN moment. These are finite-record
terms, not an assumption that all errors are independent white noise. The
diagnostic also records PN means/pair moments, conditional quantization-error
means for c3=+1/-1, the raw error moment and centered covariance separately.

The corresponding normalized-gain identity is

```
g_hat-g = B1 + <n*p1>/A + <e*p1>/A
          + 3*k*(<z^2>-eta) - (7*A^2+3*eta)*(k_hat-k)
B1 = <(r0-d)*p1>/A - [g-1+(7*A^2+3*<z^2>)*k]
```

This separates finite linear-correlation error, direct noise/quantization,
the eta approximation, and propagation of cubic-estimation error into gain.
The partitions are verified by reconstructing the measured total errors.

Controls compare unquantized, 13-bit and 24-bit backends on identical analog
captures, isolated versus joint plants, zero/default analog noise, and nested
record lengths. Balanced PN/phase repetition tests deterministic bias.
Permuting only e preserves its histogram while breaking its pairing with PN;
an independent reference deliberately removes useful cubic information too.
In balanced controls that reference is enumerated as +/-1 for every input/PN
tuple; the streaming negative control uses a separate LFSR.

Truth decomposition and error permutations cannot be supplied to a deployed
learner. They identify causes; they do not implement residue decorrelation,
prove a universal convergence rate, or justify a new hardware resource claim.
For reproduction, choose a fresh output directory:

```powershell
uv run --no-sync python src/adctoolbox/examples/11_calibration/hdc_correlation.py --output ../agent_playground/hdc-correlation
```

Defaults retain peak 0.2, seeds 601/702/803, and sample counts 2^16/2^20/2^24.
`--quick` is only a command/artifact smoke check. Dated evidence is under
[the archived correlation report](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-02-hdc-correlation.md).

## Explicit resource budget, one stage

This is a chosen unpipelined sample datapath and a shared slow controller,
not an area-minimized design. Every sign correlation uses a mux/add/subtract.

| Operation | General multipliers |
|---|---:|
| c1*r, linear squared, linear cubed, k3*cube | 4 |
| h*previous_v | 1 |
| s0 squared for eta | 1 |
| Shared slow update arithmetic | 1 |
| **Total 28x28 logical multipliers** | **7** |

There are **five signed correlation accumulators**, one occupancy counter,
six snapshot registers, and **one shared serial divider**. Multiplication
by +/-1, 0, 1/2, 3, 6, or 7 uses selection/shifts/adders. PN generation uses
XOR and shift registers. Constants may be hardwired.

For W=28, F=24, L=28 the complete declared register inventory is:

| Register bank | Vectors x width | Bits |
|---|---:|---:|
| Three injected PN LFSRs + DAC selection LFSR | 31+29+23+27 | 110 |
| Delayed p1 | 1x1 | 1 |
| Correlation accumulators | 5x56 | 280 |
| Occupancy counter | 1x29 | 29 |
| Snapshots | 5x56 + 1x29 | 309 |
| Window counter | 1x28 | 28 |
| Active and shadow coefficients | 8x28 | 224 |
| Previous v and output | 2x28 | 56 |
| Controller temporary words | 8x28 | 224 |
| Controller product | 1x56 | 56 |
| Divider dividend/quotient, remainder, divisor | 57+30+29 | 116 |
| Divider loop, controller cycle, flags | 6+9+8 | 23 |
| **Total (43 register vectors, accumulators included)** | | **1456** |

The eight flag bits include a delayed-PN validity flag: the first memory
correlation after reset is gated off, matching `previous_p=0` in the model.
They also cover window/snapshot validity, pending update, capacitor excitation,
controller overflow and divider sign. Controller sequencing is decoded from
the fixed cycle counter rather than a separately counted state register.

L=24 uses **1401 register bits**. These are logical state bits, not measured
flip-flop or FPGA DSP utilization. The inventory excludes backend/IO/debug
state, extra pipeline registers and alignment delays needed for a chosen
clock frequency, and all analog elements. The model's 100 MS/s is a time
conversion convention, not verified operating frequency. With only 512
controller clocks per dump, data-path multiplication dominates hardware cost;
controller arithmetic is heavily shared.

## Evaluation and range

The main run uses three independent seeds, a coherent high-frequency sine
and two tones, and 2^27 samples/case. The default window is 2^24 (167.77 ms
at 100 MS/s); the matched short-window control uses 2^20 (10.49 ms). The long
constant-device test uses 2^28 (2.684 s) and 2^30 total samples, seed 404.
Final metrics average per-FFT dB values over the last quarter, using 32768-point
rectangular coherent FFTs, excluding DC. Last-window shadow estimates are
reported in the update log but are not used before their commit time.

For two tones, the ratios are desired-tone power / other power, and strongest
desired tone / largest unwanted bin; they are not labeled single-tone SNDR
or SFDR. Truth and parameter tolerance bands are evaluator-only. The registered
bands are 1% gain and 10% each for cubic, memory, and mismatch.

All main tones have positive range margin, but that is not a general range
guarantee. At final parameters, worst-case constant-sign input/probe history
with input peak 0.8 produces backend magnitude 1.01914 before noise. The
range controls include DC and low-frequency clipping and a 0.78 input
derating, always retaining probe peak 0.2. Under this model and its known
coefficients, an input peak <=0.78826 reserves six backend noise sigmas.
Gaussian noise is unbounded; six sigma is a design allowance, not an absolute
guarantee. Reference, injection-capacitor, and PVT tolerances remain additional
circuit design budgets.

See the [archived HDC report](https://github.com/chenzc24/ADCToolbox/blob/daa4e4a28425265b94a5dc97c182e37da7b8d4c3/docs/experiments/2026-11-02-hdc-fixed.md) for measured results, source
hashes, convergence limitations, and regression tests.
