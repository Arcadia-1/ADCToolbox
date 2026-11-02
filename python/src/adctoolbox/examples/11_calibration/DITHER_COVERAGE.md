# Dither mechanisms and reproducible experiments

Start with [DITHER_GUIDE.md](DITHER_GUIDE.md) for the architecture/injection map
and the shared three-PN experiments. This document preserves the original
isolated-mechanism models and their different probe distributions. Published-form
fixed-point HDC is documented separately in [HDC_FIXED.md](HDC_FIXED.md).

`exp_cal02_dither_coverage.py` separates randomization from parameter
identification. It extends the gain-only `exp_cal01` example with linear
memory, capacitor-role modulation and nonlinear moment identification.
These are software behavioral examples, not public calibration APIs.

For all four errors in one conversion path, see
[JOINT_DITHER.md](JOINT_DITHER.md) and `exp_cal06_joint_dither.py`. That joint
model uses a 1.5-bit capacitor stage (nominal gain 2) and a newly derived joint
estimator; it is not a direct concatenation of the isolated gain-4 helpers.

## Run and outputs

From a copied example tree:

```bash
python 11_calibration/exp_cal02_dither_coverage.py --output output/dither_coverage
```

Defaults: 4 mechanisms x 2 waveforms x 2 scenarios x 3 seeds = 48 runs.
Each run has 256 x 32768 normal-operation samples (83.88608 ms at a behavioral
100 MS/s), preceded by 16 x 32768 dedicated foreground samples (5.24288 ms).
The static case retains initial parameters; the step case changes them at
41.94304 ms. Changes are deliberate stress tests, not measured PVT rates.

The output includes configuration and source hashes, per-block CSVs,
individual/aggregate JSON summaries, aggregate CSV, two PNG figures, 72
randomization measurements and a JSON file of failure controls. The latter
72 measurements are 3 seeds x 3 amplitudes x 2 upstream-nonlinearity cases x
4 dither modes. Existing nonempty output directories are rejected.

Short command-level check (not enough data to qualify convergence):

```bash
python 11_calibration/exp_cal02_dither_coverage.py --blocks 16 --block-size 1024 --foreground-blocks 1 --seeds 11 --mechanisms memory --waveforms sine --scenarios step --output output/dither_smoke
```

## Experimental contract

- The plant receives the true input and device parameters. Background trackers
  receive only observable subcodes/backend data, the known probe/switch sequence
  and a rail flag derived from output codes. True overload counts are evaluator
  diagnostics and never drive updates.
- The current block uses the estimate available at its start; `update` prepares
  the next block. No best-SFDR selection or truth-based stopping is used.
- PN, analog history, inverse-filter history, noise streams and waveform phase
  persist across blocks. A seed changes the LFSR initial state and noise.
- Real LFSRs are used: a 31-bit maximal-length sequence (period 2^31-1), plus
  a 29-bit sequence for the second bit of the four-level nonlinear probe.
  Finite windows are not exactly independent of the deterministic input.
- Sine input: peak 0.8, frequency nearest odd bin to 0.137 fs. Two-tone input:
  peaks 0.46 and 0.34, nearest odd bins to 0.083 fs and 0.217 fs. The latter
  metric is total desired-tone / noise+distortion, not single-tone SNDR/SFDR.
- Backend range is [-1, 1]; Gaussian backend-input noise RMS is 0.0002.
  This mechanism-isolation experiment has no input sampling noise, jitter,
  comparator offset or reference dynamics. Its noise floor is consequently
  different from `exp_cal01` and the five-stage research reconstruction.
- All main methods share the same probe-on captures. A paired probe-off plant
  uses identical analog-noise draws. In the capacitor case, "probe off" means
  fixed capacitor roles (p=+1), not zero mismatch.
- The oracle uses true coefficients only for evaluation. It corrects the
  imperfect capture; it does not remove its noise or quantization error.
- Reported windows average per-block dB values, then average across seeds.
  No waveform fitting is performed before error-SNR evaluation. FFT ratios
  exclude DC; error-SNR includes absolute gain/phase/DC error against truth.
- Different mechanisms use different plants. Comparing their absolute dB
  values is not an algorithm ranking or a comparison with commercial chips.

## 1. Subtractive dither: randomization, no learning

A monotone 14-bit quantizer has thresholds

```text
t[j] = -1 + j*LSB + 2*LSB*sin(2*pi*j/64), LSB=2/2^14.
d ~ Uniform(-32*LSB, +32*LSB).
y = Q(x+d)-d.
```

Four modes are compared: no dither, unsubtracted dither, ideal subtraction,
and a 1% error between actual analog injection and digital subtraction.
Three input peaks (0.01, 0.1, 0.8) are used. A second plant includes an
upstream cubic term `x+0.03*x^3` BEFORE injection. Dither cannot undo that
upstream transfer function in this model.

The periodic INL is chosen to expose the mechanism; it is not an extracted
LTC2208 or AD9208 transfer curve. Dither may spread spurs into a noise floor,
improving SFDR without improving integrated SNDR. This experiment is not
parameter calibration, despite its relationship to commercial dither features.

## 2. Gain and linear memory

The seven-level coarse quantizer is identical to `exp_cal01`; `q` is its DAC
level. The backend has 12 bits. Let `v=x-q+a*p`, with `a=0.0625`.

```text
r[n] = G*v[n] + H*v[n-1] + backend_noise, followed by quantization.
P[n] = [p[n], p[n-1]].
theta_observed = inv(Cov(P,P))*Cov(P,r)/a = [G,H], approximately.
theta_next = (1-0.08)*theta_used + 0.08*theta_observed.
v_hat[n] = r[n]/G_hat - (H_hat/G_hat)*v_hat[n-1].
y[n] = q[n] + v_hat[n] - a*p[n].
```

Solving with the measured PN covariance handles finite-window cross-correlation
between the two probe columns. It does not eliminate correlation with unknown
user input. The gain-only control uses only the current PN and fixes H=0.

Initial/final gain is 3.88/4.08. The gain case has H=0 throughout; the memory
case has H=0.12/0.24. Inversion is stable for these values; general dynamic
paths can have nonminimum-phase zeros or noise amplification and need a
different correction design. The model is a two-tap equivalent residue path,
not the full dielectric-relaxation/shared-opamp model of Keane et al.

Foreground identification uses a dedicated probe-off sine and fits
`r = -G*q - H*q_previous + A*sin(w*n)+B*cos(w*n)+DC`.
Frequency is known; amplitude, phase and DC are fitted. True x is not a label.
The first startup sample is excluded from this foreground regression.

## 3. Capacitor mismatch: random role switching and code-weighted correlation

Two physical capacitors are normalized to `C1=1+delta`, `C2=1-delta`.
PRBS p chooses `Cf=1-delta*p`, `Cs=1+delta*p`. The 1.5-bit coarse code q is
-1, 0 or +1, with thresholds -0.25/+0.25. The backend has 13 bits.
Ideal charge redistribution gives

```text
r = ((Cf+Cs)*x-Cs*q)/Cf = (2*x-(1+delta*p)*q)/(1-delta*p).
y_raw = (q+r)/2.
x = y_raw + delta*p*(q-y_raw), before backend noise/quantization.
```

To identify delta, use an observable instrument `z=(p-E[p|q])*q`:

```text
E[z*x] = 0 under independent switching.
delta = -E[z*y_raw] / E[z*p*(q-y_raw)].
```

Both moments use exponential averaging with factor 0.08. Mean p is computed
within each visited coarse code in the current block. This removes constant
within-code nuisance terms without accessing the true input. The initial/final
delta is 0.006/0.012; this is one capacitor-pair mismatch parameter, not a
Monte Carlo sigma for a multi-element DAC.

The unweighted control replaces z with centered p. For symmetric input its
denominator tends to zero, so it is held by a rank/excitation guard. Merely
averaging longer cannot create the missing information. If only q=0 is visited,
the selected code-weighted instrument also has no excitation and is held.

This example follows the published principle of capacitor switching plus
correlation, but differs from Sun/Lee/Ham's CFCS selection logic. We use exact
capacitor ratios and a derived observable moment equation, not their first-order
approximation. It is not a reproduction of their 15-stage ADC.

Foreground identification retains switching during a dedicated sine and fits
`y_raw+delta*p*(q-y_raw) = A*sin(w*n)+B*cos(w*n)+DC`, then freezes delta.

## 4. Nonlinear response: multiple probe levels and inverse moments

The coarse stage/backend match the gain model. The amplifier has a monotone
compressive characteristic implicitly defined by

```text
r + beta*r^3 = G*v, v=x-q+a*p.
v = c1*r+c3*r^3, c1=1/G, c3=beta/G.
```

The plant solves the forward equation with Newton iterations before adding
backend noise and quantization. This model deliberately has an exact cubic
inverse. An arbitrary amplifier does not; model mismatch remains future work.

Use four nominally equiprobable levels `p in {-1,-1/3,1/3,1}`, `a=0.1`.
Define features `phi=[r,r^3]` and instruments `Z=[p,p^3-(41/45)*p]`.
The coefficient 41/45 equals E[p^4]/E[p^2] for this distribution.
Independence of p from x-q gives two moment conditions:

```text
Cov(Z,phi) * [c1,c3] = Cov(Z,a*p).
y = q+c1*r+c3*r^3-a*p.
```

The matrix and RHS are exponentially averaged with factor 0.04. Instrument
rank, matrix condition and monotonicity/broad parameter bounds are checked.
Noise/quantization make these moments approximate; no claim of statistically
unbiased finite-sample estimates is made.

Initial G=3.88, beta=0.16; final G=4.08, beta=0.24. The scalar control uses
only p and r, and can fit only an effective gain. A binary p makes the two
selected odd instruments collinear and is explicitly rejected. This does NOT
prove that every nonlinear calibration must use four levels: binary schemes
can add windowing, other observables or structural assumptions.

This is our derived educational inverse-moment estimator. The cited parallel
correlation paper instead uses two asymmetric dither windows and a different
estimator; no exact reproduction is claimed. Foreground identification fits
`q+c1*r+c3*r^3 = A*sin(w*n)+B*cos(w*n)+DC` with probe off, then freezes c1/c3.

## Controls and coverage boundaries

`diagnostics.json` includes a one-sample reference delay error, unknown probe
amplitude, deliberately input-correlated probes, an adversarial short-period
input, binary-probe rank loss, insufficient capacitor-code coverage and physical
backend clipping. A guard can stop an invalid update; it cannot undo clipping.

The main runs isolate four mechanisms rather than combining all of them in
one ADC. They do not cover complete multi-stage joint identification, dynamic
reference settling, nonlinear memory, aperture jitter, metastability,
fixed-point implementation or chip area/power. `exp_cal01` remains the example
for continuous gain ramp tracking and probe-amplitude sweeps.
Band-limited non-subtractive dither, DEM/DNC and a full dual-dither circuit are
documented in the literature review but not numerically reproduced here.

## Primary references and implementation relationship

- [ADI, Internal Dither Improves High Speed ADC SFDR Performance](https://www.analog.com/en/resources/technical-articles/internal-dither-improves-high-speed-adc-sfdr-performance.html): practical subtractive-dither DAC matching; source for the randomization case.
- [ADI AN-835, Dither Testing](https://www.analog.com/en/resources/app-notes/an-835.html): wideband/out-of-band dither and static versus dynamic distortion.
- [TI SLAA617, section 3.4](https://www.ti.com/lit/pdf/SLAA617): input headroom, band-limited noise, measured harmonic improvements.
- [TI ADC32RF52, section 7.3.1.4](https://www.ti.com/lit/gpn/ADC32RF52): actual dither spectrum, amplitude/divider settings and sideband tradeoffs; not our plant.
- [Siragusa and Galton, JSSC 2004](https://ispg.ucsd.edu/uploads/2017/05/2004-JSSC-E.-Siragusa-A-Digitally-Enhanced-1.8-V-15-bit-40-MSampleps-CMOS-Pipelined-ADC.pdf): measured silicon using separate DAC noise cancellation and gain error correction.
- [Sun, Lee and Ham, TCAS-II 2008](https://people.seas.harvard.edu/~donhee/SLH_TCAS2.pdf): capacitor switching plus background correlation; related principle, different switching/estimator here.
- [Keane, Hurst and Lewis, TCAS-I 2006](https://doi.org/10.1109/TCSI.2005.858760): background calibration for memory effects. [Author publication index](https://www.ece.ucdavis.edu/~hurst/papers/).
- [Ali et al., VLSI Circuits 2016](https://doi.org/10.1109/VLSIC.2016.7573537): RF ADC with correlation-based gain/dynamic/memory calibration; abstract-level evidence, not a reproduced implementation.
- [He et al., ELEX 2025](https://www.jstage.jst.go.jp/article/elex/22/5/22_22.20240726/_article): calibration dither plus linearization dither and headroom-aware injection; circuit details differ from these examples.
- [Parallel correlation estimation, Microelectronics Journal 2026](https://doi.org/10.1016/j.mejo.2025.107044): nonlinear calibration using two asymmetric dither windows; publisher abstract/preview only, not implemented verbatim.
- [MathWorks, time-domain correlation analysis](https://www.mathworks.com/help/ident/ug/what-is-time-domain-correlation-analysis.html): correlation-based dynamic-system identification.

Source retrieval date: 2026-11-01. The date is not the publication date.
