# Joint dither calibration experiment

`exp_cal06_joint_dither.py` places capacitor mismatch, gain error, one-sample
linear memory and memoryless nonlinearity in the **same conversion path**.
It tests observable background identification, inverse reconstruction, parameter
tracking, injection headroom and output spectra. It is an educational behavioral
model, not a public API or a reproduction of a published ADC circuit.

Run from a copied example tree (NumPy, SciPy and Matplotlib are required):

```bash
python 11_calibration/exp_cal06_joint_dither.py --output output/joint_dither
```

The default is 12 runs: two waveforms, constant/step parameters, three seeds.
Each has 256 blocks of 32768 samples. No foreground training or truth-based
initialization is used. At the nominal time scale of 100 MS/s this is 83.88608 ms;
all four physical parameters change at 41.94304 ms in step cases. This is an
artificial tracking stress, not a measured thermal drift rate.

Short CLI check (not a convergence qualification):

```bash
python 11_calibration/exp_cal06_joint_dither.py --blocks 16 --block-size 1024 --seeds 11 --waveforms sine --scenarios step --output output/joint_smoke
```

## Shared physical assumptions and injection coordinates

The coarse converter is a 1.5-bit stage with thresholds -0.25/+0.25 and DAC
decisions q in {-1,0,1}. Two physical normalized capacitors are 1+delta and
1-delta. An independent binary sequence s selects the feedback and sampling
roles: Cf=1-delta*s, Cs=1+delta*s. Under ideal charge redistribution,

```text
w = (2*x - (1+delta*s)*q) / (1-delta*s)
u = w/2
v = u + a*p
D[n] = G*v[n] + H*v[n-1]
r_analog + beta*r_analog^3 = D
r = Q13(r_analog + backend_noise)
```

The nominal gain is **2**, not 4: the same 1.5-bit two-capacitor charge stage
already supplies nominal gain 2. `u=w/2` is a normalization, not a second ADC
or an additional gain error. G describes the actual small-signal gain from v
to the compressed output with memory disabled. beta >= 0 makes compression
monotone. The cubic follows the memory path, so the two errors interact; the
previous v also contains the previous capacitor roles and injected probe.

Probe p uses two LFSRs: p=(2*p31+p29)/3, giving levels {-1,-1/3,1/3,1}.
Capacitor roles use a third, 27-bit LFSR. All states persist across blocks.
In the installed SciPy tap convention the polynomials are x^31+x^28+1,
x^29+x^27+1, and x^27+x^26+x^25+x^22+1. Seeds choose nonzero initial states;
finite windows are not IID and not exactly independent of the business input.

`a` is the known injection amplitude in **u=w/2 coordinates**. The default
peak a=0.2 corresponds to 0.4 in nominal residue w units. It is assumed to be
an ideal additive injection after charge redistribution, with the same ensuing
gain/memory/nonlinear path as u. An actual capacitor-injection DAC whose scale
changes with Cf would require a different model and estimator. This experiment
does not prove that a monolithic switched-capacitor circuit can vary G, H, delta
and beta independently or implement this injection without cost.

| Parameter | Initial | Final |
| --- | ---: | ---: |
| G | 1.94 | 2.04 |
| H | 0.06 | 0.12 |
| delta | 0.006 | 0.012 |
| beta | 0.16 | 0.24 |

The backend is a 13-bit midrise quantizer over [-1,1], with independent noise
RMS 0.0002 before quantization. Main input peak is 0.8. Single-tone frequency
is the nearest odd bin to 0.137 fs; the two-tone peaks are 0.46/0.34 at nearest
odd bins to 0.083/0.217 fs. The time scale does not model clock jitter, settling
within phases, reference dynamics, finite op-amp gain or nonlinear memory.

## Observable joint identification

Define c1=1/G, c3=beta/G and h=H/G. Before backend noise/quantization,

```text
c1*r[n] + c3*r[n]^3 - h*a*p[n-1] - a*p[n] = u[n] + h*u[n-1].
```

Thus present PN, present cubic PN and delayed PN provide three moment
conditions for three electronics coefficients:

```text
Z = [p, p^3-(41/45)*p, p_previous]
A = Cov(Z, [r, r^3, -a*p_previous])
b = Cov(Z, a*p)
A @ [c1,c3,h] = b, approximately.
```

This requires exogeneity of the probe from current and previous business
residues, not knowledge of the clean input. Capacitor switching is independent
of the additive probe, so capacitor mismatch enters the nuisance residues u;
it is not silently set to zero during electronics identification. The measured
matrix retains finite-window cross-correlations among the probe observables.

By default, each column of Z is centered within each observed (q,q_previous)
pair before forming the moments. This removes constant within-code nuisance
terms and requires no waveform frequency or clean label. It does **not** remove
all business-signal correlation or guarantee a small estimation variance.
The `unconditioned` control uses the same joint model without that centering.

A and b are exponentially averaged at rate 0.04. A rank guard rejects binary
PN, for which p^3=p, and missing probe excitation. The matrix condition limit
is 1e5. Accepted coefficients require .35<c1<.75, -.08<c3<.3, |h|<.2 and a
positive inverse slope over the backend range (minimum >.2).

The electronics inverse is stateful:

```text
v_hat[n] = c1_hat*r[n] + c3_hat*r[n]^3 - h_hat*v_hat[n-1]
y[n] = q[n]/2 + v_hat[n] - a*p[n]
x_hat[n] = y[n] + delta_hat*s[n]*(q[n]-y[n])
```

After this deembedding, delta uses the exact two-capacitor correction relation
and the code-conditioned instrument zc=(s-mean(s within q))*q:

```text
delta_hat = -EW_mean(zc*y) / EW_mean(zc*s*(q-y)).
```

Its moment averaging rate is also 0.04; |denominator|>=.01, instrument power
>=.05 and |delta_hat|<.05 are required. An outer code is needed for this chosen
instrument; visiting one is not sufficient near the full-scale limit where
q-y is small. No claim is made that every capacitor mismatch scheme has the
same excitation requirement.

All corrections and capacitor deembedding use estimates available at the
**start** of the block. Both electronics and capacitor updates prepare the
next block. The first cap update is held because it used nominal electronics.
An independent inverse-filter state is maintained for every comparison path.
Rejected rail blocks still advance the filter state; clipping is not undone.
The only capture fields are q, q_previous, r, p, p_previous, s and rail.
Analog peaks, overload counts, true coefficients and clean input are evaluator
data kept outside the tracker interface.

This is a newly derived triangular estimator combining the existing moment
principles. It does not concatenate unmodified gain-4 helpers with a gain-2
capacitor model, and does not assume four independent calibrators can estimate
their own parameters directly from the same uncorrected backend stream.

## Comparisons and convergence reporting

All probe-on methods use identical captures:

- nominal reconstruction;
- joint background calibration, with/without coarse-code conditioning;
- gain-only, no-memory, no-capacitor and no-nonlinearity **correction ablations**
  using the same learned joint coefficients, not separately trained methods;
- the same background estimate frozen at the midpoint (not a foreground fit);
- exact device coefficients (oracle, evaluation only).

A paired probe-off plant retains capacitor switching and identical noise draws.
Nominal and oracle probe-off paths show injection cost. Probe-off does not mean
zero capacitor mismatch or removal of the independent switching sequence.

Parameter reporting uses estimates actually applied, not just the final update.
Fixed evaluation bands are 1% relative error for G and 10% for H, delta and
beta. The summary reports final-quarter parameter means/RMSE, the fraction of
blocks in which **all four** meet these bands, and the end of the first eight
consecutive in-band blocks in each half. One successful streak is not proof of
permanent convergence; the final fraction must also be inspected. Truth is used
only to calculate these diagnostics, never to accept updates or choose rates.

FFT SNDR/SFDR are computed on coherent single-tone blocks with DC excluded,
and correct one-sided power weights. For two-tone inputs the names are total
desired-tone/noise+distortion and strongest desired-tone/largest unwanted-bin
ratios, not single-tone SNDR/SFDR. SFDR near a random noise floor depends on
record length. Error-SNR additionally includes gain, phase and DC error against
the evaluator's clean input. Window averages are means of per-block dB values.

## Headroom and failure controls

`headroom.csv` scans three input peaks (0.1,0.8,0.95), six injection amplitudes
(0,0.02,0.05,0.1,0.2,0.4) and both physical parameter settings. It records the
ideal cap residue peak, normalized injected-v peak, mathematical linear drive,
actual backend-input peak including noise, 1-peak|backend|, overload counts,
rail-code count and estimator status. The mathematical linear drive is not a
separately constrained physical node in this behavioral compression model.

`diagnostics.json` checks binary-probe rank loss, missing outer coarse codes,
actual clipping, incorrect PN delay and a 10% injection-scale error. The delay
control deliberately circularly shifts each block; it is not a physical
pipeline alignment model. The scale/delay controls use short records and
demonstrate failure risk, not a statistical qualification. Whole-block rail
rejection protects identification but cannot restore clipped output samples.

Increasing a improves the weak cubic moment while consuming headroom. The
default 0.2 was selected after development runs found 0.1 too weak in this
joint plant; neither amplitude guarantees convergence for arbitrary inputs.
The final report retains lower-amplitude controls rather than hiding this
dependence. The fixed parameter bands above are not relaxed to match results.

## Artifacts, provenance and limitations

Each run writes trajectory.csv and summary.json. Top-level outputs include
aggregate.csv, configuration.json, summary.json, diagnostics.json, headroom.csv,
joint_convergence.png (if sine/step was selected), and joint_headroom.png.
Nonempty destinations are rejected. Configuration includes actual source hashes,
git HEAD when available, Python/NumPy/SciPy/Matplotlib versions and all main
parameters; HEAD alone does not identify uncommitted source files.

The model assumes an exact cubic inverse, exactly known additive injection,
ideal two-capacitor charge redistribution and a stable two-tap residue path.
It does not cover fifth-order/model mismatch, nonlinear memory, multi-stage
coupling, input sampling noise, jitter, imperfect references, fixed-point
implementation or chip area/power. Backend quantization and post-nonlinearity
noise make the instrumental moments approximate. Passing algebraic identities
or numerical rank checks does not establish practical convergence.

The physical principles and their literature relationships are documented in
[DITHER_COVERAGE.md](DITHER_COVERAGE.md). This joint composition and estimator
are independently derived for this example. In particular, they are not the
CFCS selection logic of Sun/Lee/Ham, the physical memory correction network of
Keane/Hurst/Lewis, or the asymmetric-window nonlinear calibration in PCE.
