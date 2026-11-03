# Combined bit-weight and smooth-nonlinearity calibration

This experiment places **both errors in the same conversion path** and compares
uncorrected output, each correction alone, and their frozen cascade. It reuses
existing public error/model functions rather than introducing a new cubic model.
It is foreground behavioral validation, not a measured-chip reproduction or a
background adaptive algorithm.

## Run and replay

From the copied examples directory:

```bash
python 12_foreground/exp_cal11_combined_foreground.py --output output/combined_foreground
python 12_foreground/audit_combined_foreground.py --directory output/combined_foreground --output output/combined_audit.json
```

Defaults: binary and redundant SAR, seeds 11/22/33, 64 training records, eight
validation records, 8192 samples per record, fitting stride eight, and at most
1000 iterations for each MLP. A small functional run can use `--records 8
--validation-records 2 --size 2048 --iterations 3 --seeds 11`. Such a run is not
an acceptance-performance result. Nonempty output directories are rejected.

## Physical path and reused interfaces

```text
normal acquisition:
input x -> smooth analog front end f(x) -> mismatched SAR decisions b
        -> nominal or calibrated digital reconstruction -> optional inverse map

dedicated bit calibration:
calibration sine -> injection at SAR input (front end bypassed) -> decisions b
```

| Block | Repository implementation and settings |
| --- | --- |
| Smooth front end | `ADC_Signal_Generator.apply_static_nonlinearity(input_signal=x, k2=.015, k3=.04)`, DC=0: u=x+.015x²+.04x³ |
| SAR conversion | Public `sar_convert`, range [-1,1], actual MSB weight 1% above nominal |
| Capacitor structure | `sar_ideal_weights(14)` or `sar_ideal_weights(14, redundant_bit=2)`; the latter repeats the third largest capacitor, yielding 15 decisions |
| Nominal output | Public `sar_reconstruct` plus half a nominal LSB |
| Reference monitor | Independent synchronized 16-bit monitor, 2e-4 RMS noise; existing `apply_quantization_noise` plus half-LSB midpoint; known-frequency projection |
| Sample noise | Deterministic independent RNG, 1e-5 RMS at SAR input |

The repeated-weight normalization is sum(weights)+one LSB; the two architectures
therefore have different normalized LSBs. The same deterministic +1% MSB error
is used for every seed. Seeds vary stimulus, noise and training initialization,
not physical mismatch. No comparator noise, memory, reference ripple, timing
skew or PVT drift is included. Unlike `exp_cal03`, this experiment has unit
linear front-end gain because it directly uses the original static-error API.

All normal training, validation and test captures pass through **both** the smooth
front end and the mismatched SAR. Only the separately recorded bit-calibration
capture bypasses the smooth front end. True input and actual weights enter only
the simulated plant and evaluator, never the fitters.

## Identification sequence and observation cost

1. Acquire a separate coherent 0.9-peak sine at the **SAR input**, bypassing the
   nonlinear front end. Its frequency and noise seed are separate from end-to-end
   training, validation and test captures.
2. Call public `calibrate_weight_sine` with raw decisions, known frequency,
   `force_search=False`, `harmonic_order=1`. Scale its unit-sine output using the
   **measured reference-monitor sine amplitude**, not simulator truth. Include
   the monitor's fitted DC. Save and freeze the resulting weights and offset.
3. Acquire mixed single/two-tone records through the complete device. Construct
   both nominal output y and frozen bit-corrected output z from the same decisions.
4. Fit separate scalar inverse models y→r and z→r using the same aligned external
   reference r, rows, degree/network size and optimizer budget. Validation selects
   each MLP's iterate using only reference MSE. No test-driven selection occurs.

The primary bit stage implements `z=offset+sum(w_i*b_i)`. The polynomial then
evaluates `x_hat=P5(z)`. The MLP evaluates a frozen affine baseline plus a learned
residual, exactly as the public fitter specifies. This is **sequential calibration
and cascaded correction**, not joint gradient optimization of weights and network.

The bit paths require **one extra full record (8192 points at defaults)** and
access to internal decisions plus a SAR-input calibration injection/bypass path.
The direct polynomial/MLP paths receive the nominal scalar output only. All
nonlinear fitters share the same end-to-end records, but this is not an equal
total-observation or equal hardware-access comparison.

The bypass is a material applicability condition: it must leave the normal SAR
weights/reference behavior representative. An ADC without this calibration access
cannot be assumed to implement the protocol. If the nonlinearity resides inside
the converter or residue amplifier, stage-specific observations/calibration modes
or a joint identifiable model may be needed instead.

## Methods and controls

| Saved method | Frozen operation |
| --- | --- |
| `raw` | Nominal SAR reconstruction |
| `bits_only` | Dedicated-calibration weights, no smooth inverse |
| `poly_only` | Fifth-degree polynomial of nominal scalar output |
| `mlp_only` | MLP of nominal scalar output |
| `bits_poly` | Dedicated bit weights, then fifth-degree polynomial |
| `bits_mlp` | Dedicated bit weights, then MLP |
| `end_to_end_bits` | Control: affine fit of current decisions directly to complete-chain reference, without bypass calibration |
| `end_to_end_bits_poly` | Control: freeze those end-to-end weights, then fit a polynomial |
| `known_weights_inverse` | Evaluator-only actual-weight reconstruction followed by exact inverse of the known cubic |

The no-bypass control tests whether independently fitting both stages to total
error is equivalent to isolating bit error first. Its weights can absorb smooth
distortion and need not represent physical capacitor weights. It is not the
same calibration protocol as the primary cascade.

The known-weight inverse is a diagnostic control, **not** an optimal statistical
bound. Strict binary conversion gaps and quantization remain. The exact cubic
coefficients are used only in this labeled evaluator path.

## Model size, training and measurements

This device has static errors, so polynomial and MLP use **one current scalar
input, no history**. Each polynomial has six coefficients. Each MLP has 16 tanh
hidden units, 49 network parameters plus two frozen affine coefficients. This
differs from the earlier four-input history MLP; do not interpret the comparison
as an isolated change to training iterations on the previous network.

Training has 65,536 selected rows at defaults; validation has 8192. Peak amplitude
budgets span 0.05–0.90 and tone frequencies approximately 0.02–0.45 fs. Half the
records are single-tone and half two-tone. All feature standardization uses
training statistics. Polynomial ridge=1e-10; MLP normalized-residual L2=1e-6.
Report optimizer status; reaching the iteration limit does not mean convergence.

Every case evaluates the same 15 single-tone amplitude/frequency combinations
as `exp_cal10`, plus two-, three- and five-tone records. The single-tone set has
amplitudes 0.1/0.5/0.85 and frequencies approximately 0.031/0.113/0.239/0.371/0.467
fs; the last frequency is outside the training band. All test predictions are
from JSON-reloaded frozen models. No test gain, phase or offset adjustment is made.

Single-tone SNDR includes all unwanted non-DC power, including harmonics; SFDR
uses the largest unwanted spectral bin. Coherent rectangular-window FFTs span
the full Nyquist band. Public `analyze_spectrum` cross-checks single-tone results.
Multi-tone ratios follow `FOREGROUND_BITWEIGHTS.md` and are labeled separately,
with single-tone SNDR/SFDR fields empty. Absolute error SNR remains available.
Summaries average conditions within each seed, then report seed mean/sample SD.

## Evidence and limits

Joint treatment of capacitor/gain error and smooth nonlinearity has measured
precedents. Verma and Razavi's [JSSC 2009 paper](https://www.seas.ucla.edu/brweb/papers/Journals/V%26R09.pdf),
sections II–III and V, combines programmable stage coefficients with third-order
inverse correction in a 90-nm, 10-bit, 500-MS/s prototype. Its architecture and
calibration sequence differ from this SAR-input bypass experiment.

The publisher's public abstract/highlights for Li et al.,
[Integration 2026](https://www.sciencedirect.com/science/article/pii/S0167926026001574),
describe joint PGA second-harmonic and SAR bit-weight calibration on a 180-nm
chip. Only the publicly available description was checked; its full algorithm
was not reproduced here. Neither source validates this repository's exact model.

Applicability is limited to a stable, invertible smooth front end, a SAR with
observable decisions and suitable calibration access, characterized aligned
reference measurements, and sufficient conversion redundancy/headroom. This
experiment does not show recovery from clipping or arbitrary missing-code gaps,
nor does it cover long memory, random jitter, drift or uncharacterized reference
errors. It provides no fixed-point, area/power or real-time throughput result.

Artifacts include complete model dictionaries, capture manifests, training
histories, per-condition metrics, source snapshots and hashes. The audit replays
all test captures, verifies split separation and validation selection, confirms
that models remain frozen, and independently recomputes ratios with a full FFT.
