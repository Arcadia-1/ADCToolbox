# Supervised ADC post-correction: meaning, mathematics and evidence

## The short explanation

During calibration, supply measured ADC outputs and an aligned reference.
The polynomial or MLP learns a **digital output-to-reference mapping**, not
individual capacitor errors, bit weights or amplifier gains. During normal
operation, feed new ADC outputs through the **frozen mapping** to obtain corrected
samples. No reference, fitting, backpropagation or parameter update is needed then.
The MLP implementation adds a learned residual to a frozen affine baseline.

```text
Calibration: ADC output/history + aligned reference -> fit -> saved model
Operation:   ADC output -> history buffer -> frozen model -> downstream DSP
```

This is foreground digital post-correction, after ADC conversion and usually
before spectral analysis, measurement or demodulation. It does not change the
analog conversion circuit. A background algorithm that identifies physical gain
or bit weights is a different estimator, even if it also corrects digital data.

For `history=3`, the inputs are `[y[n], y[n-1], y[n-2], y[n-3]]`: four features
with three preceding samples. This does **not** mean a third-degree polynomial
or a third-order physical ADC. Polynomial degree and memory depth are separate.

## What is the reference?

| Reference source | Requirements and limits |
|---|---|
| Independent reference ADC or instrument | Characterize gain, offset, distortion, bandwidth and relative sampling delay; compare aligned samples. A slower reference can label aligned instants if excitation remains adequate. |
| Characterized calibration source | The waveform amplitude, phase/timebase and source errors must be known to the required accuracy. Generator settings alone are not an accuracy certificate. |
| Sine fit to the DUT's own output | A useful spectral-shape target under a sine assumption, but it retains the output's fundamental gain/phase and fitted DC. It does not independently establish absolute input accuracy. |

`exp_cal03` and `exp_cal05` use an independent 16-bit monitor followed by a
known-frequency least-squares projection. For one tone,

$$r[n]=A\sin(\omega n)+B\cos(\omega n)+C.$$

Only the frequency is supplied; A, B and C come from monitor observations.
For two tones, both sine/cosine pairs and one DC column are fitted. Projection
reduces random monitor noise under this signal assumption; systematic reference
errors remain. Arbitrary business signals cannot be labeled by a sine fit.
Known-frequency sine fitting and its quantization-bias limitations are discussed
in Kollár's [2004 paper, full text](https://www.imeko.org/publications/iwadc-2004/IMEKO-IWADC-2004-009.pdf).

The simulator's clean input is used only by the evaluator. Our monitor assumes
unity gain, synchronized sampling and no bandwidth or nonlinear error; it is a
behavioral reference, not measured reference hardware. The configured gain-bias
control explicitly demonstrates that monitor gain error enters the labels.

## Polynomial: solve coefficients, then evaluate

Let y be ADC output, r the aligned reference, and z the input features standardized
using training-only statistics:

$$z_k[n]=\frac{y[n-k]-\mu_k}{\sigma_k}.$$

The implementation replaces a numerically constant feature's scale with 1.
With M features and total degree d, include the intercept and every monomial:

$$\phi_{\alpha}(\mathbf z)=\prod_{k=0}^{M-1}z_k^{\alpha_k},
\qquad \sum_k\alpha_k\le d.$$

For two features and degree two, the row is
`[1, z0, z1, z0*z0, z0*z1, z1*z1]`. Four features and degree five need
126 coefficients; one feature and degree five need only six.

Stack rows into Phi and solve

$$c^*=\arg\min_c\left[\frac1N\|\Phi c-r\|^2+
\lambda\sum_{j>0}c_j^2\right].$$

This is nonlinear in the inputs but linear in the unknown coefficients.
`_ridge` solves an augmented least-squares system, with Phi/r divided by sqrt(N)
and a sqrt(lambda) diagonal penalty whose intercept entry is zero. It does not
explicitly invert the normal-equation matrix. The API default ridge is 1e-8;
the foreground benchmarks fix it at 1e-10.

In normal operation, compute the same standardized features and monomials:

$$\hat x[n]=\phi(\mathbf z[n])^T c^*.$$

There is no iterative inversion per sample. Coefficients describe the inverse
correction, not the forward circuit. For example, if y=g*x+k*x^3, its weakly
nonlinear inverse begins as x=y/g-k*y^3/g^4. Its cubic coefficient is not k.
Bit-weight calibration instead fits `offset + sum(weight_i * decision_i)` to
raw bit/segment decisions. Those are different features from powers of final code.

## MLP: learn a residual, then add it to the baseline

One hidden unit performs `h = tanh(w*z + beta)`. A single input is multiplied by
a trainable weight, shifted by a trainable bias, then passed through tanh.
Multiple units learn different nonlinear responses and the output combines them.
This is a feedforward MLP, not a recurrent network with additional hidden state.
See [Deep Learning, Chapter 6](https://www.deeplearningbook.org/contents/mlp.html)
for the general network definition and section 6.5 for backpropagation.

The actual implementation first fits and freezes an affine baseline:

$$x_{\mathrm{base}}[n]=a_{\mathrm{bias}}+\sum_{k=0}^{M-1}a_kz_k[n].$$

It uses training data and ridge=1e-10. Then construct

$$e[n]=r[n]-x_{\mathrm{base}}[n],\quad
s=\max(\operatorname{std}(e),10^{-12}),\quad t[n]=e[n]/s.$$

The network predicts this normalized residual using H hidden units:

$$h_j[n]=\tanh\left(\sum_kW_{kj}z_k[n]+\beta_j\right),\qquad
g[n]=\sum_jv_jh_j[n]+\gamma.$$

| Symbol | Meaning |
|---|---|
| y[n], r[n] | Observable DUT output and aligned training reference |
| z_k, mu_k, sigma_k | Standardized feature and its training mean/scale |
| a_bias, a_k | Affine intercept and feature weights; fitted before network training |
| x_base[n] | Baseline prediction, not a hidden-layer bias |
| e[n], s, t[n] | Reference-minus-baseline residual, its scale and normalized target |
| W_kj, beta_j | Input-to-hidden weight and hidden-unit bias |
| h_j[n] | Hidden-unit response, between -1 and 1 |
| v_j, gamma | Hidden-to-output weight and output bias |
| g[n] | Predicted normalized residual; its weighted sum is not bounded to [-1,1] |

Train W, beta, v and gamma to minimize

$$L=\frac1N\sum_n(g[n]-t[n])^2+
\lambda\left(\sum_{k,j}W_{kj}^2+\sum_jv_j^2\right).$$

Biases are not penalized. The MLP's default lambda is 1e-6 and applies to the
normalized residual loss; it is not equivalent to the polynomial's ridge prior.
For one example without the penalty, let epsilon=g-t. The chain rule gives

$$\frac{\partial L_n}{\partial v_j}=2\epsilon h_j,\qquad
\frac{\partial L_n}{\partial W_{kj}}=2\epsilon v_j(1-h_j^2)z_k.$$

The implemented gradients average these contributions over the batch and add
the weight-penalty derivatives. SciPy's
[L-BFGS-B optimizer](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
uses the loss and analytic gradient to update the parameters. The budget counts
optimizer iterations, not mini-batch epochs. The selected model minimizes
unpenalized reference MSE on separate validation records among observed iterates.
Reaching the iteration limit is not optimizer convergence or proof of overfitting.

At inference the baseline, residual scale and network weights are frozen:

$$\boxed{\hat x[n]=x_{\mathrm{base}}[n]+s\,g[n].}$$

For example, baseline=0.503 V, s=0.002 V and g=-1.5 give 0.500 V.
This is an illustrative calculation, not a measured result. With four inputs
and 16 hidden units there are 97 network parameters plus five baseline parameters;
normalization statistics are stored separately.

## Applicability and operating conditions

| Error / architecture context | Appropriate interpretation |
|---|---|
| Gain and offset | An affine model is usually sufficient. |
| Smooth, invertible static distortion in an ADC or its front end | Start with a static polynomial; MLP is an alternative learned mapping. |
| Repeatable short memory, such as acquisition/settling or reference-path effects | History models may help if features and excitation capture the dynamics. Our plant tests the repository's previous coarse-quantized-output feedback, including its coarse/fine quantization; it does not validate every settling mechanism. |
| SAR capacitor mismatch or Pipeline stage errors visible in decision bits | Physical bit-weight/stage-code methods may be preferable. Final-code models do not identify those internal parameters. |
| Sharp code-dependent INL | A global low-degree polynomial may underfit; consider characterized LUT/bit-weight methods. Not validated by this smooth plant. |
| Time-interleaved mismatch | Channel identity, alignment or separate channel mappings are generally needed. The default four-history example is not a validated TI calibration. |
| Long memory, nonminimum-phase dynamics, nonmonotonicity or overload | A short causal inverse may be inadequate or unavailable; lost information has no general inverse. |
| Random thermal noise or aperture jitter | No per-sample deterministic correction is guaranteed. Signal-prior denoising is a different claim. |
| PVT drift or changing sampling rate/bandwidth | Frozen models may become stale; revalidate/recalibrate. No online adaptation is implemented here. |

Training requires a dedicated reference capture and separate validation/test
records. Normal operation requires identical feature order, units, timing and
sampling conditions, but no reference. Keep history state only across contiguous
blocks; reset it for independent captures. Corrected output is in reference units.
The generic APIs can accept other feature matrices, but this does not validate
every ADC architecture or make unobservable errors identifiable.

The capture helper now calls the original signal-generator APIs: static
`u=x+.015*x^2+.04*x^3`, optionally `apply_memory_effect(u, memory_strength=.04)`,
noise, then a clipped 14-bit quantizer. The memory API combines coarse/fine
quantization with delayed coarse-code feedback; it is not the removed custom
linear FIR. Details and boundary handling are in `FOREGROUND_WORKFLOWS.md`.
The static transfer is monotonic, but the memory model has code discontinuities.
Reusing repository code does not establish physical fidelity or eliminate model
selection bias. In particular, low-order smooth distortion favors polynomial
approximation. Quantization and overload are not undone by a general inverse;
the real-valued output is an estimate. Independent circuit/measurement evidence
and device-parameter sweeps remain outside this behavioral comparison.

Single-tone training is especially restrictive for four history features. Ideal
zero-DC sine samples satisfy `x0*x2 + x2*x2 - x1*x1 - x1*x3 = 0`, obtained from
two sine recurrences. Distortion/noise can make the numerical design full rank
without providing robust joint excitation. Inspect `design_rank` and
`design_condition_number` (unregularized standardized design; None for numerical
rank deficiency), and test independent waveform families. Marginal range flags
are not joint-coverage guarantees. Even good conditioning is not a generalization
certificate. Static distortion does not automatically require history features.

## Physical evidence: what the papers do and do not establish

All sources below were read as full text. They establish related methods on
real converter data, not silicon validation of this repository's exact models.

| Original source | Physical evidence | Difference from this code |
|---|---|---|
| Vallant et al., 2011, [A Linearization Strategy for Undersampling ADCs](https://imeko.org/publications/iwadc-2011/IMEKO-IWADC-2011-04.pdf), sections III-V | LTC2208, 16-bit, 120 MS/s measurements; up to 25 dB SFDR improvement using dynamic nonlinear correction. Data processed in MATLAB; hardware implementation listed as future work. | Constrained Volterra/Hammerstein structure and dynamic-phase FIR design, not our full four-input fifth-degree ridge fit. |
| Bhanushali et al., [Circuits-Informed Machine Learning Technique for Blind Open-Loop Digital Calibration of SAR ADC](https://arxiv.org/pdf/2412.14051), author manuscript, pp. 1-3 | 28 nm, 12-bit, 84 MS/s main ADC and 10.5 MS/s reference ADC; shallow tanh error learning with history/derivative features. Near-Nyquist SNDR/SFDR improvements reported as 30.5/38.6 dB. | 41 features and 40 hidden neurons, separately calibrated reference, different training design. Network power is a digital-synthesis estimate, not measured full-calibrator silicon power. No support for calling our 4-by-16 baseline/residual combination a reproduction. |

The current NumPy/SciPy floating-point implementation has no measured ADC data,
fixed-point error budget, FPGA throughput result or on-chip area/power claim.
The literature's spectral improvements do not transfer to our model, nor do they
establish absolute accuracy under an uncharacterized reference channel.

Use packaged `FOREGROUND_WORKFLOWS.md` for runnable experiments,
reference-bias controls, frozen-model replay and the optional mixed-excitation MLP
comparison. Their numerical results are behavioral experiments with stated budgets.
