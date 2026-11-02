# HDC memory and capacitor-mismatch model basis

Scope: the joint HDC model of `exp_cal08`, the role-swapping capacitor models
of `exp_cal02` and `exp_cal06`, the waveform-error and SAR weight models already
in ADCToolbox, and the papers these models rely on.

The companion [switch-phase and memory-state derivation](hdc-physical-state-and-phases.md)
adds a plate connection table, finite-reset and finite-open-loop-gain
equations, and the conditions under which the PN path observes memory.

**There is no basis for replacing these models wholesale with the existing
ADCToolbox models. The capacitor models differ mainly because the circuits
differ; the current HDC memory model is mathematically self-consistent but has
weaker physical support, so its target state and approximation conditions must
be stated.** The existing ADCToolbox models also include simplified behavioral
models and are not a physical ground truth by default.

## 1. Criteria and conclusions

Evidence is weighed in this order: circuit mechanism, state and node,
mathematical equation, range of approximation, and only then code reuse. That a
calibrator can invert a formula does not show that the formula comes from the
target circuit; that a formula differs from the existing library does not show
that it is unreasonable.

| Model | Current evidence | Scope | Status |
|---|---|---|---|
| HDC one-tap residue memory | Explicit linear FIR; corresponds to the first-order expansion of a specific recursive residue path | Equivalent model when one tap dominates and PN and signal pass through the same dynamic path | Kept as a restricted behavioral model; not yet a general settling-error or dielectric-absorption model |
| ADCToolbox previous-MSB leakage | Explicit code-dependent additive error; no derivation from device parameters to the coefficient | Equivalent coupling caused by the previous coarse decision | Kept; not a direct replacement for residue memory |
| ADCToolbox incomplete sampling | Recursion with a tracking time and a time constant | Single-state settling with a target voltage roughly constant within the tracking window | For this mechanism its physical parameters are more directly interpretable; not equivalent to the reset memory of a whole MDAC |
| Role-swapping sampling/feedback capacitors | Ideal charge conservation directly yields the rational expression | 1.5-bit stage whose two capacitors swap roles | Clear structural basis, kept; the selection logic is not an item-by-item reproduction of the paper's CFCS |
| Current HDC fixed feedback capacitor with switched DAC elements | The current equation follows from the stated ideal charge-transfer convention | Fixed sampling/feedback ratio, one differential mismatch mode | Kept; does not generalize to all MDAC/CDAC mismatch |
| ADCToolbox SAR capacitor-weight mismatch | Independent weight perturbations scaled by unit-capacitor area, applied in successive approximation | SAR behavioral model with static effective weights | Better suited to SAR; not a complete switched-capacitor circuit solver |

## 2. What the papers support

### 2.1 HDC 2009

The Panigada/Galton model targets a memoryless, weakly nonlinear residue
amplifier; the chip calibrates DAC mismatch separately with DNC. The circuit
uses independent switched-capacitor DAC networks and resets the amplifier to
reduce memory effects. The paper supports the feasibility of HDC and of
independent DAC networks, **but does not directly validate the one-tap memory,
the two-element correlator or the combined parameter set added here**. See
Sections II-B, III-C and IV and Fig. 16. [Paper](https://ispg.ucsd.edu/uploads/2017/05/2009-JSSC-A.-Panigada-A-130-mW-100-MSps-Pipelined-ADC-With-69-dB-SNDR-Enabled-by-Digital-Harmonic-Distortion-Correction.pdf)

### 2.2 Keane/Hurst/Lewis 2006

Starting from dielectric relaxation, incomplete reset and parasitic coupling
through a shared op-amp, the paper derives a state equation that contains the
previous stage output and the previous DAC code. Its unified form is

\[
X_{i+1}(z)=\frac{G_i}{1-\gamma_i z^{-1}}
\left[X_i(z)-(1-\delta_{m,i}z^{-1})K_iD_i(z)\right].
\]

The paper's memory parameter is written here as \(\delta_m\) to avoid confusion
with the capacitor mismatch \(\delta\) of this repository. The paper's two-tap
FIR \(M_i(z)\) is an equivalent correction relation and cannot be used directly
as a forward stage model. See Eqs. (14), (24), (32) and (65)–(73).
[DOI](https://doi.org/10.1109/TCSI.2005.858760), [author preprint mirror](https://citeseerx.ist.psu.edu/document?doi=54e99e923feb71e2119dcd82b1cf57876baeec3d&repid=rep1&type=pdf)

### 2.3 Sun/Lee/Ham 2008

CFCS changes the feedback-capacitor selection to obtain different transfer
curves, then extracts the mismatch by correlation. Its capacitor selection
depends on the coarse decision and the CFCS mode, and the 1.5-bit case must
also handle crossing transfer curves. The role-swapping models share the
charge-conservation and random-switching principles but use a different
selection logic and estimator; the current fixed-feedback model is even further
from a direct reproduction of CFCS. See Figs. 1–5 and Sections II–IV; the paper
is validated by behavioral simulation. [Paper](https://people.seas.harvard.edu/~donhee/SLH_TCAS2.pdf)

The equivalences, error bounds and comparisons below are derived from the code;
they are not validation results that the papers give for this repository.

## 3. Memory models: mathematics and physics

### 3.1 What system the current HDC model is

In [hdc_fixed.py](../../python/src/adctoolbox/examples/11_calibration/hdc_fixed.py):

\[
v_n=x_n-\frac{(1+\delta s_n)q_n}{2}+d_n,
\quad t_n=v_n+h v_{n-1},\quad h=H/G,
\]
\[
r_n=g t_n+k t_n^3,\qquad g=G/2,
\]

Backend noise and quantization follow. The analog state stores **the residue
plus PN before the previous nonlinearity, \(v_{n-1}\)**, not the previous
actual amplifier output and not the previous coarse code.

For \(k=0\) the forward dynamic transfer is \(g(1+h z^{-1})\). After gain and
nonlinearity are removed, the corrector uses

\[
\hat v_n=\hat t_n-\hat h\hat v_{n-1}
\]

to implement the inverse system \(1/(1+h z^{-1})\). This inverse is stable for
\(|h|<1\), but a stable inverse does not mean that the forward model has been
validated against a circuit.

The model describes residual charge that acts as a one-tap linear coupling to
the previous residue, with older history negligible. The code gives no
quantitative mapping from a target circuit's capacitor states, reset phases
and time constants to \(H/G\); the companion derivation works out such a
mapping for one candidate connection, and there the stored state is not
\(v_{n-1}\). The model therefore cannot yet be identified as an exact model of
a particular device mechanism.

### 3.2 When it approximates the paper's recursive model

Writing the paper's unified model in the time domain, without nonlinearity:

\[
R_n=G(X_n-KD_n)+\gamma R_{n-1}
       +G\delta_m K D_{n-1}.
\]

Only when \(\delta_m=0\), with \(V_n=X_n-KD_n\), and when the PN also passes
through this residue path, does this give

\[
\frac{R_n}{G}=V_n+\gamma V_{n-1}
   +\gamma^2 V_{n-2}+\cdots.
\]

The current one-tap model can be the first-order truncation of this expansion
with \(h=\gamma\). If \(\delta_m\ne0\), an independent delayed DAC-code term
appears, which in general cannot be absorbed into the same \(hV_{n-1}\).

The comparison also explains the FIR/IIR difference: the inverse of the
recursive forward system \(1/(1-\gamma z^{-1})\) is the FIR
\(1-\gamma z^{-1}\), whereas the inverse of the current forward FIR
\(1+h z^{-1}\) is recursive. That both use two taps and both correlate delayed
PN does not make them the same.

For the first-order pole system assumed above, on the unit circle

\[
\left|\frac{1}{1-hz^{-1}}-(1+hz^{-1})\right|
\le\frac{|h|^2}{1-|h|}.
\]

| Parameters | \(h=H/G\) | Truncated second-tap coefficient \(h^2\) | Bound on the transfer difference |
|---|---:|---:|---:|
| INITIAL | 0.03093 | 0.0009565 | 0.0009871 |
| FINAL | 0.05882 | 0.0034602 | 0.0036765 |

Multiplying the bound by an amplitude bound of the input sequence gives the
corresponding time-domain truncation-error bound; it is neither an SNDR
prediction nor a measured circuit error. The FINAL one-tap coefficient is about
5.9%, so explaining it as a truncated weak pole requires evaluating the later
taps against the target precision; calling them second order is not enough to
neglect them.

### 3.3 The order of memory and nonlinearity is also a model assumption

The current model chooses \(r=f(v+h v_{-1})\). Expanding the cubic term:

\[
r=g v+ghv_{-1}+kv^3+3khv^2v_{-1}
  +3kh^2vv_{-1}^2+kh^3v_{-1}^3.
\]

This contains nonlinear memory cross terms, but all their coefficients are tied
to the same \(k,h\). It equals neither \(f(v)+h f(v_{-1})\) nor a nonlinear
settling process with independent states. The cascade order corresponds to a
circuit only if the linear history coupling occurs before the static
nonlinearity. The HDC cubic formula itself cannot establish this order.

### 3.4 The two existing ADCToolbox models are not automatically more realistic

`apply_memory_effect` in [nonidealities.py](../../python/src/adctoolbox/siggen/nonidealities.py)
actually computes

\[
m_n=\lfloor16x_n\rfloor/16,\qquad
\ell_n=\lfloor4096(x_n-m_n)\rfloor/4096,
\]
\[
y_n=m_n+\ell_n+\lambda m_{n-1}.
\]

It is an additive leakage that depends on the previous coarse code and suits
constructing that kind of code-dependent error; the coefficient \(\lambda\) is
not derived from switches, parasitic capacitance or a specific charge-transfer
process. The implementation also uses `np.roll`, so the first sample refers to
the last one. That is a periodic batch boundary, not the initialization of a
circuit with continuous state.

`apply_incomplete_sampling` computes, in DC-removed coordinates,

\[
\tau_n=\tau_0(1+c x_n^2),\qquad
a_n=e^{-T_{track}/\tau_n},\qquad
y_n=(1-a_n)x_n+a_ny_{n-1}.
\]

If the target voltage within each tracking window is approximately constant,
this is the discrete form of a single-time-constant settling equation; the
quadratic voltage dependence of \(\tau\) is an additional behavioral
assumption. Its state is the previous output, and its physical parameters are
easier to interpret than an arbitrary one-tap coefficient, but it does not
contain the full two-phase MDAC reset/amplify state or the DAC charge. The code
also has no explicit current limit or maximum-slope constraint, so the word
`slew-rate` in its docstring does not make it a hard slewing model.

**Assessment: for single-node settling, the existing recursive model fits the
problem better; for previous-coarse-code coupling, the existing coarse-code
model fits better; for one-tap residue coupling, the current model is the more
direct definition. The choice needs a circuit mechanism; it cannot be decided
by which model the existing calibrator removes more easily.**

## 4. Capacitor-mismatch models: mathematics and physics

### 4.1 Two-capacitor role swapping has a direct charge basis

`CapacitorPlant` in [dither_mechanisms.py](../../python/src/adctoolbox/examples/11_calibration/dither_mechanisms.py)
and [joint_dither_model.py](../../python/src/adctoolbox/examples/11_calibration/joint_dither_model.py)
use

\[
C_f=1-\delta s,\qquad C_s=1+\delta s.
\]

If both capacitors sample the input and then one becomes the feedback capacitor
while the other switches to the coarse reference level, the ideal charge
relation is

\[
C_fR=(C_f+C_s)x-C_s q,
\quad R=\frac{2x-(1+\delta s)q}{1-\delta s}.
\]

This is not a polynomial assembled for the corrector: the denominator comes
from the feedback capacitor, and the mismatch changes both the signal gain and
the DAC weight. Defining \(y=(q+R)/2\) and solving the equation directly gives

\[
x=y+\delta s(q-y).
\]

The exact inverse is a property of the ideal charge model and is not a circular
argument. The model is limited to an ideal op-amp, complete charge transfer,
fixed capacitor and reference values, and no parasitic or dynamic errors;
random role selection is also not the complete CFCS logic of the Sun paper.

The paper's ratio mismatch cannot be compared directly with the symmetric
parameter used here. If in some state \(C_s=1+\delta\) and \(C_f=1-\delta\),
then

\[
\Delta=C_s/C_f-1=2\delta/(1-\delta).
\]

### 4.2 The fixed-feedback model also has a conditional charge interpretation

[HDC_FIXED.md](../../python/src/adctoolbox/examples/11_calibration/HDC_FIXED.md)
uses the capacitor convention

\[
C_f=15C_u,\quad C_x=30C_u,\quad
C_{D,s}=15(1+\delta s)C_u,\quad C_{p,j}=2C_u.
\]

If the input sampling charge, the reference charge of the selected DAC element
and the three PN charges are transferred, with initial charges reset, then

\[
C_fB=C_xx-C_{D,s}q+\sum_j C_{p,j}p_j,
\]
\[
B=2x-(1+\delta s)q+\frac{2}{15}\sum_jp_j
=2(u+d).
\]

Because \(C_f\) is fixed, \(\delta\) does not enter the denominator; this
difference from the role-swapping model has a structural cause. After ideal
inversion, \(y=x-\delta s q/2\) directly gives the target term of the current
correlation estimate and digital compensation.

These capacitor ratios have a charge-level interpretation, but the
implementation has no switch-phase diagram that fixes the connection, reset
path and load of the unselected elements; the companion derivation proposes
one. The evidence level is therefore an ideal charge macromodel, not a verified
switched-capacitor implementation. If the actual selection changes the feedback
factor, load or settling time, gain, memory and PN amplitude can also depend on
\(s\); the current model does not describe these couplings.

### 4.3 Degrees of freedom left out by one differential mismatch parameter

Any pair of positive capacitors can be written as

\[
C_+=\bar C(1+\delta),\quad C_-=\bar C(1-\delta),
\quad \delta=(C_+-C_-)/(C_++C_-).
\]

so the symmetric form itself is reasonable. The model, however, also sets
\(\bar C/C_f\), \(C_x/C_f\) and the injection-capacitor ratios to nominal
values, so only the **differential mismatch mode** between the two DAC elements
is active.

For example, if \(\bar C/C_f=1+\epsilon_D\), the reconstruction also contains
\(-\epsilon_D q/2\) and products with the differential mode. This coarse-code
term generally cannot be removed by the overall gain or by the single
\(\delta\). Sampling, feedback and PN-injection capacitor errors can also
change the input ratio and the identifiable probe scale separately.

The model therefore cannot claim to cover independent random mismatch of every
capacitor, but studying one chosen mismatch mode is reasonable in itself.
Whether to add these degrees of freedom depends on the target capacitor network
and error budget.

### 4.4 Strengths and limits of the ADCToolbox SAR model

`sar_apply_cap_mismatch` in [sar.py](../../python/src/adctoolbox/models/sar.py)
uses

\[
w'_j=w_j(1+\epsilon_j),\qquad
\epsilon_j\sim\mathcal N(0,\sigma_u^2/N_j),
\]

and `sar_convert` then uses the true weights in successive approximation. This
shows how SAR weights change decisions more directly than adding an arbitrary
code error to the final output, and it suits statistical mismatch experiments
with many weights.

The function returns independently perturbed weights that are not
renormalized, however. For an ideal charge-redistribution array scaled by its
total capacitance, with \(w_j=C_j/C_T\), the first-order change is

\[
\frac{\Delta w_j}{w_j}\approx
\frac{\Delta C_j}{C_j}-\frac{\Delta C_T}{C_T}.
\]

The shared denominator correlates the weights; whether the total capacitance
includes dummy, parasitic or split-bridge capacitors depends on the structure.
This is a derivation about actual capacitor ratios, not a claim that the
library's weight model is necessarily defective: it can fold the common ratio
into the gain and model effective weights directly. It also does not solve
switch phases, reference dynamics or parasitic networks.

**Assessment: the ADCToolbox SAR model suits static SAR weight problems better,
and the two pipeline models here suit their own capacitor networks better. None
of the three can replace another unconditionally.**

## 5. What correlation identification additionally requires

A reasonable model and identifiable parameters are two independent conditions.

For the current model, \(w_n=v_n+h v_{n-1}\) after ideal inversion. For

\[
E[w_np_{1,n-1}]/A=h
\]

to hold, the corresponding cross moments of the delayed PN with the other
inputs and probe terms must vanish, and the PN must actually excite this memory
state. If the PN is injected after the memory source, or the memory comes
mainly from a coarse code on another capacitor, the correlation no longer
represents this physical parameter directly.

Similarly, for \(y=x-\delta s q/2\),

\[
\hat\delta=-2\frac{E[s q y]}{E[q^2]}
=\delta-2\frac{E[s q x]}{E[q^2]}.
\]

so \(E[sqx]=0\) and \(E[q^2]>0\) are required. In the implementation the coarse
decision precedes the independent element selection, which matches this
structural assumption; if the actual switch selection affects sampling or the
coarse decision, the condition must be checked again. Finite PN windows,
quantization correlation and approximate inversion still cause estimation
errors; the existing correlation experiments diagnose them numerically and do
not validate a device.

## 6. What is supported and what still needs evidence

The current code and results can be kept, stated precisely as follows: **for a
cascade of a single differential DAC mismatch with a fixed feedback capacitor,
one-tap residue memory and cubic nonlinearity, the fixed-point numerical
behavior of correlation calibration has been tested.**

Still to be supported:

1. Which capacitor or amplifier node the memory state comes from, and why it
   stores \(v_{n-1}\) rather than the actual output or the DAC code.
2. Whether the target circuit's reset/settling equations can be truncated to
   one tap over the present \(h\) range and precision requirement, and whether
   the PN passes through the same state.
3. The phase and connection table of the fixed-feedback scheme, especially how
   the unselected DAC element and the three injection capacitors connect and
   reset.
4. To claim general capacitor mismatch, whether errors of the mean DAC ratio,
   the sampling/feedback ratio and the PN ratio are calibrated separately; they
   cannot all be assumed to be part of \(\delta\).

The [companion derivation](hdc-physical-state-and-phases.md) answers items 1
and 3 for its candidate connection: the stored states are capacitor voltages
and the previous output, and the one-tap model is a conditional first-order
approximation. Items 2 and 4 remain open for a specific target circuit.

With that evidence, choose by mechanism. If the target state is previous-output
feedback from incomplete reset, its state equation decides whether the plant
should become recursive; if it really is one-tap residue coupling, the current
model can stay. For a swapped feedback capacitor use the role-swapping
equation; for an independent DAC with a fixed feedback capacitor, keep the
current charge form.

This requires neither larger experiments nor replacing the existing library
functions first. The next threshold is the physical state and phase relations,
not better-looking calibration results.

## 7. Checks

- The code was compared with the paper equations and circuit diagrams; the
  relevant HDC and CFCS figure pages were checked against local PDF renderings.
  The Keane author-site PDF returned HTTP 403, so its equations were checked
  against the readable author preprint mirror; figures that could not be read
  are not used as evidence.
- Exact rational arithmetic checks 216 charge-relation/inversion combinations,
  covering positive, negative and zero mismatch, two selections, three coarse
  codes and four PN sums. This verifies algebraic consistency only, not a
  physical implementation or calibration performance.
- From the repository root,
  `python docs/experiments/check_hdc_model_basis_20261102.py --output <new JSON path>`
  repeats these checks; it does not overwrite an existing record.
- The bound for truncating a first-order pole to a one-tap FIR is analytic; it
  is not an ADC measurement or a convergence experiment.
- The [audit record](hdc-model-basis-audit-20261102.json) stores the computed
  results and the SHA-256 of the five model sources examined.
