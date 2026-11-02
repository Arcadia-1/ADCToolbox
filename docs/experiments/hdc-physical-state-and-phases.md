# HDC fixed feedback capacitor: switch phases and memory state

Companion to the [memory and capacitor-mismatch model basis](hdc-model-basis.md).

**The current static capacitor-mismatch formula follows from an explicit, ideal
two-phase switched-capacitor connection; the current one-tap residue memory
formula is only a first-order approximation of a particular state equation and
does not represent the reset memory of this circuit in general.** When the PN
injection capacitors also retain reset residue, the PN memory response can even
vanish while the signal memory remains.

The phase table below is a circuit candidate for the current capacitor ratios.
It is not an existing transistor circuit and not a device-by-device
reproduction of the paper's circuit. The state equations are derived
independently from this table. They make the current behavioral assumptions
checkable but do not replace device, timing or noise verification.

## 1. Relation to the papers

- Figs. 13–16 and Section IV of Panigada/Galton 2009 support the structural
  principles of an independent sampling network, an independent DAC network, a
  fixed feedback capacitor and reset. Its Fig. 15 sets the DAC sign by
  differential swapping; to make charge directions explicit, this note uses an
  equivalent plate-voltage table instead of copying its switches. The paper's
  stage gain and DAC size also differ. [Paper](https://ispg.ucsd.edu/uploads/2017/05/2009-JSSC-A.-Panigada-A-130-mW-100-MSps-Pipelined-ADC-With-69-dB-SNDR-Enabled-by-Digital-Harmonic-Distortion-Correction.pdf)
- Eqs. (13)–(18) of Keane/Hurst/Lewis 2006 derive memory from the previous
  capacitor voltages and reset residue, and Eq. (32) gives a recursive stage
  model. The two-tap FIR of its Eqs. (69)–(73) is used for equivalent
  correction. This note uses the same state analysis but recomputes the
  coefficients for the fixed-feedback connection with separate DAC and PN
  capacitors below. [DOI](https://doi.org/10.1109/TCSI.2005.858760), [readable author preprint mirror](https://citeseerx.ist.psu.edu/document?doi=54e99e923feb71e2119dcd82b1cf57876baeec3d&repid=rep1&type=pdf)

## 2. A connection that realizes the current static formula

### 2.1 Coordinates, nodes and elements

The coordinates are differential-equivalent and normalized to the reference.
`0` denotes the corresponding common-mode potential, not a connection of every
terminal of the real differential circuit to ground; `+1/-1` denote the
positive and negative differential references. Real supplies, common modes and
switch voltage ratings are not chosen yet.

- `M`: op-amp summing node, voltage \(m\); ideally \(m=0\).
- `O`: stage output, voltage \(B\). The backend receives \(B\); the normalized
  residue in the code is \(r=B/2\).
- Each capacitor has plates `a` and `b` and state voltage \(w=V_a-V_b\); the
  charge on plate `a` is \(Cw\).
- \(C_x=30C_u\) and \(C_f=15C_u\).
- Two physical DAC capacitors, \(C_{D,+}=15(1+\delta)C_u\) and
  \(C_{D,-}=15(1-\delta)C_u\).
- Three PN capacitors, \(C_{p,j}=2C_u\), with \(p_j\in\{-1,+1\}\).
- \(q\in\{-1,0,+1\}\) comes from the coarse comparison of the same input
  sample; \(s\in\{-1,+1\}\) selects one DAC capacitor. Define
  \(U_j=q\,1_{\{j=s\}}\).

### 2.2 Plate connection table

Entries are `(plate a, plate b)`. Sampling/reset voltages are targets; Section 4
computes the deviation caused by a finite reset time.

| Element | Sampling/reset phase \(\phi_S\) | Charge-transfer/amplification phase \(\phi_A\) |
|---|---|---|
| Input capacitor \(C_x\) | `(0, x[n])`; isolated from `M` | `(M, 0)` |
| Fixed feedback capacitor \(C_f\) | `(0, 0)`; isolated from the output and reset | `(M, O)` |
| DAC capacitor \(C_{D,+}\) | `(0, 0)` | `(M, U_+[n])` |
| DAC capacitor \(C_{D,-}\) | `(0, 0)` | `(M, U_-[n])` |
| PN capacitor \(C_{p,1}\) | `(0, 0)` | `(M, -p_1[n])` |
| PN capacitor \(C_{p,2}\) | `(0, 0)` | `(M, -p_2[n])` |
| PN capacitor \(C_{p,3}\) | `(0, 0)` | `(M, -p_3[n])` |
| Op-amp | Output and internal nodes reset to their common modes; residue not read | Reset released; negative feedback forms and settles |

The unselected DAC capacitor stays connected between `M` and common mode; **it
is not removed from the summing node**. When \(q=0\), the `b` plates of both DAC
capacitors connect to common mode. The PN uses a negative sign so that the
probe appears with a positive sign in the final output.

The coarse comparator uses its own sampling path, or an equivalent held input,
with its sampling aperture aligned to \(C_x\). \(q,s,p_j\) stay constant within
the current amplification window. Mismatch between the two sampling paths,
reference dynamics, switch injection and parasitics are ignored here.

### 2.3 Connection constraints at the phase edges

1. At the end of \(\phi_S\), stop sampling: first release the plate clamps that
   hold the input charge and disconnect the input tracking path, so that the
   charge is held; the DAC/PN selection logic latches the current sample.
2. After a non-overlap interval, open the reset/common-mode clamps of the
   capacitors, release the op-amp reset and close the feedback connection of
   \(C_f\); then connect plate `a` of each charge-holding capacitor to `M` and
   switch plate `b` to the amplification-phase voltage in the table.
3. **Once a capacitor holding input charge connects to `M`, no reset switch
   that hard-clamps `M` to common mode may remain closed.** Otherwise the
   sampled charge flows into the clamp source and the conservation derivation
   below fails. Reference-selection switches must break before make and must
   not short two references.
4. After amplification settles, the backend samples; the output/feedback
   connection then opens before the next reset.

These are ideal phase sequences and charge-holding conditions; the actual
bottom-plate sampling edges, clock delays and internal op-amp reset remain part
of the circuit implementation. The table does not remove memory by resetting
`M` again during the amplification phase.

## 3. Static charge equation and finite gain

### 3.1 Ideal reset and ideal op-amp

Before transfer, the total charge on the `a` plates that will join `M` is

\[
Q_S=-C_xx_n.
\]

After complete settling, neglecting input current, parasitics and switch
injection:

\[
Q_A=-C_fB_n-\sum_j C_{D,j}U_{j,n}
              +\sum_j C_{p,j}p_{j,n}=Q_S.
\]

Hence

\[
B_n=2x_n-(1+\delta s_n)q_n+\frac{2}{15}\sum_{j=1}^3p_{j,n}
=2v_n,
\]
\[
v_n=x_n-\frac{(1+\delta s_n)q_n}{2}+d_n,
\qquad d_n=\frac1{15}\sum_jp_{j,n}.
\]

The Dither peak is therefore still 0.2 in the current input-equivalent
coordinate and 0.4 at the ideal stage output. This phase table does yield the
static capacitor terms of [hdc_fixed.py](../../python/src/adctoolbox/examples/11_calibration/hdc_fixed.py)
without changing the formula.

### 3.2 Why both DAC capacitors stay connected

With this connection, the total capacitance on `M` during amplification is

\[
C_T=C_f+C_x+C_{D,+}+C_{D,-}+\sum_jC_{p,j}=81C_u.
\]

It does not change with \(s,q,p_j\), and the ideal feedback factor is
\(\beta=C_f/C_T=5/27\). If only the selected DAC capacitor were connected to
`M`, the total capacitance would become \(66C_u+15\delta s C_u\), and finite
gain and bandwidth would vary with the selection. Keeping the unselected
capacitor connected is one concrete way to make the current assumption, that
mismatch changes only the DAC charge, more reasonable.

For a constant, positive open-loop gain \(A_{ol}\), let \(B=-A_{ol}m\). The
complete static charge equation is

\[
C_Tm-C_fB-D_n+P_n=Q_S,
\quad D_n=\sum_jC_{D,j}U_{j,n},\quad P_n=\sum_jC_{p,j}p_{j,n}.
\]

Defining \(L=C_f+C_T/A_{ol}\),

\[
B_n=\frac{C_xx_n-D_n+P_n}{L}.
\]

This is a common gain reduction of all injected charges, consistent with a
single common residue gain. With the nominal capacitor ratios above, however,
it can only give \(G=C_x/L<2\). The FINAL value \(G=2.04\) in the code cannot
be explained by a positive finite open-loop gain alone; it remains a behavioral
parameter that includes other gain sources. No additional circuit is assumed
here to match that number, and the stress-test parameter is unchanged.

## 4. Memory states actually stored by this connection

### 4.1 Starting from each capacitor's actual voltage

Take a simple linear RC reset/tracking as an example:

\[
\dot w_i=-(w_i-w_{i,target})/\tau_i,
\quad\rho_i=e^{-T_{S,i}/\tau_i},
\]
\[
w^S_{i,n}=w_{i,target,n}
          +\rho_i\left(w^A_{i,n-1}-w_{i,target,n}\right).
\]

This is the solution when the tracking target is approximately constant within
the window. The targets are

\[
w_{x,target}=-x_n,\qquad
w_{f,target}=w_{D,j,target}=w_{p,j,target}=0.
\]

From the amplification-phase connection of Section 2, the previous final
states are

\[
w_x^A=m,\quad w_f^A=m-B,\quad
w_{D,j}^A=m-U_j,\quad w_{p,j}^A=m+p_j.
\]

These states store, respectively, the summing-node voltage, the feedback
output, the capacitors' actual selection code and the PN reference voltage. No
capacitor stores the abstract variable \(v\) directly; \(v\) is an
input-equivalent charge combination, not the voltage on `M`.

The dynamics in this derivation come only from finite reset/tracking, assuming
the amplification phase settles fully. If amplification also settles
incompletely, the op-amp output and internal poles need additional states;
dielectric absorption should not be equated with this single-RC model either.

### 4.2 Recursion in the ideal virtual-ground limit

With \(m=0\), summing all \(C_iw^S_i\) and equating the sum with the charge at
the end of amplification gives

\[
\begin{aligned}
B_n={}&\frac{C_x}{C_f}(1-\rho_x)x_n
       -\frac{D_n}{C_f}+\frac{P_n}{C_f}
       +\rho_f B_{n-1}\\
&+\frac1{C_f}\sum_jC_{D,j}\rho_{D,j}U_{j,n-1}
 -\frac1{C_f}\sum_jC_{p,j}\rho_{p,j}p_{j,n-1}.
\end{aligned}
\]

The feedback capacitor stores the **previous actual output**; DAC residue adds
a positive delayed reference term; with the sign convention of this phase
table, PN residue adds a negative delayed PN term. The three cannot in general
be folded into one \(h v_{n-1}\). The previous DAC term contains the previous
physical selection \(s_{n-1}\), not the current one.

In the conventional MDAC of the Keane paper, the feedback and sampling
capacitors both settle toward the input voltage during reset; this candidate
circuit resets the feedback, DAC and PN capacitors toward zero differential
voltage. The paper's analysis method applies, but the coefficients of the
conventional MDAC cannot be copied.

### 4.3 The state equation stays complete with finite open-loop gain

Keeping \(A_{ol}\), \(C_i\) and \(\rho_i\) constant, define

\[
R_C=\sum_i C_i\rho_i,\qquad
L=C_f+C_T/A_{ol},\qquad
\gamma=\frac{C_f\rho_f+R_C/A_{ol}}{L}.
\]

Using the previous \(m_{n-1}=-B_{n-1}/A_{ol}\) eliminates the summing-node
state exactly:

\[
\begin{aligned}
B_n={}&\frac{C_x(1-\rho_x)}Lx_n
       -\frac{D_n}L+\frac{P_n}L+\gamma B_{n-1}\\
&+\frac1L\sum_jC_{D,j}\rho_{D,j}U_{j,n-1}
 -\frac1L\sum_jC_{p,j}\rho_{p,j}p_{j,n-1}.
\end{aligned}
\]

So even when feedback reset is the main source, \(h\), \(\rho_f\) and \(H/G\)
should not be treated as one physical quantity without derivation. For example,
with residue only on the feedback capacitor,

\[
\gamma=\rho_f\frac{C_f(1+1/A_{ol})}{L},
\]

which approaches \(\rho_f\) only in the large open-loop gain limit. These
equations hold for the listed phases, linear RC and constant open-loop gain;
they exclude slewing, switch nonlinearity and time-varying loads.

## 5. What delayed PN correlation can see

Under the linear time-invariant conditions above, the transfer from PN path
\(j\) to the output is

\[
\frac{B(z)}{P_j(z)}
=\frac{C_{p,j}}L\frac{1-\rho_{p,j}z^{-1}}{1-\gamma z^{-1}}.
\]

Here \(P_j(z)\) is the transform of one sign sequence, not the total injected
charge \(P_n\) used earlier. Normalized to the zero-delay PN gain, its impulse
coefficients are

\[
1,\quad\gamma-\rho_{p,j},\quad
\gamma(\gamma-\rho_{p,j}),\quad\ldots.
\]

Delayed correlation with ideal white PN therefore first measures
\(\gamma-\rho_{p,j}\), not in general the feedback reset residue \(\rho_f\). If
the three PN capacitors reset differently, they also see different delayed
responses.

A directly checkable counterexample is equal residue \(\rho\) on every
capacitor. Then even with finite \(A_{ol}\), \(\gamma=\rho\), and

\[
B(z)=\frac{C_x}L\frac{1-\rho}{1-\rho z^{-1}}X(z)
      -\frac{D(z)}L+\frac{P(z)}L.
\]

The PN/DAC zero cancels the feedback pole while the input path keeps recursive
memory. In this ideal single stage the memory appears as linear filtering of
the input, so harmonic distortion should not be claimed from it; in a cascaded
ADC its effect depends on the node. It is enough to show that **zero delayed
correlation on one PN path does not prove that the whole analog state is
memoryless.**

The counterexample does not claim that the target circuit behaves this way; it
states the physical condition the correlator needs: the dynamics of the probe
path must represent the signal path to be corrected.

## 6. When the current one-tap model is a reasonable approximation

### 6.1 Linear conditions

If input tracking is nearly complete, DAC/PN reset residue is negligible, the
main residue comes from the feedback state and the nonlinearity is weak enough,
the state equation reduces to

\[
B_n=Gv_n+\gamma B_{n-1}.
\]

With fixed parameters in steady state:

\[
\frac{B_n}G=v_n+\gamma v_{n-1}+\gamma^2v_{n-2}+\cdots.
\]

For \(k=0\) the code gives \(B_n/G=v_n+h v_{n-1}\), so \(h=\gamma\) can be read
as the first-order truncation of this expansion. The condition is more specific
than small memory: both the PN/DAC residue and the older output history must be
bounded.

If the real system is this single pole but the current ideal one-tap inverse is
still used, then

\[
\frac{\hat V(z)}{V(z)}
=\frac1{1+hz^{-1}}\frac1{1-hz^{-1}}
=\frac1{1-h^2z^{-2}}.
\]

Even with \(h\) estimated exactly, responses at two and more taps remain. This
is a structural error that further sample accumulation cannot remove.

| Code parameters | \(h=H/G\) | Forward tail, bound on the coefficient sum | After the current one-tap inverse, bound on the residual coefficient sum |
|---|---:|---:|---:|
| INITIAL | 0.03093 | 0.0009871 | 0.0009574 |
| FINAL | 0.05882 | 0.0036765 | 0.0034722 |

The bounds are \(h^2/(1-|h|)\) and \(h^2/(1-h^2)\); multiplied by
\(\|v\|_\infty\) they become voltage error bounds. They are neither actual SNDR
nor errors that an actual PN sequence necessarily reaches. For a given
tolerance \(\varepsilon_v\), check whether

\[
\frac{h^2}{1-|h|}\|v\|_\infty\le\varepsilon_v
\]

holds. Half an input-referred LSB of the 13-bit backend is \(1/(8192G)\);
calling the model a first-order approximation does not guarantee that the
truncation error of the current parameters stays below this level. Without that
error budget it is a mechanism example, not a high-precision physical
equivalent.

### 6.2 Cubic nonlinearity makes the stored quantity matter more

Let the memoryless static map be \(r=f(v)=gv+kv^3\) with \(g=G/2\). In an
input-equivalent approximation that ignores the finite summing-node voltage and
other additional states, feedback residue gives

\[
r_n=f(v_n+\rho_{eff}r_{n-1}).
\]

\(\rho_{eff}\) is an effective coefficient of charge retention and cannot be
equated with \(e^{-T/\tau}\) in every circuit. In the purely linear part
\(h=g\rho_{eff}\). Replacing the previous actual output by its memoryless
approximation gives

\[
v_n+\rho_{eff}r_{n-1}
=v_n+h v_{n-1}+\rho_{eff}k v_{n-1}^3+O(\rho_{eff}^2).
\]

The code's \(f(v_n+h v_{n-1})\) also omits the \(O(\rho_{eff}k)\) nonlinear
history term. Matching the current model exactly therefore requires storing
**the residue before the nonlinearity**, or showing that this term and the
recursive tail are both small enough. The network of Section 2 has no storage
node for that abstract residue, so it cannot be treated as implemented.

An exact one-tap analog feedforward history usually needs an additional
sample/hold or charge-storage path. That would be an extra circuit structure,
and it is not added here just to keep the existing formula.

## 7. Assessment of the current simplifications

| Assumption | Assessment | Basis and limits |
|---|---|---|
| Fixed \(C_f\); mismatch enters only the selected DAC charge | Holds under ideal static conditions | Follows directly from the phase table and charge conservation |
| Selecting a DAC element does not change the feedback load | Can hold, but the connection must be explicit | Both DAC capacitors always connect to `M`, the unselected one to common mode; excludes parasitics and reference-impedance differences |
| The three-PN peak is 0.2 | Holds for the nominal capacitor ratios | Input-equivalent amplitude per PN is \(1/15\); capacitor/reference errors not included |
| \(v_{n-1}\) represents reset memory exactly | Not in general | The actual states are capacitor voltages and the previous output, with separate DAC and PN delay terms |
| One-tap residue approximates recursive memory | Conditionally | Needs a specific dominant state, a commonly observed path and an explicit truncation-error budget |
| One delayed-PN correlation covers all memory | No | The PN path has a zero that can cancel the feedback pole; several states need not be identifiable from one quantity |
| The existing incomplete-sampling model can replace it directly | No | It is a single-node settling process and does not contain the multi-capacitor states and charge injection of this phase table |

The static capacitor model can therefore stay, with this phase table as its
conditional physical interpretation. The memory model can also keep its
experimental baseline, but its results remain limited to calibration of a
one-tap residue behavioral model, not verified calibration of the full reset
memory.

To reproduce the real reset process of this fixed-feedback capacitor, use the
state equations of Section 4 as the candidate physical reference before
deciding whether to change the model. If the goal is only to demonstrate the
one-tap mechanism, no code change is needed. **The examples implement neither
model change.**

## 8. Checks

- Exact rational arithmetic cross-checks the charge expressions of the phase
  table and the state recursions for ideal virtual ground and for finite
  open-loop gain.
- 128 state transitions are checked for each of four reset conditions (ideal,
  feedback residue only, equal residue on every capacitor, different residue
  per capacitor), 512 in total; 1536 finite-open-loop-gain state transitions
  and 384 static finite-gain charge relations are also checked.
- With equal residue, the PN/DAC delay cancellation agrees with the
  per-capacitor state computation in ideal virtual ground and under three
  finite open-loop gains. These are equation checks, not transistor simulation
  or calibration acceptance.
- The [audit record](hdc-physical-state-phases-audit-20261102.json) contains the
  numerical bounds, check counts and the hashes of the five model sources. Its
  script hash belongs to a copy of the
  [check script](check_hdc_physical_equations_20261102.py) without the
  `--output` option; the committed script differs from it only by that option,
  so a new record carries a different script hash.
- From the repository root,
  `python docs/experiments/check_hdc_physical_equations_20261102.py --output <new JSON path>`
  repeats these checks. The script first requires the five model sources to
  match the hashes in the model-basis audit record and stops otherwise; the
  current result is in [Dither validation](dither-validation.md).
- Figs. 13–16 of the HDC paper were checked against local renderings. The Keane
  author site could not be accessed directly, so its equations come from the
  readable preprint mirror; the candidate phases above do not paraphrase
  figures that could not be read.
