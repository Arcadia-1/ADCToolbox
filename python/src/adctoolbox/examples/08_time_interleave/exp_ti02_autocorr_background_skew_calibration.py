"""
Time-interleave demo: autocorrelation-based background skew calibration.

Models the technique described in [1]: during normal operation (no foreground
tone sweep, no offline calibration), measure the mean absolute difference
(MAD) between samples from adjacent sub-ADCs and use its cumulative pattern
across channels to steer per-channel delay-line trim codes.

Data flow each sweep:

    1. Capture a batch of records at the current trim codes
    2. Arrange as (M+1, K) rows:
        - rows 0..M-1: sub-ADC m samples at lag 0
        - row M:       sub-ADC 0 samples at lag 1 (wrap-around to next period)
    3. MAD_i = sum_over_batch |row[i+1] - row[i]|   (i = 0..M-1)
    4. Cumulative: MADk_i = sum(MAD[0..i])
    5. Fair line: (i+1) * mean(MAD). Deviation gives each channel a ±1 direction.
    6. Trim codes are shifted by ±1 LSB and clipped to the VDL range.

Reference
---------
[1] "A 1-GS/s 11-b Time-Interleaved SAR ADC With Robust Fast and Accurate
    Autocorrelation-Based Background Timing-Skew Calibration"
    (see references/ in the analog-agents project)

Ported from a hardware testbench MATLAB script — everything here is synthetic
so it runs without a chip. The VDL and TI-SAR models are in
``variable_delay_line.py`` next to this file.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from adctoolbox import (
    analyze_spectrum,
    deinterleave,
    find_coherent_frequency,
)

from variable_delay_line import TISARModel, VariableDelayLine


# =============================================================================
# Knobs
# =============================================================================
M = 4                            # number of sub-ADCs
Fs = 1e9                         # aggregate sample rate
N_FFT = 2**13                    # samples per capture
Amp = 0.5                        # sine amplitude
N_SWEEP = 120                    # calibration iterations
BATCH = 10                       # captures per iteration
DIRECTION = -1                   # MATLAB "direction" constant; ±1

# =============================================================================
# Build the TI-ADC model (truth that the algorithm cannot see)
# =============================================================================
# Fin close to Nyquist/4 -> strong sensitivity of MAD to timing error.
# Must still stay below per-channel Nyquist fs/(2M) = 125 MHz.
Fin, Fin_bin = find_coherent_frequency(Fs, 110e6, N_FFT)

# Intrinsic per-channel skew: small enough to fit inside a ±6 ps VDL trim
# range comfortably, so every channel has a reachable optimum.
rng_truth = np.random.default_rng(7)
intrinsic_skew = 2.0e-12 * rng_truth.standard_normal(M)
intrinsic_skew -= intrinsic_skew.mean()    # a common delay is unobservable

vdls = [
    VariableDelayLine(
        n_codes=128,
        lsb_mean_sec=100e-15,      # 100 fs / LSB mean -> ±6.4 ps range
        step_cv=0.15,              # 15% per-step DNL
        seed=1000 + m,
    )
    for m in range(M)
]

ti = TISARModel(M=M, fs=Fs, intrinsic_skew_sec=intrinsic_skew, vdls=vdls)

# Save the uncalibrated spectrum (trim codes at their centers)
x_before = ti.capture(Fin, Amp, N_FFT)


# =============================================================================
# Background calibration loop
# =============================================================================
def _sfdr(x):
    return analyze_spectrum(x, fs=Fs, create_plot=False)["sfdr_dbc"]


def _wrap_row_stack(channels_cat: np.ndarray) -> np.ndarray:
    """
    Build the (M+1, K-1) row stack that the MATLAB code uses:
    rows 0..M-1 take samples [0..K-2]; row M is channel 0 samples [1..K-1]
    (a 1-sample look-ahead for the wrap-around MAD term).
    """
    M_ = channels_cat.shape[0]
    out = np.empty((M_ + 1, channels_cat.shape[1] - 1), dtype=channels_cat.dtype)
    out[:M_] = channels_cat[:, :-1]
    out[M_] = channels_cat[0, 1:]
    return out


trim_history = []
sfdr_history = []
best_sfdr = -np.inf
best_trim = ti.trim_codes.copy()

for i_sweep in range(N_SWEEP):
    # --- collect a batch ---
    batch_chs = []          # each element: deinterleaved (M, K)
    batch_sfdrs = []
    trim_used = ti.trim_codes.copy()   # codes in effect for every capture in this batch
    for _ in range(BATCH):
        x = ti.capture(Fin, Amp, N_FFT)
        batch_chs.append(deinterleave(x, M))
        sfdr_here = _sfdr(x)
        batch_sfdrs.append(sfdr_here)
        # Record best inside the batch, matching MATLAB behavior
        if sfdr_here > best_sfdr:
            best_sfdr = sfdr_here
            best_trim = trim_used.copy()
    # Stack per-channel samples across the whole batch in time order
    channels = np.concatenate(batch_chs, axis=1)   # shape (M, BATCH * K)

    # --- MAD across adjacent rows, with ch0-shifted wrap ---
    rows = _wrap_row_stack(channels)
    mad = np.sum(np.abs(np.diff(rows, axis=0)), axis=1)    # length M
    madk = np.cumsum(mad)
    mean_mad = mad.mean()
    # k[i] for i in 0..M-2 drives channels 1..M-1 (channel 0 is reference)
    k = madk[:-1] > np.arange(1, M) * mean_mad

    # --- Update trim codes ±1 LSB ---
    for m_ch in range(1, M):
        delta = DIRECTION * (2 * int(k[m_ch - 1]) - 1)
        new_code = ti.trim_codes[m_ch] + delta
        ti.trim_codes[m_ch] = int(
            np.clip(new_code, ti.vdls[m_ch].code_min, ti.vdls[m_ch].code_max)
        )

    # --- log trajectory ---
    sfdr_sweep = float(np.max(batch_sfdrs))
    sfdr_history.append(sfdr_sweep)
    trim_history.append(ti.trim_codes.copy())

    if (i_sweep + 1) % 10 == 0 or i_sweep == 0:
        print(
            f"[sweep {i_sweep+1:3d}/{N_SWEEP}] "
            f"SFDR={sfdr_sweep:6.2f} dBc  best={best_sfdr:6.2f} dBc  "
            f"trim={ti.trim_codes.tolist()}"
        )


# Apply the best trim codes for the "after" capture
ti.trim_codes = best_trim.copy()
x_after = ti.capture(Fin, Amp, N_FFT)


# =============================================================================
# Diagnostics
# =============================================================================
print()
print(f"[Setup]     Fs={Fs/1e9:.1f} GHz, M={M}, Fin={Fin/1e6:.3f} MHz, N_FFT={N_FFT}")
print(f"[Intrinsic] skew (ps) = {np.round(intrinsic_skew*1e12, 2).tolist()}")
print(
    f"[VDL LSB]   mean={vdls[0].lsb_mean_sec*1e15:.0f} fs  "
    f"range≈{vdls[0].total_range_sec*1e12:.1f} ps / channel"
)
ideal_codes = [v.nearest_code(-s) for v, s in zip(vdls, intrinsic_skew)]
print(f"[Ideal trim] per-channel nearest = {ideal_codes}")
print(f"[Best trim]  chosen by algorithm = {best_trim.tolist()}")
print(f"[SFDR]       uncalibrated = {_sfdr(x_before):.2f} dBc")
print(f"[SFDR]       calibrated   = {_sfdr(x_after):.2f} dBc "
      f"(best-of-batch during calibration = {best_sfdr:.2f} dBc)")


# =============================================================================
# Plots
# =============================================================================
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

trim_arr = np.array(trim_history)   # (N_SWEEP, M)

# ---- Figure 1: calibration trajectory ----
fig1, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(14, 5))

for m_ch in range(M):
    ax_t.plot(trim_arr[:, m_ch], "-o", markersize=3, label=f"ch{m_ch}"
              + (" (ref)" if m_ch == 0 else ""))
    # dashed guideline at ideal code
    ax_t.axhline(ideal_codes[m_ch], linestyle=":", alpha=0.4,
                 color=f"C{m_ch}")
ax_t.axhline(vdls[0].code_center, color="k", linestyle="--", alpha=0.4,
             label="center")
ax_t.set_xlabel("sweep")
ax_t.set_ylabel("VDL trim code")
ax_t.set_title("Per-channel trim-code trajectory\n(dotted = ideal code, dashed = VDL center)")
ax_t.legend(loc="best", fontsize=8)
ax_t.grid(True, alpha=0.3)

ax_s.plot(sfdr_history, "-o", markersize=4, label="sweep SFDR")
ax_s.plot(np.maximum.accumulate(sfdr_history), "--s", markersize=4,
          label="best-so-far")
ax_s.set_xlabel("sweep")
ax_s.set_ylabel("SFDR (dBc)")
ax_s.set_title(f"SFDR trajectory — best = {best_sfdr:.1f} dBc")
ax_s.legend()
ax_s.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(output_dir / "exp_ti02_trajectory.png", dpi=150)
plt.close(fig1)

# ---- Figure 2: before / after spectrum ----
fig2, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
plt.sca(axes[0])
res_before = analyze_spectrum(x_before, fs=Fs)
axes[0].set_title(
    f"Before calibration — SNDR={res_before['sndr_dbc']:.1f} dBc, "
    f"SFDR={res_before['sfdr_dbc']:.1f} dBc, ENOB={res_before['enob']:.2f} b"
)
plt.sca(axes[1])
res_after = analyze_spectrum(x_after, fs=Fs)
axes[1].set_title(
    f"After background calibration (trim = {best_trim.tolist()}) — "
    f"SNDR={res_after['sndr_dbc']:.1f} dBc, "
    f"SFDR={res_after['sfdr_dbc']:.1f} dBc, ENOB={res_after['enob']:.2f} b"
)
fig2.tight_layout()
fig2.savefig(output_dir / "exp_ti02_before_after_spectrum.png", dpi=150)
plt.close(fig2)

# ---- Figure 3: VDL transfer curves (shows the non-ideality) ----
fig3, ax = plt.subplots(figsize=(8, 5))
codes = np.arange(vdls[0].n_codes)
for m_ch, v in enumerate(vdls):
    ax.plot(codes, v(codes) * 1e15, label=f"ch{m_ch}")
ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
ax.set_xlabel("trim code")
ax.set_ylabel("VDL delay (fs)")
ax.set_title(
    f"Per-channel VDL transfer functions "
    f"(monotonic; LSB ≈ {vdls[0].lsb_mean_sec*1e15:.0f} fs, step CV = 15 %)"
)
ax.legend()
ax.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig(output_dir / "exp_ti02_vdl_curves.png", dpi=150)
plt.close(fig3)

print(f"\n[Saved] {output_dir}/exp_ti02_trajectory.png")
print(f"[Saved] {output_dir}/exp_ti02_before_after_spectrum.png")
print(f"[Saved] {output_dir}/exp_ti02_vdl_curves.png")
