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

from variable_delay_line import TIMultiSampler, VariableDelayLine


# =============================================================================
# Knobs
# =============================================================================
M = 4                            # number of sub-ADCs (round-robin sampling)
Fs = 1e9                         # aggregate sample rate (1 GS/s — matches paper)
N_FFT = 2**13                    # samples per capture
Amp = 0.5                        # sine amplitude
N_SWEEP = 50                     # calibration iterations (matches MATLAB)
BATCH = 10                       # captures per iteration (matches MATLAB)
DIRECTION = -1                   # negative feedback; R'(Ts)<0 with MAD'(Ts)>0
                                 # so direction flips once — matches MATLAB

# =============================================================================
# Build the TI-ADC model (truth that the algorithm cannot see)
# =============================================================================
# Fin placed high on the spectrum to *demonstrate* the paper's main claim:
# the algorithm works up to fs/2, relaxing the prior-art fs/N = 250 MHz limit.
# At 450 MHz we're deep in the [fs/4, fs/2] band where older single-channel
# autocorrelation methods fail.
Fin, Fin_bin = find_coherent_frequency(Fs, 300e6, N_FFT)

# Intrinsic per-channel skew: ~2 ps rms, well within the ±25 ps VDL range
# but big enough to dominate SFDR before calibration.
rng_truth = np.random.default_rng(7)
intrinsic_skew = 2.0e-12 * rng_truth.standard_normal(M)
intrinsic_skew -= intrinsic_skew.mean()    # a common delay is unobservable

# Per-channel VDL: 10-bit (1024 codes), 50 fs LSB mean, 15% per-step DNL.
# Each channel has its own random DNL realization (monotonicity guaranteed).
vdls = [
    VariableDelayLine(
        n_codes=1024,              # 10-bit
        lsb_mean_sec=50e-15,       # 50 fs / LSB mean -> ±25.6 ps range
        step_cv=0.15,              # 15% per-step DNL
        seed=1000 + m,
    )
    for m in range(M)
]

ti = TIMultiSampler(M=M, fs=Fs, intrinsic_skew_sec=intrinsic_skew, vdls=vdls)

# Save the uncalibrated spectrum (trim codes at their centers)
x_before = ti.capture(Fin, Amp, N_FFT)


# =============================================================================
# Background calibration loop
# =============================================================================
def _sfdr(x):
    return analyze_spectrum(x, fs=Fs, create_plot=False)["sfdr_dbc"]


trim_history = []
sfdr_history = []
best_sfdr = -np.inf
best_trim = ti.trim_codes.copy()

for i_sweep in range(N_SWEEP):
    # --- Collect a batch. MATLAB-faithful arrangement: MAD is accumulated
    #     per-capture (rows [:, :-1] for all channels + ch0 shifted by 1
    #     sample WITHIN the same capture), never across captures.
    batch_sfdrs = []
    trim_used = ti.trim_codes.copy()   # codes in effect for every capture in this batch
    mad = np.zeros(M, dtype=float)

    for _ in range(BATCH):
        x = ti.capture(Fin, Amp, N_FFT)
        sfdr_here = _sfdr(x)
        batch_sfdrs.append(sfdr_here)
        if sfdr_here > best_sfdr:
            best_sfdr = sfdr_here
            best_trim = trim_used.copy()

        ch = deinterleave(x, M)            # (M, K) this capture only
        # Rows 0..M-1: this capture's channel-m samples [0..K-2]
        # Row M:       this capture's channel-0 samples [1..K-1]
        # (matches MATLAB's tmp(1:end-1) and data1_cal(2:end) per capture)
        rows = np.vstack([ch[:, :-1], ch[0:1, 1:]])     # (M+1, K-1)
        mad += np.sum(np.abs(np.diff(rows, axis=0)), axis=1)

    # --- MAD cumsum / fair-line decision (paper Eq 15) ---
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

ax_s.plot(sfdr_history, "-o", markersize=4)
ax_s.set_xlabel("sweep")
ax_s.set_ylabel("SFDR (dBc)")
ax_s.set_title(f"SFDR per sweep (best seen = {best_sfdr:.1f} dBc)")
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
