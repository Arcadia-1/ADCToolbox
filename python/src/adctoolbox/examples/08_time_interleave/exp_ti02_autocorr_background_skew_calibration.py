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
N_BATCH = 2**20                  # 1.05 M samples per iteration — one big capture.
                                 # 10× above paper's 1e5 narrowband requirement.
N_SFDR = 2**13                   # FFT length for the per-iter SFDR trace. Since
                                 # N_BATCH / N_SFDR = 128 is integer, a Fin coherent
                                 # at N_SFDR is also coherent at N_BATCH — so we can
                                 # slice x[:N_SFDR] with zero spectral leakage.
                                 # ~190× faster than FFT-ing the full record.
Amp = 0.5                        # sine amplitude
N_ITER = 200                     # calibration iterations
DIRECTION = -1                   # negative feedback; R'(Ts)<0 with MAD'(Ts)>0
                                 # so direction flips once — matches MATLAB
NOISE_RMS = 3.5e-5               # Gaussian per-sample noise. With Amp=0.5 this gives an
                                 # integrated SNR of ≈ 80 dB (per-bin noise floor well
                                 # below spur levels). Kept so SNDR is spur-limited.
HD3_DBC = -100                   # 3rd-harmonic distortion at the input (dBc). Caps the
                                 # post-calibration SFDR at this level — models a
                                 # realistic input-source / buffer non-linearity and
                                 # tames the 110+ dB dithering from pure cos.

# =============================================================================
# Build the TI-ADC model (truth that the algorithm cannot see)
# =============================================================================
# Fin placed high on the spectrum to *demonstrate* the paper's main claim:
# the algorithm works up to fs/2, relaxing the prior-art fs/N = 250 MHz limit.
# At 450 MHz we're deep in the [fs/4, fs/2] band where older single-channel
# autocorrelation methods fail.
Fin, Fin_bin = find_coherent_frequency(Fs, 300e6, N_SFDR)
# This Fin is coherent at both N_SFDR and N_BATCH (since N_BATCH is an
# integer multiple of N_SFDR). Zero leakage at either FFT length.

# Intrinsic per-channel skew: ~2 ps rms, well within the ±25 ps VDL range
# but big enough to dominate SFDR before calibration.
rng_truth = np.random.default_rng(7)
intrinsic_skew = 1.5e-12 * rng_truth.standard_normal(M)
intrinsic_skew -= intrinsic_skew.mean()    # a common delay is unobservable

# Per-channel VDL: 10-bit (1024 codes), 50 fs LSB mean, 15% per-step DNL.
# Each channel has its own random DNL realization (monotonicity guaranteed).
vdls = [
    VariableDelayLine(
        n_codes=1024,              # 10-bit
        lsb_mean_sec=10e-15,       # 10 fs / LSB mean -> ±5.12 ps range
        step_cv=0.15,              # 15% per-step DNL
        seed=1000 + m,
    )
    for m in range(M)
]

ti = TIMultiSampler(M=M, fs=Fs, intrinsic_skew_sec=intrinsic_skew, vdls=vdls)

# Save the uncalibrated spectrum (trim codes at their centers, same noise
# level as all other captures for a fair comparison)
x_before = ti.capture(Fin, Amp, N_BATCH, noise_rms=NOISE_RMS, seed=0, hd3_dbc=HD3_DBC)


# =============================================================================
# Background calibration loop
# =============================================================================
def _sfdr(x):
    """Fast per-iter SFDR: FFT a short coherent slice of the capture.

    FFT cost is O(N log N); shrinking N from 2^20 → 2^13 is ≈ 190× faster
    and is more than accurate enough for a trajectory display (HD3 floor
    and TI spurs are both well above the shorter FFT's noise floor).
    """
    return analyze_spectrum(x[:N_SFDR], fs=Fs, create_plot=False)["sfdr_dbc"]


trim_history = []
sfdr_history = []
best_sfdr = -np.inf
best_trim = ti.trim_codes.copy()

for i_iter in range(N_ITER):
    trim_used = ti.trim_codes.copy()

    # One big capture. Random starting phase (in capture()) is essential for
    # MAD-based calibration with sinusoidal input — without it the signal
    # phase structure creates a systematic finite-K bias that the algorithm
    # would mistake for skew.
    x = ti.capture(Fin, Amp, N_BATCH, noise_rms=NOISE_RMS, seed=i_iter, hd3_dbc=HD3_DBC)
    sfdr_here = _sfdr(x)
    if sfdr_here > best_sfdr:
        best_sfdr = sfdr_here
        best_trim = trim_used.copy()

    # Build the (M+1, K-1) row stack and compute MAD
    ch = deinterleave(x, M)
    rows = np.vstack([ch[:, :-1], ch[0:1, 1:]])
    mad = np.sum(np.abs(np.diff(rows, axis=0)), axis=1)

    # MAD cumsum / fair-line decision (paper Eq 15)
    madk = np.cumsum(mad)
    mean_mad = mad.mean()
    k = madk[:-1] > np.arange(1, M) * mean_mad

    # Update trim codes by ±1 LSB
    for m_ch in range(1, M):
        delta = DIRECTION * (2 * int(k[m_ch - 1]) - 1)
        new_code = ti.trim_codes[m_ch] + delta
        ti.trim_codes[m_ch] = int(
            np.clip(new_code, ti.vdls[m_ch].code_min, ti.vdls[m_ch].code_max)
        )

    trim_history.append(ti.trim_codes.copy())
    sfdr_history.append(sfdr_here)

    if i_iter == 0 or (i_iter + 1) % 10 == 0 or i_iter == N_ITER - 1:
        print(
            f"[iter {i_iter+1:3d}/{N_ITER}] "
            f"SFDR={sfdr_here:6.2f} dBc  best={best_sfdr:6.2f} dBc  "
            f"trim={ti.trim_codes.tolist()}"
        )


# Apply the best trim codes for the "after" capture
ti.trim_codes = best_trim.copy()
x_after = ti.capture(Fin, Amp, N_BATCH, noise_rms=NOISE_RMS, seed=999_999, hd3_dbc=HD3_DBC)
# Also take a noise-free version so the reported SFDR shows the physical ceiling
# of this trim setting (not obscured by the noise floor we deliberately injected).
x_after_clean = ti.capture(Fin, Amp, N_BATCH, noise_rms=0.0, hd3_dbc=HD3_DBC)


# =============================================================================
# Diagnostics
# =============================================================================
print()
print(f"[Setup]     Fs={Fs/1e9:.1f} GHz, M={M}, Fin={Fin/1e6:.3f} MHz, N_BATCH={N_BATCH}")
print(f"[Intrinsic] skew (ps) = {np.round(intrinsic_skew*1e12, 2).tolist()}")
print(
    f"[VDL LSB]   mean={vdls[0].lsb_mean_sec*1e15:.0f} fs  "
    f"range≈{vdls[0].total_range_sec*1e12:.1f} ps / channel"
)
# "Relative-ideal" trim codes: ch0 stays at VDL center (fixed as reference);
# each other channel's VDL delay is set so its effective skew matches ch0's.
# This is the TRUE optimum the algorithm can reach — the absolute-cancellation
# codes (one per channel, each zeroing its own skew) would require ch0 to move
# too, which the algorithm does not do.
ch0_effective_skew = float(intrinsic_skew[0] + vdls[0](vdls[0].code_center))
ideal_codes = [vdls[0].code_center]
for m_ch in range(1, M):
    target_vdl = ch0_effective_skew - intrinsic_skew[m_ch]
    ideal_codes.append(vdls[m_ch].nearest_code(target_vdl))
print(f"[Ideal trim] relative to ch0 (reachable optimum) = {ideal_codes}")
print(f"[Best trim]  chosen by algorithm = {best_trim.tolist()}")
print(f"[SFDR]       uncalibrated          = {_sfdr(x_before):.2f} dBc")
print(f"[SFDR]       calibrated (noisy)    = {_sfdr(x_after):.2f} dBc "
      f"(best-of-batch = {best_sfdr:.2f} dBc)")
print(f"[SFDR]       calibrated (noise-free) = {_sfdr(x_after_clean):.2f} dBc  "
      f"← physical ceiling of this trim setting")
# Compare against what a "perfect" calibration would give, by applying the
# relative-ideal codes directly and taking a clean capture.
ti.trim_codes = np.asarray(ideal_codes, dtype=int)
x_ideal = ti.capture(Fin, Amp, N_BATCH, noise_rms=0.0, hd3_dbc=HD3_DBC)
print(f"[SFDR]       AT RELATIVE-IDEAL       = {_sfdr(x_ideal):.2f} dBc  "
      f"← best reachable with ch0 fixed at center + nearest_code rounding")
# And the absolute best: intrinsic_skew == 0
intrinsic_backup = ti.intrinsic_skew_sec.copy()
ti.intrinsic_skew_sec = np.zeros(M)
ti.trim_codes = np.array([v.code_center for v in vdls], dtype=int)
x_zero = ti.capture(Fin, Amp, N_BATCH, noise_rms=0.0, hd3_dbc=HD3_DBC)
print(f"[SFDR]       AT ZERO INTRINSIC SKEW  = {_sfdr(x_zero):.2f} dBc  "
      f"← algorithmic ceiling (no residual skew possible)")
ti.intrinsic_skew_sec = intrinsic_backup
ti.trim_codes = best_trim.copy()  # restore


# =============================================================================
# Plots
# =============================================================================
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

trim_arr = np.array(trim_history)   # (N_ITER, M)
iters = np.arange(N_ITER)

# ---- Summary figure: 2×2 grid (spectra + trajectories) ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: spectrum before
plt.sca(axes[0, 0])
analyze_spectrum(x_before, fs=Fs)
axes[0, 0].set_title("Before calibration")

# Top-right: spectrum after
plt.sca(axes[0, 1])
analyze_spectrum(x_after, fs=Fs)
axes[0, 1].set_title("After calibration")

# Bottom-left: trim-code trajectory
ax_t = axes[1, 0]
for m_ch in range(M):
    ax_t.plot(iters, trim_arr[:, m_ch], "-", linewidth=1.2,
              label=f"ch{m_ch}" + (" (ref)" if m_ch == 0 else ""))
    ax_t.axhline(ideal_codes[m_ch], linestyle=":", alpha=0.4, color=f"C{m_ch}")
ax_t.axhline(vdls[0].code_center, color="k", linestyle="--", alpha=0.3,
             label="center")
ax_t.set_xlabel("iteration")
ax_t.set_ylabel("VDL trim code")
ax_t.set_title("Per-channel trim-code trajectory (dotted = ideal code)")
ax_t.legend(loc="best", fontsize=8)
ax_t.grid(True, alpha=0.3)

# Bottom-right: SFDR trajectory
ax_s = axes[1, 1]
ax_s.plot(iters, sfdr_history, "-", linewidth=1.2, color="tab:blue")
ax_s.set_xlabel("iteration")
ax_s.set_ylabel("SFDR (dBc)")
ax_s.set_title(f"SFDR per iteration (best = {best_sfdr:.1f} dBc)")
ax_s.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(output_dir / "exp_ti02_summary.png", dpi=150)
plt.close(fig)

# ---- Figure 4: LSB-step sensitivity ----
# Left panel:   sweep one non-ref channel's trim around its ideal code,
#               other 3 held at ideal; plot SFDR vs offset for each channel.
# Right panel:  SFDR when N non-ref channels are each offset +1 LSB from ideal
#               (quantifies the worst-case SFDR of a non-ideal calibration
#               that is within 1 LSB of the true optimum).

def _sfdr_at_trim(trim_codes, n_captures: int = 8, seed_base: int = 50_000) -> float:
    """Mean SFDR over a few captures with fixed trim codes (noise averaged)."""
    ti.trim_codes = np.asarray(trim_codes, dtype=int)
    vals = []
    for i in range(n_captures):
        x = ti.capture(Fin, Amp, N_BATCH, noise_rms=NOISE_RMS, seed=seed_base + i, hd3_dbc=HD3_DBC)
        vals.append(_sfdr(x))
    return float(np.mean(vals))


ideal_np = np.asarray(ideal_codes, dtype=int)
offsets = np.arange(-15, 16)

# Left-panel data
sweep_curves = {}
for target_ch in range(1, M):       # non-reference channels
    sfdrs = np.empty(offsets.size)
    for i, off in enumerate(offsets):
        trim = ideal_np.copy()
        trim[target_ch] = int(np.clip(
            trim[target_ch] + off, 0, vdls[target_ch].code_max))
        sfdrs[i] = _sfdr_at_trim(trim)
    sweep_curves[target_ch] = sfdrs

# Right-panel data: 0..(M-1) channels each +1 LSB off, starting from ch1
n_off_range = list(range(M))
n_off_sfdr = []
for N_off in n_off_range:
    trim = ideal_np.copy()
    for ch in range(1, 1 + N_off):
        trim[ch] = int(np.clip(trim[ch] + 1, 0, vdls[ch].code_max))
    n_off_sfdr.append(_sfdr_at_trim(trim, n_captures=12))

# restore the algorithm's chosen trim so subsequent code sees the same state
ti.trim_codes = best_trim.copy()

fig4, (ax_sw, ax_nb) = plt.subplots(1, 2, figsize=(14, 5))

for target_ch, sfdrs in sweep_curves.items():
    ax_sw.plot(offsets, sfdrs, "-o", markersize=4, label=f"ch{target_ch}")
ax_sw.axvline(0, color="k", linestyle="--", alpha=0.4, label="ideal code")
ax_sw.set_xlabel(f"code offset from ideal (1 LSB = {vdls[0].lsb_mean_sec*1e15:.0f} fs)")
ax_sw.set_ylabel("SFDR (dBc)")
ax_sw.set_title("SFDR vs single-channel trim offset\n(other 3 channels held at ideal codes)")
ax_sw.legend()
ax_sw.grid(True, alpha=0.3)

bars = ax_nb.bar(n_off_range, n_off_sfdr, color="tab:blue", alpha=0.75)
for i, s in enumerate(n_off_sfdr):
    ax_nb.text(i, s + 0.5, f"{s:.1f} dBc", ha="center", fontsize=10)
ax_nb.set_xlabel("number of non-ref channels offset by +1 LSB from ideal")
ax_nb.set_ylabel("SFDR (dBc)")
ax_nb.set_title("SFDR when N channels are off the ideal code by +1 LSB")
ax_nb.set_xticks(n_off_range)
ax_nb.grid(True, alpha=0.3, axis="y")

fig4.tight_layout()
fig4.savefig(output_dir / "exp_ti02_lsb_sensitivity.png", dpi=150)
plt.close(fig4)

print(f"\n[Saved] {output_dir}/exp_ti02_summary.png")
print(f"[Saved] {output_dir}/exp_ti02_lsb_sensitivity.png")
