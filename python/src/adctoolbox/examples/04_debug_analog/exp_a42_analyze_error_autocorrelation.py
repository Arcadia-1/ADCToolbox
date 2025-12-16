"""Error autocorrelation: 12 different ADC non-idealities"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_coherent_frequency, amplitudes_to_snr, snr_to_nsd
from adctoolbox.aout.analyze_error_autocorr import analyze_error_autocorr
from adctoolbox.aout.fit_sine_4param import fit_sine_4param as fit_sine
from adctoolbox.siggen import ADC_Signal_Generator

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**13
Fs = 800e6
Fin_target = 80e6
Fin, J = find_coherent_frequency(Fs, Fin_target, N)
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

print(f"[Error Autocorrelation - 12 Cases] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]")
print(f"[Signal Parameters] A={A:.3f} V, DC={DC:.3f} V")

snr_base = amplitudes_to_snr(sig_amplitude=A, noise_amplitude=base_noise)
nsd_base = snr_to_nsd(snr_base, fs=Fs, osr=1)
print(f"[Base Signal] Noise RMS=[{base_noise*1e6:.2f} uVrms], Theoretical SNR=[{snr_base:.2f} dB], Theoretical NSD=[{nsd_base:.2f} dBFS/Hz]\n")

# Initialize signal generator
gen = ADC_Signal_Generator(N=N, Fs=Fs, Fin=Fin, A=A, DC=DC, base_noise=base_noise)

signals = []
titles = []

# 1. Noise (white, uncorrelated)
noise_rms = 180e-6
signal_noise = gen.apply_thermal_noise(noise_rms=noise_rms)
signals.append(signal_noise)
titles.append(f'1. Noise\nRMS={noise_rms*1e6:.0f}uV')

# 2. Jitter
jitter_rms = 2e-12
signal_jitter = gen.apply_jitter(jitter_rms=jitter_rms)
signals.append(signal_jitter)
titles.append(f'2. Jitter\n{jitter_rms*1e12:.1f}ps')

# 3. Static Nonlinearity (HD2+HD3)
hd2_dB, hd3_dB = -80, -66
signal_hd = gen.apply_static_nonlinearity(hd2_dB=hd2_dB, hd3_dB=hd3_dB)
signals.append(signal_hd)
titles.append(f'3. Static Nonlin\nHD2={hd2_dB}dB HD3={hd3_dB}dB')

# 4. Kickback
kickback_strength = 0.009
signal_kickback = gen.apply_kickback(kickback_strength=kickback_strength)
signals.append(signal_kickback)
titles.append(f'4. Kickback\nStrength={kickback_strength}')

# 5. AM Noise
am_noise_freq = 1e6  # 1 MHz modulation
am_noise_depth = 0.1
signal_am_noise = gen.apply_am_noise(am_noise_freq=am_noise_freq, am_noise_depth=am_noise_depth)
signals.append(signal_am_noise)
titles.append(f'5. AM Noise\n{am_noise_freq/1e6:.0f}MHz, {am_noise_depth*100:.0f}%')

# 6. AM Tone
am_tone_freq = 500e3  # 500 kHz modulation
am_tone_depth = 0.05
signal_am_tone = gen.apply_am_tone(am_tone_freq=am_tone_freq, am_tone_depth=am_tone_depth)
signals.append(signal_am_tone)
titles.append(f'6. AM Tone\n{am_tone_freq/1e3:.0f}kHz, {am_tone_depth*100:.0f}%')

# 7. Clipping
clip_level = 0.8
signal_clip = gen.apply_clipping(clip_level=clip_level)
signals.append(signal_clip)
titles.append(f'7. Clipping\nLevel={clip_level*100:.0f}%')

# 8. Drift (random walk / 1/f noise)
signal_drift = gen.apply_drift(drift_scale=5e-5)
signals.append(signal_drift)
titles.append(f'8. Drift\nRandom Walk')

# 9. Gain Error
gain_error = 0.03  # 3% gain error
signal_gain = gen.apply_gain_error(gain_error=gain_error)
signals.append(signal_gain)
titles.append(f'9. Gain Error\n{gain_error*100:.0f}%')

# 10. Glitch (periodic glitches)
glitch_period = 512  # Glitch every 512 samples
glitch_amplitude = 0.05
signal_glitch = gen.apply_glitch(glitch_period=glitch_period, glitch_amplitude=glitch_amplitude)
signals.append(signal_glitch)
titles.append(f'10. Glitch\nEvery {glitch_period} samples')

# 11. Dynamic Nonlinearity
signal_dynamic = gen.apply_dynamic_nonlinearity(T_track=(1/Fs)*0.2, tau_nom=40e-12, coeff_k=0.15)
signals.append(signal_dynamic)
titles.append(f'11. Dynamic Nonlin\nτ=40.0ps k=0.15')

# 12. Reference Error (DAC errors in reference)
ref_error_amplitude = 0.02
ref_error_freq = 2e6  # 2 MHz reference ripple
signal_ref = gen.apply_reference_error(ref_error_amplitude=ref_error_amplitude, ref_error_freq=ref_error_freq)
signals.append(signal_ref)
titles.append(f'12. Ref Error\n{ref_error_freq/1e6:.0f}MHz, {ref_error_amplitude*100:.0f}%')

# Create 3x4 subplot
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for i, (signal, title) in enumerate(zip(signals, titles)):
    # Fit sine and get error
    fit_result = fit_sine(signal, Fin/Fs)
    sig_fit = fit_result['fitted_signal']
    err = signal - sig_fit

    # Compute autocorrelation
    acf, lags = analyze_error_autocorr(err, max_lag=100, normalize=True)

    # Plot autocorrelation
    axes[i].stem(lags, acf, linefmt='b-', markerfmt='b.', basefmt='k-')
    axes[i].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[i].set_xlabel('Lag (samples)', fontsize=9)
    axes[i].set_ylabel('ACF', fontsize=9)
    axes[i].set_title(title, fontsize=10, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim([-100, 100])
    axes[i].set_ylim([-0.3, 1.1])

    # Print stats
    acf_0 = acf[lags==0][0]
    acf_1 = acf[lags==1][0] if len(acf[lags==1]) > 0 else 0
    print(f"  {i+1:2d}. {title.split(chr(10))[0]:25s} - ACF[0]={acf_0:6.3f}, ACF[1]={acf_1:7.4f}")

plt.tight_layout()
fig_path = output_dir / 'exp_a42_analyze_error_autocorrelation.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")

# ============================================================================
# 3-Plot Comparison: Thermal Noise vs Phase Noise vs Amplitude Noise
# ============================================================================

print("\n" + "="*80)
print("3-PLOT COMPARISON: Thermal Noise vs Phase Noise vs Amplitude Noise")
print("="*80 + "\n")

# Set seed for reproducibility
np.random.seed(42)

# Define noise levels
target_thermal = 150e-6
target_pm_rad = 200e-6
target_am = 100e-6

# Generate phase signals
phase_clean = 2 * np.pi * Fin * t

# Case 1: Thermal Noise Only (additive white noise)
n_thermal = np.random.randn(N) * target_thermal
sig_thermal_only = A * np.sin(phase_clean) + DC + n_thermal

# Case 2: Phase Noise Only (phase jitter)
n_pm = np.random.randn(N) * target_pm_rad
sig_pm_only = A * np.sin(phase_clean + n_pm) + DC

# Case 3: Amplitude Noise Only (amplitude modulation)
n_am = np.random.randn(N) * target_am
sig_am_only = (A + n_am) * np.sin(phase_clean) + DC

signals_3 = [sig_thermal_only, sig_pm_only, sig_am_only]
titles_3 = [f'Thermal Noise Only\n({target_thermal*1e6:.0f} µV)',
            f'Phase Noise Only\n({target_pm_rad*1e6:.0f} µV)',
            f'Amplitude Noise Only\n({target_am*1e6:.0f} µV)']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (signal, title) in enumerate(zip(signals_3, titles_3)):
    # Fit sine and get error
    fit_result = fit_sine(signal, Fin/Fs)
    sig_fit = fit_result['fitted_signal']
    err = signal - sig_fit

    # Compute autocorrelation
    acf, lags = analyze_error_autocorr(err, max_lag=100, normalize=True)

    # Plot autocorrelation
    axes[i].stem(lags, acf, linefmt='b-', markerfmt='b.', basefmt='k-')
    axes[i].axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[i].set_xlabel('Lag (samples)', fontsize=11)
    axes[i].set_ylabel('ACF', fontsize=11)
    axes[i].set_title(title, fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim([-100, 100])
    axes[i].set_ylim([-0.3, 1.1])

    # Print stats
    acf_0 = acf[lags==0][0]
    acf_1 = acf[lags==1][0] if len(acf[lags==1]) > 0 else 0
    print(f"  {title.split(chr(10))[0]:30s} - ACF[0]={acf_0:6.3f}, ACF[1]={acf_1:7.4f}")

plt.suptitle('Error Autocorrelation Comparison: Phase Noise vs Amplitude Noise vs Thermal Noise',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig_path = output_dir / 'exp_a42_analyze_error_autocorrelation_3plot.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[Save fig] -> [{fig_path}]\n")
