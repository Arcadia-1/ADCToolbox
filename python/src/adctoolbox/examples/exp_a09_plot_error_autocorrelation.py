"""Error autocorrelation: 12 different ADC non-idealities"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from adctoolbox import find_bin, plot_error_autocorr
from adctoolbox.common.sine_fit import sine_fit

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# Parameters
N = 2**13
Fs = 800e6
Fin_target = 80e6
J = find_bin(Fs, Fin_target, N)
Fin = J * Fs / N
t = np.arange(N) / Fs
A, DC = 0.49, 0.5
base_noise = 50e-6

print(f"[Error Autocorrelation - 12 Cases] [Fs = {Fs/1e6:.0f} MHz, Fin = {Fin/1e6:.1f} MHz, N = {N}]\n")

signals = []
titles = []

# 1. Noise (white, uncorrelated)
noise_rms = 180e-6
signal_noise = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * noise_rms
signals.append(signal_noise)
titles.append(f'1. Noise\nRMS={noise_rms*1e6:.0f}uV')

# 2. Jitter
jitter_rms = 2e-12
phase_jitter = np.random.randn(N) * 2 * np.pi * Fin * jitter_rms
signal_jitter = A * np.sin(2*np.pi*Fin*t + phase_jitter) + DC + np.random.randn(N) * base_noise
signals.append(signal_jitter)
titles.append(f'2. Jitter\n{jitter_rms*1e12:.1f}ps')

# 3. Static Nonlinearity (HD2+HD3)
hd2_dB, hd3_dB = -80, -66
hd2_amp = 10**(hd2_dB/20)
hd3_amp = 10**(hd3_dB/20)
k2 = hd2_amp / (A / 2)
k3 = hd3_amp / (A**2 / 4)
sinewave = A * np.sin(2*np.pi*Fin*t)
signal_hd = sinewave + k2 * sinewave**2 + k3 * sinewave**3 + DC + np.random.randn(N) * base_noise
signals.append(signal_hd)
titles.append(f'3. Static Nonlin\nHD2={hd2_dB}dB HD3={hd3_dB}dB')

# 4. Kickback
kickback_strength = 0.009
t_ext = np.arange(N+1) / Fs
sig_clean_ext = A * np.sin(2*np.pi*Fin*t_ext) + DC + np.random.randn(N+1) * base_noise
msb_ext = np.floor(sig_clean_ext * 2**4) / 2**4
lsb_ext = np.floor((sig_clean_ext - msb_ext) * 2**12) / 2**12
msb_shifted = msb_ext[:-1]
msb = msb_ext[1:]
lsb = lsb_ext[1:]
signal_kickback = msb + lsb + kickback_strength * msb_shifted
signals.append(signal_kickback)
titles.append(f'4. Kickback\nStrength={kickback_strength}')

# 5. AM Noise
am_noise_freq = 1e6  # 1 MHz modulation
am_noise_depth = 0.1
am_env = 1 + am_noise_depth * np.sin(2*np.pi*am_noise_freq*t)
signal_am_noise = A * np.sin(2*np.pi*Fin*t) * am_env + DC + np.random.randn(N) * base_noise
signals.append(signal_am_noise)
titles.append(f'5. AM Noise\n{am_noise_freq/1e6:.0f}MHz, {am_noise_depth*100:.0f}%')

# 6. AM Tone
am_tone_freq = 500e3  # 500 kHz modulation
am_tone_depth = 0.05
am_tone_env = 1 + am_tone_depth * np.sin(2*np.pi*am_tone_freq*t)
signal_am_tone = A * np.sin(2*np.pi*Fin*t) * am_tone_env + DC + np.random.randn(N) * base_noise
signals.append(signal_am_tone)
titles.append(f'6. AM Tone\n{am_tone_freq/1e3:.0f}kHz, {am_tone_depth*100:.0f}%')

# 7. Clipping
clip_level = 0.8
sinewave_clip = A * np.sin(2*np.pi*Fin*t)
signal_clip = np.clip(sinewave_clip, -clip_level*A, clip_level*A) + DC + np.random.randn(N) * base_noise
signals.append(signal_clip)
titles.append(f'7. Clipping\nLevel={clip_level*100:.0f}%')

# 8. Drift (random walk / 1/f noise)
# Create low-frequency drift using cumulative sum of small random steps
drift_steps = np.random.randn(N) * 5e-5
drift_walk = np.cumsum(drift_steps)
# Low-pass filter to make it smoother
from scipy import signal as scipy_signal
b, a = scipy_signal.butter(2, 0.001)  # Very low cutoff frequency
drift = scipy_signal.filtfilt(b, a, drift_walk)
signal_drift = A * np.sin(2*np.pi*Fin*t) + DC + drift + np.random.randn(N) * base_noise
signals.append(signal_drift)
titles.append(f'8. Drift\nRandom Walk')

# 9. Gain Error
gain_error = 0.03  # 3% gain error
signal_gain = (1 + gain_error) * A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * base_noise
signals.append(signal_gain)
titles.append(f'9. Gain Error\n{gain_error*100:.0f}%')

# 10. Glitch (periodic glitches)
glitch_period = 512  # Glitch every 512 samples
glitch_amplitude = 0.05
signal_glitch = A * np.sin(2*np.pi*Fin*t) + DC + np.random.randn(N) * base_noise
glitch_indices = np.arange(0, N, glitch_period)
signal_glitch[glitch_indices] += glitch_amplitude
signals.append(signal_glitch)
titles.append(f'10. Glitch\nEvery {glitch_period} samples')

# 11. Dynamic Nonlinearity
T_track = (1 / Fs) * 0.2
tau_nom = 40e-12
coeff_k = 0.15
vout = np.zeros(N)
v_prev = 0
for n in range(N):
    v_target = sinewave[n]
    tau_dynamic = tau_nom * (1 + coeff_k * v_target**2)
    vout[n] = v_target + (v_prev - v_target) * np.exp(-T_track / tau_dynamic)
    v_prev = vout[n]
signal_dynamic = vout + DC + np.random.randn(N) * base_noise
signals.append(signal_dynamic)
titles.append(f'11. Dynamic Nonlin\nτ={tau_nom*1e12:.0f}ps k={coeff_k}')

# 12. Reference Error (DAC errors in reference)
ref_error_amplitude = 0.02
ref_error_freq = 2e6  # 2 MHz reference ripple
ref_error = ref_error_amplitude * np.sin(2*np.pi*ref_error_freq*t)
signal_ref = A * np.sin(2*np.pi*Fin*t) * (1 + ref_error) + DC + np.random.randn(N) * base_noise
signals.append(signal_ref)
titles.append(f'12. Ref Error\n{ref_error_freq/1e6:.0f}MHz, {ref_error_amplitude*100:.0f}%')

# Create 3x4 subplot
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for i, (signal, title) in enumerate(zip(signals, titles)):
    # Fit sine and get error
    sig_fit, _, _, _, _ = sine_fit(signal, Fin/Fs)
    err = signal - sig_fit

    # Compute autocorrelation
    acf, lags = plot_error_autocorr(err, max_lag=100, normalize=True)

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
fig_path = output_dir / 'exp_a09_plot_error_autocorrelation.png'
plt.savefig(fig_path, dpi=150)
plt.close()

print(f"\n[Save fig] -> [{fig_path}]")
