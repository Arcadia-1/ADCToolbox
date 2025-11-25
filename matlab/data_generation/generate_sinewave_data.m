close all ; clear; clc; warning("off")
rng(42); % set random seed for reproducibility

data_dir = "dataset";
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf("Created output directory: [%s]\n", data_dir);
end

%% Sinewave with jitter
% Tj_list = logspace(-15, -12, 2); % jitter values (in s) to sweep
Tj_list = 1e-12;

N = 2^13;
Fs = 1e9;
J = findBin(Fs, 300e6, N);
Fin = J / N * Fs;

ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs; % ideal phase

for k = 1:length(Tj_list)
    Tj = Tj_list(k);

    phase_noise_rms = 2 * pi * Fin * Tj; % convert jitter(sec) -> phase jitter(rad)
    phase_jitter = randn(1, N) * phase_noise_rms; % random jitter

    data = sin(ideal_phase+phase_jitter) * 0.49 + 0.5 + randn(1, N) * 1e-6; % jittered signal

    filename = fullfile(data_dir, sprintf("sinewave_jitter_%dfs.csv", round(Tj*1e15)));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename)
end

%% Sinewave with clipping
clipping_list = [0.055, 0.060]; % clipping thresholds to sweep

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;

ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

for k = 1:length(clipping_list)
    clip_th = clipping_list(k);

    sig = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    % apply clipping distortion
    data = min(max(sig, clip_th), 1-clip_th);

    str = replace(sprintf("%.3f", clip_th), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_clipping_%s.csv", str));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename)
end

%% Sinewave with 2-step quantization gain error
gain_error_list = [0.99, 1.01]; % interstage gain error values
% gain_error_list = linspace(0.99,1.01, 10);

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;

ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

for k = 1:length(gain_error_list)
    g_err = gain_error_list(k); % interstage gain error (e.g. 0.98, 0.99, 1.00, ...)

    sig = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    msb = floor(sig*2^4) / 2^4; % coarse quantizer (4-bit)
    lsb = floor((sig - msb)*2^12) / 2^12; % fine quantizer  (12-bit)

    data = msb * g_err + lsb; % apply interstage gain error

    gstr = replace(sprintf("%.2f", g_err), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_gain_error_%s.csv", gstr));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename)

end

%% Sinewave with 2-step quantization kickback
kickback_strength_list = 0.005:0.005:0.01; % kickback coupling strength

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;

ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

for k = 1:length(kickback_strength_list)
    kb = kickback_strength_list(k);

    sig = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    % two-step quantizer
    msb = floor(sig*2^4) / 2^4; % coarse quantizer (4-bit)
    lsb = floor((sig - msb)*2^12) / 2^12; % fine quantizer  (12-bit)

    % apply kickback (previous MSB affects the next residue)
    msb_shifted = [msb(1), msb(1:end-1)]; % delayed MSB (one-step memory)
    data = msb + lsb + kb * msb_shifted; % kickback injection

    kstr = replace(sprintf("%.3f", kb), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_kickback_%s.csv", kstr));
    fprintf("[Save data into file] -> [%s]\n", filename);

    writematrix(data, filename)
end
%% Sinewave with random glitch injection
glitch_prob_list = [0.001, 0.01]; % 0.1%, 1%, 10%

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;

ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs; % phase of ideal sinewave

for k = 1:length(glitch_prob_list)
    gprob = glitch_prob_list(k);

    sig = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    % glitch injection: upward spike of +0.1 with probability gprob
    glitch = (rand(1, N) < gprob) * 0.1;
    data = sig + glitch;

    pstr = replace(sprintf("%.3f", gprob), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_glitch_%s.csv", pstr));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename)
end
%% Sinewave with baseline drift
baseline_list = [0.1, 0.2]; % different drift amplitudes to sweep

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;

ideal_phase = 2 * pi * Fin * (0:N - 1) / Fs;

for k = 1:length(baseline_list)
    drift_amp = baseline_list(k);

    sig = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * 1e-5;

    % exponential drift:  from 0 → drift_amp
    drift = (1 - exp(-(0:N - 1)/N*10)) * drift_amp;
    data = sig + drift;

    bstr = replace(sprintf("%.3f", drift_amp), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_drift_%s.csv", bstr));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename)
end

%% Sinewave with reference modulation error

ref_error_list = [0.01, 0.02]; % different reference mismatch levels

for k = 1:length(ref_error_list)
    ref_amp = ref_error_list(k);

    sig = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    msb = floor(sig*2^4) / 2^4;
    lsb = floor((sig - msb)*2^12) / 2^12;

    data = msb + lsb + (msb - 0.5) .* ((-1).^(0:N - 1)) * ref_amp;

    rstr = replace(sprintf("%.3f", ref_amp), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_ref_error_%s.csv", rstr));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename)
end

%% Sinewave with additive noise
noise_list = [1e-3, 10e-3]; % 100uV, 1mV, 10mV

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;
ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

for k = 1:length(noise_list)
    noise_amp = noise_list(k);

    data = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * noise_amp;

    if noise_amp < 1e-3
        nstr = sprintf("%duV", round(noise_amp*1e6)); % µV
    else
        nstr = sprintf("%dmV", round(noise_amp*1e3)); % mV
    end
    filename = fullfile(data_dir, sprintf("sinewave_noise_%s.csv", nstr));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename)
end
%% Sinewave with specific HD2 / HD3 distortion levels
HD2_dB_list = -80; % Target HD2 levels in dB
HD3_dB_list = [-70, -80]; % Target HD3 levels in dB

N = 2^13;
J = findBin(1, 0.0789, N);
A = 0.499;
sinewave = A * sin((0:N - 1)*J*2*pi/N); % Base sinewave (zero mean)

for k = 1:length(HD2_dB_list)
    for k2 = 1:length(HD3_dB_list)
        % Convert target HD to linear amplitude ratio (Harmonic Amp / Fundamental Amp)
        hd2_amp = 10^(HD2_dB_list(k) / 20);
        hd3_amp = 10^(HD3_dB_list(k2) / 20);

        % ---- Corrected coef2/coef3 (to achieve target HD levels) ----
        % The target HD ratio (hd2_amp) = coef2 * A / 2  → coef2 = hd2_amp / (A/2)
        coef2 = (hd2_amp / (A / 2));
        % The target HD ratio (hd3_amp) = coef3 * A^2 / 4 → coef3 = hd3_amp / (A^2/4)
        coef3 = (hd3_amp / (A^2 / 4));

        % Generate distorted waveform (zero-mean → nonlinear → add DC)
        % Add small noise for practical simulation
        data = sinewave + coef2 * (sinewave.^2) + coef3 * (sinewave.^3);
        data = data + 0.5 + randn(1, N) * 1e-6;

        hd2_str = sprintf("HD2_n%ddB", abs(HD2_dB_list(k)));
        hd3_str = sprintf("HD3_n%ddB", abs(HD3_dB_list(k2)));
        filename = fullfile(data_dir, sprintf("sinewave_%s_%s.csv", hd2_str, hd3_str));
        fprintf("[Save data into file] -> [%s]\n", filename);
        writematrix(data, filename);
    end
end

%% Sinewave with True Amplitude Modulation (AM Tone)
am_strength_list = [0.0005, 0.001]; % modulation depth (m)
am_freq = 99e6; % AM modulation frequency (Hz)

N = 2^13;
Fs = 1e9;
J = findBin(Fs, 200e6, N);
Fin = J / N * Fs;
t = (0:N - 1) / Fs;
ideal_phase = 2 * pi * Fin * t;

for k = 1:length(am_strength_list)
    am_strength = am_strength_list(k);

    sine_zero_mean = sin(ideal_phase) * 0.49; % baseband sine

    am_factor = 1 + am_strength * sin(2*pi*am_freq*t); % AM(t) = 1 + m*sin(Ωt)

    data = sine_zero_mean .* am_factor + 0.5; % add DC offset
    data = data + randn(1, N) * 1e-6; % tiny white noise

    astr = replace(sprintf('%.3f', am_strength), '.', 'P');
    filename = fullfile(data_dir, ...
        sprintf("sinewave_amplitude_modulation_%s.csv", astr));

    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename);
end

%% Sinewave with Amplitude Noise (Random Amplitude Noise)
am_noise_list = [0.005, 0.01]; % AM noise strength (e.g., 0.1% to 1%)

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;
ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

for k = 1:length(am_noise_list)
    am_strength = am_noise_list(k);

    sinewave_zero_mean = sin(ideal_phase) * 0.49; % Ideal sinewave (zero mean)

    % Generate AM noise factor: 1 + random_modulation
    am_factor = 1 + randn(1, N) * am_strength;

    % Apply amplitude modulation noise: sinewave * am_factor
    % Add DC offset and small additive white noise (for realism)
    data = sinewave_zero_mean .* am_factor + 0.5 + randn(1, N) * 1e-6;

    astr = replace(sprintf("%.3f", am_strength), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_amplitude_noise_%s.csv", astr));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename);
end

%% Sinewave in Nyquist Zones 2 to 4
N = 2^13;
Fs = 1e9;
Offset = 1e6;

Nyquist_Zone_Fins = [; ...
    0.5e9 + Offset, ... % Zone 2
    1.0e9 + Offset, ... % Zone 3
    ];

Tj = 100e-15; % 100 fs Jitter

for k = 1:length(Nyquist_Zone_Fins)
    Fin_target = Nyquist_Zone_Fins(k);

    J = findBin(Fs, Fin_target, N);
    Fin = J / N * Fs; % Coherent Fin

    F_alias = alias(J, N) / N * Fs; % Map to 0 to Fs/2

    ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

    phase_noise_rms = 2 * pi * Fin * Tj; % convert jitter(sec) -> phase jitter(rad)
    phase_jitter = randn(1, N) * phase_noise_rms; % random jitter
    data = sin(ideal_phase+phase_jitter) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    Zone = ceil(Fin/(Fs / 2));

    filename = fullfile(data_dir, sprintf("sinewave_Zone%d_Tj_%dfs.csv", ...
        Zone, round(Tj*1e15)));

    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data, filename);
end
%% Sinewave with multirun
N_run_list = [2, 16];

N = 2^12;
Fs = 1e9;
J = findBin(Fs, 99e6, N);
Fin = J / N * Fs;
A = 0.5;

% HD to linear amplitude ratio (Harmonic Amp / Fundamental Amp)
hd2_amp = 10^(-118 / 20);
hd3_amp = 10^(-109 / 20);

coef2 = (hd2_amp / (A / 2));
coef3 = (hd3_amp / (A^2 / 4));

sinewave = A * sin((0:N - 1)*J*2*pi/N); % Base sinewave (zero mean)
sinewave = sinewave + coef2 * (sinewave.^2) + coef3 * (sinewave.^3);

for k = 1:length(N_run_list)
    N_run = N_run_list(k);
    data_batch = zeros([N_run, N]);

    for iter_run = 1:N_run
        data_batch(iter_run, :) = sinewave + 0.5 + randn(1, N) * 1e-5;
    end

    filename = fullfile(data_dir, sprintf("batch_sinewave_Nrun_%d.csv", N_run));

    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(data_batch, filename);
end
