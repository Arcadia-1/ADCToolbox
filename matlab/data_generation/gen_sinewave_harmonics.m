%% Generate sinewave with specific HD2 / HD3 distortion levels
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset";

%% Sinewave with specific HD2 / HD3 distortion levels
HD2_dB_list = -65; % Target HD2 levels in dB
HD3_dB_list = -65; % Target HD3 levels in dB

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
        ENoB = specPlot(data,"isplot",0);
        writematrix(data, filename)
        fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
    end
end
