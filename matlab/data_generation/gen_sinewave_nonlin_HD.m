%% Generate sinewave with specific HD2 / HD3 distortion levels
close all; clear; clc; warning("off");
rng(42);
subFolder = fullfile("dataset", "sinewave");
if ~exist(subFolder, 'dir'), mkdir(subFolder); end

%% Sinewave with specific HD2 / HD3 distortion levels
HD2_dB_list = -150; % Target HD2 levels in dB
HD3_dB_list = -66.19; % Target HD3 levels in dB

N = 2^13;
Fs = 1e9;
J = findBin(Fs, 80e6, N);
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
        data = data + 0.5 + randn(1, N) * 1e-4;

        hd2_str = sprintf("HD2_n%0.0fdB", abs(HD2_dB_list(k)));
        hd3_str = sprintf("HD3_n%0.0fdB", abs(HD3_dB_list(k2)));
        filename = fullfile(subFolder, sprintf("sinewave_%s_%s.csv", hd2_str, hd3_str));
        writematrix(data, filename)

        ENoB = specPlot(data,"isplot",0);
        fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
    end
end
