%% Generate sinewave with additive noise
close all; clear; clc; warning("off");
rng(42);

subFolder = "dataset/aout/sinewave";
if ~isfolder(subFolder), mkdir(subFolder); end

%% Sinewave with additive noise
noise_list = [2.7e-4]; % 100uV, 1mV, 10mV

N = 2^13;
Fs = 1e9;
J = findBin(Fs, 300e6, N);
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
    filename = fullfile(subFolder, sprintf("sinewave_noise_%s.csv", nstr));
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
