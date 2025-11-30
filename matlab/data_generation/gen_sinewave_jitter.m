%% Generate sinewave with jitter
close all; clear; clc; warning("off");
rng(42); % set random seed for reproducibility

data_dir = "dataset/aout";

%% Sinewave with jitter
% Tj_list = logspace(-15, -12, 2); % jitter values (in s) to sweep
Tj_list = 400e-15;

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
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
