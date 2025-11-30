%% Generate sinewave with clipping
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset/aout";

%% Sinewave with clipping
clipping_list = [0.012]; % clipping thresholds to sweep

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
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
