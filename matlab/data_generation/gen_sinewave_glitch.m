%% Generate sinewave with random glitch injection
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset/aout";

%% Sinewave with random glitch injection
glitch_prob_list = [0.00015]; % 0.1%, 1%, 10%

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
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
