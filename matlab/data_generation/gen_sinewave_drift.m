%% Generate sinewave with baseline drift
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset";

%% Sinewave with baseline drift
baseline_list = [0.004]; % different drift amplitudes to sweep

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
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
