%% Generate sinewave with Amplitude Noise (Random Amplitude Noise)
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset/aout";

%% Sinewave with Amplitude Noise (Random Amplitude Noise)
am_noise_list = [0.00075]; % AM noise strength (e.g., 0.1% to 1%)

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
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
