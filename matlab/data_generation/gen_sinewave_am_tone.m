%% Generate sinewave with True Amplitude Modulation (AM Tone)
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset/aout";

%% Sinewave with True Amplitude Modulation (AM Tone)
am_strength_list = [0.0011]; % modulation depth (m)
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
    filename = fullfile(data_dir, sprintf("sinewave_amplitude_modulation_%s.csv", astr));
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
