%% Generate sinewave with 2-step quantization kickback
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset/aout";

%% Sinewave with 2-step quantization kickback
kickback_strength_list = 0.015; % kickback coupling strength

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
    
    ENoB = specPlot(data,"isplot",0);
    kstr = replace(sprintf("%.3f", kb), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_kickback_%s.csv", kstr));
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
    writematrix(data, filename)
end
