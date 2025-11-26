%% Generate sinewave with 2-step quantization gain error
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset";

%% Sinewave with 2-step quantization gain error
gain_error_list = [0.985]; % interstage gain error values
% gain_error_list = linspace(0.99,1.01, 10);

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;

ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

for k = 1:length(gain_error_list)
    g_err = gain_error_list(k); % interstage gain error (e.g. 0.98, 0.99, 1.00, ...)

    sig = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    msb = floor(sig*2^4) / 2^4; % coarse quantizer (4-bit)
    lsb = floor((sig - msb)*2^12) / 2^12; % fine quantizer  (12-bit)

    data = msb * g_err + lsb; % apply interstage gain error

    gstr = replace(sprintf("%.2f", g_err), ".", "P");
    filename = fullfile(data_dir, sprintf("sinewave_gain_error_%s.csv", gstr));
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
