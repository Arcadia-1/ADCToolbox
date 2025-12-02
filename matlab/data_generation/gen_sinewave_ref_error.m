%% Generate sinewave with reference modulation error
close all; clear; clc; warning("off");
rng(42);
subFolder = fullfile("dataset", "sinewave");
if ~exist(subFolder, 'dir'), mkdir(subFolder); end

%% Sinewave with reference modulation error
ref_error_list = [0.00075]; % different reference mismatch levels

N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;
ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

for k = 1:length(ref_error_list)
    ref_amp = ref_error_list(k);

    sig = sin(ideal_phase) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    msb = floor(sig*2^4) / 2^4;
    lsb = floor((sig - msb)*2^12) / 2^12;

    data = msb + lsb + (msb - 0.5) .* ((-1).^(0:N - 1)) * ref_amp;

    rstr = replace(sprintf("%.3f", ref_amp), ".", "P");
    filename = fullfile(subFolder, sprintf("sinewave_ref_error_%s.csv", rstr));
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
