%% Generate sinewave in Nyquist Zones 2 to 4
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset";

%% Sinewave in Nyquist Zones 2 to 4
N = 2^13;
Fs = 1e9;
Offset = 1e6;

Nyquist_Zone_Fins = [
    0.5e9 + Offset % Zone 2
    1.0e9 + Offset % Zone 3
];

Tj = 250e-15; % 100 fs Jitter

for k = 1:length(Nyquist_Zone_Fins)
    Fin_target = Nyquist_Zone_Fins(k);

    J = findBin(Fs, Fin_target, N);
    Fin = J / N * Fs; % Coherent Fin

    F_alias = alias(J, N) / N * Fs; % Map to 0 to Fs/2

    ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;

    phase_noise_rms = 2 * pi * Fin * Tj; % convert jitter(sec) -> phase jitter(rad)
    phase_jitter = randn(1, N) * phase_noise_rms; % random jitter
    data = sin(ideal_phase+phase_jitter) * 0.49 + 0.5 + randn(1, N) * 1e-6;

    Zone = ceil(Fin/(Fs / 2));

    filename = fullfile(data_dir, sprintf("sinewave_Zone%d_Tj_%dfs.csv", ...
        Zone, round(Tj*1e15)));
    ENoB = specPlot(data,"isplot",0);
    writematrix(data, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end
