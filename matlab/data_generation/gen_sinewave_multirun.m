%% Generate sinewave with multirun (batch data)
close all; clear; clc; warning("off");
rng(42);
subFolder = fullfile("dataset", "batch_sinewave");
if ~exist(subFolder, 'dir'), mkdir(subFolder); end

%% Sinewave with multirun
N_run_list = [16];

N = 2^12;
Fs = 1e9;
J = findBin(Fs, 99e6, N);
Fin = J / N * Fs;
A = 0.5;

% HD to linear amplitude ratio (Harmonic Amp / Fundamental Amp)
hd2_amp = 10^(-118 / 20);
hd3_amp = 10^(-109 / 20);

coef2 = (hd2_amp / (A / 2));
coef3 = (hd3_amp / (A^2 / 4));

sinewave = A * sin((0:N - 1)*J*2*pi/N); % Base sinewave (zero mean)
sinewave = sinewave + coef2 * (sinewave.^2) + coef3 * (sinewave.^3);

for k = 1:length(N_run_list)
    N_run = N_run_list(k);
    data_batch = zeros([N_run, N]);

    for iter_run = 1:N_run
        data_batch(iter_run, :) = sinewave + 0.5 + randn(1, N) * 2.8e-4;
    end

    filename = fullfile(subFolder, sprintf("batch_sinewave_Nrun_%d.csv", N_run));
    ENoB = specPlot(data_batch,"isplot",0);
    writematrix(data_batch, filename)
    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
end