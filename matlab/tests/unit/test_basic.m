%% test_basic.m - Generate and plot a basic sine wave
close all; clc; clear;

%% Configuration & Setup
verbose = 1;
subFolder = fullfile("test_output", mfilename);
if ~isfolder(subFolder), mkdir(subFolder); end

%% Generate sine wave
N = 1024; Fs = 1e3; Fin = 99; A = 0.49; DC = 0.5;
t = (0:N-1)'/Fs;
sinewave = A*sin(2*pi*Fin*t) + DC + randn(N,1)*1e-6;

%% Plot sine wave
figure('Position', [100, 100, 1000, 600], "Visible", verbose);

% Full waveform
subplot(2,1,1);
plot(t*1e3, sinewave, 'b-', 'LineWidth', 1.5);
grid on; xlim([0, max(t)*1e3]); ylim([min(sinewave)-0.1, max(sinewave)+0.1]);
xlabel('Time (ms)'); ylabel('Amplitude');
title(sprintf('Full Sine Wave (Fin=%d Hz, Fs=%d Hz, N=%d)', Fin, Fs, N));
set(gca, 'FontSize', 14);

% Zoomed (first 3 periods)
subplot(2,1,2);
n_zoom = min(3*round(Fs/Fin), N);
plot(t(1:n_zoom)*1e3, sinewave(1:n_zoom), '-o', 'LineWidth', 2, 'MarkerSize', 4);
grid on;
xlabel('Time (ms)'); ylabel('Amplitude');
title('Zoomed View (First 3 Periods)');
set(gca, 'FontSize', 14);
ylim([min(sinewave(1:n_zoom))-0.1, max(sinewave(1:n_zoom))+0.1]);

%% Save
saveFig(subFolder, "sinewave_basic.png", verbose);
saveVariable(subFolder, sinewave, verbose);
