%% test_basic.m - Generate and plot a basic sine wave
close all; clc; clear;
%% Configuration
verbose = 0;
subFolder = fullfile("test_output", mfilename);
if ~isfolder(subFolder), mkdir(subFolder); end
%% Generate sine wave
N = 1024; % Number of samples
Fs = 1e3; % Sampling frequency (Hz)
Fin = 99; % Input frequency (Hz)
A = 0.49; % Amplitude
DC = 0.5; % DC offset

t = (0:N - 1)' / Fs;
sinewave = A * sin(2*pi*Fin*t) + DC;
%% Plot sine wave
figure('Position', [100, 100, 1000, 800], "Visible", verbose);

% Full waveform in subplot 1
subplot(2, 1, 1);
hold on;grid on;
plot(t*1e3, sinewave, 'b-', 'LineWidth', 2);
xlim([0, max(t) * 1e3]);
ylim([min(sinewave) - 0.1, max(sinewave) + 0.1]);
xlabel('Time (ms)');
ylabel('Amplitude');
title(sprintf('Full Sine Wave (Fin=%d Hz, Fs=%d Hz, N=%d)', Fin, Fs, N));
set(gca, 'FontSize', 14);

% Zoomed (first 3 periods) in subplot 2
subplot(2, 1, 2);
hold on;
grid on;
period_samples = round(Fs/Fin);
n_periods = 3;
n_zoom = min(period_samples*n_periods, N);
t_zoom = t(1:n_zoom);
sinewave_zoom = sinewave(1:n_zoom);
plot(t_zoom*1e3, sinewave_zoom, '-o', 'LineWidth', 2, 'MarkerSize', 4);
xlabel('Time (ms)');
ylabel('Amplitude');
title(sprintf('Zoomed View (First %d Periods)', n_periods));
set(gca, 'FontSize', 14);
ylim([min(sinewave_zoom) - 0.1, max(sinewave_zoom) + 0.1]);
%% Save results
saveFig(subFolder, "sinewave_basic_matlab.png", verbose);
saveVariable(subFolder, sinewave, verbose);
