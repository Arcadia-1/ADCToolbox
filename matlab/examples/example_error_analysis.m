close all; clc; clear;

%% generate your distorted sinewave here
jitter_sec = 100e-15;
thermal_noise = 1e-4;
gain_error = 0.001;

N = 2^13;
Fs = 1e9;
J = findBin(Fs, 123e6, N);
Fin = J / N * Fs;

ideal_phase = 2 * pi * Fin * (0:N - 1) * 1 / Fs;
phase_jitter = randn(1, N) * 2 * pi * Fin * jitter_sec; % jitter(sec) -> phase jitter(rad)

sig = sin(ideal_phase+phase_jitter) * 0.49 + 0.5 + randn(1, N) * thermal_noise; % signal with noise

msb = floor(sig*2^4) / 2^4; % coarse quantizer (4-bit)
lsb = floor((sig - msb)*2^12) / 2^12; % fine quantizer  (12-bit)

data = msb * (1 + gain_error) + lsb; % apply interstage gain error

%% basic spectrum
figure;
specPlot(data);

%% error analysis
[data_fit, freq_est, mag, dc, phi] = sineFit(data);
err_data = data - data_fit;
Fs = 1;

figure;
[noise_lsb, mu, sigma, x, fx, gauss_pdf] = errPDF(err_data);

figure;
[acf, lags] = errAutoCorrelation(err_data, "MaxLag", 300);

figure;
specPlot(err_data, "label", 0, "Fs", 1e9);
title("Error Spectrum");

figure;
errEnvelopeSpectrum(err_data, "Fs", 1e9);
