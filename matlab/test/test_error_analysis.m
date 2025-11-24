close all; clc; clear;

data = readmatrix(fullfile("ADCToolbox_example_data","sinewave_glitch_0P100.csv"));

[data_fit, freq_est, mag, dc, phi] = sineFit(data);
err_data = data - data_fit;
Fs = 1;

figure;
specPlot(data, "label", 1, "Fs", 1e9);
% 

figure;
[noise_lsb, mu, sigma, x, fx, gauss_pdf] = errPDF(err_data);

figure;
[acf, lags] = errAutoCorrelation(err_data,"MaxLag", 300);

figure;
specPlot(err_data, "label", 0, "Fs", 1e9);
title("Error Spectrum");

figure;
errEnvelopeSpectrum(err_data, "Fs", 1e9);




