%% Centralized Configuration for Sinewave Generation
% Edit this file to update paths and parameters for all sinewave generation scripts

close all; clear; clc; warning("off");

%% Directory Paths
subFolder = fullfile("test_dataset");
if ~exist(subFolder, 'dir'), mkdir(subFolder); end


%% Signal Parameters
rng(42);

N = 2^13; % Number of samples (8192)
Fs = 1e9; % Sampling frequency (1 GHz)
Fin_want = 300e6; % Input frequency (300 MHz)

J = findbin(Fs, Fin_want, N);
Fin = J / N * Fs;
ideal_phase = 2 * pi * Fin * (0:N - 1)' * 1 / Fs;

A = 0.499;
DC = 0.5;
Noise_rms = 50e-6;