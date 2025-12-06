clear; clc; close all; warning('off');

N = 2^16;
Fs = 800e6;
Fin = 127/N * Fs;
A = 0.5;
DC = 0;

Target_AM_V = 0e-6;
Target_PM_V = 200e-6;
Target_Thermal_V = 300e-6;

Target_PM_Rad = Target_PM_V / A;

t = (0:N-1)' / Fs;
phase_clean = 2*pi*Fin*t;

rng(42);
n_am = randn(N, 1) * Target_AM_V;
n_pm = randn(N, 1) * Target_PM_Rad;
n_th = randn(N, 1) * Target_Thermal_V;

sig_noisy = (A + n_am) .* sin(phase_clean + n_pm) + DC + n_th;

tic;
[emean, erms, xx, meas_am, meas_pm_rad] = errsin_hp(sig_noisy, 'bin', 100, 'fin', Fin/Fs, 'disp', 1);
toc;

meas_pm_v = meas_pm_rad * A;

fprintf('\nResults Comparison:\n');
fprintf('--------------------------------------------------\n');
fprintf('%-20s | %-12s | %-12s\n', 'Metric', 'Input', 'Measured');
fprintf('--------------------------------------------------\n');
fprintf('AM Noise (V)         | %12.2e | %12.2e\n', Target_AM_V, meas_am);
fprintf('PM Noise (V)         | %12.2e | %12.2e\n', Target_PM_V, meas_pm_v);
fprintf('PM Noise (rad)       | %12.2e | %12.2e\n', Target_PM_Rad, meas_pm_rad);
fprintf('--------------------------------------------------\n');