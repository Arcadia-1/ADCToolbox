%% Centralized Configuration for Sinewave Generation
common_gen_sinewave;

%% Generate 3 Noise Types: Thermal, Phase Noise, Amplitude Noise
% Target peak contribution (same value across all noise types for comparison)
target_v_rms = 50e-6;  % 50 µV peak contribution

%% 1. Thermal Noise Only (White noise baseline)
data_thermal = A * sin(ideal_phase) + DC + randn(N, 1) * target_v_rms;

tstr = replace(sprintf('%.2f', target_v_rms * 1e6), '.', 'P');  % in uV
filename = fullfile(subFolder, sprintf("sinewave_thermal_noise_%suV.csv", tstr));
ENoB = plotspec(data_thermal, "isplot", 0);
writematrix(data_thermal, filename)
fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);

%% 2. Phase Noise Only (Phase jitter)
% PM model: error² = (pm*A)² * cos²(φ) → pm*A is the peak PM contribution
% For target_peak contribution: pm_rad = target_peak / A
pm_uV = target_v_rms * 1e6;
pm_rad = target_v_rms / A;
pnoi_urad = pm_rad * 1e6;
jitter_rms_fs = (pm_rad / (2 * pi * Fin)) * 1e15;
fprintf("    PM_rms = [%0.4f urad] -> [%0.4f uV] -> [%0.4f fs]\n", pnoi_urad, pm_uV, jitter_rms_fs)

phase_jitter = randn(N, 1) * pm_rad;
data_phase_noise = A * sin(ideal_phase + phase_jitter) + DC;

pstr = replace(sprintf('%.2f', target_v_rms * 1e6), '.', 'P');  % in uV (peak contribution = pm_rad * A)
filename = fullfile(subFolder, sprintf("sinewave_phase_noise_%suV.csv", pstr));
ENoB = plotspec(data_phase_noise, "isplot", 0);
writematrix(data_phase_noise, filename)
fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);

%% 3. Amplitude Noise Only (AM noise)
% AM model: error² = am² * sin²(φ) → am is directly the peak AM contribution
am_noise = randn(N, 1) * target_v_rms;
data_am_noise = (A + am_noise) .* sin(ideal_phase) + DC;

astr = replace(sprintf('%.2f', target_v_rms * 1e6), '.', 'P');  % in uV
filename = fullfile(subFolder, sprintf("sinewave_amplitude_noise_%suV.csv", astr));
ENoB = plotspec(data_am_noise, "isplot", 0);
writematrix(data_am_noise, filename)
fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
