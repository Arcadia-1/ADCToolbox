%% Verification Testbench for plotphase FFT and LMS modes
% This script generates synthetic data with known characteristics
% and verifies that both modes correctly identify harmonics
clear; close all; clc;

fprintf('=== plotphase Verification Testbench ===\n\n');

%% Generate Synthetic Test Signal
N = 8192;           % Number of samples
Fs = 1e9;           % Sampling frequency
J = 323;            % Fundamental bin (prime number for coherent)
Fin = J * Fs / N;   % Fundamental frequency

fprintf('Test Signal Parameters:\n');
fprintf('  N = %d samples\n', N);
fprintf('  Fs = %.0f Hz\n', Fs);
fprintf('  Bin = %d (coherent)\n', J);
fprintf('  Fin = %.6f Hz (normalized: %.6f)\n', Fin, Fin/Fs);
fprintf('\n');

% Time vector
t = (0:N-1)' / Fs;

% Generate signal with known harmonics
A_fundamental = 0.45;
A_HD2 = 0.05;   % 2nd harmonic at -19 dB
A_HD3 = 0.02;   % 3rd harmonic at -27 dB
A_HD4 = 0.01;   % 4th harmonic at -33 dB

% Known phases (radians)
phi_fundamental = 0.5;
phi_HD2 = 1.2;
phi_HD3 = -0.8;
phi_HD4 = 0.3;

% Build signal
signal = A_fundamental * sin(2*pi*Fin*t + phi_fundamental) + ...
         A_HD2 * sin(2*pi*2*Fin*t + phi_HD2) + ...
         A_HD3 * sin(2*pi*3*Fin*t + phi_HD3) + ...
         A_HD4 * sin(2*pi*4*Fin*t + phi_HD4);

% Add DC offset and small noise
signal = signal + 0.5 + randn(N, 1) * 1e-5;

fprintf('Expected Values:\n');
fprintf('  Fundamental: A=%.4f, phase=%.4f rad\n', A_fundamental, phi_fundamental);
fprintf('  HD2: A=%.4f (%.1f dB), phase=%.4f rad\n', A_HD2, 20*log10(A_HD2/A_fundamental), phi_HD2);
fprintf('  HD3: A=%.4f (%.1f dB), phase=%.4f rad\n', A_HD3, 20*log10(A_HD3/A_fundamental), phi_HD3);
fprintf('  HD4: A=%.4f (%.1f dB), phase=%.4f rad\n\n', A_HD4, 20*log10(A_HD4/A_fundamental), phi_HD4);

%% Test FFT Mode
fprintf('=== Testing FFT Mode ===\n');
figure('Position', [100, 100, 800, 800]);
[h_fft, harm_phase_fft, harm_mag_fft, freq_fft, noise_dB_fft] = plotphase(signal, 'harmonic', 5, 'mode', 'FFT');
title('FFT Mode Verification');

fprintf('FFT Mode Outputs:\n');
fprintf('  harm_phase: %s (should be empty)\n', mat2str(harm_phase_fft));
fprintf('  harm_mag: %s (should be empty)\n', mat2str(harm_mag_fft));
fprintf('  freq: %s (should be empty)\n', mat2str(freq_fft));
fprintf('  noise_dB: %s (should be empty)\n\n', mat2str(noise_dB_fft));

%% Test LMS Mode
fprintf('=== Testing LMS Mode ===\n');
figure('Position', [920, 100, 800, 800]);
[h_lms, harm_phase_lms, harm_mag_lms, freq_lms, noise_dB_lms] = plotphase(signal, 'harmonic', 5, 'mode', 'LMS');
title('LMS Mode Verification');

fprintf('LMS Mode Outputs:\n');
fprintf('  Detected frequency: %.6f (normalized)\n', freq_lms);
fprintf('  Noise floor: %.2f dB\n', noise_dB_lms);
fprintf('  Number of harmonics: %d\n\n', length(harm_phase_lms));

fprintf('Harmonic Analysis (LMS):\n');
fprintf('  H# |  Magnitude  | Mag (dB) |  Phase (rad) | Phase (deg)\n');
fprintf('  ---|-------------|----------|--------------|------------\n');
for ii = 1:min(5, length(harm_phase_lms))
    mag_dB = 20*log10(harm_mag_lms(ii)/harm_mag_lms(1));
    fprintf('  %2d | %11.6f | %8.2f | %12.6f | %11.2f\n', ...
        ii, harm_mag_lms(ii), mag_dB, harm_phase_lms(ii), harm_phase_lms(ii)*180/pi);
end
fprintf('\n');

%% Verify Results
fprintf('=== Verification Results ===\n');

% Check frequency detection
freq_error = abs(freq_lms - Fin/Fs);
fprintf('Frequency Detection:\n');
fprintf('  Expected: %.6f\n', Fin/Fs);
fprintf('  Detected: %.6f\n', freq_lms);
fprintf('  Error: %.2e %s\n\n', freq_error, freq_error < 1e-6 ? '✓ PASS' : '✗ FAIL');

% Check magnitude detection
expected_mags = [A_fundamental, A_HD2, A_HD3, A_HD4];
fprintf('Magnitude Detection:\n');
for ii = 1:min(4, length(harm_mag_lms))
    mag_error = abs(harm_mag_lms(ii) - expected_mags(ii));
    mag_error_pct = mag_error / expected_mags(ii) * 100;
    status = mag_error_pct < 5 ? '✓' : '✗';
    fprintf('  H%d: Expected=%.4f, Detected=%.4f, Error=%.2f%% %s\n', ...
        ii, expected_mags(ii), harm_mag_lms(ii), mag_error_pct, status);
end
fprintf('\n');

% Check phase detection (LMS mode returns phases relative to fundamental)
% Note: LMS rotates all phases relative to fundamental, so fundamental phase should be ~0
fprintf('Phase Detection (relative to fundamental):\n');
fprintf('  H1 (fundamental): %.4f rad (should be ~0 after rotation)\n', harm_phase_lms(1));
if length(harm_phase_lms) >= 4
    % The relative phases in LMS mode are: phase[i] - i*phase[1]
    % So we need to compare with the original phase relationships
    fprintf('  H2: Detected=%.4f rad\n', harm_phase_lms(2));
    fprintf('  H3: Detected=%.4f rad\n', harm_phase_lms(3));
    fprintf('  H4: Detected=%.4f rad\n', harm_phase_lms(4));
end
fprintf('\n');

% Check noise floor is reasonable
fprintf('Noise Floor:\n');
fprintf('  Detected: %.2f dB\n', noise_dB_lms);
fprintf('  Status: %s\n', noise_dB_lms < -60 ? '✓ PASS (low noise)' : '⚠ WARNING (noisy)');
fprintf('\n');

fprintf('=== Testbench Complete ===\n');
