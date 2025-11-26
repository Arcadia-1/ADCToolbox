close all; clear; clc; warning("off")
rng(42); % set random seed for reproducibility

data_dir = "dataset/non_lin";  % Subfolder for nonlinearity test data
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf("Created output directory: [%s]\n", data_dir);
end

%% Sinewave with controllable HD2, HD3, HD4, HD5 distortion
% User-configurable parameters
HD2_dB_list = [-60];     % HD2 levels in dB (can be freely set or swept)
HD3_dB_list = [-50];     % HD3 levels in dB
% HD4_dB_list = [-90];     % HD4 levels in dB (comment out to exclude)
% HD5_dB_list = [-90];     % HD5 levels in dB (comment out to exclude)
N_list = 2.^(12);  % FFT points (can be freely set or swept)

% Check which harmonics are defined
use_HD4 = exist('HD4_dB_list', 'var');
use_HD5 = exist('HD5_dB_list', 'var');

% Set default list for undefined harmonics (single loop iteration)
if ~use_HD4, HD4_dB_list = NaN; end
if ~use_HD5, HD5_dB_list = NaN; end

% Fundamental frequency parameters
Fs = 1e9;                     % Sampling frequency (Hz)
Fin_ratio = 0.0789;           % Normalized frequency (Fin/Fs)

file_count = 0;
for n_idx = 1:length(N_list)
    N = N_list(n_idx);
    J = findBin(1, Fin_ratio, N);
    A = 0.499;  % Amplitude (zero-mean)

    % Base sinewave (zero mean)
    sinewave = A * sin((0:N - 1) * J * 2 * pi / N);

    for idx2 = 1:length(HD2_dB_list)
        for idx3 = 1:length(HD3_dB_list)
            for idx4 = 1:length(HD4_dB_list)
                for idx5 = 1:length(HD5_dB_list)
                    % Convert target HD levels (dB) to linear amplitude ratios
                    hd2_amp = 10^(HD2_dB_list(idx2) / 20);
                    hd3_amp = 10^(HD3_dB_list(idx3) / 20);

                    % Calculate polynomial coefficients for each harmonic
                    % HD2: 2nd harmonic amplitude = coef2 * A / 2
                    coef2 = hd2_amp / (A / 2);

                    % HD3: 3rd harmonic amplitude = coef3 * A^2 / 4
                    coef3 = hd3_amp / (A^2 / 4);

                    % Generate distorted waveform: start with base harmonics
                    distorted = sinewave + ...
                                coef2 * (sinewave.^2) + ...
                                coef3 * (sinewave.^3);

                    % Add HD4 if defined
                    if use_HD4
                        hd4_amp = 10^(HD4_dB_list(idx4) / 20);
                        coef4 = hd4_amp / (A^3 / 8);
                        distorted = distorted + coef4 * (sinewave.^4);
                    end

                    % Add HD5 if defined
                    if use_HD5
                        hd5_amp = 10^(HD5_dB_list(idx5) / 20);
                        coef5 = hd5_amp / (A^4 / 16);
                        distorted = distorted + coef5 * (sinewave.^5);
                    end

                    % Add DC offset and small noise for realism
                    data = distorted + 0.5 + randn(1, N) * 1e-6;

                    % Build filename dynamically based on which harmonics are defined
                    hd2_str = sprintf("HD2_n%ddB", abs(HD2_dB_list(idx2)));
                    hd3_str = sprintf("HD3_n%ddB", abs(HD3_dB_list(idx3)));

                    filename_str = hd2_str + "_" + hd3_str;

                    if use_HD4
                        hd4_str = sprintf("HD4_n%ddB", abs(HD4_dB_list(idx4)));
                        filename_str = filename_str + "_" + hd4_str;
                    end

                    if use_HD5
                        hd5_str = sprintf("HD5_n%ddB", abs(HD5_dB_list(idx5)));
                        filename_str = filename_str + "_" + hd5_str;
                    end

                    n_str = sprintf("N_%d", N);
                    filename_str = filename_str + "_" + n_str;

                    filename = fullfile(data_dir, sprintf("sinewave_%s.csv", filename_str));

                    fprintf("[Save data into file] -> [%s]\n", filename);
                    writematrix(data, filename);
                    file_count = file_count + 1;
                end
            end
        end
    end
end

fprintf("\n=== Generation Complete ===\n");
fprintf("Total files generated: %d\n", file_count);
