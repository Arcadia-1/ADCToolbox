close all; clear; clc; warning("off")
rng(42); % set random seed for reproducibility

data_dir = "dataset";
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf("Created output directory: [%s]\n", data_dir);
end

%% Sinewave with controllable HD2, HD3, HD4, HD5 distortion
% User-configurable parameters
HD2_dB_list = [-60, -70];     % HD2 levels in dB (can be freely set or swept)
HD3_dB_list = [-60, -70];     % HD3 levels in dB
% HD4_dB_list = [-90];     % HD4 levels in dB
% HD5_dB_list = [-90];     % HD5 levels in dB
N_list = 2^(6:8);  % FFT points (can be freely set or swept)

% Fundamental frequency parameters
Fs = 1e9;                     % Sampling frequency (Hz)
Fin_ratio = 0.0789;           % Normalized frequency (Fin/Fs)

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
                    hd4_amp = 10^(HD4_dB_list(idx4) / 20);
                    hd5_amp = 10^(HD5_dB_list(idx5) / 20);

                    % Calculate polynomial coefficients for each harmonic
                    % HD2: 2nd harmonic amplitude = coef2 * A / 2
                    coef2 = hd2_amp / (A / 2);

                    % HD3: 3rd harmonic amplitude = coef3 * A^2 / 4
                    coef3 = hd3_amp / (A^2 / 4);

                    % HD4: 4th harmonic amplitude = coef4 * A^3 / 8
                    coef4 = hd4_amp / (A^3 / 8);

                    % HD5: 5th harmonic amplitude = coef5 * A^4 / 16
                    coef5 = hd5_amp / (A^4 / 16);

                    % Generate distorted waveform using polynomial nonlinearity
                    % y = x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5
                    distorted = sinewave + ...
                                coef2 * (sinewave.^2) + ...
                                coef3 * (sinewave.^3) + ...
                                coef4 * (sinewave.^4) + ...
                                coef5 * (sinewave.^5);

                    % Add DC offset and small noise for realism
                    data = distorted + 0.5 + randn(1, N) * 1e-6;

                    % Format filename: HD2_xx_HD3_xx_HD4_xx_HD5_xx_N_xxxx.csv
                    hd2_str = sprintf("HD2_n%ddB", abs(HD2_dB_list(idx2)));
                    hd3_str = sprintf("HD3_n%ddB", abs(HD3_dB_list(idx3)));
                    hd4_str = sprintf("HD4_n%ddB", abs(HD4_dB_list(idx4)));
                    hd5_str = sprintf("HD5_n%ddB", abs(HD5_dB_list(idx5)));
                    n_str = sprintf("N_%d", N);

                    filename = fullfile(data_dir, sprintf("sinewave_%s_%s_%s_%s_%s.csv", ...
                        hd2_str, hd3_str, hd4_str, hd5_str, n_str));

                    fprintf("[Save data into file] -> [%s]\n", filename);
                    writematrix(data, filename);
                end
            end
        end
    end
end

fprintf("\n=== Generation Complete ===\n");
fprintf("Total files generated: %d\n", ...
    length(N_list) * length(HD2_dB_list) * length(HD3_dB_list) * ...
    length(HD4_dB_list) * length(HD5_dB_list));
