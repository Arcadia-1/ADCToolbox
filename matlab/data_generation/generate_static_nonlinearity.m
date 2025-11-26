close all; clear; clc; warning("off")
rng(42); % set random seed for reproducibility

data_dir = "dataset/static_nonlin";  % Subfolder for static nonlinearity test data
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
    fprintf("Created output directory: [%s]\n", data_dir);
end

%% Generate sine wave with STATIC nonlinearity
% Direct transfer function: y = k1*x + k2*x^2 + k3*x^3 + k4*x^4 + k5*x^5
% This is point-by-point transfer function (no frequency-dependent effects)

% User-configurable parameters - specify coefficients directly
k1_list = [1.0];           % Linear gain (ideal = 1.0)
k2_list = [0.0];     % 2nd order nonlinearity coefficient
k3_list = [0.01];     % 3rd order nonlinearity coefficient
% k4_list = [0.01];        % 4th order (uncomment to include)
% k5_list = [0.01];        % 5th order (uncomment to include)

% Check which orders are defined
use_k4 = exist('k4_list', 'var');
use_k5 = exist('k5_list', 'var');

% Set default list for undefined coefficients
if ~use_k4, k4_list = 0; end
if ~use_k5, k5_list = 0; end

% Fundamental frequency parameters
Fs = 1e9;                     % Sampling frequency (Hz)
Fin_ratio = 0.0789;           % Normalized frequency (Fin/Fs)
N_list = 2.^(12);             % FFT points

file_count = 0;
for n_idx = 1:length(N_list)
    N = N_list(n_idx);
    J = findBin(1, Fin_ratio, N);
    A = 0.499;  % Amplitude (peak, zero-mean)

    % Generate IDEAL input sine wave (this is x in the transfer function)
    x_ideal = A * sin((0:N - 1) * J * 2 * pi / N);

    for idx1 = 1:length(k1_list)
        for idx2 = 1:length(k2_list)
            for idx3 = 1:length(k3_list)
                for idx4 = 1:length(k4_list)
                    for idx5 = 1:length(k5_list)

                        k1 = k1_list(idx1);
                        k2 = k2_list(idx2);
                        k3 = k3_list(idx3);
                        k4 = k4_list(idx4);
                        k5 = k5_list(idx5);

                        % Apply STATIC transfer function point-by-point
                        % y = k1*x + k2*x^2 + k3*x^3 + k4*x^4 + k5*x^5
                        y_output = k1 * x_ideal + ...
                                   k2 * (x_ideal.^2) + ...
                                   k3 * (x_ideal.^3) + ...
                                   k4 * (x_ideal.^4) + ...
                                   k5 * (x_ideal.^5);

                        % Add DC offset and small noise for realism
                        data = y_output + 0.5 + randn(1, N) * 1e-4;

                        % Build filename with coefficient values
                        k1_str = sprintf("k1_%.3f", k1);
                        k2_str = sprintf("k2_%.3f", k2);
                        k3_str = sprintf("k3_%.3f", k3);

                        filename_str = k1_str + "_" + k2_str + "_" + k3_str;

                        if use_k4 && k4 ~= 0
                            k4_str = sprintf("k4_%.3f", k4);
                            filename_str = filename_str + "_" + k4_str;
                        end

                        if use_k5 && k5 ~= 0
                            k5_str = sprintf("k5_%.3f", k5);
                            filename_str = filename_str + "_" + k5_str;
                        end

                        n_str = sprintf("N_%d", N);
                        filename_str = filename_str + "_" + n_str;

                        filename = fullfile(data_dir, sprintf("sinewave_static_nonlin_%s.csv", filename_str));

                        fprintf("[Save data into file] -> [%s]\n", filename);
                        fprintf("  Coefficients: k1=%.4f, k2=%.4f, k3=%.4f, k4=%.4f, k5=%.4f\n", ...
                                k1, k2, k3, k4, k5);
                        writematrix(data, filename);
                        file_count = file_count + 1;
                    end
                end
            end
        end
    end
end

fprintf("\n=== Generation Complete ===\n");
fprintf("Total files generated: %d\n", file_count);
fprintf("These files contain STATIC nonlinearity (point-by-point transfer function)\n");
fprintf("Use polynomial fitting to extract k1, k2, k3, k4, k5 coefficients\n");
