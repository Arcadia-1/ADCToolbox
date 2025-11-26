%% Generate sinewave with Static Nonlinearity (INL - Transfer Function)
close all; clear; clc; warning("off");
rng(42);

data_dir = "dataset";
%% Sinewave with Static Nonlinearity (INL - Transfer Function)
% Direct transfer function: y = k1*x + k2*x^2 + k3*x^3 + k4*x^4 + k5*x^5
% This models point-by-point static nonlinearity (INL)

% User-configurable parameters - specify coefficients directly
k1_list = [1.0]; % Linear gain (ideal = 1.0)
k2_list = [0.001]; % 2nd order nonlinearity coefficient
k3_list = [0.01]; % 3rd order nonlinearity coefficient
% k4_list = [0.001];       % 4th order (uncomment to include)
% k5_list = [0.001];       % 5th order (uncomment to include)

% Check which orders are defined
use_k4 = exist('k4_list', 'var');
use_k5 = exist('k5_list', 'var');

% Set default for undefined coefficients
if ~use_k4, k4_list = 0; end
if ~use_k5, k5_list = 0; end

% Signal parameters
N = 2^13;
Fs = 10e9;
J = findBin(Fs, 1000e6, N);
Fin = J / N * Fs;
A = 0.499; % Amplitude (peak, zero-mean)

% Generate IDEAL input sine wave (this is x in the transfer function)
x_ideal = A * sin((0:N - 1)*J*2*pi/N);

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
                    data = y_output + 0.5 + randn(1, N) * 1.5e-4;

                    % Build filename: sinewave_INL_k2_xxx_k3_xxx_...
                    % Format: Positive: 0.001 -> 0P0010, Negative: -0.01 -> nP0100
                    formatK = @(val) string(repmat('n',1,val<0)) + ...
                 replace(sprintf("%.4f", abs(val)), ".", "P");

                    filename_parts = "INL";

                    % Only include non-zero k1 if it's not 1.0
                    if k1 ~= 1.0
                        filename_parts = filename_parts + "_k1_" + formatK(k1);
                    end

                    % Always include k2 and k3
                    filename_parts = filename_parts + "_k2_" + formatK(k2) + "_k3_" + formatK(k3);

                    % Include k4 if defined and non-zero
                    if use_k4 && k4 ~= 0
                        filename_parts = filename_parts + "_k4_" + formatK(k4);
                    end

                    % Include k5 if defined and non-zero
                    if use_k5 && k5 ~= 0
                        filename_parts = filename_parts + "_k5_" + formatK(k5);
                    end

                    filename = fullfile(data_dir, sprintf("sinewave_%s.csv",filename_parts));
                    ENoB = specPlot(data, "isplot", 0);
                    writematrix(data, filename)
                    fprintf("  [ENoB = %0.2f] [Save] %s\n", ENoB, filename);
                end
            end
        end
    end
end
