close all; clear; clc;

% --- WEIGHT LISTS TO SWEEP ---
CDAC_lists = {; ...
    [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1]; ...
    [800, 440, 230, 122, 63, 32, 16, 8, 4, 2, 1, 1]; ... % sub-radix 2
    [1024, 512, 256, 256, 128, 64, 64, 32, 16, 8, 8, 4, 2, 1, 1]; ... % redundancy
    };

N = 2^13;
J = findBin(1, 0.0789, N);
FS = 1;
A = 0.99; % Reference voltage and signal amplitude
sinewave = A * sin((0:N - 1)*J*2*pi/N); % Base sinewave (zero mean)

for k = 1:length(CDAC_lists)
    CDAC = CDAC_lists{k};
    B = length(CDAC);

    % --- Core Calculations based on Current Weight ---
    % Calculate resolution based on the number of bits (B)
    resolution = log2(sum(CDAC)/CDAC(end)*2);

    weight_voltage = CDAC / sum(CDAC) * FS; % Calculate weighted voltage levels

    residue = sinewave';
    dout = zeros(N, B); % Initialize quantized bits (N samples x B bits)

    % SAR Quantization Loop
    for j = 1:B
        dout(:, j) = (residue > randn(N, 1)*1e-6);
        delta_cdac = (2 * dout(:, j) - 1) * weight_voltage(j);
        if j < B
            residue = residue - delta_cdac;
        end
    end

    nominal_weight = CDAC;
    nominal_weight(end) = nominal_weight(end) / 2; % LSB dummy

    aout = dout * nominal_weight';

    figure
    [ENoB, SNDR, ~] = specPlot(aout, 'label', 1, 'harmonic', 5, 'winType', @hann, 'OSR', 1, 'coAvg', 0);

    N_bit = round(resolution);
    filename = fullfile("dataset", sprintf("dout_SAR_%db_weight_%d.csv", N_bit, k));
    fprintf("[Save data into file] -> [%s]\n", filename);
    writematrix(dout, filename);
end
