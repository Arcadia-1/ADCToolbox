%% Centralized Configuration for Dout Generation
common_gen_dout;

%% --- WEIGHT LISTS TO SWEEP ---
CDAC_lists = {; ...
    [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 1]; ...
    [800, 440, 230, 122, 63, 32, 16, 8, 4, 2, 1, 1]; ... % sub-radix 2
    % [1024, 512, 256, 256, 128, 64, 64, 32, 16, 8, 8, 4, 2, 1, 1]; ... % redundancy
    };

sig = 2*A*sin(ideal_phase);
FS = 1;

for k = 1:length(CDAC_lists)
    CDAC = CDAC_lists{k};
    B = length(CDAC);

    % --- Core Calculations based on Current Weight ---
    % Calculate resolution based on the number of bits (B)
    resolution = log2(sum(CDAC)/CDAC(end)*2);

    weight_voltage = CDAC / sum(CDAC) * FS; % Calculate weighted voltage levels

    residue = sig;
    dout = zeros(N, B); % Initialize quantized bits (N samples x B bits)

    % SAR Quantization Loop
    for j = 1:B
        dout(:, j) = (residue > 0);
        delta_cdac = (2 * dout(:, j) - 1) * weight_voltage(j);
        if j < B
            residue = residue - delta_cdac;
        end
    end

    nominal_weight = CDAC;
    nominal_weight(end) = nominal_weight(end) / 2; % LSB dummy

    aout = dout * nominal_weight';

    figure(Visible="on")
    [ENoB, SNDR, ~] = specPlot(aout);
    % close all;

    N_bit = round(resolution);
    filename = fullfile(subFolder, sprintf("dout_SAR_%db_weight_%d.csv", N_bit, k));
    fprintf("[Save data into file] [ENoB = %0.2f] -> [%s]\n", ENoB, filename);
    writematrix(dout, filename);
end
